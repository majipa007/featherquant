"""Budget-planned streaming quantization loop.

Working set per chunk = source read buffer + numpy temporaries inside
``quantize_q8_0`` + packed output.  The planner sizes the largest row chunk
whose estimated cost fits ``max_ram - rss_at_start - RESERVE``, and RSS is
sampled after every tensor for the report.
"""
import gc
import json
import os
import sys
import time
from typing import Any

from gguf import GGML_QUANT_SIZES, GGMLQuantizationType
from gguf.gguf_reader import ReaderTensor

from .controller import BlockController
from .formats import FORMATS
from .ggml_backend import GgmlLib, load_ggml
from .gguf_io import ALIGN, ITEMSIZE, IncrementalWriter, ResumeWriter, TensorSource
from .manifest import Manifest, TensorEntry, sha256_file_region
from .q8_0 import BLOCK, TYPE_SIZE, quantize_q8_0
from .st_source import SafetensorsSource, StReaderTensor

# A plannable tensor from either source (duck-compatible fields).
AnyTensor = ReaderTensor | StReaderTensor

# ponytail: fixed 64 MiB safety reserve; adaptive governor is Phase 3.
RESERVE = 64 << 20
# Chunk size for verbatim byte copies of unquantized tensors.
COPY_CHUNK = 8 << 20


def rss_bytes() -> int:
    """Current resident set size of this process, in bytes (Linux)."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError) as exc:
        raise RuntimeError(f"cannot read RSS from /proc/self/statm: {exc}") from exc


def packed_nbytes(n_elements: int, ggml_type: GGMLQuantizationType) -> int:
    """Packed size for any ggml type: deterministic from element count."""
    blk, tsz = GGML_QUANT_SIZES[ggml_type]
    return n_elements // blk * tsz


def q8_0_nbytes(n_elements: int) -> int:
    """Packed Q8_0 size is deterministic: 34 bytes per 32 elements."""
    return n_elements // BLOCK * TYPE_SIZE


def per_row_cost(ne0: int, itemsize: int,
                 out_type: GGMLQuantizationType = GGMLQuantizationType.Q8_0) -> int:
    """Estimated working-set bytes to process one row of length ne0."""
    # ponytail: crude static model — read buf + ~5 float32-sized numpy
    # temporaries inside the kernel + packed row; replaced by measured
    # feedback in the Phase 3 adaptive controller.
    return ne0 * itemsize + 20 * ne0 + packed_nbytes(ne0, out_type)


def _align(n: int) -> int:
    """Round n up to the GGUF tensor alignment boundary."""
    return (n + ALIGN - 1) // ALIGN * ALIGN


def _prepare_resume(src: str, dst: str, manifest_path: str,
                    plan: list[tuple[AnyTensor, GGMLQuantizationType, int]],
                    fmt: str) -> tuple[Manifest, int]:
    """Validate a saved manifest against the re-planned job.

    Returns (manifest, index of first tensor to (re)write). Raises
    RuntimeError on any identity/plan/header mismatch — never resumes into
    a file that does not provably match the interrupted run.
    """
    man = Manifest.load(manifest_path)
    if man.status == "complete":
        sys.exit(f"{dst} is already complete per {manifest_path}; "
                 "nothing to resume")
    try:
        st = os.stat(src)
    except OSError as exc:
        raise RuntimeError(f"cannot stat source {src}: {exc}") from exc
    if (man.source_path != os.path.abspath(src)
            or man.source_size != st.st_size
            or man.source_mtime_ns != st.st_mtime_ns):
        raise RuntimeError(
            f"source changed since interrupted run: manifest recorded "
            f"{man.source_path} size={man.source_size} "
            f"mtime={man.source_mtime_ns}, found size={st.st_size} "
            f"mtime={st.st_mtime_ns}")
    if man.config != {"fmt": fmt}:
        raise RuntimeError(f"config changed: manifest {man.config}, "
                           f"current fmt={fmt!r}")
    if len(plan) != len(man.tensors):
        raise RuntimeError("re-planned tensor count differs from manifest")
    off = man.header_end
    for e, (t, tt, nbytes) in zip(man.tensors, plan):
        if (e.name != t.name or e.ggml_type != int(tt)
                or e.nbytes != nbytes or e.offset != off):
            raise RuntimeError(f"re-planned job diverges from manifest at "
                               f"{t.name} — refusing to resume")
        off = _align(off + nbytes)
    if sha256_file_region(dst, 0, man.header_end) != man.header_sha256:
        raise RuntimeError(f"header of {dst} does not match manifest — "
                           "output file was modified or belongs to another job")
    # Verify committed tensors in order; first bad/missing hash is the
    # resume point (a corrupt middle tensor forces rewrite from there on).
    start_i = len(man.tensors)
    for i, e in enumerate(man.tensors):
        if e.sha256 is None:
            start_i = i
            break
        if sha256_file_region(dst, e.offset, e.nbytes) != e.sha256:
            e.sha256 = None
            start_i = i
            break
    return man, start_i


def quantize_model(src: str, dst: str, max_ram: int, report: str | None = None,
                   fmt: str = "q8_0", ggml_lib: str | None = None,
                   manifest_path: str | None = None, resume: bool = False,
                   adaptive: bool = True, vocab_gguf: str | None = None,
                   _force_chunk_rows: int | None = None,
                   _fail_after: int | None = None) -> dict[str, Any]:
    """Quantize ``src`` GGUF to format ``fmt`` at ``dst``, peak RSS <= ``max_ram``.

    A sidecar manifest (``manifest_path``, default ``dst + ".manifest.json"``)
    is checkpointed atomically after every committed tensor for crash
    recovery. Returns a stats dict; optionally writes it as JSON to
    ``report``. ``_force_chunk_rows`` is a test hook that overrides the
    planner.
    """
    t0 = time.monotonic()
    if fmt not in FORMATS:
        sys.exit(f"unknown format {fmt!r}; supported: {sorted(FORMATS)}")
    spec = FORMATS[fmt]
    lib: GgmlLib | None = None
    if spec.needs_ggml:
        try:
            lib = load_ggml(ggml_lib)
        except RuntimeError as exc:
            sys.exit(f"format {fmt!r} needs the ggml library: {exc}")
    # Source dispatch: a directory is a sharded-safetensors HF checkpoint,
    # a file is a GGUF.
    source: TensorSource | SafetensorsSource
    if os.path.isdir(src):
        if not vocab_gguf:
            sys.exit(f"{src} is a safetensors directory: pass --vocab-gguf "
                     "(create one with scripts/make_vocab_gguf.sh)")
        source = SafetensorsSource(src, vocab_gguf)
    else:
        source = TensorSource(src)

    # Layer count feeds the layer-dependent type rules (0 when absent, e.g.
    # in synthetic test fixtures — rules degrade to their layer-free parts).
    arch = str(source.reader.fields["general.architecture"].contents())
    bc_field = source.reader.fields.get(f"{arch}.block_count")
    n_layers = int(bc_field.contents()) if bc_field is not None else 0

    # Plan first: every output size/offset is known before any data moves.
    plan: list[tuple[AnyTensor, GGMLQuantizationType, int]] = []
    for t in source.tensors:
        tt = spec.tensor_type(t, n_layers)
        nbytes = (packed_nbytes(int(t.n_elements), tt)
                  if tt != t.tensor_type else int(t.n_bytes))
        plan.append((t, tt, nbytes))

    stats: dict[str, Any] = {"max_ram": max_ram, "reserve": RESERVE,
                             "working_budget": 0, "bytes_read": 0,
                             "bytes_written": 0, "peak_rss": rss_bytes(),
                             "budget_violations": 0, "chunks": 0}

    if manifest_path is None:
        manifest_path = dst + ".manifest.json"
    resuming = resume and os.path.exists(manifest_path)

    iw: IncrementalWriter | ResumeWriter
    if resuming:
        man, start_i = _prepare_resume(src, dst, manifest_path, plan, fmt)
        entries = man.tensors
        if start_i < len(entries):
            pos = entries[start_i].offset
        else:  # everything committed; only the final status save was lost
            pos = _align(entries[-1].offset + entries[-1].nbytes) if entries \
                else man.header_end
        iw = ResumeWriter(dst, pos)
    else:
        start_i = 0
        iw = IncrementalWriter(dst, source.reader, spec.file_type)

    # From here on the writer holds an open file: every exit path must close
    # it (close() is lifecycle-safe even before begin_data()).
    try:
        # KV metadata is fully consumed (writer ctor copied it; plan and
        # layer rules are final). Drop GGUFReader's huge KV object graph —
        # ~0.5 GiB on 150k-token vocabs — BEFORE sizing the working budget,
        # so streaming gets that memory back. The startup transient itself
        # is unavoidable with GGUFReader and defines the minimum feasible
        # budget; it is recorded in the stats for transparency.
        rss_metadata_peak = rss_bytes()
        source.release_metadata()
        gc.collect()
        working = max_ram - rss_bytes() - RESERVE
        if working <= 0:
            sys.exit(f"budget {max_ram} B too small: runtime still at "
                     f"{rss_bytes()} B RSS after metadata release "
                     f"+ {RESERVE} B reserve")
        stats.update(working_budget=working, rss_metadata_peak=rss_metadata_peak)

        if not resuming:
            assert isinstance(iw, IncrementalWriter)
            for t, tt, nbytes in plan:
                iw.add_tensor_info(t.name, t.shape, nbytes, tt)
            iw.begin_data()
            iw.flush()
            # Precompute every tensor's absolute output offset (same
            # alignment rule the writer applies) and open the manifest.
            header_end = _align(iw.tell())
            entries, off = [], header_end
            for t, tt, nbytes in plan:
                entries.append(TensorEntry(t.name, int(tt), off, nbytes, None))
                off = _align(off + nbytes)
            st = os.stat(src)
            man = Manifest(source_path=os.path.abspath(src),
                           source_size=st.st_size,
                           source_mtime_ns=st.st_mtime_ns, config={"fmt": fmt},
                           header_end=header_end, header_sha256="",
                           tensors=entries, status="in_progress")

        for i in range(start_i, len(plan)):
            t, tt, _ = plan[i]
            if _fail_after is not None and (i - start_i) >= _fail_after:
                raise RuntimeError("_fail_after test hook: simulated crash")
            iw.begin_tensor()
            if not resuming and i == 0:
                # Padding now exists: the header region [0, header_end) is
                # final. Hash it and write the initial checkpoint.
                iw.flush()
                man.header_sha256 = sha256_file_region(dst, 0, man.header_end)
                man.save(manifest_path)
            if iw.tell() != entries[i].offset:
                raise RuntimeError(
                    f"offset drift on {t.name}: file at {iw.tell()}, "
                    f"planned {entries[i].offset}")
            if tt != t.tensor_type:
                _stream_quantize(source, iw, t, tt, working, stats,
                                 _force_chunk_rows, lib, adaptive)
            else:
                _stream_copy(source, iw, t, stats)
            # Commit: flush data, hash the written region, checkpoint.
            iw.flush()
            entries[i].sha256 = sha256_file_region(
                dst, entries[i].offset, entries[i].nbytes)
            man.save(manifest_path)
            # Telemetry: sample RSS after each tensor; count budget breaches.
            r = rss_bytes()
            stats["peak_rss"] = max(stats["peak_rss"], r)
            if r > max_ram:
                stats["budget_violations"] += 1

        man.status = "complete"
        man.save(manifest_path)
    finally:
        iw.close()
        source.close()

    stats["elapsed_s"] = round(time.monotonic() - t0, 3)
    if report:
        try:
            with open(report, "w") as f:
                json.dump(stats, f, indent=2)
        except OSError as exc:
            raise RuntimeError(f"failed to write report {report}: {exc}") from exc
    return stats


def _stream_quantize(source: "TensorSource | SafetensorsSource",
                     iw: "IncrementalWriter | ResumeWriter",
                     t: AnyTensor, tt: GGMLQuantizationType, working: int,
                     stats: dict[str, Any], force_rows: int | None,
                     lib: GgmlLib | None, adaptive: bool = True) -> None:
    """Quantize one tensor to type ``tt`` in row chunks sized by the budget.

    With ``adaptive`` on, a BlockController refines the chunk size from
    measured RSS deltas; the static ``per_row_cost`` model is only the
    prior. Chunk boundaries never affect output bytes (rows and blocks are
    independent), so adaptation is a memory-behavior knob, not a
    correctness one.
    """
    ne0 = int(t.shape[0])
    rows = int(t.n_elements) // ne0
    isz = ITEMSIZE[t.tensor_type]
    est = per_row_cost(ne0, isz, tt)
    max_chunk = force_rows or min(rows, working // est)
    if max_chunk < 1:
        # Report the minimum feasible budget instead of thrashing.
        sys.exit(f"budget too small for tensor {t.name}: one row needs about "
                 f"{est} B working set on top of runtime + reserve")
    ctrl = (BlockController(working, est)
            if adaptive and force_rows is None else None)
    # One reusable read buffer, sized at the static maximum; the controller
    # may shrink chunks but never grows past the buffer (budget already
    # spent on it).
    buf = bytearray(max_chunk * ne0 * isz)
    max_ram = int(stats["max_ram"])
    seen_min, seen_max = rows + 1, 0
    r0 = 0
    while r0 < rows:
        if ctrl is not None:
            n = min(ctrl.next_rows(rows - r0), max_chunk)
        else:
            n = min(max_chunk, rows - r0)
        seen_min, seen_max = min(seen_min, n), max(seen_max, n)
        before = rss_bytes()
        x = source.read_rows_f32(t, r0, n, buf)
        if tt == GGMLQuantizationType.Q8_0:
            packed = quantize_q8_0(x)
        else:
            # K-quants: byte-exact kernels from the ggml shared library.
            assert lib is not None  # guaranteed by quantize_model for needs_ggml
            packed = lib.quantize_rows(x, tt, ne0)
        iw.write(packed)
        stats["bytes_read"] += n * ne0 * isz
        stats["bytes_written"] += len(packed)
        stats["chunks"] += 1
        if ctrl is not None:
            after = rss_bytes()
            ctrl.observe(n, before, after)
            if after > max_ram:
                stats["budget_violations"] += 1
                ctrl.violation()
            elif after > max_ram - RESERVE // 2:
                # Emergency headroom: reclaim garbage and back off.
                gc.collect()
                ctrl.violation()
        r0 += n
    if ctrl is not None:
        stats.setdefault("adaptive", {})[t.name] = {
            "chunk_rows_min": seen_min, "chunk_rows_max": seen_max,
            "per_row_final": round(ctrl.per_row, 1)}


def _stream_copy(source: "TensorSource | SafetensorsSource",
                 iw: "IncrementalWriter | ResumeWriter", t: AnyTensor,
                 stats: dict[str, Any]) -> None:
    """Copy an unquantized tensor verbatim in bounded chunks."""
    remaining = int(t.n_bytes)
    pos = 0
    buf = bytearray(min(remaining, COPY_CHUNK))
    while remaining:
        n = min(COPY_CHUNK, remaining)
        iw.write(source.read_raw(t, pos, n, buf))
        pos += n
        remaining -= n
        stats["bytes_read"] += n
        stats["bytes_written"] += n
        stats["chunks"] += 1
