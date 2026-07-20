"""Budget-planned streaming quantization loop.

Working set per chunk = source read buffer + numpy temporaries inside
``quantize_q8_0`` + packed output.  The planner sizes the largest row chunk
whose estimated cost fits ``max_ram - rss_at_start - RESERVE``, and RSS is
sampled after every tensor for the report.
"""
import json
import os
import sys
import time

from gguf import GGMLQuantizationType, LlamaFileType

from .gguf_io import ITEMSIZE, IncrementalWriter, TensorSource
from .q8_0 import BLOCK, TYPE_SIZE, quantize_q8_0

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


def target_type(t):
    """Quantization rule mirroring llama-quantize Q8_0 defaults.

    2-D+ float tensors whose contiguous row length divides by 32 become
    Q8_0; everything else keeps its source type and bytes.
    """
    if len(t.shape) >= 2 and t.tensor_type in ITEMSIZE and int(t.shape[0]) % BLOCK == 0:
        return GGMLQuantizationType.Q8_0
    return t.tensor_type


def q8_0_nbytes(n_elements: int) -> int:
    """Packed Q8_0 size is deterministic: 34 bytes per 32 elements."""
    return n_elements // BLOCK * TYPE_SIZE


def per_row_cost(ne0: int, itemsize: int) -> int:
    """Estimated working-set bytes to process one row of length ne0."""
    # ponytail: crude static model — read buf + ~5 float32-sized numpy
    # temporaries inside quantize_q8_0 + packed row; replaced by measured
    # feedback in the Phase 3 adaptive controller.
    return ne0 * itemsize + 20 * ne0 + (ne0 // BLOCK) * TYPE_SIZE


def quantize_model(src, dst, max_ram, report=None, _force_chunk_rows=None):
    """Quantize ``src`` GGUF to Q8_0 at ``dst`` with peak RSS <= ``max_ram``.

    Returns a stats dict; optionally writes it as JSON to ``report``.
    ``_force_chunk_rows`` is a test hook that overrides the planner.
    """
    t0 = time.monotonic()
    source = TensorSource(src)
    # Everything not yet allocated is available for per-chunk working set.
    working = max_ram - rss_bytes() - RESERVE
    if working <= 0:
        sys.exit(f"budget {max_ram} B too small: runtime already at "
                 f"{rss_bytes()} B RSS + {RESERVE} B reserve")
    stats = {"max_ram": max_ram, "reserve": RESERVE, "working_budget": working,
             "bytes_read": 0, "bytes_written": 0, "peak_rss": rss_bytes(),
             "budget_violations": 0, "chunks": 0}

    # Plan first: every output size/offset is known before any data moves.
    plan = []
    for t in source.tensors:
        tt = target_type(t)
        nbytes = q8_0_nbytes(int(t.n_elements)) if tt != t.tensor_type else int(t.n_bytes)
        plan.append((t, tt, nbytes))

    iw = IncrementalWriter(dst, source.reader, LlamaFileType.MOSTLY_Q8_0)
    for t, tt, nbytes in plan:
        iw.add_tensor_info(t.name, t.shape, nbytes, tt)
    iw.begin_data()

    try:
        for t, tt, _ in plan:
            iw.begin_tensor()
            if tt != t.tensor_type:
                _stream_quantize(source, iw, t, working, stats, _force_chunk_rows)
            else:
                _stream_copy(source, iw, t, stats)
            # Telemetry: sample RSS after each tensor; count budget breaches.
            r = rss_bytes()
            stats["peak_rss"] = max(stats["peak_rss"], r)
            if r > max_ram:
                stats["budget_violations"] += 1
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


def _stream_quantize(source, iw, t, working, stats, force_rows):
    """Quantize one tensor to Q8_0 in row chunks sized by the budget."""
    ne0 = int(t.shape[0])
    rows = int(t.n_elements) // ne0
    isz = ITEMSIZE[t.tensor_type]
    chunk = force_rows or min(rows, working // per_row_cost(ne0, isz))
    if chunk < 1:
        # Report the minimum feasible budget instead of thrashing.
        sys.exit(f"budget too small for tensor {t.name}: one row needs about "
                 f"{per_row_cost(ne0, isz)} B working set on top of runtime + reserve")
    # One reusable read buffer for the whole tensor.
    buf = bytearray(chunk * ne0 * isz)
    for r0 in range(0, rows, chunk):
        n = min(chunk, rows - r0)
        packed = quantize_q8_0(source.read_rows_f32(t, r0, n, buf))
        iw.write(packed)
        stats["bytes_read"] += n * ne0 * isz
        stats["bytes_written"] += len(packed)
        stats["chunks"] += 1


def _stream_copy(source, iw, t, stats):
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
