"""Output validation (spec §4.8): structural, comparative, deterministic.

Loadability and numerical-vs-reference checks live in the bench harness
(they need llama.cpp binaries); this module is the pure-Python half that
CI can run on any machine.
"""
import os

import numpy as np
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType, GGUFReader, ReaderField, ReaderTensor

from .gguf_io import ALIGN


class _MetadataOnlyReader(GGUFReader):
    """GGUFReader that never materializes a tensor's raw data.

    ``GGUFReader.__init__`` memmaps and reshapes every tensor's actual
    bytes as its last construction step, so a file truncated only in its
    tensor-data section (header, KV block and tensor-info table all
    intact) raises ``ValueError`` from numpy's reshape *during
    construction* — before ``structural_check`` ever gets to run its own
    offset/size arithmetic. That defeats the truncation check for the
    exact case it exists to catch. ``structural_check`` never reads
    ``ReaderTensor.data``, so this subclass overrides only the final
    data-materializing step of construction, reusing everything else
    (header, KV, tensor-info parsing) verbatim, so tensor metadata
    (name, type, shape, offset) is available even for a truncated file.
    A malformed header, unsupported version, or bad alignment field
    still raises from the base class before this override is reached —
    that is genuine corruption, not truncation.
    """

    def _build_tensors(self, start_offs: int, fields: list[ReaderField]) -> None:
        tensors = []
        for field in fields:
            _name_len, name_data, _n_dims, dims, raw_dtype, offset_tensor = field.parts
            tensor_name = str(bytes(name_data), encoding="utf-8")
            ggml_type = GGMLQuantizationType(raw_dtype[0])
            n_elems = int(np.prod(dims))
            data_offs = int(start_offs + offset_tensor[0])
            try:
                block_size, type_size = GGML_QUANT_SIZES[ggml_type]
                n_bytes = n_elems * type_size // block_size
            except KeyError:
                # Unsupported quant type: leave a sentinel so
                # structural_check's own loop reports this per-tensor
                # instead of construction crashing on it.
                n_bytes = -1
            tensors.append(ReaderTensor(
                name=tensor_name, tensor_type=ggml_type, shape=dims,
                n_elements=n_elems, n_bytes=n_bytes, data_offset=data_offs,
                data=np.empty(0, dtype=np.uint8), field=field,
            ))
        self.tensors = tensors


def _chunks_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Compare two arrays 64 MiB at a time (works on files bigger than RAM)."""
    x, y = a.reshape(-1), b.reshape(-1)
    if x.dtype != y.dtype or x.size != y.size:
        return False
    step = max(1, (64 << 20) // x.itemsize)
    return all(np.array_equal(x[i:i + step], y[i:i + step])
               for i in range(0, x.size, step))


def compare_gguf(a: str, b: str) -> list[str]:
    """Tensor-for-tensor comparison; empty list means identical.

    Unlike ``structural_check``, a construction failure here raises
    rather than becoming a message: this function needs real tensor
    bytes to compare (``ReaderTensor.data``), so a file that can't be
    opened for its data can't be compared at all — there is no partial
    "here's what's wrong" result to return, only "this can't run."
    """
    try:
        ra, rb = GGUFReader(a), GGUFReader(b)
    except Exception as exc:
        raise RuntimeError(f"cannot open GGUF for comparison: {exc}") from exc
    ta = {t.name: t for t in ra.tensors}
    tb = {t.name: t for t in rb.tensors}
    msgs: list[str] = []
    if ta.keys() != tb.keys():
        msgs.append(f"tensor name sets differ: {sorted(ta.keys() ^ tb.keys())}")
    for name in sorted(ta.keys() & tb.keys()):
        x, y = ta[name], tb[name]
        if x.tensor_type != y.tensor_type:
            msgs.append(f"{name}: type {x.tensor_type.name} != {y.tensor_type.name}")
        elif not _chunks_equal(x.data, y.data):
            msgs.append(f"{name}: byte mismatch")
    return msgs


def structural_check(path: str) -> list[str]:
    """Offsets, alignment, declared sizes vs the file's actual length.

    ``t.data_offset`` from GGUFReader is an absolute file offset (data
    section start + the tensor's relative offset); verified against
    ~/models/qwen3-0.6b-bf16.gguf and the tests/conftest.py tiny-GGUF
    fixture that both the data section start and every tensor's absolute
    offset are already 32-byte aligned, so no relative adjustment is
    needed here.
    """
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise RuntimeError(f"cannot access {path}: {exc}") from exc
    try:
        reader = _MetadataOnlyReader(path)
    except Exception as exc:
        # _MetadataOnlyReader tolerates a truncated tensor-data section
        # (see its docstring); anything that still raises here failed
        # earlier in GGUFReader.__init__ — bad magic, unsupported
        # version, or a bad alignment field — none of which is a
        # truncation, so the message says so rather than reusing that
        # word.
        return [f"file is not a readable GGUF (bad header, version, "
                f"or alignment): {exc}"]
    msgs: list[str] = []
    if not reader.tensors:
        msgs.append("no tensors in file")
    for t in reader.tensors:
        try:
            blk, tsz = GGML_QUANT_SIZES[t.tensor_type]
        except KeyError:
            msgs.append(f"{t.name}: unsupported tensor type {t.tensor_type!r}")
            continue
        n_elements = int(np.prod([int(d) for d in t.shape]))
        if n_elements % blk:
            msgs.append(f"{t.name}: {n_elements} elements is not a multiple "
                        f"of block size {blk}")
        expect = n_elements // blk * tsz
        if int(t.n_bytes) != expect:
            msgs.append(f"{t.name}: declared {t.n_bytes} B, format implies {expect} B")
        if int(t.data_offset) % ALIGN:
            msgs.append(f"{t.name}: data_offset {t.data_offset} is not "
                        f"{ALIGN}-byte aligned")
        end = int(t.data_offset) + int(t.n_bytes)
        if end > size:
            msgs.append(f"{t.name}: truncated — needs {end} B, file is {size} B")
    return msgs
