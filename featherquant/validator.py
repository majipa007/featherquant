"""Output validation (spec §4.8): structural, comparative, deterministic.

Loadability and numerical-vs-reference checks live in the bench harness
(they need llama.cpp binaries); this module is the pure-Python half that
CI can run on any machine.
"""
import os

import numpy as np
from gguf import GGML_QUANT_SIZES, GGUFReader

from .gguf_io import ALIGN


def _chunks_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Compare two arrays 64 MiB at a time (works on files bigger than RAM)."""
    x, y = a.reshape(-1), b.reshape(-1)
    if x.dtype != y.dtype or x.size != y.size:
        return False
    step = max(1, (64 << 20) // x.itemsize)
    return all(np.array_equal(x[i:i + step], y[i:i + step])
               for i in range(0, x.size, step))


def compare_gguf(a: str, b: str) -> list[str]:
    """Tensor-for-tensor comparison; empty list means identical."""
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
        reader = GGUFReader(path)
    except ValueError as exc:
        # GGUFReader memmaps and reshapes every tensor's data eagerly at
        # construction time, so a truncated (or otherwise corrupt)
        # tensor-data section raises here rather than surfacing as an
        # offset/size mismatch in the per-tensor checks below.
        return [f"file truncated or corrupt: {exc}"]
    except Exception as exc:
        raise RuntimeError(f"cannot open {path}: {exc}") from exc
    msgs: list[str] = []
    if not reader.tensors:
        msgs.append("no tensors in file")
    for t in reader.tensors:
        blk, tsz = GGML_QUANT_SIZES[t.tensor_type]
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
