#!/usr/bin/env python3
"""Tensor-by-tensor GGUF comparison: names, types, bytes.

Usage: python scripts/compare_reference.py reference.gguf featherquant.gguf
Exit 0 iff everything matches. Byte comparison is chunked so this also
works on models bigger than RAM (memmap slices, 64 MiB at a time).
"""
import sys

import numpy as np
from gguf import GGUFReader


def chunks_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Compare two (possibly memmapped) arrays 64 MiB at a time."""
    x, y = a.reshape(-1), b.reshape(-1)
    if x.dtype != y.dtype or x.size != y.size:
        return False
    step = max(1, (64 << 20) // x.itemsize)
    return all(np.array_equal(x[i:i + step], y[i:i + step])
               for i in range(0, x.size, step))


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    try:
        ra, rb = GGUFReader(sys.argv[1]), GGUFReader(sys.argv[2])
    except Exception as exc:
        sys.exit(f"failed to open GGUF: {exc}")
    ta = {t.name: t for t in ra.tensors}
    tb = {t.name: t for t in rb.tensors}
    bad = 0
    # Name sets must match exactly before comparing content.
    if ta.keys() != tb.keys():
        print("tensor name sets differ:", sorted(ta.keys() ^ tb.keys()))
        bad += 1
    for name in sorted(ta.keys() & tb.keys()):
        a, b = ta[name], tb[name]
        if a.tensor_type != b.tensor_type:
            print(f"{name}: type {a.tensor_type.name} != {b.tensor_type.name}")
            bad += 1
        elif not chunks_equal(a.data, b.data):
            print(f"{name}: byte mismatch")
            bad += 1
    print(f"{len(ta)} tensors, {bad} mismatches")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
