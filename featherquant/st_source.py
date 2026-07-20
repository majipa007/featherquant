"""Safetensors input: minimal shard parser and row-sliced reads.

The safetensors format is trivial — an 8-byte little-endian header length,
a JSON header of ``{name: {dtype, shape, data_offsets}}``, then raw
row-major tensor data. Parsing it directly (no ``safetensors`` dependency)
gives exact control over reads: tensor bytes move through caller-owned
buffers just like the GGUF path, never through mmap or full-tensor loads.
"""
import json
import struct
from dataclasses import dataclass
from typing import BinaryIO

import numpy as np

from .q8_0 import bf16_to_f32

# Byte size per element for supported safetensors dtypes.
ST_ITEMSIZE = {"F32": 4, "F16": 2, "BF16": 2}


@dataclass(frozen=True)
class StTensor:
    """One tensor as declared in a shard header (offsets are data-relative)."""
    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int


def parse_shard_header(path: str) -> tuple[dict[str, StTensor], int]:
    """Parse one shard's JSON header.

    Returns ({name: StTensor}, data_base) where data_base is the absolute
    file offset at which tensor data begins.
    """
    try:
        with open(path, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(hlen))
    except (OSError, struct.error, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot parse safetensors header {path}: {exc}") from exc
    tensors: dict[str, StTensor] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        try:
            dtype = info["dtype"]
            if dtype not in ST_ITEMSIZE:
                raise RuntimeError(
                    f"unsupported safetensors dtype {dtype!r} for {name} "
                    f"in {path} (supported: {sorted(ST_ITEMSIZE)})")
            start, end = info["data_offsets"]
            tensors[name] = StTensor(name, dtype, tuple(info["shape"]),
                                     int(start), int(end))
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"malformed entry {name!r} in {path}: {exc}") from exc
    return tensors, 8 + hlen


def read_st_rows(f: BinaryIO, t: StTensor, data_base: int, row_start: int,
                 n_rows: int, buf: bytearray) -> np.ndarray:
    """Read ``n_rows`` rows (last-dim-major) into ``buf``; return float32.

    Rows run along the LAST dim (row-major layout), which matches GGUF's
    contiguous ne0 dimension after shape reversal.
    """
    row_len = t.shape[-1] if t.shape else 1
    isz = ST_ITEMSIZE[t.dtype]
    nb = n_rows * row_len * isz
    try:
        f.seek(data_base + t.start + row_start * row_len * isz)
        got = f.readinto(memoryview(buf)[:nb])
    except OSError as exc:
        raise RuntimeError(f"read error on tensor {t.name}: {exc}") from exc
    if got != nb:
        raise RuntimeError(f"short read on {t.name}: {got}/{nb} bytes")
    n = n_rows * row_len
    if t.dtype == "F32":
        return np.frombuffer(buf, np.float32, count=n)
    if t.dtype == "F16":
        return np.frombuffer(buf, np.float16, count=n).astype(np.float32)
    return bf16_to_f32(np.frombuffer(buf, np.uint16, count=n))
