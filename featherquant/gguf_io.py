"""GGUF I/O: sliced tensor reads and incremental streamed output.

Reads use an explicit file handle with seek+readinto into caller-owned
buffers — never the GGUFReader memmap — so resident memory is exactly the
buffer the caller sized, not whatever pages mmap happened to touch.
GGUFReader is used for metadata only.
"""
import numpy as np
from gguf import (GGML_QUANT_VERSION, GGMLQuantizationType, GGUFReader,
                  GGUFValueType, GGUFWriter)

from .q8_0 import bf16_to_f32

# Default GGUF tensor-data alignment (bytes).
ALIGN = 32

# Byte size per element for the source types we support.
ITEMSIZE = {
    GGMLQuantizationType.F32: 4,
    GGMLQuantizationType.F16: 2,
    GGMLQuantizationType.BF16: 2,
}


class TensorSource:
    """Read-only view of a GGUF file with row-sliced tensor access."""

    def __init__(self, path: str):
        try:
            # GGUFReader parses metadata via memmap; we never touch its
            # tensor data pages.
            self.reader = GGUFReader(path)
        except Exception as exc:
            raise RuntimeError(f"failed to parse GGUF metadata from {path}: {exc}") from exc
        self.tensors = list(self.reader.tensors)
        try:
            # Separate handle for explicit, budget-controlled data reads.
            self.f = open(path, "rb")
        except OSError as exc:
            raise RuntimeError(f"failed to open {path} for reading: {exc}") from exc

    def close(self):
        """Release the data file handle."""
        try:
            self.f.close()
        except OSError:
            pass  # closing is best-effort; nothing actionable on failure

    def read_rows_f32(self, tensor, row_start: int, n_rows: int,
                      buf: bytearray) -> np.ndarray:
        """Read ``n_rows`` rows starting at ``row_start`` into ``buf``.

        Returns a float32 array of length ``n_rows * ne0``.  The returned
        array may alias ``buf`` (F32 source) or be a fresh converted copy
        (F16/BF16 source) — callers must consume it before the next read.
        """
        ne0 = int(tensor.shape[0])  # ggml order: shape[0] = contiguous row length
        isz = ITEMSIZE[tensor.tensor_type]
        nb = n_rows * ne0 * isz
        try:
            self.f.seek(int(tensor.data_offset) + row_start * ne0 * isz)
            got = self.f.readinto(memoryview(buf)[:nb])
        except OSError as exc:
            raise RuntimeError(f"read error on tensor {tensor.name}: {exc}") from exc
        if got != nb:
            raise RuntimeError(f"short read on {tensor.name}: {got}/{nb} bytes")
        n = n_rows * ne0
        tt = tensor.tensor_type
        if tt == GGMLQuantizationType.F32:
            return np.frombuffer(buf, np.float32, count=n)
        if tt == GGMLQuantizationType.F16:
            return np.frombuffer(buf, np.float16, count=n).astype(np.float32)
        # Remaining supported type is BF16: reinterpret bits then widen.
        return bf16_to_f32(np.frombuffer(buf, np.uint16, count=n))

    def read_raw(self, tensor, byte_start: int, nbytes: int,
                 buf: bytearray) -> memoryview:
        """Read raw tensor bytes (for verbatim copies of unquantized tensors)."""
        try:
            self.f.seek(int(tensor.data_offset) + byte_start)
            got = self.f.readinto(memoryview(buf)[:nbytes])
        except OSError as exc:
            raise RuntimeError(f"read error on tensor {tensor.name}: {exc}") from exc
        if got != nbytes:
            raise RuntimeError(f"short read on {tensor.name}: {got}/{nbytes} bytes")
        return memoryview(buf)[:nbytes]
