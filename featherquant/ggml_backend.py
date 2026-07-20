"""ctypes bridge to llama.cpp's ggml quantization kernels.

K-quant math (iterative scale search, 6-bit scale packing) is subtle and
float-summation-order-sensitive; reimplementing it byte-exact in numpy is
high-risk. Instead we call ``ggml_quantize_chunk()`` from the shared library
llama.cpp already built — byte parity by construction. The call operates on
the caller's row chunk only, so the memory contract is unchanged.
"""
import ctypes
import functools
import os

import numpy as np
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType

# Default location of the user's llama.cpp CPU build.
_DEFAULT_LIB = "/home/sukuna/llama.cpp/build-cpu/bin/libggml-base.so"


class GgmlLib:
    """Loaded libggml exposing the one entry point featherquant needs."""

    def __init__(self, path: str):
        try:
            self._lib = ctypes.CDLL(path)
        except OSError as exc:
            raise RuntimeError(
                f"cannot load ggml library {path!r}: {exc}. Build llama.cpp "
                "or point --ggml-lib / $GGML_LIB at libggml-base.so"
            ) from exc
        try:
            fn = self._lib.ggml_quantize_chunk
        except AttributeError as exc:
            raise RuntimeError(
                f"{path!r} does not export ggml_quantize_chunk"
            ) from exc
        # size_t ggml_quantize_chunk(enum ggml_type, const float *src, void *dst,
        #                            int64_t start, int64_t nrows,
        #                            int64_t n_per_row, const float *imatrix)
        fn.restype = ctypes.c_size_t
        fn.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float),
                       ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                       ctypes.c_int64, ctypes.POINTER(ctypes.c_float)]
        self._quantize_chunk = fn

    def quantize_rows(self, x: np.ndarray, ggml_type: GGMLQuantizationType,
                      n_per_row: int) -> bytes:
        """Quantize 1-D float32 rows to packed ggml bytes (imatrix-free).

        Rows are independent inside ggml_quantize_chunk, so calling this per
        chunk is byte-identical to quantizing the whole tensor at once.
        """
        blk, tsz = GGML_QUANT_SIZES[ggml_type]
        if (x.dtype != np.float32 or x.ndim != 1 or x.size % n_per_row != 0
                or n_per_row % blk != 0):
            raise ValueError(
                f"need 1-D float32, size % {n_per_row} == 0, "
                f"n_per_row % {blk} == 0; got dtype={x.dtype} size={x.size}")
        nrows = x.size // n_per_row
        out = ctypes.create_string_buffer(nrows * (n_per_row // blk) * tsz)
        x = np.ascontiguousarray(x)
        written = self._quantize_chunk(
            int(ggml_type), x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out, 0, nrows, n_per_row, None)
        if written != len(out):
            raise RuntimeError(
                f"ggml_quantize_chunk wrote {written} B, expected {len(out)} B")
        return out.raw


@functools.lru_cache(maxsize=4)
def load_ggml(path: str | None = None) -> GgmlLib:
    """Load (and cache) the ggml library from path, $GGML_LIB, or the default."""
    return GgmlLib(path or os.environ.get("GGML_LIB", _DEFAULT_LIB))
