"""ctypes bridge to llama.cpp's ggml quantization kernels.

K-quant math (iterative scale search, 6-bit scale packing) is subtle and
float-summation-order-sensitive; reimplementing it byte-exact in numpy is
high-risk. Instead we call ``ggml_quantize_chunk()`` from the shared library
llama.cpp already built — byte parity by construction. The call operates on
the caller's row chunk only, so the memory contract is unchanged.

Rows are independent inside the kernel, so one chunk can be split across
worker threads (ctypes releases the GIL during the foreign call) without
changing a single output byte — the same parallelism ``llama-quantize`` uses.
"""
import ctypes
import functools
import glob
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType

# Where a stock ``cmake -B build*`` of llama.cpp leaves the library.
_SEARCH_GLOB = "~/llama.cpp/build*/bin/libggml-base.so"


def default_lib_path() -> str | None:
    """``$GGML_LIB`` if set, else the first match of ``~/llama.cpp/build*/bin/libggml-base.so``."""
    env = os.environ.get("GGML_LIB")
    if env:
        return env
    hits = sorted(glob.glob(os.path.expanduser(_SEARCH_GLOB)))
    return hits[0] if hits else None


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
                      n_per_row: int, threads: int = 1) -> bytes:
        """Quantize 1-D float32 rows to packed ggml bytes (imatrix-free).

        Rows are independent inside ggml_quantize_chunk, so calling this per
        chunk — and splitting the chunk across ``threads`` workers — is
        byte-identical to quantizing the whole tensor at once.
        """
        blk, tsz = GGML_QUANT_SIZES[ggml_type]
        if (x.dtype != np.float32 or x.ndim != 1 or x.size % n_per_row != 0
                or n_per_row % blk != 0):
            raise ValueError(
                f"need 1-D float32, size % {n_per_row} == 0, "
                f"n_per_row % {blk} == 0; got dtype={x.dtype} size={x.size}")
        nrows = x.size // n_per_row
        row_bytes = n_per_row // blk * tsz
        out = ctypes.create_string_buffer(nrows * row_bytes)
        x = np.ascontiguousarray(x)
        src_addr, dst_addr = x.ctypes.data, ctypes.addressof(out)

        def run(r0: int, r1: int) -> int:
            # Each worker gets its own src/dst window (start=0), so the split
            # never depends on how ggml maps ``start`` onto dst internally.
            src = ctypes.cast(src_addr + r0 * n_per_row * 4, ctypes.POINTER(ctypes.c_float))
            dst = ctypes.c_void_p(dst_addr + r0 * row_bytes)
            return int(self._quantize_chunk(int(ggml_type), src, dst, 0, r1 - r0,
                                            n_per_row, None))

        workers = max(1, min(int(threads), nrows))
        if workers == 1:
            written = run(0, nrows)
        else:
            bounds = [nrows * i // workers for i in range(workers + 1)]
            with ThreadPoolExecutor(max_workers=workers) as ex:
                written = sum(ex.map(run, bounds[:-1], bounds[1:]))
        if written != len(out):
            raise RuntimeError(
                f"ggml_quantize_chunk wrote {written} B, expected {len(out)} B")
        return out.raw


def load_ggml(path: str | None = None) -> GgmlLib:
    """Load (and cache) the ggml library from ``path``, ``$GGML_LIB`` or the llama.cpp build dir."""
    resolved = path or default_lib_path()
    if resolved is None:
        raise RuntimeError(
            "no ggml library found: set $GGML_LIB or pass --ggml-lib "
            f"(searched {_SEARCH_GLOB})")
    return _load(resolved)


@functools.lru_cache(maxsize=4)
def _load(path: str) -> GgmlLib:
    return GgmlLib(path)
