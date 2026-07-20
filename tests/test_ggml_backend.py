"""Tests for the ctypes ggml quantization backend."""
import numpy as np
import pytest
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType

from featherquant.ggml_backend import load_ggml


@pytest.fixture(scope="module")
def lib():
    try:
        return load_ggml()
    except RuntimeError as exc:
        pytest.skip(f"libggml not available: {exc}")


def test_q4_k_size_and_determinism(lib):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4 * 256).astype(np.float32)
    out = lib.quantize_rows(x, GGMLQuantizationType.Q4_K, 256)
    blk, tsz = GGML_QUANT_SIZES[GGMLQuantizationType.Q4_K]
    assert len(out) == x.size // blk * tsz
    assert out == lib.quantize_rows(x, GGMLQuantizationType.Q4_K, 256)


def test_q6_k_size(lib):
    x = np.zeros(2 * 256, np.float32)
    out = lib.quantize_rows(x, GGMLQuantizationType.Q6_K, 256)
    blk, tsz = GGML_QUANT_SIZES[GGMLQuantizationType.Q6_K]
    assert len(out) == x.size // blk * tsz


def test_chunked_equals_full(lib):
    # rows are independent in ggml_quantize_chunk: chunking must not change bytes
    rng = np.random.default_rng(1)
    x = rng.standard_normal(8 * 256).astype(np.float32)
    full = lib.quantize_rows(x, GGMLQuantizationType.Q4_K, 256)
    parts = b"".join(
        lib.quantize_rows(x[i * 256:(i + 3) * 256], GGMLQuantizationType.Q4_K, 256)
        for i in (0, 3, 6)
    )
    assert parts == full


def test_rejects_bad_input(lib):
    with pytest.raises(ValueError):
        lib.quantize_rows(np.zeros(100, np.float32), GGMLQuantizationType.Q4_K, 256)
    with pytest.raises(ValueError):
        lib.quantize_rows(np.zeros(256, np.float64), GGMLQuantizationType.Q4_K, 256)
