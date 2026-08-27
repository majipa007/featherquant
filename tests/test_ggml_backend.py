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


def test_threads_do_not_change_bytes(lib):
    # Rows are independent: splitting them across worker threads must be
    # byte-identical, including an uneven split (37 rows / 4 workers) and
    # more workers than rows.
    rng = np.random.default_rng(2)
    x = rng.standard_normal(37 * 256).astype(np.float32)
    for tt in (GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q6_K,
               GGMLQuantizationType.Q8_0):
        one = lib.quantize_rows(x, tt, 256, threads=1)
        assert lib.quantize_rows(x, tt, 256, threads=4) == one
        assert lib.quantize_rows(x, tt, 256, threads=64) == one


def test_q8_0_matches_numpy_kernel(lib):
    # ggml's Q8_0 reference kernel and featherquant's numpy kernel must agree
    # byte-for-byte, so either can serve the q8_0 path.
    from featherquant.q8_0 import quantize_q8_0
    rng = np.random.default_rng(3)
    x = rng.standard_normal(16 * 64).astype(np.float32)
    assert lib.quantize_rows(x, GGMLQuantizationType.Q8_0, 64, threads=3) == quantize_q8_0(x)


def test_default_lib_path_resolution(monkeypatch, tmp_path):
    from featherquant.ggml_backend import default_lib_path
    monkeypatch.setenv("GGML_LIB", "/explicit/libggml-base.so")
    assert default_lib_path() == "/explicit/libggml-base.so"
    monkeypatch.delenv("GGML_LIB")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert default_lib_path() is None
    hit = tmp_path / "llama.cpp" / "build-cpu" / "bin" / "libggml-base.so"
    hit.parent.mkdir(parents=True)
    hit.write_bytes(b"")
    assert default_lib_path() == str(hit)


def test_load_ggml_without_any_lib_names_the_search(monkeypatch, tmp_path):
    from featherquant.ggml_backend import load_ggml
    monkeypatch.delenv("GGML_LIB", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(RuntimeError, match="GGML_LIB"):
        load_ggml()
