"""Unit tests for the Q8_0 kernel (llama.cpp byte compatibility)."""
import numpy as np
import pytest

from featherquant.q8_0 import BLOCK, TYPE_SIZE, bf16_to_f32, dequantize_q8_0, quantize_q8_0


def test_constants():
    assert BLOCK == 32 and TYPE_SIZE == 34


def test_known_block():
    # amax = 127 -> d = 1.0 exactly; -63.5 must round AWAY from zero (llama.cpp roundf)
    x = np.zeros(32, np.float32)
    x[0], x[1] = 127.0, -63.5
    raw = quantize_q8_0(x)
    assert len(raw) == TYPE_SIZE
    d = np.frombuffer(raw[:2], np.float16)[0]
    q = np.frombuffer(raw[2:], np.int8)
    assert d == np.float16(1.0)
    assert q[0] == 127 and q[1] == -64
    assert not q[2:].any()


def test_zero_block():
    raw = quantize_q8_0(np.zeros(32, np.float32))
    assert raw == b"\x00" * TYPE_SIZE


def test_roundtrip_error_bound():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1024).astype(np.float32)
    y = dequantize_q8_0(quantize_q8_0(x))
    # per-block error <= ~0.5*d plus fp16 scale rounding (~127*d*2^-11)
    d = np.abs(x).reshape(-1, BLOCK).max(axis=1) / 127.0
    assert np.all(np.abs(x - y).reshape(-1, BLOCK) <= 0.6 * d[:, None] + 1e-7)


def test_chunked_equals_full():
    # blocks are independent: quantizing in arbitrary 32-multiple chunks
    # must be byte-identical to one-shot quantization
    rng = np.random.default_rng(1)
    x = rng.standard_normal(32 * 100).astype(np.float32)
    full = quantize_q8_0(x)
    step = 32 * 7
    parts = b"".join(quantize_q8_0(x[i:i + step]) for i in range(0, x.size, step))
    assert parts == full


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        quantize_q8_0(np.zeros(33, np.float32))
    with pytest.raises(ValueError):
        quantize_q8_0(np.zeros(32, np.float64))


def test_bf16_to_f32():
    f = np.array([1.0, -2.5, 0.0, 15.75], np.float32)  # all exactly representable in bf16
    raw = (f.view(np.uint32) >> 16).astype(np.uint16)
    assert np.array_equal(bf16_to_f32(raw), f)
