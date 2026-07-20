"""Tests for sliced GGUF reads (TensorSource)."""
import gguf
import numpy as np

from featherquant.gguf_io import ITEMSIZE, TensorSource
from tests.conftest import make_gguf


def test_read_rows_f16(tmp_path):
    a = (np.arange(8 * 64, dtype=np.float32).reshape(8, 64) / 16).astype(np.float16)
    p = tmp_path / "m.gguf"
    make_gguf(p, {"w": a})
    src = TensorSource(str(p))
    t = src.tensors[0]
    assert t.name == "w"
    assert int(t.shape[0]) == 64  # ne0 = contiguous row length
    isz = ITEMSIZE[t.tensor_type]
    buf = bytearray(3 * 64 * isz)
    x = src.read_rows_f32(t, 2, 3, buf)
    assert x.dtype == np.float32
    assert np.array_equal(x, a[2:5].astype(np.float32).ravel())
    src.close()


def test_read_rows_f32_and_raw(tmp_path):
    a = np.arange(4 * 32, dtype=np.float32).reshape(4, 32)
    p = tmp_path / "m.gguf"
    make_gguf(p, {"w": a})
    src = TensorSource(str(p))
    t = src.tensors[0]
    buf = bytearray(4 * 32 * 4)
    assert np.array_equal(src.read_rows_f32(t, 0, 4, buf), a.ravel())
    raw = src.read_raw(t, 32 * 4, 32 * 4, buf)  # second row's bytes
    assert bytes(raw) == a[1].tobytes()
    src.close()


def test_read_rows_bf16(tmp_path):
    f = (np.arange(64, dtype=np.float32) / 4).reshape(2, 32)  # exact in bf16
    u16 = (f.view(np.uint32) >> 16).astype(np.uint16)
    p = tmp_path / "m.gguf"
    w = gguf.GGUFWriter(str(p), "llama")
    # Pass the uint16 array directly; GGUFWriter takes shape/nbytes from it.
    w.add_tensor("w", u16, raw_dtype=gguf.GGMLQuantizationType.BF16)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    src = TensorSource(str(p))
    buf = bytearray(2 * 32 * 2)
    x = src.read_rows_f32(src.tensors[0], 0, 2, buf)
    assert np.array_equal(x, f.ravel())
    src.close()
