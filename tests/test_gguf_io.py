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


# ---------------------------------------------------------------------------
# IncrementalWriter tests
# ---------------------------------------------------------------------------
from gguf import GGUFReader, LlamaFileType

from featherquant.gguf_io import IncrementalWriter
from featherquant.q8_0 import quantize_q8_0


def test_incremental_writer_roundtrip(tmp_path):
    src_arr = (np.arange(4 * 64, dtype=np.float32).reshape(4, 64) / 8).astype(np.float16)
    sp = tmp_path / "src.gguf"
    make_gguf(sp, {"w": src_arr})
    reader = GGUFReader(str(sp))
    out = tmp_path / "out.gguf"
    iw = IncrementalWriter(str(out), reader, LlamaFileType.MOSTLY_Q8_0)
    t = reader.tensors[0]
    payload = quantize_q8_0(src_arr.astype(np.float32).ravel())
    iw.add_tensor_info(t.name, t.shape, len(payload), gguf.GGMLQuantizationType.Q8_0)
    iw.begin_data()
    iw.begin_tensor()
    iw.write(payload[:170])   # two chunks proves streamed writes land correctly
    iw.write(payload[170:])
    iw.close()

    r2 = GGUFReader(str(out))
    t2 = r2.tensors[0]
    assert t2.name == "w"
    assert t2.tensor_type == gguf.GGMLQuantizationType.Q8_0
    assert [int(d) for d in t2.shape] == [64, 4]
    assert t2.data.tobytes() == payload
    assert int(r2.fields["general.file_type"].contents()) == int(LlamaFileType.MOSTLY_Q8_0)
    assert r2.fields["general.architecture"].contents() == "llama"


def test_two_tensor_alignment_and_copy(tmp_path):
    # Q8_0 payloads are 34-byte multiples (not 32-aligned): second tensor
    # exercises inter-tensor padding. Second tensor is a verbatim F32 copy.
    a = np.ones((1, 32), np.float16)
    b = np.arange(32, dtype=np.float32)
    sp = tmp_path / "src.gguf"
    make_gguf(sp, {"a": a, "b": b})
    reader = GGUFReader(str(sp))
    ta, tb = reader.tensors[0], reader.tensors[1]
    out = tmp_path / "out.gguf"
    iw = IncrementalWriter(str(out), reader, LlamaFileType.MOSTLY_Q8_0)
    pa = quantize_q8_0(a.astype(np.float32).ravel())
    iw.add_tensor_info(ta.name, ta.shape, len(pa), gguf.GGMLQuantizationType.Q8_0)
    iw.add_tensor_info(tb.name, tb.shape, int(tb.n_bytes), tb.tensor_type)
    iw.begin_data()
    iw.begin_tensor()
    iw.write(pa)
    iw.begin_tensor()
    iw.write(b.tobytes())
    iw.close()

    r2 = GGUFReader(str(out))
    by_name = {t.name: t for t in r2.tensors}
    assert by_name["a"].data.tobytes() == pa
    assert by_name["b"].tensor_type == gguf.GGMLQuantizationType.F32
    assert np.array_equal(by_name["b"].data.reshape(-1), b)
