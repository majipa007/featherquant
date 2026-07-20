"""Tests for the budget-planned streaming engine."""
import numpy as np
import pytest
from gguf import GGMLQuantizationType, GGUFReader

from featherquant.engine import RESERVE, per_row_cost, q8_0_nbytes, quantize_model, rss_bytes
from featherquant.q8_0 import quantize_q8_0
from tests.conftest import make_gguf


def test_q8_0_nbytes():
    assert q8_0_nbytes(64) == 2 * 34


def test_per_row_cost_positive():
    assert per_row_cost(4096, 2) > 4096 * 2  # read buf plus working temps


def _make_model(tmp_path):
    rng = np.random.default_rng(0)
    w = rng.standard_normal((10, 64)).astype(np.float16)
    norm = rng.standard_normal(64).astype(np.float32)  # 1-D: copied, not quantized
    odd = rng.standard_normal((4, 30)).astype(np.float16)  # row % 32 != 0: copied
    sp = tmp_path / "src.gguf"
    make_gguf(sp, {"blk.0.attn_q.weight": w,
                   "blk.0.attn_norm.weight": norm,
                   "blk.0.odd.weight": odd})
    return sp, w, norm, odd


def test_engine_streams_and_matches_in_memory_reference(tmp_path):
    sp, w, norm, odd = _make_model(tmp_path)
    big = rss_bytes() + (512 << 20)
    o1, o2 = tmp_path / "a.gguf", tmp_path / "b.gguf"
    s1 = quantize_model(str(sp), str(o1), big, _force_chunk_rows=3)  # forces 4 chunks
    quantize_model(str(sp), str(o2), big)  # one chunk
    assert o1.read_bytes() == o2.read_bytes()  # chunking must not change output
    assert s1["chunks"] >= 4 and s1["peak_rss"] <= big

    r = GGUFReader(str(o1))
    by_name = {t.name: t for t in r.tensors}
    tq = by_name["blk.0.attn_q.weight"]
    assert tq.tensor_type == GGMLQuantizationType.Q8_0
    assert tq.data.tobytes() == quantize_q8_0(w.astype(np.float32).ravel())
    tn = by_name["blk.0.attn_norm.weight"]
    assert tn.tensor_type == GGMLQuantizationType.F32
    assert np.array_equal(tn.data.reshape(-1), norm)
    to = by_name["blk.0.odd.weight"]
    assert to.tensor_type == GGMLQuantizationType.F16
    assert np.array_equal(to.data.reshape(-1), odd.ravel())


def test_deterministic_across_runs(tmp_path):
    sp, *_ = _make_model(tmp_path)
    big = rss_bytes() + (512 << 20)
    o1, o2 = tmp_path / "a.gguf", tmp_path / "b.gguf"
    quantize_model(str(sp), str(o1), big)
    quantize_model(str(sp), str(o2), big)
    assert o1.read_bytes() == o2.read_bytes()


def test_impossible_budget_exits_with_minimum(tmp_path):
    sp, *_ = _make_model(tmp_path)
    with pytest.raises(SystemExit):
        quantize_model(str(sp), str(tmp_path / "o.gguf"), max_ram=RESERVE)


def test_report_written(tmp_path):
    import json
    sp, *_ = _make_model(tmp_path)
    rp = tmp_path / "r.json"
    quantize_model(str(sp), str(tmp_path / "o.gguf"),
                   rss_bytes() + (512 << 20), report=str(rp))
    stats = json.loads(rp.read_text())
    assert stats["bytes_read"] > 0 and stats["peak_rss"] > 0 and "elapsed_s" in stats
