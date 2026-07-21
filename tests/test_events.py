"""Engine progress-event emission: order, counts, resume offsets."""
import numpy as np
import pytest

from featherquant.engine import quantize_model, rss_bytes
from featherquant.events import ChunkDone, JobDone, JobStart, TensorDone, TensorStart
from tests.conftest import make_gguf


def _make_model(tmp_path):
    rng = np.random.default_rng(0)
    sp = tmp_path / "src.gguf"
    make_gguf(sp, {"blk.0.attn_q.weight": rng.standard_normal((10, 64)).astype(np.float16),
                   "blk.0.attn_norm.weight": rng.standard_normal(64).astype(np.float32)})
    return sp


def _budget():
    return rss_bytes() + (512 << 20)


def test_event_sequence(tmp_path):
    sp = _make_model(tmp_path)
    events = []
    stats = quantize_model(str(sp), str(tmp_path / "o.gguf"), _budget(),
                           progress=events.append)
    assert isinstance(events[0], JobStart)
    js = events[0]
    assert js.total_tensors == 2 and js.resumed_at == 0 and js.done_out_bytes == 0
    assert js.fmt == "q8_0" and js.max_ram == stats["max_ram"]
    starts = [e for e in events if isinstance(e, TensorStart)]
    assert [s.index for s in starts] == [0, 1]
    assert starts[0].name == "blk.0.attn_q.weight"
    assert starts[0].src_type == "F16" and starts[0].dst_type == "Q8_0"
    chunks = [e for e in events if isinstance(e, ChunkDone)]
    assert len(chunks) == stats["chunks"]
    assert sum(c.out_bytes for c in chunks) == stats["bytes_written"]
    assert all(c.rss > 0 for c in chunks)
    dones = [e for e in events if isinstance(e, TensorDone)]
    assert [d.index for d in dones] == [0, 1]
    assert isinstance(events[-1], JobDone)
    assert events[-1].stats["chunks"] == stats["chunks"]
    # totals are byte-accurate against the written file
    assert js.total_out_bytes == sum(s.out_bytes for s in starts)


def test_no_progress_means_no_change(tmp_path):
    # default path emits nothing and produces identical output
    sp = _make_model(tmp_path)
    o1, o2 = tmp_path / "a.gguf", tmp_path / "b.gguf"
    quantize_model(str(sp), str(o1), _budget())
    quantize_model(str(sp), str(o2), _budget(), progress=lambda e: None)
    assert o1.read_bytes() == o2.read_bytes()


def test_resume_reports_offset(tmp_path):
    sp = _make_model(tmp_path)
    out = tmp_path / "o.gguf"
    with pytest.raises(RuntimeError):
        quantize_model(str(sp), str(out), _budget(), _fail_after=1)
    events = []
    quantize_model(str(sp), str(out), _budget(), resume=True,
                   progress=events.append)
    js = events[0]
    assert js.resumed_at == 1
    assert js.done_out_bytes > 0  # first tensor's bytes already banked
    starts = [e for e in events if isinstance(e, TensorStart)]
    assert [s.index for s in starts] == [1]
