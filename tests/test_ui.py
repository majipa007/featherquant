"""Reporter render tests — no real quantization, just synthetic events."""
import io

from featherquant.events import ChunkDone, JobDone, JobStart, TensorDone, TensorStart
from featherquant.ui import PlainReporter

MIB = 1 << 20


def _events():
    return [
        JobStart(total_tensors=2, total_in_bytes=4 * MIB, total_out_bytes=2 * MIB,
                 done_out_bytes=0, max_ram=1024 * MIB, fmt="q8_0",
                 dst="/tmp/out.gguf", resumed_at=0),
        TensorStart(0, "blk.0.attn_q.weight", "F16", "Q8_0", MIB),
        ChunkDone(2 * MIB, MIB, 500 * MIB),
        TensorDone(0),
        TensorStart(1, "blk.0.attn_norm.weight", "F32", "F32", MIB),
        ChunkDone(2 * MIB, MIB, 600 * MIB),
        TensorDone(1),
        JobDone({"elapsed_s": 12.3, "peak_rss": 600 * MIB,
                 "budget_violations": 0, "chunks": 2}),
    ]


def test_plain_reporter_lines():
    buf = io.StringIO()
    r = PlainReporter(out=buf)
    for ev in _events():
        r(ev)
    r.close()
    out = buf.getvalue()
    lines = out.strip().splitlines()
    assert "2 tensors" in lines[0] and "q8_0" in lines[0]
    assert lines[1].startswith("[1/2] blk.0.attn_q.weight F16->Q8_0")
    assert lines[2].startswith("[2/2] blk.0.attn_norm.weight F32->F32")
    assert "peak RSS 600 MiB" in lines[-1] and "violations 0" in lines[-1]
    assert "\x1b[" not in out  # no escape codes in plain mode


def test_plain_reporter_resume_note():
    buf = io.StringIO()
    r = PlainReporter(out=buf)
    ev = _events()[0]
    r(JobStart(**{**ev.__dict__, "resumed_at": 5, "done_out_bytes": MIB}))
    assert "resuming at #5" in buf.getvalue()
