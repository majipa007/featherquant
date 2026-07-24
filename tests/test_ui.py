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


def test_dashboard_renders_progress_and_memory():
    from rich.console import Console

    from featherquant.ui import Dashboard
    console = Console(file=io.StringIO(), force_terminal=True, width=120)
    d = Dashboard(console=console)
    for ev in _events():
        d(ev)
    d.close()
    out = console.file.getvalue()
    assert "blk.0.attn_norm.weight" in out   # current-tensor line
    assert "RSS" in out and "600" in out     # memory gauge with last sample
    assert "q8_0" in out                     # bar description


def test_dashboard_close_idempotent():
    from rich.console import Console

    from featherquant.ui import Dashboard
    console = Console(file=io.StringIO(), force_terminal=True, width=80)
    d = Dashboard(console=console)
    d(_events()[0])
    d.close()
    d.close()  # second close must not raise


def test_summary_table_contents():
    from rich.console import Console

    from featherquant.ui import summary_table
    console = Console(file=io.StringIO(), width=100)
    console.print(summary_table({
        "max_ram": 1024 * MIB, "peak_rss": 600 * MIB,
        "rss_metadata_peak": 561 * MIB, "budget_violations": 0,
        "chunks": 455, "bytes_read": 4 * MIB, "bytes_written": 2 * MIB,
        "elapsed_s": 331.0, "working_budget": 400 * MIB}))
    out = console.file.getvalue()
    assert "600 MiB" in out and "1,024 MiB" in out and "331.0" in out


def test_plain_reporter_prints_phase_lines():
    from featherquant.events import Phase
    from featherquant.ui import PlainReporter
    buf = io.StringIO()
    r = PlainReporter(out=buf)
    r(Phase("read metadata: 311 tensors"))
    assert "read metadata: 311 tensors" in buf.getvalue()


def test_dashboard_shows_phase_in_status():
    from rich.console import Console

    from featherquant.events import Phase
    from featherquant.ui import Dashboard
    console = Console(file=io.StringIO(), force_terminal=True, width=120)
    d = Dashboard(console=console)
    d(Phase("planning 311 tensors"))
    d.close()
    assert "planning 311 tensors" in console.file.getvalue()
