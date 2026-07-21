# Terminal UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace featherquant's silent multi-minute runs with a live terminal dashboard (overall progress bar with ETA, current tensor, color-coded memory gauge) plus a plain line-mode for pipes/CI and a human summary at the end.

**Architecture:** The engine stays UI-free: it gains an optional `progress` callback and emits typed events (`JobStart`, `TensorStart`, `ChunkDone`, `TensorDone`, `JobDone`) from `featherquant/events.py`. Two consumers implement the same callable protocol in `featherquant/ui.py`: `PlainReporter` (stdlib, line-per-tensor, safe for pipes) and `Dashboard` (rich `Live` view). The CLI picks one from `--ui auto|rich|plain|none` (auto = rich when stderr is a TTY). All UI output goes to **stderr**; stdout stays machine-clean (`--json` prints the stats dict there).

**Tech Stack:** `rich>=13` (only new dependency), stdlib dataclasses, existing pytest/ruff/mypy gates.

## Global Constraints

- Repo: featherQuant as of commit `~36` (post scale-proof). All existing gates must stay green: `.venv/bin/pytest`, `.venv/bin/ruff check featherquant tests`, `.venv/bin/mypy featherquant` (strict, `disallow_untyped_defs = true`).
- Only new runtime dependency: `rich>=13` in `pyproject.toml` `[project] dependencies`.
- `featherquant/engine.py` must NOT import rich — events only. `progress=None` (default) must add zero behavior change; all existing engine tests pass untouched.
- UI renders to stderr; stdout carries only `--json` output. Progress events fire even under `--ui none` cost: none (engine checks `progress is not None` before building events).
- Total output bytes are known at plan time (packed sizes are deterministic), so the overall bar is byte-accurate, including resume (bar starts at the committed byte count).
- Environment quirk: WSL2, venv via uv (`uv pip install -p .venv/bin/python -e '.[dev]'`). Commits authored by majipa007, no co-author line.
- Real-model demo paths: `/home/sukuna/models/qwen3-0.6b-bf16.gguf` (fast) and `/home/sukuna/models/qwen3-14b-bf16.gguf` (long run, shows ETA/gauge properly).

## File Structure

```
featherquant/events.py   — event dataclasses + ProgressFn type (new, stdlib only)
featherquant/engine.py   — emit events when progress is set (modify)
featherquant/ui.py       — PlainReporter (Task 2), Dashboard + summary_table (Task 3)
featherquant/cli.py      — --ui/--json flags, reporter wiring (modify)
tests/test_events.py     — engine emission order/content (new)
tests/test_ui.py         — reporter/dashboard render tests (new)
tests/test_cli.py        — update for --json; add --ui none path (modify)
```

---

### Task 1: Progress events + engine emission

**Files:**
- Create: `featherquant/events.py`
- Modify: `featherquant/engine.py`
- Test: `tests/test_events.py`

**Interfaces:**
- Consumes: `quantize_model` internals as they exist today (plan list, `entries`, `start_i`, `_stream_quantize`, `_stream_copy`, `stats`).
- Produces (used by Tasks 2–4):
  - `featherquant.events.JobStart(total_tensors: int, total_in_bytes: int, total_out_bytes: int, done_out_bytes: int, max_ram: int, fmt: str, dst: str, resumed_at: int)`
  - `TensorStart(index: int, name: str, src_type: str, dst_type: str, out_bytes: int)`
  - `ChunkDone(in_bytes: int, out_bytes: int, rss: int)`
  - `TensorDone(index: int)`
  - `JobDone(stats: dict[str, Any])`
  - `Event = JobStart | TensorStart | ChunkDone | TensorDone | JobDone`, `ProgressFn = Callable[[Event], None]`
  - `quantize_model(..., progress: ProgressFn | None = None)`

- [ ] **Step 1: Write the failing test**

`tests/test_events.py`:

```python
"""Engine progress-event emission: order, counts, resume offsets."""
import numpy as np
import pytest

from featherquant.engine import quantize_model, rss_bytes
from featherquant.events import (ChunkDone, JobDone, JobStart, TensorDone,
                                 TensorStart)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_events.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'featherquant.events'`.

- [ ] **Step 3: Write `featherquant/events.py`**

```python
"""Typed progress events emitted by the engine.

UI-agnostic on purpose: the engine emits these through an optional
callback and never imports a rendering library. Consumers (PlainReporter,
Dashboard) live in featherquant.ui.
"""
from dataclasses import dataclass
from typing import Any, Callable, Union


@dataclass(frozen=True)
class JobStart:
    """Fired once, after planning, before the first tensor streams."""
    total_tensors: int
    total_in_bytes: int
    total_out_bytes: int
    done_out_bytes: int   # committed bytes when resuming (bar starts here)
    max_ram: int
    fmt: str
    dst: str
    resumed_at: int       # index of first tensor this run writes (0 = fresh)


@dataclass(frozen=True)
class TensorStart:
    """Fired before each tensor begins streaming."""
    index: int
    name: str
    src_type: str
    dst_type: str
    out_bytes: int


@dataclass(frozen=True)
class ChunkDone:
    """Fired after every streamed chunk (quantize or copy)."""
    in_bytes: int
    out_bytes: int
    rss: int


@dataclass(frozen=True)
class TensorDone:
    """Fired after a tensor is committed to the manifest."""
    index: int


@dataclass(frozen=True)
class JobDone:
    """Fired last, with the final stats dict."""
    stats: dict[str, Any]


Event = Union[JobStart, TensorStart, ChunkDone, TensorDone, JobDone]
ProgressFn = Callable[[Event], None]
```

- [ ] **Step 4: Wire emission into `featherquant/engine.py`**

Import at the top (with the other package imports):

```python
from .events import (ChunkDone, JobDone, JobStart, ProgressFn, TensorDone,
                     TensorStart)
```

Signature (add `progress` after `_fail_after`):

```python
def quantize_model(src: str, dst: str, max_ram: int, report: str | None = None,
                   fmt: str = "q8_0", ggml_lib: str | None = None,
                   manifest_path: str | None = None, resume: bool = False,
                   adaptive: bool = True, vocab_gguf: str | None = None,
                   _force_chunk_rows: int | None = None,
                   _fail_after: int | None = None,
                   progress: ProgressFn | None = None) -> dict[str, Any]:
```

Inside the `try`, right before the `for i in range(start_i, len(plan)):` loop (at this point both branches have `entries`, `plan`, `start_i`):

```python
        if progress is not None:
            progress(JobStart(
                total_tensors=len(plan),
                total_in_bytes=sum(int(t.n_bytes) for t, _, _ in plan),
                total_out_bytes=sum(e.nbytes for e in entries),
                done_out_bytes=sum(e.nbytes for e in entries[:start_i]),
                max_ram=max_ram, fmt=fmt, dst=dst, resumed_at=start_i))
```

Inside the loop, right after `t, tt, _ = plan[i]` (before the `_fail_after` check):

```python
            if progress is not None:
                progress(TensorStart(i, t.name, t.tensor_type.name, tt.name,
                                     entries[i].nbytes))
```

After `man.save(manifest_path)` in the per-tensor commit block:

```python
            if progress is not None:
                progress(TensorDone(i))
```

Just before `return stats` (after the report file is written):

```python
    if progress is not None:
        progress(JobDone(stats))
    return stats
```

Thread `progress` into both streamers. `_stream_quantize` gains a trailing
parameter `progress: ProgressFn | None = None`; after its
`stats["chunks"] += 1` line add:

```python
        if progress is not None:
            progress(ChunkDone(n * ne0 * isz, len(packed), rss_bytes()))
```

`_stream_copy` gains the same trailing parameter; after its
`stats["chunks"] += 1` line add:

```python
        if progress is not None:
            progress(ChunkDone(n, n, rss_bytes()))
```

Update the two call sites in the main loop to pass `progress`:

```python
            if tt != t.tensor_type:
                _stream_quantize(source, iw, t, tt, working, stats,
                                 _force_chunk_rows, lib, adaptive, progress)
            else:
                _stream_copy(source, iw, t, stats, progress)
```

(Adjust `_stream_quantize`'s def to
`..., lib: GgmlLib | None, adaptive: bool = True, progress: ProgressFn | None = None`
and `_stream_copy`'s to `..., stats: dict[str, Any], progress: ProgressFn | None = None`.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_events.py -v`
Expected: 3 passed. Then the full suite: `.venv/bin/pytest` — everything passes (59 + 3).

- [ ] **Step 6: Gates + commit**

```bash
.venv/bin/ruff check featherquant tests && .venv/bin/mypy featherquant
git add featherquant/events.py featherquant/engine.py tests/test_events.py
git commit -m "feat: typed progress events from the quantization engine"
```

---

### Task 2: PlainReporter (pipe/CI mode)

**Files:**
- Create: `featherquant/ui.py` (PlainReporter half; Dashboard lands in Task 3)
- Test: `tests/test_ui.py`

**Interfaces:**
- Consumes: `featherquant.events` types from Task 1.
- Produces: `featherquant.ui.PlainReporter(out: IO[str] | None = None)` — callable `(Event) -> None`, method `close() -> None`; `featherquant.ui._mib(n: int | float) -> str`. Task 4's CLI constructs it with no args (defaults to stderr).

- [ ] **Step 1: Write the failing test**

`tests/test_ui.py`:

```python
"""Reporter render tests — no real quantization, just synthetic events."""
import io

from featherquant.events import (ChunkDone, JobDone, JobStart, TensorDone,
                                 TensorStart)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_ui.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'featherquant.ui'`.

- [ ] **Step 3: Write the PlainReporter half of `featherquant/ui.py`**

```python
"""Terminal front-ends for engine progress events.

PlainReporter: stdlib line-per-tensor output for pipes and CI.
Dashboard (Task 3): live rich view for interactive terminals.
Both write to stderr by default so stdout stays machine-clean.
"""
import sys
from typing import IO

from .events import Event, JobDone, JobStart, TensorStart


def _mib(n: int | float) -> str:
    """Human size in MiB (fixed unit keeps lines grep-able)."""
    return f"{n / (1 << 20):,.0f} MiB"


class PlainReporter:
    """Line-per-tensor progress with zero escape codes."""

    def __init__(self, out: IO[str] | None = None):
        self.out = out if out is not None else sys.stderr
        self.total = 0

    def __call__(self, ev: Event) -> None:
        if isinstance(ev, JobStart):
            self.total = ev.total_tensors
            note = f" | resuming at #{ev.resumed_at}" if ev.resumed_at else ""
            print(f"featherquant: {ev.fmt} -> {ev.dst} | "
                  f"{ev.total_tensors} tensors, {_mib(ev.total_out_bytes)} out, "
                  f"budget {_mib(ev.max_ram)}{note}",
                  file=self.out, flush=True)
        elif isinstance(ev, TensorStart):
            print(f"[{ev.index + 1}/{self.total}] {ev.name} "
                  f"{ev.src_type}->{ev.dst_type} {_mib(ev.out_bytes)}",
                  file=self.out, flush=True)
        elif isinstance(ev, JobDone):
            s = ev.stats
            print(f"done in {s['elapsed_s']} s | peak RSS {_mib(s['peak_rss'])} "
                  f"| violations {s['budget_violations']}",
                  file=self.out, flush=True)
        # ChunkDone/TensorDone are intentionally silent in plain mode:
        # one line per tensor keeps CI logs readable at 311+ tensors.

    def close(self) -> None:
        """Nothing to release; symmetry with Dashboard."""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_ui.py -v`
Expected: 2 passed.

- [ ] **Step 5: Gates + commit**

```bash
.venv/bin/ruff check featherquant tests && .venv/bin/mypy featherquant
git add featherquant/ui.py tests/test_ui.py
git commit -m "feat: plain line-mode progress reporter"
```

---

### Task 3: Rich dashboard + summary table

**Files:**
- Modify: `pyproject.toml` (add `rich>=13`), `featherquant/ui.py` (append)
- Test: `tests/test_ui.py` (append)

**Interfaces:**
- Consumes: Task 1 events, Task 2's `_mib`.
- Produces: `featherquant.ui.Dashboard(console: rich.console.Console | None = None)` — callable `(Event) -> None`, `close() -> None`; `featherquant.ui.summary_table(stats: dict[str, Any]) -> rich.table.Table`. Task 4 constructs `Dashboard()` with no args and prints `summary_table(stats)`.

- [ ] **Step 1: Add the dependency**

In `pyproject.toml` change:

```toml
dependencies = ["numpy>=1.26", "gguf>=0.16,<0.18"]
```

to:

```toml
dependencies = ["numpy>=1.26", "gguf>=0.16,<0.18", "rich>=13"]
```

Run: `uv pip install -p .venv/bin/python -e '.[dev]'`
Expected: rich installed without error.

- [ ] **Step 2: Write the failing tests (append to `tests/test_ui.py`)**

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_ui.py -v`
Expected: new tests FAIL — `ImportError: cannot import name 'Dashboard'`.

- [ ] **Step 4: Append Dashboard + summary_table to `featherquant/ui.py`**

Extend the imports at the top of the file:

```python
from typing import IO, Any

from rich.console import Console, Group
from rich.live import Live
from rich.progress import (BarColumn, DownloadColumn, Progress, TaskID,
                           TextColumn, TimeRemainingColumn,
                           TransferSpeedColumn)
from rich.table import Table
from rich.text import Text

from .events import ChunkDone, Event, JobDone, JobStart, TensorStart
```

Append:

```python
class Dashboard:
    """Live rich dashboard: current tensor, memory gauge, byte-accurate bar.

    The bar total is the planned packed-output size (known exactly before
    streaming), so ETA and percent are real, and resume starts the bar at
    the committed byte count.
    """

    def __init__(self, console: Console | None = None):
        self.console = console if console is not None else Console(stderr=True)
        self.progress = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=None),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=self.console)
        self.task_id: TaskID | None = None
        self.status = Text("starting…")
        self.mem = Text("")
        self.max_ram = 0
        self.peak = 0
        self.total = 0
        self._closed = False
        self.live = Live(self._view(), console=self.console,
                         refresh_per_second=8)
        self.live.start()

    def _view(self) -> Group:
        return Group(self.status, self.mem, self.progress)

    def __call__(self, ev: Event) -> None:
        if isinstance(ev, JobStart):
            self.max_ram = ev.max_ram
            self.total = ev.total_tensors
            self.task_id = self.progress.add_task(
                ev.fmt, total=ev.total_out_bytes, completed=ev.done_out_bytes)
        elif isinstance(ev, TensorStart):
            self.status = Text.assemble(
                (f"[{ev.index + 1}/{self.total}] ", "cyan"),
                (ev.name, "bold"),
                (f"  {ev.src_type}→{ev.dst_type}  {_mib(ev.out_bytes)}", "dim"))
        elif isinstance(ev, ChunkDone):
            if self.task_id is not None:
                self.progress.update(self.task_id, advance=ev.out_bytes)
            self.peak = max(self.peak, ev.rss)
            frac = ev.rss / self.max_ram if self.max_ram else 0.0
            color = ("green" if frac < 0.8 else
                     "yellow" if frac <= 1.0 else "red")
            self.mem = Text.assemble(
                ("RSS ", "dim"), (_mib(ev.rss), color),
                (f" / {_mib(self.max_ram)} budget"
                 f"  (peak {_mib(self.peak)})", "dim"))
        elif isinstance(ev, JobDone):
            self.close()
            return
        self.live.update(self._view())

    def close(self) -> None:
        """Stop the live view; safe to call more than once."""
        if not self._closed:
            self._closed = True
            self.live.stop()


def summary_table(stats: dict[str, Any]) -> Table:
    """Human end-of-run summary (replaces the raw JSON dump)."""
    t = Table(title="featherquant run", show_header=False)
    t.add_column(style="dim")
    t.add_column(justify="right")
    t.add_row("peak RSS", _mib(stats["peak_rss"]))
    t.add_row("metadata peak", _mib(stats.get("rss_metadata_peak", 0)))
    t.add_row("budget", _mib(stats["max_ram"]))
    t.add_row("budget violations", str(stats["budget_violations"]))
    t.add_row("read / written",
              f"{_mib(stats['bytes_read'])} / {_mib(stats['bytes_written'])}")
    t.add_row("chunks", str(stats["chunks"]))
    t.add_row("elapsed", f"{stats['elapsed_s']} s")
    return t
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_ui.py -v`
Expected: 5 passed. If `Live` output is empty in capture, the console must be created with `force_terminal=True` (it is, in the tests) — fix the test setup, not the Dashboard.

- [ ] **Step 6: Gates + commit**

```bash
.venv/bin/ruff check featherquant tests && .venv/bin/mypy featherquant && .venv/bin/pytest
git add pyproject.toml featherquant/ui.py tests/test_ui.py
git commit -m "feat: live rich dashboard and end-of-run summary table"
```

---

### Task 4: CLI wiring, flags, docs, real-model demo

**Files:**
- Modify: `featherquant/cli.py`, `tests/test_cli.py`, `README.md`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `PlainReporter`, `Dashboard`, `summary_table` (Tasks 2–3), `quantize_model(..., progress=...)` (Task 1).
- Produces: `featherquant --ui auto|rich|plain|none` (default `auto`: rich when `sys.stderr.isatty()`, else plain) and `--json` (stats JSON on stdout; without it a summary table goes to stderr).

- [ ] **Step 1: Write the failing tests (replace `test_cli_end_to_end` in `tests/test_cli.py` and add the new ones)**

```python
def test_cli_end_to_end_json(tmp_path, monkeypatch, capsys):
    src = tmp_path / "src.gguf"
    make_gguf(src, {"w": np.ones((4, 64), np.float16)})
    out = tmp_path / "out.gguf"
    rp = tmp_path / "out.report.json"
    budget = str(rss_bytes() + (512 << 20))
    monkeypatch.setattr(sys, "argv",
                        ["featherquant", "--model", str(src), "--output", str(out),
                         "--format", "q8_0", "--max-ram", budget,
                         "--report", str(rp), "--ui", "none", "--json"])
    cli.main()
    assert out.exists()
    stats = json.loads(capsys.readouterr().out)   # stdout is pure JSON
    assert stats["peak_rss"] <= int(budget)
    assert json.loads(rp.read_text())["chunks"] == stats["chunks"]


def test_cli_plain_ui_goes_to_stderr(tmp_path, monkeypatch, capsys):
    src = tmp_path / "src.gguf"
    make_gguf(src, {"w": np.ones((4, 64), np.float16)})
    out = tmp_path / "out.gguf"
    budget = str(rss_bytes() + (512 << 20))
    monkeypatch.setattr(sys, "argv",
                        ["featherquant", "--model", str(src), "--output", str(out),
                         "--max-ram", budget, "--ui", "plain"])
    cli.main()
    captured = capsys.readouterr()
    assert "[1/1] w" in captured.err          # progress lines on stderr
    assert captured.out == ""                 # stdout clean without --json


def test_cli_auto_falls_back_to_plain_when_piped(tmp_path, monkeypatch, capsys):
    src = tmp_path / "src.gguf"
    make_gguf(src, {"w": np.ones((4, 64), np.float16)})
    out = tmp_path / "out.gguf"
    budget = str(rss_bytes() + (512 << 20))
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False, raising=False)
    monkeypatch.setattr(sys, "argv",
                        ["featherquant", "--model", str(src), "--output", str(out),
                         "--max-ram", budget])
    cli.main()
    assert "\x1b[" not in capsys.readouterr().err   # plain mode, no escapes
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_cli.py -v`
Expected: new tests FAIL — `error: unrecognized arguments: --ui none --json` (argparse exits non-zero).

- [ ] **Step 3: Rewire `featherquant/cli.py`**

Add flags after `--resume`:

```python
    p.add_argument("--ui", choices=["auto", "rich", "plain", "none"],
                   default="auto",
                   help="progress display: rich dashboard, plain lines, or "
                        "none (auto = rich on a TTY, plain otherwise)")
    p.add_argument("--json", action="store_true",
                   help="print the stats JSON to stdout")
```

Replace the run/print block:

```python
    a = p.parse_args()
    mode = a.ui if a.ui != "auto" else (
        "rich" if sys.stderr.isatty() else "plain")
    reporter: Dashboard | PlainReporter | None
    if mode == "rich":
        reporter = Dashboard()
    elif mode == "plain":
        reporter = PlainReporter()
    else:
        reporter = None
    try:
        stats = quantize_model(a.model, a.output, a.max_ram, report=a.report,
                               fmt=a.format, ggml_lib=a.ggml_lib,
                               resume=a.resume, vocab_gguf=a.vocab_gguf,
                               progress=reporter)
    except RuntimeError as exc:
        # Turn internal errors into a clean CLI failure, no traceback spam.
        sys.exit(f"featherquant: error: {exc}")
    finally:
        if reporter is not None:
            reporter.close()   # always restore the terminal, even on error
    if a.json:
        print(json.dumps(stats, indent=2))       # stdout: machine-readable
    elif mode != "none":
        Console(stderr=True).print(summary_table(stats))
```

And the imports:

```python
from rich.console import Console

from .engine import quantize_model
from .ui import Dashboard, PlainReporter, summary_table
```

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/pytest`
Expected: all pass (Task 1–3 tests plus 3 rewired CLI tests; the old `test_cli_end_to_end` is gone). Then gates:
`.venv/bin/ruff check featherquant tests && .venv/bin/mypy featherquant`

- [ ] **Step 5: Real-model demo (manual verification)**

```bash
source .venv/bin/activate
featherquant --model /home/sukuna/models/qwen3-0.6b-bf16.gguf \
  --output /tmp/claude-1000/ui-demo.gguf --max-ram 1GB
```

Expected: live dashboard on stderr — advancing byte bar with ETA, tensor counter reaching `[311/311]`, green RSS gauge, then the summary table. Also verify pipe mode:
`featherquant ... --max-ram 1GB 2>&1 | head -5` shows plain lines, no escape codes. Clean up `/tmp/claude-1000/ui-demo*`.

- [ ] **Step 6: Update README and commit**

In `README.md`, under the Usage section, add one paragraph:

```markdown
Runs show a live dashboard (progress bar with ETA, current tensor, RSS
gauge vs budget) when stderr is a terminal; pipes/CI get plain
line-per-tensor output automatically. Control it with
`--ui rich|plain|none`; add `--json` to print the stats dict to stdout.
```

Then:

```bash
git add featherquant/cli.py tests/test_cli.py README.md
git commit -m "feat: --ui flag wiring live dashboard, plain mode, and summary table"
```

---

## Self-review notes

- Spec coverage: live progress ✓ (Task 3), ETA ✓ (byte-accurate totals, Task 1/3), memory gauge ✓ (ChunkDone.rss, Task 3), pipe/CI safety ✓ (Task 2 + auto fallback, Task 4), resume-aware bar ✓ (`done_out_bytes`), machine output preserved ✓ (`--json`, stderr-only UI).
- Type consistency: `ProgressFn = Callable[[Event], None]`; both reporters are plain callables with `close()`; engine references only `featherquant.events`.
- Known trade-off: per-chunk `rss_bytes()` read (~µs on /proc) also fires in non-adaptive copy paths when a reporter is attached; `progress=None` skips everything.
