"""Terminal front-ends for engine progress events.

PlainReporter: stdlib line-per-tensor output for pipes and CI.
Dashboard (Task 3): live rich view for interactive terminals.
Both write to stderr by default so stdout stays machine-clean.
"""
import sys
from typing import IO, Any

from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table
from rich.text import Text

from .events import ChunkDone, Event, JobDone, JobStart, TensorStart


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
