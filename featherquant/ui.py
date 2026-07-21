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
