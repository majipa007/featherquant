"""Adaptive chunk-size controller (Phase 3).

Pure state machine: the engine feeds it RSS observations around each chunk
and asks for the next chunk size. Purity keeps it fully unit-testable with
scripted observations — no flaky real-RSS assertions.

Policy: EWMA of measured per-row memory cost; a safety factor keeps the
target below the working budget; violations double the cost estimate
(halving the next chunk).
"""


class BlockController:
    """Chooses row-chunk sizes from live memory feedback."""

    def __init__(self, working_budget: int, est_per_row: int,
                 min_rows: int = 1, safety: float = 0.85,
                 ewma_alpha: float = 0.3):
        if working_budget <= 0 or est_per_row <= 0:
            raise ValueError("working_budget and est_per_row must be positive")
        self.working_budget = working_budget
        self.min_rows = min_rows
        self.safety = safety
        self.ewma_alpha = ewma_alpha
        self._per_row = float(est_per_row)

    @property
    def per_row(self) -> float:
        """Current per-row working-set estimate in bytes (telemetry)."""
        return self._per_row

    def next_rows(self, rows_remaining: int) -> int:
        """Largest chunk the current estimate says fits the budget."""
        rows = int(self.safety * self.working_budget / self._per_row)
        return max(self.min_rows, min(rows_remaining, rows))

    def observe(self, rows: int, rss_before: int, rss_after: int) -> None:
        """Fold one measured chunk into the estimate.

        Non-positive deltas carry no signal (allocator reuse / GC dip) and
        are ignored rather than shrinking the estimate toward zero.
        """
        delta = rss_after - rss_before
        if rows <= 0 or delta <= 0:
            return
        measured = delta / rows
        a = self.ewma_alpha
        self._per_row = (1 - a) * self._per_row + a * measured

    def violation(self) -> None:
        """Budget breached: double the cost estimate (halve the next chunk)."""
        self._per_row *= 2.0
