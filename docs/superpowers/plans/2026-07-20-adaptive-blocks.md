# Adaptive Block Sizing Implementation Plan (Phase 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the static `per_row_cost` guess with a controller that observes real RSS per chunk and adapts chunk size, staying closer to the ceiling without crossing it — retiring both `ponytail:` debts in `engine.py`.

**Architecture:** A pure, side-effect-free `BlockController` class holds the loop logic (estimate → observe → adjust with EWMA; halve on violation, grow 1.25x when comfortably under). The engine feeds it `rss_bytes()` observations around each chunk. Purity makes it fully unit-testable with scripted fake observations — no flaky RSS assertions in CI. Output bytes are provably chunking-independent (existing tests), so the controller can never affect correctness, only memory behavior.

**Tech Stack:** stdlib only.

## Global Constraints

- All MVP global constraints apply. Determinism of OUTPUT is untouched (chunking never changes bytes — guarded by existing `test_engine_streams_and_matches_in_memory_reference`); chunk-size SEQUENCE may vary run-to-run and that is acceptable.
- Emergency stop: if observed RSS exceeds `max_ram - RESERVE // 2`, the engine calls `gc.collect()` and halves; if RSS exceeds `max_ram` itself, count a violation (existing stat) and halve again. Never silently exceed.
- Execute AFTER the checkpoint-resume plan; rebase this plan's engine edits on top of it.
- Gates green per task; commits by majipa007.

## File Structure

```
featherquant/controller.py   — BlockController (new, pure)
featherquant/engine.py       — wire controller into _stream_quantize (modify)
tests/test_controller.py     — scripted-observation unit tests (new)
scripts/bench_adaptive.py    — static vs adaptive peak-RSS/closeness report (new)
```

---

### Task 1: Pure controller

**Files:**
- Create: `featherquant/controller.py`
- Test: `tests/test_controller.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `class BlockController`:
  - `__init__(self, working_budget: int, est_per_row: int, min_rows: int = 1, safety: float = 0.85, ewma_alpha: float = 0.3)`
  - `next_rows(self, rows_remaining: int) -> int` — `max(min_rows, min(rows_remaining, int(safety * working_budget / per_row)))` using the current per-row estimate.
  - `observe(self, rows: int, rss_before: int, rss_after: int) -> None` — measured per-row = `max(0, rss_after - rss_before) / rows`, folded into the estimate by EWMA; ignores chunks where delta ≤ 0 (allocator reuse tells us nothing).
  - `violation(self) -> None` — halve the effective chunk (doubles the per-row estimate).
  - Property `per_row: float` (current estimate, for telemetry).

- [ ] **Step 1: Write the failing tests**

```python
"""BlockController unit tests with scripted observations — no real RSS."""
from featherquant.controller import BlockController


def test_initial_chunk_from_estimate():
    c = BlockController(working_budget=1000, est_per_row=10, safety=0.9)
    assert c.next_rows(rows_remaining=10**9) == 90  # 0.9 * 1000 / 10


def test_clamps_to_remaining_and_min():
    c = BlockController(working_budget=1000, est_per_row=10)
    assert c.next_rows(rows_remaining=5) == 5
    c2 = BlockController(working_budget=10, est_per_row=1000, min_rows=1)
    assert c2.next_rows(10**9) == 1  # never zero: caller handles min-budget exit


def test_learns_from_observation():
    c = BlockController(working_budget=1000, est_per_row=10, ewma_alpha=1.0)
    c.observe(rows=10, rss_before=0, rss_after=500)  # actual 50/row, alpha=1 adopts it
    assert c.per_row == 50.0
    assert c.next_rows(10**9) == int(0.85 * 1000 / 50)


def test_negative_delta_ignored():
    c = BlockController(working_budget=1000, est_per_row=10)
    c.observe(rows=10, rss_before=500, rss_after=100)  # GC dip
    assert c.per_row == 10.0


def test_violation_halves():
    c = BlockController(working_budget=1000, est_per_row=10)
    before = c.next_rows(10**9)
    c.violation()
    assert c.next_rows(10**9) <= before // 2 + 1
```

- [ ] **Step 2: Verify failure** — module missing.
- [ ] **Step 3: Implement** (~30 lines, one class, full docstrings + types).
- [ ] **Step 4: Verify pass, gates, commit** — `feat: pure EWMA block-size controller`.

### Task 2: Engine wiring

**Files:**
- Modify: `featherquant/engine.py`
- Test: extend `tests/test_engine.py`

**Interfaces:**
- Consumes: Task 1; existing `_stream_quantize`.
- Produces: `_stream_quantize` builds one `BlockController` per tensor seeded with `per_row_cost(ne0, isz)` (the static model becomes the PRIOR, not the law — delete its `ponytail:` comment, keep the function). Each iteration: `n = ctrl.next_rows(rows - r0)`, then RSS before/after the chunk → `ctrl.observe(...)`; over `max_ram` → `stats["budget_violations"] += 1; ctrl.violation()`; over `max_ram - RESERVE // 2` → `gc.collect()` + `ctrl.violation()`. The read buffer is allocated at the controller's INITIAL chunk size and re-sliced; if the controller ever asks for MORE rows than the buffer holds, cap at buffer size (no realloc — the budget was already spent). New stats keys: `chunk_rows_min`, `chunk_rows_max`, `per_row_final` (per tensor keyed dict under `stats["adaptive"]`).

- [ ] **Step 1: Failing test** — monkeypatch `featherquant.engine.rss_bytes` with a scripted sequence (list popped per call) that simulates an over-prediction; assert `stats["adaptive"]` present, chunk shrank (`chunk_rows_min < chunk_rows_max`), and output bytes still equal the unchunked reference (correctness invariant).
- [ ] **Step 2: Watch fail. Step 3: Implement. Step 4: Full suite green (existing `_force_chunk_rows` tests keep passing — force bypasses the controller).**
- [ ] **Step 5: Commit** — `feat: adaptive chunk sizing from live RSS feedback`.

### Task 3: Static-vs-adaptive benchmark

**Files:**
- Create: `scripts/bench_adaptive.py`

- [ ] **Step 1: Write the script** — run `quantize_model` twice on a given model+budget (env `FQ_STATIC=1` forces controller off via a `quantize_model(adaptive=False)` kwarg added in Task 2), print a two-row table: peak_rss, budget headroom %, violations, elapsed, chunk count. Concrete code, argparse with `--model --max-ram`.
- [ ] **Step 2: Run on Qwen3-0.6B at 512M and 1G** — record both rows in baseline notes. Success criterion (spec hypothesis): adaptive peak-RSS closeness to ceiling ≥ static, violations = 0.
- [ ] **Step 3: Commit** — `feat: adaptive-vs-static memory benchmark`.
