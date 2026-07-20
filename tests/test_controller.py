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
    c = BlockController(working_budget=1000, est_per_row=10,
                        safety=0.85, ewma_alpha=1.0)
    c.observe(rows=10, rss_before=0, rss_after=500)  # 50/row; alpha=1 adopts fully
    assert c.per_row == 50.0
    assert c.next_rows(10**9) == int(0.85 * 1000 / 50)


def test_ewma_blends():
    c = BlockController(working_budget=1000, est_per_row=10, ewma_alpha=0.5)
    c.observe(rows=10, rss_before=0, rss_after=300)  # measured 30/row
    assert c.per_row == 20.0  # 0.5*10 + 0.5*30


def test_negative_delta_ignored():
    c = BlockController(working_budget=1000, est_per_row=10)
    c.observe(rows=10, rss_before=500, rss_after=100)  # GC dip: no signal
    assert c.per_row == 10.0


def test_violation_halves():
    c = BlockController(working_budget=1000, est_per_row=10)
    before = c.next_rows(10**9)
    c.violation()
    after = c.next_rows(10**9)
    assert after <= before // 2 + 1
