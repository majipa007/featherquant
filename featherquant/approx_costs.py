"""Read docs/approximation_costs.md — the planner's only source of costs.

Spec §4.2: "Never guess the quality cost." A rung whose measurement does
not exist yet is reported as UNMEASURED, in the refusal message, in front
of the user.
"""
import re
from dataclasses import dataclass

# MB in the doc means 10^6 bytes (what file-size tools report); ceilings
# elsewhere are binary. Spec §6 requires the distinction be labelled.
_UNITS = {"B": 1, "KB": 10 ** 3, "MB": 10 ** 6, "GB": 10 ** 9}


@dataclass(frozen=True)
class ApproxCost:
    """One row of the ladder table."""
    rung: str
    flag: str
    peak_delta_bytes: int | None
    runtime_delta_pct: float | None
    ppl_delta: float | None
    task_delta: str
    measured: bool


def _size(cell: str) -> int | None:
    """Parse '-598 MB' / '0' into bytes; None when UNMEASURED."""
    cell = cell.strip()
    if cell == "UNMEASURED":
        return None
    m = re.fullmatch(r"([+-]?[\d.]+)\s*([KMG]?B)?", cell)
    if not m:
        raise RuntimeError(f"cannot parse size cell {cell!r} in the cost table")
    return int(float(m[1]) * _UNITS.get((m[2] or "B").upper(), 1))


def _number(cell: str) -> float | None:
    """Parse '+0.31' / '+4%' into a float; None when UNMEASURED."""
    cell = cell.strip().rstrip("%")
    if cell == "UNMEASURED":
        return None
    try:
        return float(cell)
    except ValueError as exc:
        raise RuntimeError(f"cannot parse numeric cell {cell!r}: {exc}") from exc


def load_costs(path: str = "docs/approximation_costs.md") -> dict[str, ApproxCost]:
    """Parse the markdown table into a rung -> ApproxCost lookup."""
    try:
        lines = open(path).read().splitlines()
    except OSError as exc:
        raise RuntimeError(f"cannot read cost table {path}: {exc}") from exc
    costs: dict[str, ApproxCost] = {}
    for line in lines:
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 7 or cells[0] in ("rung", "---"):
            continue
        if set(cells[0]) <= {"-", ":"}:
            continue
        rung, flag, peak, runtime, ppl, task, source = cells
        cost = ApproxCost(rung=rung, flag=flag.strip("`"),
                          peak_delta_bytes=_size(peak),
                          runtime_delta_pct=_number(runtime),
                          ppl_delta=_number(ppl), task_delta=task,
                          measured=source != "UNMEASURED")
        costs[rung] = cost
    if not costs:
        raise RuntimeError(f"no cost rows parsed from {path}")
    return costs


def format_option(cost: ApproxCost, flag: str) -> str:
    """One line of the planner's INFEASIBLE options block."""
    peak = ("UNMEASURED" if cost.peak_delta_bytes is None
            else f"{cost.peak_delta_bytes / 10 ** 6:+.0f} MB")
    ppl = ("PPL cost UNMEASURED" if not cost.measured or cost.ppl_delta is None
           else f"measured PPL cost {cost.ppl_delta:+.2f}")
    return f"{flag:<28} ({peak}, {ppl})"
