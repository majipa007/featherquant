#!/usr/bin/env python3
"""Static vs adaptive chunk sizing: peak RSS, headroom, time.

Runs quantize_model twice in subprocesses (fresh RSS baselines) and prints
a two-row table. Usage:

    python scripts/bench_adaptive.py --model M.gguf --max-ram 512M
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from featherquant.cli import parse_size

_CHILD = """
import json, sys
from featherquant.engine import quantize_model
stats = quantize_model(sys.argv[1], sys.argv[2], int(sys.argv[3]),
                       adaptive=(sys.argv[4] == "1"))
print(json.dumps(stats))
"""


def run(model: str, out: str, max_ram: int, adaptive: bool) -> dict:
    """One quantization in a fresh process; returns its stats dict."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD, model, out, str(max_ram),
             "1" if adaptive else "0"],
            capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        sys.exit(f"child failed ({'adaptive' if adaptive else 'static'}): "
                 f"{exc.stderr.strip()}")
    stats: dict = json.loads(proc.stdout.splitlines()[-1])
    return stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--max-ram", required=True, type=parse_size)
    a = p.parse_args()

    rows = []
    with tempfile.TemporaryDirectory() as td:
        for adaptive in (False, True):
            out = str(Path(td) / f"bench-{int(adaptive)}.gguf")
            s = run(a.model, out, a.max_ram, adaptive)
            rows.append((("adaptive" if adaptive else "static"), s))

    print(f"{'mode':<9} {'peak_rss_MiB':>12} {'headroom_%':>10} "
          f"{'violations':>10} {'chunks':>7} {'elapsed_s':>9}")
    for name, s in rows:
        headroom = 100.0 * (s["max_ram"] - s["peak_rss"]) / s["max_ram"]
        print(f"{name:<9} {s['peak_rss'] / (1 << 20):>12.1f} {headroom:>10.1f} "
              f"{s['budget_violations']:>10} {s['chunks']:>7} "
              f"{s['elapsed_s']:>9.1f}")


if __name__ == "__main__":
    main()
