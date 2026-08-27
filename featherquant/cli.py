"""Command-line interface for FeatherQuant."""
import argparse
import json
import os
import re
import sys

from rich.console import Console

from .engine import quantize_model
from .ui import Dashboard, PlainReporter, summary_table


def positive_int(s: str) -> int:
    """argparse type: an integer >= 1."""
    try:
        n = int(s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not an integer: {s!r}") from exc
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
    return n


def parse_size(s: str) -> int:
    """Parse a human-readable size ('2GB', '512M', '1.5GiB') to bytes."""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([KMGT])?I?B?", s.strip(), re.IGNORECASE)
    if not m:
        raise argparse.ArgumentTypeError(f"bad size: {s!r} (try 2GB or 512M)")
    # Suffix letter -> power of 1024; no suffix -> plain bytes.
    exp = "KMGT".find(m[2].upper()) + 1 if m[2] else 0
    n = int(float(m[1]) * 1024 ** exp)
    if n < 1 << 20:
        # A sub-MiB budget is always a typo ("2" meaning 2 GB): the minimum
        # feasible working set is hundreds of MiB.
        raise argparse.ArgumentTypeError(
            f"{s!r} is {n} bytes — did you mean {m[1].rstrip('.0') or m[1]}GB?")
    return n


def main() -> None:
    """Entry point for the ``featherquant`` console script."""
    p = argparse.ArgumentParser(prog="featherquant",
                                description="Memory-bounded GGUF quantization")
    p.add_argument("--model", required=True,
                   help="source F16/BF16 GGUF file, or a sharded-safetensors "
                        "HF model directory (needs --vocab-gguf)")
    p.add_argument("--vocab-gguf",
                   help="metadata/tokenizer-only GGUF for safetensors input "
                        "(scripts/make_vocab_gguf.sh)")
    p.add_argument("--output", required=True, help="output GGUF path")
    p.add_argument("--format", default="q8_0", choices=["q8_0", "q4_k_m"])
    p.add_argument("--max-ram", required=True, type=parse_size,
                   help="peak RSS budget, e.g. 2GB")
    p.add_argument("--report", help="write JSON stats here")
    p.add_argument("--ggml-lib",
                   help="path to libggml-base.so for K-quant formats "
                        "(default: $GGML_LIB or the built-in path)")
    p.add_argument("--resume", action="store_true",
                   help="continue an interrupted run from its manifest "
                        "(verifies committed tensors first)")
    p.add_argument("--threads", type=positive_int, default=os.cpu_count() or 1,
                   help="worker threads inside the ggml kernels; never changes "
                        "output bytes (default: all CPUs)")
    p.add_argument("--ui", choices=["auto", "rich", "plain", "none"],
                   default="auto",
                   help="progress display: rich dashboard, plain lines, or "
                        "none (auto = rich on a TTY, plain otherwise)")
    p.add_argument("--json", action="store_true",
                   help="print the stats JSON to stdout")
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
                               threads=a.threads, progress=reporter)
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


if __name__ == "__main__":
    main()
