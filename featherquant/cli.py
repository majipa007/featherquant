"""Command-line interface for FeatherQuant."""
import argparse
import json
import re
import sys

from rich.console import Console

from .engine import quantize_model
from .indexer import index_model
from .ui import Dashboard, PlainReporter, summary_table


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


# Spec §8 subcommands. Only "index" is implemented so far (Task 6); plan,
# run, verify and bench land in Tasks 9-10 — see _dispatch below.
SUBCOMMANDS = {"index", "plan", "run", "verify", "bench"}


def main() -> None:
    """Entry point: spec §8 subcommands, or the legacy flat-flag form."""
    argv = sys.argv[1:]
    if argv and argv[0] in SUBCOMMANDS:
        _dispatch(argv[0], argv[1:])
        return
    _legacy_quantize(argv)      # the existing --model/--output/--max-ram path


def _legacy_quantize(argv: list[str]) -> None:
    """The original flat-flag quantize command (pre-subcommand CLI)."""
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
    p.add_argument("--ui", choices=["auto", "rich", "plain", "none"],
                   default="auto",
                   help="progress display: rich dashboard, plain lines, or "
                        "none (auto = rich on a TTY, plain otherwise)")
    p.add_argument("--json", action="store_true",
                   help="print the stats JSON to stdout")
    a = p.parse_args(argv)
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


def _cmd_index(argv: list[str]) -> None:
    """``featherquant index <model_path> -o manifest.json``"""
    p = argparse.ArgumentParser(prog="featherquant index",
                                description="Emit a model index (metadata only)")
    p.add_argument("model_path", help="HF checkpoint directory or GGUF file")
    p.add_argument("-o", "--output", required=True, help="index JSON path")
    a = p.parse_args(argv)
    try:
        idx = index_model(a.model_path)
    except RuntimeError as exc:
        sys.exit(f"featherquant index: error: {exc}")
    idx.save(a.output)
    print(f"{len(idx.tensors)} tensors, {idx.n_layers} layers, "
          f"largest tensor {idx.largest_tensor_bytes / 2**20:.1f} MiB -> "
          f"{a.output}")


def _dispatch(name: str, argv: list[str]) -> None:
    """Route a subcommand; later tasks register plan/run/verify/bench here."""
    handlers = {"index": _cmd_index}
    try:
        handlers[name](argv)
    except KeyError:
        sys.exit(f"featherquant: {name} is not implemented yet")


if __name__ == "__main__":
    main()
