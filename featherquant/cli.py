"""Command-line interface for FeatherQuant."""
import argparse
import json
import re
import sys

from .engine import quantize_model


def parse_size(s: str) -> int:
    """Parse a human-readable size ('2GB', '512M', '1.5GiB', '1024') to bytes."""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([KMGT])?I?B?", s.strip(), re.IGNORECASE)
    if not m:
        raise argparse.ArgumentTypeError(f"bad size: {s!r} (try 2GB, 512M, 1024)")
    # Suffix letter -> power of 1024; no suffix -> plain bytes.
    exp = "KMGT".find(m[2].upper()) + 1 if m[2] else 0
    return int(float(m[1]) * 1024 ** exp)


def main() -> None:
    """Entry point for the ``featherquant`` console script."""
    p = argparse.ArgumentParser(prog="featherquant",
                                description="Memory-bounded GGUF quantization")
    p.add_argument("--model", required=True, help="source F16/BF16 GGUF")
    p.add_argument("--output", required=True, help="output GGUF path")
    p.add_argument("--format", default="q8_0", choices=["q8_0"])
    p.add_argument("--max-ram", required=True, type=parse_size,
                   help="peak RSS budget, e.g. 2GB")
    p.add_argument("--report", help="write JSON stats here")
    a = p.parse_args()
    try:
        stats = quantize_model(a.model, a.output, a.max_ram, report=a.report)
    except RuntimeError as exc:
        # Turn internal errors into a clean CLI failure, no traceback spam.
        sys.exit(f"featherquant: error: {exc}")
    # Print the run stats so a bare CLI run is self-documenting.
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
