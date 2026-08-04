"""bench/harness/run_gptq_reference.py: script-level sanity checks only.

This script's real dependencies (gptqmodel, torch) are deliberately absent
from this repo's venv -- it is meant to run in a throwaway GPU venv a human
sets up by hand (see docs/baselines.md, Baseline 4). Mocking a GPU quantizer
here would test nothing real, so these checks stay limited to what is
honestly verifiable without it: the script parses as valid Python, and it
validates its own argv before touching any heavy (torch/gptqmodel) import.

The script's actual quantization/perplexity/layer-error behaviour is
UNVERIFIED until a human runs it on a GPU.
"""
import subprocess
import sys
from pathlib import Path

SCRIPT = Path("bench/harness/run_gptq_reference.py")


def test_script_is_syntactically_valid_python():
    """compile() the source -- catches syntax errors without needing the
    script's heavy (torch/gptqmodel) dependencies installed."""
    source = SCRIPT.read_text()
    compile(source, str(SCRIPT), "exec")


def test_no_args_exits_nonzero_with_usage_message_not_a_traceback():
    """Argument validation must happen before any heavy import, so a bare
    invocation (as would happen in this dependency-less venv) fails fast
    with a usage message instead of an ImportError/IndexError traceback."""
    r = subprocess.run([sys.executable, str(SCRIPT)],
                        capture_output=True, text=True)
    assert r.returncode != 0
    assert "usage" in (r.stdout + r.stderr).lower()
    assert "traceback" not in r.stderr.lower()
