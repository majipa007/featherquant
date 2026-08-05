"""bench/harness/run_gptq_reference.py: script-level sanity checks, plus real
unit tests for the two pieces of pure-Python logic the script contains
(calibration chunking, directory hashing) -- neither needs torch or
gptqmodel, so both are imported and exercised for real, not mocked.

The rest of this script's dependencies (gptqmodel, torch) are deliberately
absent from this repo's venv -- it is meant to run in a throwaway GPU venv a
human sets up by hand (see docs/baselines.md, Baseline 4). Mocking a GPU
quantizer here would test nothing real, so those parts stay limited to what
is honestly verifiable without it: the script parses as valid Python, and it
validates its own argv before touching any heavy (torch/gptqmodel) import.

The script's actual quantization/perplexity/layer-error behaviour is
UNVERIFIED until a human runs it on a GPU.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path("bench/harness/run_gptq_reference.py")


def _load_module():
    """Import the script as a module for direct unit testing.

    Safe to do without torch/gptqmodel installed: the module's top-level
    code only defines constants and functions -- the heavy imports
    (torch, gptqmodel) are all deferred inside main() and inside the
    functions that actually need them, so simply importing the module
    executes none of them.
    """
    spec = importlib.util.spec_from_file_location("run_gptq_reference", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_chunking_obtains_exactly_the_requested_count_and_logs_it(capsys):
    """A corpus with exactly enough characters for N full chunks must
    yield exactly N chunks, each of the expected size, and must log the
    requested-vs-obtained counts (Baseline 4's real corpus produced a
    silent single-giant-sample bug before this was fixed -- the log line
    is what makes that class of bug visible instead of silent again)."""
    module = _load_module()
    chunk_chars = 512 * 4
    text = "a" * (chunk_chars * 128)
    chunks = module._chunk_calibration_corpus(
        text, samples=128, seqlen_tokens=512, chars_per_token=4)
    assert len(chunks) == 128
    assert all(len(c) == chunk_chars for c in chunks)
    logged = capsys.readouterr().err
    assert "requested 128" in logged
    assert "obtained 128" in logged


def test_chunking_drops_short_trailing_chunk():
    """A trailing partial chunk (corpus length not an exact multiple of the
    chunk size) must be dropped, not kept undersized -- it isn't a real
    ~512-token sample."""
    module = _load_module()
    chunk_chars = 512 * 4
    text = "a" * (chunk_chars * 3 + 10)  # 3 full chunks + a short remainder
    chunks = module._chunk_calibration_corpus(
        text, samples=3, seqlen_tokens=512, chars_per_token=4)
    assert len(chunks) == 3
    assert all(len(c) == chunk_chars for c in chunks)


def test_chunking_fails_loudly_when_corpus_too_short():
    """Must never silently calibrate on fewer chunks than requested -- the
    real corpus regression this guards against produced ONE giant sample
    instead of 128 and said nothing about it."""
    module = _load_module()
    text = "a" * 100  # nowhere near enough for even one 512*4-char chunk
    with pytest.raises(RuntimeError, match=r"requested 128.*only 0"):
        module._chunk_calibration_corpus(
            text, samples=128, seqlen_tokens=512, chars_per_token=4)


def test_hash_directory_is_deterministic_and_content_sensitive(tmp_path):
    """The output directory digest must be the same across repeated calls
    (order-independent -- gptqmodel.save()'s directory has no guaranteed
    file order) and must change if any file's content changes."""
    module = _load_module()
    d = tmp_path / "out"
    d.mkdir()
    (d / "b.json").write_text('{"b": 1}')
    (d / "a.safetensors").write_bytes(b"weights-placeholder")
    sub = d / "shard"
    sub.mkdir()
    (sub / "c.bin").write_bytes(b"more-weights")

    first = module._hash_directory(str(d))
    second = module._hash_directory(str(d))
    assert first == second
    assert len(first) == 64  # sha256 hex digest

    (sub / "c.bin").write_bytes(b"different-weights")
    changed = module._hash_directory(str(d))
    assert changed != first
