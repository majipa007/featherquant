"""Subprocess tests for featherquant.sh (bootstrap seams + passthrough)."""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SH = str(ROOT / "featherquant.sh")


def run_sh(args=(), stdin="", extra_env=None, timeout=60):
    """Run the launcher with the dry-run seam on; return CompletedProcess."""
    env = {**os.environ, "FQ_DRY_RUN": "1", "FQ_SKIP_BOOTSTRAP": "1",
           **(extra_env or {})}
    return subprocess.run(["bash", SH, *args], input=stdin, text=True,
                          capture_output=True, env=env, cwd=ROOT,
                          timeout=timeout)


def test_passthrough_composes_cli_command():
    r = run_sh(["--model", "m.gguf", "--output", "o.gguf", "--max-ram", "1GB"])
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert ".venv/bin/featherquant" in out
    assert "--model m.gguf" in out and "--max-ram 1GB" in out


def test_stdout_carries_only_the_command():
    r = run_sh(["--model", "m.gguf", "--output", "o.gguf", "--max-ram", "1GB"])
    assert r.returncode == 0
    assert len(r.stdout.strip().splitlines()) == 1  # prompts/status live on stderr


def test_no_escape_codes_when_piped():
    r = run_sh(["--model", "m.gguf", "--output", "o.gguf", "--max-ram", "1GB"])
    assert "\x1b[" not in r.stdout + r.stderr


def test_bootstrap_fast_path_skips_reinstall():
    # Real repo already has a synced venv + stamp: a run WITHOUT the skip
    # seam must not install anything (idempotence) and still succeed.
    r = run_sh(["--model", "m.gguf", "--output", "o.gguf", "--max-ram", "1GB"],
               extra_env={"FQ_SKIP_BOOTSTRAP": ""})
    assert r.returncode == 0, r.stderr
    assert "installing" not in r.stderr.lower()
    assert "syncing" not in r.stderr.lower()
