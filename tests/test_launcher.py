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


def _touch(p: Path, data: bytes = b"GGUF") -> Path:
    p.write_bytes(data)
    return p


def test_wizard_composes_full_command(tmp_path):
    model = _touch(tmp_path / "tiny.gguf")
    # inputs: model, format=1 (q8_0), output=default, ram=default
    stdin = f"{model}\n1\n\n\n"
    r = run_sh(stdin=stdin)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "--model" in out and "tiny.gguf" in out
    assert "--format q8_0" in out
    assert "--max-ram 2GB" in out
    assert "tiny-q8_0.gguf" in out            # derived default output name


def test_wizard_reasks_on_missing_model(tmp_path):
    model = _touch(tmp_path / "real.gguf")
    stdin = f"/nope/missing.gguf\n{model}\n1\n\n\n"
    r = run_sh(stdin=stdin)
    assert r.returncode == 0, r.stderr
    assert "not found" in r.stderr
    assert "real.gguf" in r.stdout


def test_wizard_directory_model_requires_vocab(tmp_path):
    d = tmp_path / "hfmodel"
    d.mkdir()
    vocab = _touch(tmp_path / "vocab.gguf")
    # inputs: model dir, vocab, format=2 (q4_k_m), output default, ram 1GB
    stdin = f"{d}\n{vocab}\n2\n\n1GB\n"
    r = run_sh(stdin=stdin)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "--vocab-gguf" in out and "vocab.gguf" in out
    assert "--format q4_k_m" in out and "--max-ram 1GB" in out


def test_wizard_offers_resume_when_manifest_exists(tmp_path):
    model = _touch(tmp_path / "m.gguf")
    out_path = tmp_path / "out.gguf"
    _touch(out_path.with_suffix(".gguf.manifest.json"), b"{}")
    # inputs: model, format=1, output=explicit, ram default, resume default(y)
    stdin = f"{model}\n1\n{out_path}\n\n\n"
    r = run_sh(stdin=stdin)
    assert r.returncode == 0, r.stderr
    assert "--resume" in r.stdout


def test_wizard_can_decline_resume(tmp_path):
    model = _touch(tmp_path / "m.gguf")
    out_path = tmp_path / "out.gguf"
    _touch(out_path.with_suffix(".gguf.manifest.json"), b"{}")
    stdin = f"{model}\n1\n{out_path}\n\nn\n"
    r = run_sh(stdin=stdin)
    assert r.returncode == 0, r.stderr
    assert "--resume" not in r.stdout
