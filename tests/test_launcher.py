"""Subprocess tests for featherquant.sh (bootstrap seams + passthrough)."""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SH = str(ROOT / "featherquant.sh")


def run_sh(args=(), stdin="", extra_env=None, timeout=60, cwd=ROOT):
    """Run the launcher with the dry-run seam on; return CompletedProcess."""
    env = {**os.environ, "FQ_DRY_RUN": "1", "FQ_SKIP_BOOTSTRAP": "1",
           **(extra_env or {})}
    return subprocess.run(["bash", SH, *args], input=stdin, text=True,
                          capture_output=True, env=env, cwd=cwd,
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
    # Make the stamp fresh ourselves so this test doesn't depend on the
    # ambient repo state (fresh clone => real network install otherwise).
    stamp = ROOT / ".venv" / ".fq-deps-stamp"
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.touch()
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


def test_wizard_expands_tilde_model_path(tmp_path):
    _touch(tmp_path / "m.gguf")
    # inputs: model as ~/m.gguf, format=1, output=default, ram=default
    stdin = "~/m.gguf\n1\n\n\n"
    r = run_sh(stdin=stdin, extra_env={"HOME": str(tmp_path)})
    assert r.returncode == 0, r.stderr
    assert str(tmp_path / "m.gguf") in r.stdout


def test_relative_paths_resolve_against_callers_cwd(tmp_path):
    _touch(tmp_path / "rel.gguf")
    # inputs: bare relative model name, format=1, output=default, ram=default
    stdin = "rel.gguf\n1\n\n\n"
    r = run_sh(stdin=stdin, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "rel.gguf" in r.stdout


def _stub_generator(tmp_path: Path, exit_code: int = 0) -> str:
    """Fake make_vocab_gguf.sh: touches the output file (or fails)."""
    stub = tmp_path / "stub-gen.sh"
    if exit_code == 0:
        stub.write_text('#!/bin/bash\ntouch "$2"\n')
    else:
        stub.write_text(f'#!/bin/bash\nexit {exit_code}\n')
    stub.chmod(0o755)
    return str(stub)


def test_wizard_generates_missing_vocab_on_enter(tmp_path):
    d = tmp_path / "hfmodel"
    d.mkdir()
    # inputs: model dir, vocab=default(empty), generate=default(y),
    #         format=1, output default, ram default
    stdin = f"{d}\n\n\n1\n\n\n"
    r = run_sh(stdin=stdin, cwd=str(tmp_path),
               extra_env={"FQ_VOCAB_GEN": _stub_generator(tmp_path)})
    assert r.returncode == 0, r.stderr
    assert "--vocab-gguf" in r.stdout and "hfmodel-vocab.gguf" in r.stdout
    assert (tmp_path / "hfmodel-vocab.gguf").exists()  # stub actually ran


def test_wizard_vocab_directory_input_gets_clear_message(tmp_path):
    d = tmp_path / "hfmodel"
    d.mkdir()
    vocab = _touch(tmp_path / "v.gguf")
    # first vocab answer is a directory, second is a real file
    stdin = f"{d}\n{tmp_path}\n{vocab}\n1\n\n\n"
    r = run_sh(stdin=stdin, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "is a directory, need a .gguf file" in r.stderr
    assert "v.gguf" in r.stdout


def test_wizard_generation_failure_reasks(tmp_path):
    d = tmp_path / "hfmodel"
    d.mkdir()
    vocab = _touch(tmp_path / "fallback.gguf")
    # generation fails once, user then supplies an existing file
    stdin = f"{d}\n\n\n{vocab}\n1\n\n\n"
    r = run_sh(stdin=stdin, cwd=str(tmp_path),
               extra_env={"FQ_VOCAB_GEN": _stub_generator(tmp_path, exit_code=1)})
    assert r.returncode == 0, r.stderr
    assert "failed" in r.stderr
    assert "fallback.gguf" in r.stdout


def test_wizard_tab_completes_paths_on_tty(tmp_path):
    """On a real terminal, Tab must filename-complete in path prompts."""
    import pty
    import select
    import signal
    import time

    _touch(tmp_path / "unique-model.gguf")
    pid, fd = pty.fork()
    if pid == 0:  # child: run the wizard on a pty
        os.chdir(tmp_path)
        os.environ["FQ_DRY_RUN"] = "1"
        os.environ["FQ_SKIP_BOOTSTRAP"] = "1"
        os.environ["TERM"] = "dumb"
        os.execvp("bash", ["bash", SH])

    buf = b""

    def read_until(marker: bytes, deadline: float = 8.0) -> bool:
        nonlocal buf
        end = time.monotonic() + deadline
        while marker not in buf and time.monotonic() < end:
            r, _, _ = select.select([fd], [], [], 0.2)
            if r:
                try:
                    chunk = os.read(fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
        return marker in buf

    try:
        assert read_until(b"Model ("), buf.decode(errors="replace")
        os.write(fd, b"uniq\t")      # readline should complete the filename
        time.sleep(0.5)
        os.write(fd, b"\n1\n\n\n")   # accept model, format, output, ram
        read_until(b"--model")
        time.sleep(0.3)
        r, _, _ = select.select([fd], [], [], 1.0)
        if r:
            try:
                buf += os.read(fd, 8192)
            except OSError:
                pass
    finally:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        os.waitpid(pid, 0)
        os.close(fd)

    text = buf.decode(errors="replace")
    assert "not found" not in text, text   # completion failed -> re-ask loop
    assert "unique-model.gguf" in text, text
