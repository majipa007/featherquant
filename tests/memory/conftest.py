"""Shared skips for memory tests: these require cgroup v2 and llama.cpp."""
import os
import shutil
import subprocess

import pytest

LLAMA_BIN = os.environ.get("LLAMA_BIN",
                           os.path.expanduser("~/llama.cpp/build-cpu/bin"))
MODEL = os.environ.get("FQ_TEST_MODEL",
                       os.path.expanduser("~/models/qwen3-0.6b-bf16.gguf"))


def _systemd_user_session_up() -> bool:
    """`shutil.which("systemd-run")` only proves the binary exists — the
    --user session bus can still be down (no active login/lingering session,
    common in containers/WSL2), which fails differently (bus error) than an
    OOM-kill and would otherwise make the ceiling assertion flaky."""
    if not shutil.which("systemd-run"):
        return False
    probe = subprocess.run(
        ["systemd-run", "--user", "--scope", "--collect", "true"],
        capture_output=True, text=True)
    return probe.returncode == 0


needs_cgroup = pytest.mark.skipif(
    not _systemd_user_session_up(),
    reason="cgroup v2 enforcement needs a live systemd --user session bus")
needs_llama = pytest.mark.skipif(
    not os.path.exists(f"{LLAMA_BIN}/llama-quantize") or not os.path.exists(MODEL),
    reason="llama.cpp build or test model not available")
