import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

HARNESS = Path("bench/harness/run_baseline.sh")


def _systemd_user_session_up() -> bool:
    """`shutil.which("systemd-run")` only proves the binary exists — the
    --user session bus can still be down (no active login/lingering session,
    common in containers/WSL), which fails differently (bus error) than an
    OOM-kill and would otherwise make an OOM assertion flaky."""
    if not shutil.which("systemd-run"):
        return False
    probe = subprocess.run(
        ["systemd-run", "--user", "--scope", "--collect", "true"],
        capture_output=True, text=True)
    return probe.returncode == 0


@pytest.mark.skipif(not shutil.which("/usr/bin/time"), reason="GNU time absent")
def test_harness_emits_run_manifest(tmp_path):
    out = tmp_path / "artifact.bin"
    cmd = json.dumps(["sh", "-c", f"printf hello > {out}"])
    r = subprocess.run(
        ["bash", str(HARNESS), cmd, "t_harness", "noop", "test/model",
         str(out), "nvme"],
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "FQ_MANIFEST_DIR": str(tmp_path)})
    assert r.returncode == 0, r.stderr
    m = json.loads((tmp_path / "t_harness.json").read_text())
    assert m["method"] == "noop"
    assert m["runtime_seconds"] >= 0
    assert m["peak_observed_bytes"] > 0
    assert m["oom_killed"] is False
    assert len(m["output_sha256"]) == 64


def test_ceiling_wrapper_reports_oom_exit_code(tmp_path):
    if not shutil.which("systemd-run"):
        pytest.skip("systemd-run absent")
    r = subprocess.run(
        ["bash", "bench/harness/run_under_ceiling.sh", "32M",
         "python", "-c", "b=bytearray(512<<20)"],
        capture_output=True, text=True)
    assert r.returncode != 0


@pytest.mark.skipif(not shutil.which("/usr/bin/time"), reason="GNU time absent")
def test_harness_records_oom_kill_under_ceiling(tmp_path):
    """Baseline-2 end-to-end: run_baseline.sh wrapping run_under_ceiling.sh
    around a memory hog under a tight ceiling must still emit a manifest,
    and that manifest must say oom_killed=True — not silently succeed."""
    if not _systemd_user_session_up():
        pytest.skip("systemd --user session unavailable (no session bus)")
    out = tmp_path / "never_created.bin"
    cmd = json.dumps(["bash", "bench/harness/run_under_ceiling.sh", "32M",
                       "python", "-c", "b=bytearray(512<<20)"])
    # Full inherited env here (not the restricted PATH used above): systemd-run
    # --user needs XDG_RUNTIME_DIR/the session bus, which only the real
    # environment provides. run_baseline.sh's own Python calls stay
    # PATH-independent regardless (resolved via its own script location).
    env = dict(os.environ)
    env["FQ_MANIFEST_DIR"] = str(tmp_path)
    r = subprocess.run(
        ["bash", str(HARNESS), cmd, "t_oom", "noop", "test/model",
         str(out), "nvme", str(32 << 20)],
        capture_output=True, text=True, env=env)
    assert r.returncode == 137, r.stderr
    m = json.loads((tmp_path / "t_oom.json").read_text())
    assert m["oom_killed"] is True
    assert m["runtime_seconds"] is not None and m["runtime_seconds"] >= 0
    assert m["enforcement"] == "cgroup_v2_memory_max"
    assert m["output_sha256"] is None  # artifact never existed
