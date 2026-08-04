import json
import shutil
import subprocess
from pathlib import Path

import pytest

HARNESS = Path("bench/harness/run_baseline.sh")


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
