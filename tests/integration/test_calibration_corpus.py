"""bench/harness/fetch_calibration_corpus.sh: pin-and-verify behaviour.

No network access here — every case drives the script against a tiny fake
corpus in tmp_path via the DEST override. The real download path is exercised
only by hand (see docs/baselines.md, Step 1); these tests only cover the
provenance guarantee: a corpus that changes after being pinned must fail
loudly, not silently produce a different perplexity number later.
"""
import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path("bench/harness/fetch_calibration_corpus.sh")

pytestmark = pytest.mark.skipif(
    shutil.which("sha256sum") is None, reason="sha256sum absent")


def _run(dest: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), str(dest)], capture_output=True, text=True)


def _pin(dest: Path) -> None:
    """Write dest.sha256 in the same format `sha256sum` itself produces."""
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    Path(f"{dest}.sha256").write_text(f"{digest}  {dest}\n")


def test_verifies_clean_against_pinned_checksum(tmp_path):
    dest = tmp_path / "wiki.test.raw"
    dest.write_text("a small fake calibration corpus\n")
    _pin(dest)
    r = _run(dest)
    assert r.returncode == 0, r.stderr
    assert "verified" in (r.stdout + r.stderr).lower()


def test_corrupted_file_fails_loudly(tmp_path):
    dest = tmp_path / "wiki.test.raw"
    dest.write_text("original corpus content\n")
    _pin(dest)
    dest.write_text("tampered corpus content\n")  # same size, different bytes
    r = _run(dest)
    assert r.returncode != 0
    assert str(dest) in (r.stdout + r.stderr)


def test_missing_sha256_pins_existing_corpus(tmp_path):
    dest = tmp_path / "wiki.test.raw"
    dest.write_text("corpus present but never pinned before\n")
    sha_file = Path(f"{dest}.sha256")
    assert not sha_file.exists()
    r = _run(dest)
    assert r.returncode == 0, r.stderr
    assert sha_file.exists()
    assert str(dest) in sha_file.read_text()
    # Re-running against the now-pinned checksum must still pass.
    r2 = _run(dest)
    assert r2.returncode == 0, r2.stderr
