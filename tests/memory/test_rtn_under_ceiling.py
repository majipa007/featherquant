"""M2 gate: RTN output is bit-identical to llama-quantize, under a ceiling.

A test that passes without a ceiling is not a memory test (spec §9), so
the featherquant run happens inside systemd-run with MemoryMax set and
swap disabled. If the kernel OOM-kills it, the test fails.
"""
import subprocess
import sys

import pytest

from featherquant.validator import compare_gguf, structural_check
from tests.memory.conftest import LLAMA_BIN, MODEL, needs_cgroup, needs_llama

pytestmark = [pytest.mark.memory, pytest.mark.slow, needs_cgroup, needs_llama]


@pytest.mark.parametrize("fmt,ref_type", [("q8_0", "Q8_0"), ("q4_k_m", "Q4_K_M")])
def test_bit_identical_under_ceiling(tmp_path, fmt, ref_type):
    ref = tmp_path / f"ref_{fmt}.gguf"
    out = tmp_path / f"fq_{fmt}.gguf"
    subprocess.run([f"{LLAMA_BIN}/llama-quantize", MODEL, str(ref), ref_type],
                   check=True, capture_output=True)
    r = subprocess.run(
        ["bash", "bench/harness/run_under_ceiling.sh", "1G",
         sys.executable, "-m", "featherquant.cli", "--model", MODEL,
         "--output", str(out), "--format", fmt, "--max-ram", "1GB",
         "--ui", "none"],
        capture_output=True, text=True)
    assert r.returncode == 0, (
        f"exit {r.returncode} (137 = OOM-killed by the 1G ceiling)\n{r.stderr}")
    assert structural_check(str(out)) == []
    assert compare_gguf(str(ref), str(out)) == []
