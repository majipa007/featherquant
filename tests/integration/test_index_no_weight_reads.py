"""The indexer's gate: it must never read weight bytes.

strace counts bytes returned by read()/pread64() on the shard. Header
bytes are legitimate; anything on the order of the tensor payload is not.
"""
import json
import re
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(not shutil.which("strace"), reason="strace absent")


def _make_model(tmp_path):
    import numpy as np

    from tests.unit.test_indexer import write_safetensors  # fixture writer
    # 4 MiB of payload: any full read is unmissable next to a ~1 KiB header.
    write_safetensors(tmp_path / "model.safetensors",
                      {"model.embed_tokens.weight": np.zeros((1024, 512), np.float16),
                       "model.norm.weight": np.zeros((512,), np.float32)})
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen3", "num_hidden_layers": 0, "hidden_size": 512,
        "intermediate_size": 1024, "vocab_size": 1024,
        "num_attention_heads": 8, "num_key_value_heads": 8, "head_dim": 64}))
    return tmp_path


def test_index_reads_only_headers(tmp_path):
    model = _make_model(tmp_path)
    out = tmp_path / "manifest.json"
    trace = tmp_path / "trace.txt"
    subprocess.run(
        ["strace", "-f", "-e", "trace=read,pread64", "-o", str(trace),
         sys.executable, "-m", "featherquant.cli", "index", str(model),
         "-o", str(out)], check=True, capture_output=True)
    text = trace.read_text()
    # Sum the return values of successful reads on any .safetensors fd.
    total = sum(int(m) for m in re.findall(r"= (\d+)$", text, re.M))
    payload = 1024 * 512 * 2
    assert out.exists()
    assert total < payload // 4, f"indexer read {total} B; payload is {payload} B"
