"""Invariant 4: same input + config + seed -> byte-identical output."""
import hashlib
from pathlib import Path

import numpy as np

from featherquant.engine import quantize_model
from tests.conftest import make_gguf


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_two_runs_are_byte_identical(tmp_path):
    src = tmp_path / "src.gguf"
    rng = np.random.default_rng(0)
    make_gguf(src, {"blk.0.ffn_down.weight":
                    rng.standard_normal((8, 256), dtype=np.float32),
                    "blk.0.attn_norm.weight": np.ones((256,), np.float32)})
    a, b = tmp_path / "a.gguf", tmp_path / "b.gguf"
    for out in (a, b):
        quantize_model(str(src), str(out), max_ram=512 << 20, fmt="q8_0")
    assert _sha(a) == _sha(b)


def test_chunking_does_not_change_bytes(tmp_path):
    """Different chunk sizes must produce the same file (rows independent)."""
    src = tmp_path / "src.gguf"
    rng = np.random.default_rng(1)
    make_gguf(src, {"blk.0.ffn_down.weight":
                    rng.standard_normal((16, 256), dtype=np.float32)})
    a, b = tmp_path / "a.gguf", tmp_path / "b.gguf"
    quantize_model(str(src), str(a), max_ram=512 << 20, fmt="q8_0",
                   _force_chunk_rows=1)
    quantize_model(str(src), str(b), max_ram=512 << 20, fmt="q8_0",
                   _force_chunk_rows=16)
    assert _sha(a) == _sha(b)
