"""Checkpoint + resume integration tests on synthetic models."""
import numpy as np

from featherquant.engine import quantize_model, rss_bytes
from featherquant.manifest import Manifest, sha256_file_region
from tests.conftest import make_gguf


def _make_model(tmp_path):
    rng = np.random.default_rng(0)
    sp = tmp_path / "src.gguf"
    make_gguf(sp, {"blk.0.attn_q.weight": rng.standard_normal((10, 64)).astype(np.float16),
                   "blk.0.attn_norm.weight": rng.standard_normal(64).astype(np.float32),
                   "blk.1.attn_q.weight": rng.standard_normal((6, 64)).astype(np.float16)})
    return sp


def _budget():
    return rss_bytes() + (512 << 20)


def test_manifest_written_and_complete(tmp_path):
    sp = _make_model(tmp_path)
    out = tmp_path / "out.gguf"
    quantize_model(str(sp), str(out), _budget())
    m = Manifest.load(str(out) + ".manifest.json")
    assert m.status == "complete"
    assert len(m.tensors) == 3
    for e in m.tensors:
        assert e.sha256 is not None
        assert sha256_file_region(str(out), e.offset, e.nbytes) == e.sha256
        assert (e.offset - m.header_end) % 32 == 0  # aligned relative to data start
    assert m.header_sha256 == sha256_file_region(str(out), 0, m.header_end)
