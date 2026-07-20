"""Checkpoint + resume integration tests on synthetic models."""
import numpy as np
import pytest

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


def test_resume_completes_identically(tmp_path):
    sp = _make_model(tmp_path)
    ref, out = tmp_path / "ref.gguf", tmp_path / "out.gguf"
    quantize_model(str(sp), str(ref), _budget())
    with pytest.raises(RuntimeError):
        quantize_model(str(sp), str(out), _budget(), _fail_after=1)
    m = Manifest.load(str(out) + ".manifest.json")
    assert m.status == "in_progress"
    assert m.tensors[0].sha256 is not None and m.tensors[1].sha256 is None
    quantize_model(str(sp), str(out), _budget(), resume=True)
    assert out.read_bytes() == ref.read_bytes()
    assert Manifest.load(str(out) + ".manifest.json").status == "complete"


def test_resume_rejects_changed_source(tmp_path):
    sp = _make_model(tmp_path)
    out = tmp_path / "out.gguf"
    with pytest.raises(RuntimeError):
        quantize_model(str(sp), str(out), _budget(), _fail_after=1)
    rng = np.random.default_rng(9)  # different content = different size/mtime
    make_gguf(sp, {"blk.0.attn_q.weight": rng.standard_normal((12, 64)).astype(np.float16)})
    with pytest.raises((RuntimeError, SystemExit)):
        quantize_model(str(sp), str(out), _budget(), resume=True)


def test_resume_detects_corrupt_committed_tensor(tmp_path):
    sp = _make_model(tmp_path)
    ref, out = tmp_path / "ref.gguf", tmp_path / "out.gguf"
    quantize_model(str(sp), str(ref), _budget())
    with pytest.raises(RuntimeError):
        quantize_model(str(sp), str(out), _budget(), _fail_after=2)
    m = Manifest.load(str(out) + ".manifest.json")
    e0 = m.tensors[0]
    with open(out, "r+b") as f:  # flip one byte inside committed tensor 0
        f.seek(e0.offset + 5)
        b = f.read(1)
        f.seek(e0.offset + 5)
        f.write(bytes([b[0] ^ 0xFF]))
    quantize_model(str(sp), str(out), _budget(), resume=True)
    assert out.read_bytes() == ref.read_bytes()


def test_resume_without_manifest_runs_fresh(tmp_path):
    sp = _make_model(tmp_path)
    out = tmp_path / "out.gguf"
    quantize_model(str(sp), str(out), _budget(), resume=True)  # no manifest yet
    assert Manifest.load(str(out) + ".manifest.json").status == "complete"


def test_resume_on_complete_exits(tmp_path):
    sp = _make_model(tmp_path)
    out = tmp_path / "out.gguf"
    quantize_model(str(sp), str(out), _budget())
    with pytest.raises(SystemExit):
        quantize_model(str(sp), str(out), _budget(), resume=True)
