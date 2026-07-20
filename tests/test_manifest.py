"""Manifest atomic save/load/verify unit tests."""
import hashlib
import json

import pytest

from featherquant.manifest import Manifest, TensorEntry, sha256_file_region


def _mk():
    return Manifest(source_path="/x/src.gguf", source_size=100, source_mtime_ns=5,
                    config={"fmt": "q8_0"}, header_end=64, header_sha256="h" * 64,
                    tensors=[TensorEntry("a", 8, 64, 34, None)],
                    status="in_progress")


def test_roundtrip_atomic(tmp_path):
    m = _mk()
    p = tmp_path / "out.gguf.manifest.json"
    m.save(str(p))
    leftovers = [f for f in tmp_path.iterdir() if f.name != p.name]
    assert not leftovers, f"temp files left behind: {leftovers}"
    m2 = Manifest.load(str(p))
    assert m2 == m


def test_load_rejects_wrong_version(tmp_path):
    p = tmp_path / "m.json"
    m = _mk()
    m.save(str(p))
    d = json.loads(p.read_text())
    d["version"] = 999
    p.write_text(json.dumps(d))
    with pytest.raises(RuntimeError):
        Manifest.load(str(p))


def test_load_rejects_garbage(tmp_path):
    p = tmp_path / "m.json"
    p.write_text("not json {")
    with pytest.raises(RuntimeError):
        Manifest.load(str(p))


def test_sha256_file_region(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"A" * 10 + b"B" * 20 + b"C" * 5)
    assert sha256_file_region(str(p), 10, 20) == hashlib.sha256(b"B" * 20).hexdigest()


def test_sha256_file_region_short_file(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"A" * 10)
    with pytest.raises(RuntimeError):
        sha256_file_region(str(p), 0, 100)
