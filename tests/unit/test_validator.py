"""Tests for featherquant.validator: structural + comparative GGUF checks."""
import numpy as np

from featherquant.validator import compare_gguf, structural_check
from tests.conftest import make_gguf


def test_identical_files_compare_clean(tmp_path):
    a, b = tmp_path / "a.gguf", tmp_path / "b.gguf"
    arr = np.arange(64, dtype=np.float32).reshape(2, 32)
    make_gguf(a, {"t.weight": arr})
    make_gguf(b, {"t.weight": arr})
    assert compare_gguf(str(a), str(b)) == []


def test_byte_difference_is_reported(tmp_path):
    a, b = tmp_path / "a.gguf", tmp_path / "b.gguf"
    arr = np.arange(64, dtype=np.float32).reshape(2, 32)
    make_gguf(a, {"t.weight": arr})
    make_gguf(b, {"t.weight": arr + 1})
    msgs = compare_gguf(str(a), str(b))
    assert len(msgs) == 1 and "byte mismatch" in msgs[0]


def test_structural_check_passes_on_valid_file(tmp_path):
    p = tmp_path / "a.gguf"
    make_gguf(p, {"t.weight": np.zeros((2, 32), np.float32)})
    assert structural_check(str(p)) == []


def test_structural_check_catches_truncation(tmp_path):
    p = tmp_path / "a.gguf"
    make_gguf(p, {"t.weight": np.zeros((2, 32), np.float32)})
    with open(p, "r+b") as f:
        f.truncate(p.stat().st_size - 16)
    msgs = structural_check(str(p))
    assert any("truncated" in m for m in msgs)
