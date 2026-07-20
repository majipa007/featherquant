"""Tests for the featherquant CLI."""
import json
import sys

import numpy as np
import pytest

from featherquant import cli
from featherquant.engine import rss_bytes
from tests.conftest import make_gguf


def test_parse_size():
    assert cli.parse_size("2GB") == 2 << 30
    assert cli.parse_size("1.5GiB") == int(1.5 * (1 << 30))
    assert cli.parse_size("512M") == 512 << 20
    assert cli.parse_size("64KB") == 64 << 10
    assert cli.parse_size("1024") == 1024
    with pytest.raises(Exception):
        cli.parse_size("lots")


def test_cli_end_to_end(tmp_path, monkeypatch):
    src = tmp_path / "src.gguf"
    make_gguf(src, {"w": np.ones((4, 64), np.float16)})
    out = tmp_path / "out.gguf"
    rp = tmp_path / "out.report.json"
    budget = str(rss_bytes() + (512 << 20))
    monkeypatch.setattr(sys, "argv",
                        ["featherquant", "--model", str(src), "--output", str(out),
                         "--format", "q8_0", "--max-ram", budget, "--report", str(rp)])
    cli.main()
    assert out.exists()
    stats = json.loads(rp.read_text())
    assert stats["peak_rss"] <= int(budget)
