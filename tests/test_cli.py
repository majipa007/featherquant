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
    assert cli.parse_size("64MB") == 64 << 20
    with pytest.raises(Exception):
        cli.parse_size("lots")


def test_parse_size_rejects_sub_mebibyte_budgets():
    # "--max-ram 2" meaning 2 BYTES is always a user mistake; the minimum
    # feasible budget is hundreds of MiB. Reject with a hint.
    for bad in ("2", "512", "1024", "900KB"):
        with pytest.raises(Exception, match="[Gg]B"):
            cli.parse_size(bad)


def test_cli_end_to_end_json(tmp_path, monkeypatch, capsys):
    src = tmp_path / "src.gguf"
    make_gguf(src, {"w": np.ones((4, 64), np.float16)})
    out = tmp_path / "out.gguf"
    rp = tmp_path / "out.report.json"
    budget = str(rss_bytes() + (512 << 20))
    monkeypatch.setattr(sys, "argv",
                        ["featherquant", "--model", str(src), "--output", str(out),
                         "--format", "q8_0", "--max-ram", budget,
                         "--report", str(rp), "--ui", "none", "--json"])
    cli.main()
    assert out.exists()
    stats = json.loads(capsys.readouterr().out)   # stdout is pure JSON
    assert stats["peak_rss"] <= int(budget)
    assert json.loads(rp.read_text())["chunks"] == stats["chunks"]


def test_cli_plain_ui_goes_to_stderr(tmp_path, monkeypatch, capsys):
    src = tmp_path / "src.gguf"
    make_gguf(src, {"w": np.ones((4, 64), np.float16)})
    out = tmp_path / "out.gguf"
    budget = str(rss_bytes() + (512 << 20))
    monkeypatch.setattr(sys, "argv",
                        ["featherquant", "--model", str(src), "--output", str(out),
                         "--max-ram", budget, "--ui", "plain"])
    cli.main()
    captured = capsys.readouterr()
    assert "[1/1] w" in captured.err          # progress lines on stderr
    assert captured.out == ""                 # stdout clean without --json


def test_cli_auto_falls_back_to_plain_when_piped(tmp_path, monkeypatch, capsys):
    src = tmp_path / "src.gguf"
    make_gguf(src, {"w": np.ones((4, 64), np.float16)})
    out = tmp_path / "out.gguf"
    budget = str(rss_bytes() + (512 << 20))
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False, raising=False)
    monkeypatch.setattr(sys, "argv",
                        ["featherquant", "--model", str(src), "--output", str(out),
                         "--max-ram", budget])
    cli.main()
    assert "\x1b[" not in capsys.readouterr().err   # plain mode, no escapes
