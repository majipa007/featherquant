import pytest

from featherquant.approx_costs import format_option, load_costs


def test_loads_every_rung():
    costs = load_costs()
    assert "hessian_diagonal" in costs
    assert costs["hessian_full"].measured is True
    assert costs["hessian_diagonal"].measured is False


def test_unmeasured_option_line_says_so():
    costs = load_costs()
    line = format_option(costs["hessian_diagonal"], "--hessian-approx=diagonal")
    assert "UNMEASURED" in line


def test_measured_option_line_has_numbers(tmp_path):
    doc = tmp_path / "costs.md"
    doc.write_text(
        "| rung | flag | peak Δ | runtime Δ | PPL Δ | downstream task Δ | source |\n"
        "|---|---|---|---|---|---|---|\n"
        "| hessian_diagonal | `--hessian-approx=diagonal` | -598 MB | +4% | "
        "+0.31 | -0.4 | m6_diag.json |\n")
    costs = load_costs(str(doc))
    c = costs["hessian_diagonal"]
    assert c.peak_delta_bytes == -598 * 1000 * 1000
    assert c.ppl_delta == pytest.approx(0.31)
    assert c.measured is True
    assert "measured PPL cost +0.31" in format_option(c, "--hessian-approx=diagonal")
