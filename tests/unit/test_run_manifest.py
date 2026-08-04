import json

from featherquant.run_manifest import RunManifest, host_info, today_ddmmyyyy


def test_new_manifest_has_spec_shape(tmp_path):
    m = RunManifest.new(run_id="m0_rtn_unconstrained",
                        model={"id": "Qwen/Qwen3-0.6B", "revision": "main",
                               "sha256": "0" * 64},
                        method="rtn_q8_0", budget_bytes=2147483648,
                        storage="nvme")
    assert m.enforcement == "cgroup_v2_memory_max"
    assert m.oom_killed is False
    assert m.quality == {"ppl": None, "ppl_dataset": None, "tasks": {}}
    p = tmp_path / "run.json"
    m.save(str(p))
    d = json.loads(p.read_text())
    assert set(d) == {"run_id", "date", "model", "method", "approximations",
                      "budget_bytes", "enforcement", "peak_observed_bytes",
                      "oom_killed", "runtime_seconds", "bytes_read",
                      "bytes_written", "storage", "output_sha256", "quality",
                      "host"}
    assert d["host"]["kernel"] and d["host"]["ram_gb"] > 0


def test_date_is_singapore_format():
    s = today_ddmmyyyy()
    dd, mm, yyyy = s.split("/")
    assert len(dd) == 2 and len(mm) == 2 and len(yyyy) == 4


def test_roundtrip(tmp_path):
    m = RunManifest.new("r", {"id": "x", "revision": "y", "sha256": "z"},
                        "rtn_q8_0", 1 << 30, "nvme")
    m.runtime_seconds = 12.5
    p = tmp_path / "r.json"
    m.save(str(p))
    assert RunManifest.load(str(p)).runtime_seconds == 12.5


def test_host_info_fields():
    h = host_info()
    assert set(h) == {"cpu", "ram_gb", "kernel"}
