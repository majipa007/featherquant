"""Measurement manifest (spec §6): one JSON per measured run.

No number enters a table, a doc, or the README without one of these. The
schema is fixed by the spec — fields are never dropped, only filled in.
"""
import datetime
import hashlib
import json
import os
import platform
from dataclasses import asdict, dataclass, field
from typing import Any


def today_ddmmyyyy() -> str:
    """Today's date in Singapore format (DD/MM/YYYY)."""
    return datetime.date.today().strftime("%d/%m/%Y")


def sha256_file(path: str) -> str:
    """Stream a file's sha256 in 8 MiB chunks (never materializes it)."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(8 << 20):
                h.update(chunk)
    except OSError as exc:
        raise RuntimeError(f"cannot hash {path}: {exc}") from exc
    return h.hexdigest()


def host_info() -> dict[str, Any]:
    """CPU model, total RAM in GiB, kernel release."""
    cpu = platform.processor() or "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass  # non-Linux or restricted /proc: keep platform.processor()
    try:
        ram_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
                       / (1 << 30), 2)
    except (ValueError, OSError) as exc:
        raise RuntimeError(f"cannot read physical memory size: {exc}") from exc
    return {"cpu": cpu, "ram_gb": ram_gb, "kernel": platform.release()}


@dataclass
class RunManifest:
    """Spec §6 run record. Field order matches the spec listing."""
    run_id: str
    date: str
    model: dict[str, str]
    method: str
    approximations: list[dict[str, Any]]
    budget_bytes: int
    enforcement: str
    peak_observed_bytes: int | None
    oom_killed: bool
    runtime_seconds: float | None
    bytes_read: int | None
    bytes_written: int | None
    storage: str
    output_sha256: str | None
    quality: dict[str, Any]
    host: dict[str, Any] = field(default_factory=host_info)

    @classmethod
    def new(cls, run_id: str, model: dict[str, str], method: str,
            budget_bytes: int, storage: str) -> "RunManifest":
        """A manifest with everything measurable still unfilled."""
        return cls(run_id=run_id, date=today_ddmmyyyy(), model=dict(model),
                   method=method, approximations=[],
                   budget_bytes=budget_bytes,
                   enforcement="cgroup_v2_memory_max",
                   peak_observed_bytes=None, oom_killed=False,
                   runtime_seconds=None, bytes_read=None, bytes_written=None,
                   storage=storage, output_sha256=None,
                   quality={"ppl": None, "ppl_dataset": None, "tasks": {}})

    def save(self, path: str) -> None:
        """Write the manifest as pretty JSON (atomic: tmp + replace)."""
        tmp = path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(asdict(self), f, indent=2, sort_keys=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except OSError as exc:
            raise RuntimeError(f"cannot save run manifest {path}: {exc}") from exc

    @classmethod
    def load(cls, path: str) -> "RunManifest":
        """Load a manifest, failing loudly on a schema mismatch."""
        try:
            with open(path) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot load run manifest {path}: {exc}") from exc
        try:
            return cls(**d)
        except TypeError as exc:
            raise RuntimeError(f"malformed run manifest {path}: {exc}") from exc
