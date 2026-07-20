"""Atomic sidecar manifest for checkpoint/resume.

One JSON file per output (``<output>.manifest.json``) records source
identity, config, the deterministic tensor plan with absolute output
offsets, and a sha256 per committed tensor. Saves are atomic
(tmp + fsync + ``os.replace``) so a crash never leaves a torn manifest;
the manifest, not the output file, is the source of truth for progress.
"""
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Any

# Bump when the on-disk shape changes; load() refuses other versions.
MANIFEST_VERSION = 1


def sha256_file_region(path: str, offset: int, nbytes: int) -> str:
    """Hash a byte range without materializing it (8 MiB chunks)."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            remaining = nbytes
            while remaining:
                chunk = f.read(min(8 << 20, remaining))
                if not chunk:
                    raise RuntimeError(
                        f"short read hashing {path} at offset {offset}: "
                        f"{nbytes - remaining}/{nbytes} bytes")
                h.update(chunk)
                remaining -= len(chunk)
    except OSError as exc:
        raise RuntimeError(f"cannot hash {path}: {exc}") from exc
    return h.hexdigest()


@dataclass
class TensorEntry:
    """One planned output tensor; sha256 is None until committed."""
    name: str
    ggml_type: int
    offset: int   # absolute byte offset in the output file
    nbytes: int
    sha256: str | None


@dataclass
class Manifest:
    """Whole-job checkpoint state."""
    source_path: str
    source_size: int
    source_mtime_ns: int
    config: dict[str, Any]
    header_end: int      # file offset where tensor data begins (aligned)
    header_sha256: str   # sha256 of bytes [0, header_end)
    tensors: list[TensorEntry]
    status: str          # "in_progress" | "complete"
    version: int = field(default=MANIFEST_VERSION)

    def save(self, path: str) -> None:
        """Atomically write the manifest: tmp file + fsync + rename."""
        data = json.dumps(asdict(self), indent=1)
        d = os.path.dirname(os.path.abspath(path))
        try:
            fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, path)
            except BaseException:
                # Never leave a stray temp file behind on failure.
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise
        except OSError as exc:
            raise RuntimeError(f"cannot save manifest {path}: {exc}") from exc

    @classmethod
    def load(cls, path: str) -> "Manifest":
        """Load and validate a manifest; RuntimeError on any mismatch."""
        try:
            with open(path) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot load manifest {path}: {exc}") from exc
        try:
            if d["version"] != MANIFEST_VERSION:
                raise RuntimeError(
                    f"manifest {path} has version {d['version']}, "
                    f"expected {MANIFEST_VERSION}")
            tensors = [TensorEntry(**t) for t in d.pop("tensors")]
            return cls(tensors=tensors, **d)
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"malformed manifest {path}: {exc}") from exc
