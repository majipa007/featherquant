"""Shared test helpers: tiny GGUF fixture builder."""
from pathlib import Path

import gguf
import numpy as np


def make_gguf(path: Path | str, tensors: dict[str, np.ndarray],
              arch: str = "llama", kv: dict[str, int] | None = None) -> None:
    """Write a tiny GGUF for tests.

    float16/float32 arrays get F16/F32 tensor types automatically;
    in-memory buffering inside GGUFWriter is fine at fixture sizes.
    ``kv`` adds arch-scoped integer metadata (e.g. {"block_count": 1,
    "attention.key_length": 32}) for tests that exercise KV-derived fields.
    """
    w = gguf.GGUFWriter(str(path), arch)
    try:
        for key, value in (kv or {}).items():
            w.add_uint32(f"{arch}.{key}", value)
        for name, arr in tensors.items():
            w.add_tensor(name, arr)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
    finally:
        # Always release the file handle, even if a write step failed.
        w.close()
