"""Shared test helpers: tiny GGUF fixture builder."""
import gguf
import numpy as np


def make_gguf(path, tensors, arch="llama"):
    """Write a tiny GGUF for tests.

    float16/float32 arrays get F16/F32 tensor types automatically;
    in-memory buffering inside GGUFWriter is fine at fixture sizes.
    """
    w = gguf.GGUFWriter(str(path), arch)
    try:
        for name, arr in tensors.items():
            w.add_tensor(name, arr)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
    finally:
        # Always release the file handle, even if a write step failed.
        w.close()
