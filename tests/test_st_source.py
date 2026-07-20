"""Safetensors parser tests against hand-built shards."""
import json
import struct

import numpy as np

from featherquant.st_source import parse_shard_header, read_st_rows


def write_shard(path, tensors):
    """Minimal safetensors writer: u64 header length, JSON header, raw data."""
    header, blobs, off = {}, [], 0
    for name, arr in tensors.items():
        raw = arr.tobytes()
        dtype = {"<f4": "F32", "<f2": "F16", "<u2": "BF16"}[arr.dtype.str]
        header[name] = {"dtype": dtype, "shape": list(arr.shape),
                        "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hj = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)


def test_parse_and_slice_f16(tmp_path):
    a = (np.arange(6 * 32, dtype=np.float32) / 8).reshape(6, 32).astype(np.float16)
    p = tmp_path / "s.safetensors"
    write_shard(p, {"model.layers.0.q.weight": a})
    tensors, data_base = parse_shard_header(str(p))
    t = tensors["model.layers.0.q.weight"]
    assert t.shape == (6, 32) and t.dtype == "F16"
    with open(p, "rb") as f:
        buf = bytearray(2 * 32 * 2)
        x = read_st_rows(f, t, data_base, 2, 2, buf)
    assert np.array_equal(x, a[2:4].astype(np.float32).ravel())


def test_parse_and_slice_bf16(tmp_path):
    fvals = (np.arange(64, dtype=np.float32) / 4).reshape(2, 32)  # exact in bf16
    u16 = (fvals.view(np.uint32) >> 16).astype(np.uint16)
    p = tmp_path / "s.safetensors"
    write_shard(p, {"w": u16})
    tensors, data_base = parse_shard_header(str(p))
    t = tensors["w"]
    assert t.dtype == "BF16"
    with open(p, "rb") as f:
        buf = bytearray(64 * 2)
        x = read_st_rows(f, t, data_base, 0, 2, buf)
    assert np.array_equal(x, fvals.ravel())


def test_metadata_key_skipped(tmp_path):
    a = np.zeros((1, 32), np.float32)
    p = tmp_path / "s.safetensors"
    # Hand-add a __metadata__ entry alongside a real tensor.
    raw = a.tobytes()
    header = {"__metadata__": {"format": "pt"},
              "w": {"dtype": "F32", "shape": [1, 32],
                    "data_offsets": [0, len(raw)]}}
    hj = json.dumps(header).encode()
    with open(p, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        f.write(raw)
    tensors, _ = parse_shard_header(str(p))
    assert set(tensors) == {"w"}
