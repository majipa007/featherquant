"""Safetensors parser tests against hand-built shards."""
import json
import struct

import gguf
import numpy as np
import pytest
from gguf import GGMLQuantizationType

from featherquant.st_source import SafetensorsSource, parse_shard_header, read_st_rows


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


# ---------------------------------------------------------------------------
# SafetensorsSource tests
# ---------------------------------------------------------------------------


def make_vocab_gguf(path, arch="qwen3", n_blocks=1):
    w = gguf.GGUFWriter(str(path), arch)
    w.add_block_count(n_blocks)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def _make_model_dir(tmp_path):
    rng = np.random.default_rng(0)
    d = tmp_path / "hf"
    d.mkdir()
    q = rng.standard_normal((4, 64)).astype(np.float16)
    nrm = rng.standard_normal(64).astype(np.float32)
    emb = rng.standard_normal((8, 32)).astype(np.float16)
    write_shard(d / "model-00001-of-00002.safetensors",
                {"model.layers.0.self_attn.q_proj.weight": q,
                 "model.norm.weight": nrm})
    write_shard(d / "model-00002-of-00002.safetensors",
                {"model.embed_tokens.weight": emb})
    index = {"weight_map": {
        "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
        "model.norm.weight": "model-00001-of-00002.safetensors",
        "model.embed_tokens.weight": "model-00002-of-00002.safetensors"}}
    (d / "model.safetensors.index.json").write_text(json.dumps(index))
    vp = tmp_path / "vocab.gguf"
    make_vocab_gguf(vp)
    return d, vp, q, nrm, emb


def test_safetensors_source_maps_and_slices(tmp_path):
    d, vp, q, nrm, emb = _make_model_dir(tmp_path)
    src = SafetensorsSource(str(d), str(vp))
    names = [t.name for t in src.tensors]
    # deterministic order: shards sorted, header order within a shard
    assert names == ["blk.0.attn_q.weight", "output_norm.weight",
                     "token_embd.weight"]
    tq = src.tensors[0]
    assert [int(x) for x in tq.shape] == [64, 4]  # ggml ne-order (reversed HF)
    assert tq.tensor_type == GGMLQuantizationType.F16
    assert int(tq.n_elements) == 256 and int(tq.n_bytes) == 512
    buf = bytearray(2 * 64 * 2)
    x = src.read_rows_f32(tq, 1, 2, buf)
    assert np.array_equal(x, q[1:3].astype(np.float32).ravel())
    tn = src.tensors[1]
    raw = src.read_raw(tn, 4, 8, bytearray(8))
    assert bytes(raw) == nrm.tobytes()[4:12]
    assert src.reader.fields["general.architecture"].contents() == "qwen3"
    src.close()


def test_safetensors_source_refuses_unknown_arch(tmp_path):
    d, _, *_ = _make_model_dir(tmp_path)
    vp = tmp_path / "vocab-llama.gguf"
    make_vocab_gguf(vp, arch="stablelm")
    with pytest.raises(RuntimeError):
        SafetensorsSource(str(d), str(vp))


def test_safetensors_source_refuses_unmapped_tensor(tmp_path):
    d, vp, *_ = _make_model_dir(tmp_path)
    write_shard(d / "model-00003-of-00002.safetensors",
                {"model.mystery.weight": np.zeros((1, 32), np.float16)})
    idx = json.loads((d / "model.safetensors.index.json").read_text())
    idx["weight_map"]["model.mystery.weight"] = "model-00003-of-00002.safetensors"
    (d / "model.safetensors.index.json").write_text(json.dumps(idx))
    with pytest.raises(RuntimeError, match="mystery"):
        SafetensorsSource(str(d), str(vp))
