import json
import struct

import numpy as np
import pytest

from featherquant.indexer import BLOCK_ALIGN, ModelIndex, index_model
from featherquant.roles import Role
from tests.conftest import make_gguf


def write_safetensors(path, arrays):
    """Minimal safetensors writer for fixtures (header + raw data)."""
    header, offset = {}, 0
    blobs = []
    for name, arr in arrays.items():
        raw = arr.tobytes()
        dtype = {"float32": "F32", "float16": "F16"}[arr.dtype.name]
        header[name] = {"dtype": dtype, "shape": list(arr.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        blobs.append(raw)
    hjson = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for b in blobs:
            f.write(b)


@pytest.fixture
def tiny_model(tmp_path):
    """A 2-layer model with the shapes an indexer must derive, not assume.

    Dimensions are BLOCK_ALIGN-aligned (unlike the miniature 8/16/32 this
    fixture used before): quant_eligible is a real per-row block-alignment
    check now, so a shape indivisible by BLOCK_ALIGN would wrongly exclude
    every projection tensor here.
    """
    h, i, v = 8 * BLOCK_ALIGN, 16 * BLOCK_ALIGN, 32 * BLOCK_ALIGN  # 256, 512, 1024
    arrays = {"model.embed_tokens.weight": np.zeros((v, h), np.float16),
              "model.norm.weight": np.zeros((h,), np.float32),
              "lm_head.weight": np.zeros((v, h), np.float16)}
    for layer in range(2):
        p = f"model.layers.{layer}."
        arrays[p + "self_attn.q_proj.weight"] = np.zeros((h, h), np.float16)
        arrays[p + "self_attn.k_proj.weight"] = np.zeros((h // 2, h), np.float16)
        arrays[p + "self_attn.v_proj.weight"] = np.zeros((h // 2, h), np.float16)
        arrays[p + "self_attn.o_proj.weight"] = np.zeros((h, h), np.float16)
        arrays[p + "mlp.gate_proj.weight"] = np.zeros((i, h), np.float16)
        arrays[p + "mlp.up_proj.weight"] = np.zeros((i, h), np.float16)
        arrays[p + "mlp.down_proj.weight"] = np.zeros((h, i), np.float16)
        arrays[p + "input_layernorm.weight"] = np.zeros((h,), np.float32)
    write_safetensors(tmp_path / "model.safetensors", arrays)
    (tmp_path / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
        "num_hidden_layers": 2, "hidden_size": h, "intermediate_size": i,
        "vocab_size": v, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 4, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0}))
    return tmp_path


def test_index_derives_dimensions(tiny_model):
    idx = index_model(str(tiny_model))
    assert idx.model_arch == "qwen3"
    assert (idx.n_layers, idx.hidden_size, idx.intermediate_size,
            idx.vocab_size) == (2, 256, 512, 1024)
    assert idx.head_dims == {"n_heads": 2, "n_kv_heads": 1, "head_dim": 4}


def test_largest_tensor_is_the_embedding(tiny_model):
    idx = index_model(str(tiny_model))
    assert idx.largest_tensor_bytes == 1024 * 256 * 2    # v x h, fp16
    assert idx.total_bytes == sum(t.byte_length for t in idx.tensors)


def test_roles_and_layer_indices(tiny_model):
    idx = index_model(str(tiny_model))
    downs = idx.by_role(Role.FFN_DOWN)
    assert sorted(t.layer_index for t in downs) == [0, 1]
    assert all(t.quant_eligible for t in downs)
    assert not idx.by_role(Role.NORM)[0].quant_eligible   # 1-D tensor
    assert {t.role for t in idx.layer_tensors(0)} == {
        "attn_q", "attn_k", "attn_v", "attn_o",
        "ffn_gate", "ffn_up", "ffn_down", "norm"}


def test_roundtrip(tiny_model, tmp_path):
    idx = index_model(str(tiny_model))
    p = tmp_path / "manifest.json"
    idx.save(str(p))
    again = ModelIndex.load(str(p))
    assert again.tensors == idx.tensors
    assert again.largest_tensor_bytes == idx.largest_tensor_bytes


def test_missing_config_fails_loudly(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="config.json"):
        index_model(str(tmp_path))


def test_quant_eligible_requires_block_alignment(tmp_path):
    """A 2-D tensor whose row length isn't a BLOCK_ALIGN multiple is excluded.

    Real ggml quant formats work in blocks of BLOCK_ALIGN (Q8_0) or larger
    (K-quants); a row that isn't even a multiple of the smallest block can
    never be quantized to any of them, regardless of role or dtype.
    """
    h = BLOCK_ALIGN - 1   # 31: deliberately not a multiple of BLOCK_ALIGN
    write_safetensors(tmp_path / "model.safetensors", {
        "model.embed_tokens.weight": np.zeros((4, h), np.float16),
        "model.norm.weight": np.zeros((h,), np.float32),
        "model.layers.0.self_attn.q_proj.weight": np.zeros((h, h), np.float16),
    })
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen3", "num_hidden_layers": 1, "hidden_size": h,
        "intermediate_size": h, "vocab_size": 4, "num_attention_heads": 1,
        "num_key_value_heads": 1, "head_dim": h}))
    idx = index_model(str(tmp_path))
    q = idx.by_role(Role.ATTN_Q)[0]
    assert q.shape[-1] % BLOCK_ALIGN != 0
    assert not q.quant_eligible                # misaligned row, 2-D, non-norm


def _gguf_kv(hidden, ffn, n_layers, n_heads, n_kv, key_length=None):
    """Arch-scoped KV block index_gguf reads (see featherquant/indexer.py)."""
    kv = {"block_count": n_layers, "embedding_length": hidden,
          "feed_forward_length": ffn, "attention.head_count": n_heads,
          "attention.head_count_kv": n_kv}
    if key_length is not None:
        kv["attention.key_length"] = key_length
    return kv


def test_index_gguf_derives_dimensions_and_roles(tmp_path):
    """GGUF path: dimensions, per-layer roles, and embedding-sized largest tensor."""
    hidden, ffn, vocab = 64, 128, 256
    tensors = {
        "token_embd.weight": np.zeros((vocab, hidden), np.float16),
        "blk.0.attn_q.weight": np.zeros((hidden, hidden), np.float16),
        "blk.0.attn_k.weight": np.zeros((hidden // 2, hidden), np.float16),
        "blk.0.attn_v.weight": np.zeros((hidden // 2, hidden), np.float16),
        "blk.0.attn_output.weight": np.zeros((hidden, hidden), np.float16),
        "blk.0.ffn_gate.weight": np.zeros((ffn, hidden), np.float16),
        "blk.0.ffn_up.weight": np.zeros((ffn, hidden), np.float16),
        "blk.0.ffn_down.weight": np.zeros((hidden, ffn), np.float16),
        "blk.0.attn_norm.weight": np.zeros((hidden,), np.float32),
    }
    path = tmp_path / "tiny.gguf"
    make_gguf(path, tensors, arch="qwen3",
              kv=_gguf_kv(hidden, ffn, n_layers=1, n_heads=4, n_kv=2, key_length=32))
    idx = index_model(str(path))
    assert idx.model_arch == "qwen3"
    assert (idx.n_layers, idx.hidden_size, idx.intermediate_size) == (1, hidden, ffn)
    assert idx.vocab_size == vocab
    assert idx.head_dims == {"n_heads": 4, "n_kv_heads": 2, "head_dim": 32}
    assert {t.role for t in idx.layer_tensors(0)} == {
        "attn_q", "attn_k", "attn_v", "attn_o",
        "ffn_gate", "ffn_up", "ffn_down", "norm"}
    assert idx.largest_tensor_bytes == vocab * hidden * 2    # embed, fp16
    embed = idx.by_role(Role.EMBED)[0]
    assert embed.shape == (vocab, hidden)     # source order, not ggml ne order


def test_index_gguf_head_dim_falls_back_without_key_length(tmp_path):
    """Without an explicit key_length KV, head_dim falls back to hidden // n_heads."""
    hidden = 64
    tensors = {"token_embd.weight": np.zeros((32, hidden), np.float16),
              "blk.0.attn_q.weight": np.zeros((hidden, hidden), np.float16)}
    path = tmp_path / "tiny.gguf"
    make_gguf(path, tensors, arch="qwen3",
              kv=_gguf_kv(hidden, 128, n_layers=1, n_heads=4, n_kv=4))
    idx = index_model(str(path))
    assert idx.head_dims["head_dim"] == hidden // 4


def test_index_gguf_empty_tensor_list_fails_loudly(tmp_path):
    """A GGUF with metadata but no tensors must raise, not crash inside max()."""
    path = tmp_path / "empty.gguf"
    make_gguf(path, {}, arch="qwen3",
              kv=_gguf_kv(64, 128, n_layers=1, n_heads=4, n_kv=4))
    with pytest.raises(RuntimeError, match="no tensors"):
        index_model(str(path))
