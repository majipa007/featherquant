import json
import struct

import numpy as np
import pytest

from featherquant.indexer import ModelIndex, index_model
from featherquant.roles import Role


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
    """A 2-layer model with the shapes an indexer must derive, not assume."""
    h, i, v = 8, 16, 32
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
            idx.vocab_size) == (2, 8, 16, 32)
    assert idx.head_dims == {"n_heads": 2, "n_kv_heads": 1, "head_dim": 4}


def test_largest_tensor_is_the_embedding(tiny_model):
    idx = index_model(str(tiny_model))
    assert idx.largest_tensor_bytes == 32 * 8 * 2       # v x h, fp16
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
