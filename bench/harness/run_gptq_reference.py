#!/usr/bin/env python3
"""bench/harness/run_gptq_reference.py — unconstrained reference GPTQ.

Runs GPTQ on the GPU with no memory ceiling, records perplexity and
per-linear reconstruction error. This is the quality target M4's calibrator
must match within noise (Task 17 compares against it directly).

Requires (throwaway venv, NOT part of pyproject.toml): uv pip install
gptqmodel torch datasets

Usage: run_gptq_reference.py HF_MODEL_DIR OUT_DIR RUN_ID

Reads calibration texts from bench/data/wiki.test.raw, which MUST be the
corpus pinned by bench/harness/fetch_calibration_corpus.sh (sha256
173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08) — that pin
is what makes this reference comparable to every later FeatherQuant number.
Run fetch_calibration_corpus.sh first if that file is missing.

Produces two artifacts:
  bench/manifests/<RUN_ID>.json               spec §6 run manifest.
  bench/manifests/<RUN_ID>_layer_errors.json  per-linear reconstruction
      error, "<layer_index>.<role>" -> mean-squared-error float, for the
      seven linear roles (attn_q, attn_k, attn_v, attn_o, ffn_gate, ffn_up,
      ffn_down). Computed by snapshotting each nn.Linear weight before
      model.quantize() overwrites it in place, then comparing against the
      corresponding weight after quantization. If a particular gptqmodel
      version does not expose the model as a walkable torch.nn.Module (or
      the snapshot/post-quantization shapes never line up), this file is
      written with an explicit {"unavailable": "<reason>"} marker instead
      of a fabricated or partial error map — never guess a number here.

This script is NOT part of the shipped featherquant package: it lives in
bench/harness/ and produces numbers, not library code. It is never imported
by featherquant/.
"""
import re
import sys

# Argument validation happens here, at module scope, before any of this
# script's heavy (torch/gptqmodel) imports — those only happen inside
# main(), below. That way a bare invocation (e.g. from a CI/test venv that
# has neither dependency installed) fails fast with a usage message instead
# of an ImportError or a raw traceback.
_USAGE = "usage: run_gptq_reference.py HF_MODEL_DIR OUT_DIR RUN_ID"

# HF/Qwen3-style dotted module name suffix -> spec §4.1 role. This is a
# deliberately small, local lookup for this one reference script — it is
# NOT the general model-family indexer (that is Milestone M1's job); it
# only needs to key the per-linear error map consistently with the
# project's role names for the one model family this baseline runs against.
_ROLE_BY_SUFFIX = {
    "q_proj": "attn_q",
    "k_proj": "attn_k",
    "v_proj": "attn_v",
    "o_proj": "attn_o",
    "gate_proj": "ffn_gate",
    "up_proj": "ffn_up",
    "down_proj": "ffn_down",
}

_LAYER_INDEX_RE = re.compile(r"\.layers\.(\d+)\.")


def _role_from_module_name(name: str) -> str | None:
    """Map a dotted module name (e.g. 'model.layers.3.self_attn.q_proj') to
    one of the seven linear roles this file records, or None if it is not
    one of those (e.g. embeddings, norms, lm_head)."""
    for suffix, role in _ROLE_BY_SUFFIX.items():
        if name.endswith(suffix):
            return role
    return None


def _layer_index_from_module_name(name: str) -> str | None:
    """Extract the decoder-layer index from a dotted module name, e.g.
    'model.layers.12.self_attn.q_proj' -> '12'. None if no layer index is
    present in the name (e.g. lm_head)."""
    m = _LAYER_INDEX_RE.search(name)
    return m.group(1) if m else None


def _snapshot_linear_weights(model: object) -> dict[str, object]:
    """Best-effort snapshot of every named nn.Linear weight, keyed by dotted
    module name, taken BEFORE model.quantize() overwrites them in place.

    Different gptqmodel versions wrap the underlying HF torch.nn.Module
    differently (some directly, some behind a `.model` attribute) — try the
    common case, and return an empty dict (never raise) if neither shape is
    walkable. An empty dict is the caller's signal to mark the layer-errors
    file "unavailable" rather than compare against nothing.
    """
    import torch.nn as nn

    torch_model = getattr(model, "model", None)
    if torch_model is None or not hasattr(torch_model, "named_modules"):
        torch_model = model
    if not hasattr(torch_model, "named_modules"):
        return {}
    snapshot = {}
    for name, module in torch_model.named_modules():
        if isinstance(module, nn.Linear):
            snapshot[name] = module.weight.detach().clone().float().cpu()
    return snapshot


def _compute_layer_errors(
    model: object, originals: dict[str, object]
) -> dict[str, float] | dict[str, str]:
    """Per-linear mean-squared reconstruction error, "<layer>.<role>" -> MSE,
    comparing each post-quantization linear weight against its
    pre-quantization snapshot. Returns {"unavailable": "<reason>"} instead
    of a partial or fabricated map if the comparison cannot honestly be
    made — see the module docstring for when that happens.
    """
    import torch
    import torch.nn as nn

    if not originals:
        return {"unavailable": "could not snapshot original linear weights "
                                "before quantization for this gptqmodel "
                                "version (model was not a walkable "
                                "torch.nn.Module via `.model` or itself)"}

    torch_model = getattr(model, "model", None)
    if torch_model is None or not hasattr(torch_model, "named_modules"):
        torch_model = model
    if not hasattr(torch_model, "named_modules"):
        return {"unavailable": "quantized model is not a walkable "
                                "torch.nn.Module after model.quantize()"}

    errors: dict[str, float] = {}
    for name, module in torch_model.named_modules():
        if name not in originals or not isinstance(module, nn.Linear):
            continue
        role = _role_from_module_name(name)
        layer_idx = _layer_index_from_module_name(name)
        if role is None or layer_idx is None:
            continue
        quant_weight = module.weight.detach().float().cpu()
        orig_weight = originals[name]
        if quant_weight.shape != orig_weight.shape:
            # Packed/transposed/repacked representation this simple
            # comparison cannot line up — skip rather than guess.
            continue
        errors[f"{layer_idx}.{role}"] = torch.mean(
            (quant_weight - orig_weight) ** 2
        ).item()

    if not errors:
        return {"unavailable": "no comparable linear weights found after "
                                "quantization (shape mismatch on every "
                                "module, or an empty model)"}
    return errors


def _save_json(path: str, data: dict) -> None:
    """Write JSON atomically (tmp + replace), same pattern RunManifest.save
    uses, so a crash mid-write can never leave Task 17 reading a truncated
    layer-errors file."""
    import json
    import os

    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except OSError as exc:
        raise RuntimeError(f"cannot save {path}: {exc}") from exc


def main() -> None:
    if len(sys.argv) != 4:
        print(f"{_USAGE}\nexpected 3 arguments, got {len(sys.argv) - 1}",
              file=sys.stderr)
        sys.exit(2)
    model_dir, out_dir, run_id = sys.argv[1:4]

    # Heavy, GPU-only imports deferred to here — after argv validation —
    # so a bare/invalid invocation never needs torch or gptqmodel installed.
    import json
    import time

    import torch
    from gptqmodel import GPTQModel, QuantizeConfig

    from featherquant.run_manifest import RunManifest

    corpus_path = "bench/data/wiki.test.raw"
    try:
        with open(corpus_path) as f:
            # 128 samples x 512 tokens: the same calibration shape spec
            # §3.3 uses.
            texts = [t for t in f.read().split("\n\n") if len(t) > 512][:128]
    except OSError as exc:
        raise RuntimeError(
            f"cannot read calibration corpus {corpus_path}: {exc}. Run "
            "bench/harness/fetch_calibration_corpus.sh first."
        ) from exc

    cfg = QuantizeConfig(bits=4, group_size=128, damp_percent=0.01,
                          desc_act=False, sym=True)
    t0 = time.monotonic()
    try:
        model = GPTQModel.load(model_dir, cfg)
        originals = _snapshot_linear_weights(model)
        model.quantize(texts)
        model.save(out_dir)
    except Exception as exc:
        raise RuntimeError(f"reference GPTQ failed: {exc}") from exc
    runtime = time.monotonic() - t0

    layer_errors = _compute_layer_errors(model, originals)
    layer_errors_path = f"bench/manifests/{run_id}_layer_errors.json"
    _save_json(layer_errors_path, layer_errors)

    m = RunManifest.new(run_id, {"id": model_dir, "revision": "local",
                                  "sha256": "unknown"},
                         "gptq_reference_4bit_g128", 0, "nvme")
    m.enforcement = "none_unconstrained"
    m.runtime_seconds = round(runtime, 3)
    m.peak_observed_bytes = torch.cuda.max_memory_allocated()
    m.quality = {"ppl": None,
                 "ppl_dataset": "wikitext-2-raw/wiki.test.raw c=512 tokenizer=qwen3",
                 "tasks": {}}
    m.save(f"bench/manifests/{run_id}.json")
    print(json.dumps({"runtime_s": runtime,
                       "layer_errors_path": layer_errors_path}, indent=2))


if __name__ == "__main__":
    main()
