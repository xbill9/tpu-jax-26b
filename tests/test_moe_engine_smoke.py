"""End-to-end: a 26B-shaped MoE checkpoint through JaxGemmaEngine on CPU.

The parity tests cover the MoE math against the reference, but they call the
model directly. This one goes through the paths a real 26B run actually uses and
that nothing else touches:

  * ``config_from_hf`` reading enable_moe_block / num_experts off a config.json
  * the streaming ``safe_open`` shard reader
  * Q4_0 -> W4A16 requantization during load
  * the KV-cached DECODE step, where the MoE block sees one token per sequence
    and therefore takes the gather path rather than the dense one

Tiny synthetic weights, real code path. Runs on CPU in seconds.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp

from jax_engine import JaxGemmaEngine, config_from_hf

safetensors_numpy = pytest.importorskip("safetensors.numpy")

H, E, K, MOE_I, INTER, L = 64, 8, 2, 32, 32, 2
HEADS, KV, HD, V = 4, 2, 32, 64
GROUP = 32


def _grid(shape, rng, scale=0.03):
    """Weights on an exact Q4_0 grid, so requantization has something to recover."""
    n = int(np.prod(shape))
    g = n // GROUP
    d = rng.uniform(scale / 4, scale, (g, 1)).astype(np.float16).astype(np.float32)
    q = rng.integers(-8, 8, (g, GROUP)).astype(np.float32)
    q[:, 0] = -8
    return (q * d).reshape(shape).astype(np.float32)


def _write_checkpoint(path, moe=True):
    rng = np.random.default_rng(0)
    cfg = {
        "model_type": "gemma4",
        "text_config": {
            "vocab_size": V, "hidden_size": H, "intermediate_size": INTER,
            "num_hidden_layers": L, "num_attention_heads": HEADS,
            "num_key_value_heads": KV, "head_dim": HD,
            "num_global_key_value_heads": KV, "global_head_dim": HD,
            "num_kv_shared_layers": 0, "hidden_size_per_layer_input": 0,
            "vocab_size_per_layer_input": V, "sliding_window": 8,
            "layer_types": ["sliding_attention", "full_attention"],
            "rms_norm_eps": 1e-6, "final_logit_softcapping": 30.0,
            "attention_k_eq_v": False, "use_double_wide_mlp": False,
            "enable_moe_block": moe, "num_experts": E if moe else 0,
            "top_k_experts": K if moe else 0,
            "moe_intermediate_size": MOE_I if moe else 0,
            "rope_parameters": {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": {"rope_type": "proportional",
                                   "partial_rotary_factor": 0.25,
                                   "rope_theta": 1000000.0},
            },
        },
    }
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as fh:
        json.dump(cfg, fh)

    t = {
        "model.language_model.embed_tokens.weight": _grid((V, H), rng),
        "model.language_model.norm.weight": np.ones(H, np.float32),
        # a vision tower the text decoder must skip rather than choke on
        "model.vision_tower.patch_embedder.input_proj.weight":
            rng.normal(0, 0.1, (H, H)).astype(np.float32),
    }
    for i in range(L):
        p = f"model.language_model.layers.{i}"
        norms = ["input_layernorm", "post_attention_layernorm",
                 "pre_feedforward_layernorm", "post_feedforward_layernorm"]
        if moe:
            norms += ["pre_feedforward_layernorm_2",
                      "post_feedforward_layernorm_1", "post_feedforward_layernorm_2"]
        for n in norms:
            t[f"{p}.{n}.weight"] = rng.uniform(0.5, 1.5, H).astype(np.float32)
        t[f"{p}.layer_scalar"] = np.array([0.9], np.float32)
        t[f"{p}.self_attn.q_proj.weight"] = _grid((HEADS * HD, H), rng)
        t[f"{p}.self_attn.k_proj.weight"] = _grid((KV * HD, H), rng)
        t[f"{p}.self_attn.v_proj.weight"] = _grid((KV * HD, H), rng)
        t[f"{p}.self_attn.o_proj.weight"] = _grid((H, HEADS * HD), rng)
        t[f"{p}.self_attn.q_norm.weight"] = np.ones(HD, np.float32)
        t[f"{p}.self_attn.k_norm.weight"] = np.ones(HD, np.float32)
        t[f"{p}.mlp.gate_proj.weight"] = _grid((INTER, H), rng)
        t[f"{p}.mlp.up_proj.weight"] = _grid((INTER, H), rng)
        t[f"{p}.mlp.down_proj.weight"] = _grid((H, INTER), rng)
        if moe:
            t[f"{p}.router.proj.weight"] = _grid((E, H), rng)
            t[f"{p}.router.scale"] = rng.uniform(0.5, 1.5, H).astype(np.float32)
            t[f"{p}.router.per_expert_scale"] = rng.uniform(0.8, 1.2, E).astype(np.float32)
            t[f"{p}.experts.gate_up_proj"] = _grid((E, 2 * MOE_I, H), rng)
            t[f"{p}.experts.down_proj"] = _grid((E, H, MOE_I), rng)

    # two shards, so the streaming reader is exercised as it will be in anger
    keys = sorted(t)
    half = len(keys) // 2
    safetensors_numpy.save_file(
        {k: t[k] for k in keys[:half]},
        os.path.join(path, "model-00001-of-00002.safetensors"))
    safetensors_numpy.save_file(
        {k: t[k] for k in keys[half:]},
        os.path.join(path, "model-00002-of-00002.safetensors"))
    return cfg


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("moe26b"))
    _write_checkpoint(path)
    return path


def test_config_from_hf_picks_up_moe(ckpt):
    with open(os.path.join(ckpt, "config.json")) as fh:
        cfg = config_from_hf(json.load(fh))
    assert cfg.enable_moe_block is True
    assert (cfg.num_experts, cfg.top_k_experts, cfg.moe_intermediate_size) == (
        E, K, MOE_I)


def test_moe_flag_without_shapes_does_not_enable(ckpt):
    """A config declaring the flag but no expert dims must not build a MoE.

    Zero experts would make the expert branch contribute nothing while the model
    still reported itself as sparse — a silent halving of the feed-forward block.
    """
    with open(os.path.join(ckpt, "config.json")) as fh:
        raw = json.load(fh)
    raw["text_config"]["num_experts"] = 0
    assert config_from_hf(raw).enable_moe_block is False


def test_engine_loads_requantized_and_generates(ckpt):
    eng = JaxGemmaEngine(
        model_id="synthetic-26b-moe", kv_cache_dtype="bf16", quant_mode="w4a16",
        max_model_len=64, requantize_q4_0=True, add_bos=False,
    )
    eng.load(local_dir=ckpt)

    assert eng.config.enable_moe_block
    st = eng.requantize_stats
    assert st["failed_groups"] == 0, "synthetic weights are on a Q4_0 grid"
    assert st["packed"] > 0
    # 4 bits + one BF16 scale per 32 values = 4.5 bits, i.e. ~3.56x off BF16.
    assert 3.4 < st["bytes_dense"] / st["bytes_packed"] < 3.7

    # every expert bank must have survived as packed, on every layer
    for i in range(L):
        moe = eng.params[f"layer_{i}"]["moe"]
        assert "experts_gate_up_proj_packed" in moe
        assert "experts_down_proj_packed" in moe
        assert moe["experts_gate_up_proj_packed"].dtype == jnp.int32
        # the router stays dense on purpose
        assert "router_proj" in moe and "router_proj_packed" not in moe

    # Decode: this is the gather path, and nothing else in the suite reaches it
    # through the KV cache.
    toks, stats = eng.generate([1, 2, 3, 4, 5], max_new_tokens=6, temperature=0.0)
    assert len(toks) == 6
    assert all(0 <= t < V for t in toks)
    assert stats.completion_tokens == 6


def test_dense_and_gather_decode_agree(ckpt):
    """Pinning the same weights, the two expert paths must decode identically."""
    outs = {}
    for impl in ("gather", "dense"):
        eng = JaxGemmaEngine(
            model_id="synthetic-26b-moe", kv_cache_dtype="bf16",
            quant_mode="w4a16", max_model_len=64, requantize_q4_0=True,
            add_bos=False, moe_impl=impl,
        )
        eng.load(local_dir=ckpt)
        outs[impl], _ = eng.generate([1, 2, 3, 4, 5], max_new_tokens=6,
                                     temperature=0.0)
    assert outs["gather"] == outs["dense"]


def test_non_q4_0_checkpoint_is_refused(tmp_path):
    """Requantizing weights that are not on a Q4_0 grid must fail loudly."""
    path = str(tmp_path / "dense")
    _write_checkpoint(path)
    # overwrite one shard with off-grid Gaussian noise
    rng = np.random.default_rng(1)
    f = os.path.join(path, "model-00001-of-00002.safetensors")
    loaded = safetensors_numpy.load_file(f)
    for k in list(loaded):
        if loaded[k].ndim >= 2:
            loaded[k] = rng.normal(0, 0.05, loaded[k].shape).astype(np.float32)
    safetensors_numpy.save_file(loaded, f)

    eng = JaxGemmaEngine(
        model_id="synthetic-off-grid", quant_mode="w4a16", max_model_len=64,
        requantize_q4_0=True,
    )
    with pytest.raises(ValueError, match="not on a Q4_0"):
        eng.load(local_dir=path)
