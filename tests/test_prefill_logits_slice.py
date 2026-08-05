"""The `logits_at` prefill optimization must change memory, never numerics.

`prefill_with_kv_cache` used to build [B, S, vocab] logits and keep one row per
sequence. It now pushes the position into the model so the lm_head runs on that
row alone. Matmul rows are independent and the softcap is elementwise, so the
kept row must come out BITWISE identical — not merely close. These tests pin
that, because a silent drift here would be invisible in a benchmark and would
corrupt every generation.
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ports.gemma4.jax_e_model import (
    Gemma4EConfig,
    Gemma4EModelJAX,
    make_prefill_causal_mask,
    prefill_with_kv_cache,
)


def _tiny_config() -> Gemma4EConfig:
    """A 31B-shaped miniature: no PLE, no KV sharing, k_eq_v, mixed layer types."""
    return Gemma4EConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_global_key_value_heads=1,
        global_head_dim=16,
        num_kv_shared_layers=0,
        hidden_size_per_layer_input=0,
        sliding_window=4,
        layer_types=["sliding_attention", "sliding_attention",
                     "full_attention", "sliding_attention"],
        logit_softcapping=30.0,
        attention_k_eq_v=True,
    )


def _random_params(cfg: Gemma4EConfig, seed: int = 0):
    key = jax.random.PRNGKey(seed)

    def rnd(*shape):
        nonlocal key
        key, sub = jax.random.split(key)
        return jax.random.normal(sub, shape, dtype=jnp.float32) * 0.05

    params = {
        "embed_tokens": rnd(cfg.vocab_size, cfg.hidden_size),
        "final_norm": rnd(cfg.hidden_size) + 1.0,
    }
    for i in range(cfg.num_hidden_layers):
        is_sliding = cfg.layer_types[i] == "sliding_attention"
        n_kv = cfg.num_key_value_heads if is_sliding else cfg.num_global_key_value_heads
        h_dim = cfg.head_dim if is_sliding else cfg.global_head_dim
        q_out = cfg.num_attention_heads * h_dim
        kv_out = n_kv * h_dim
        k = rnd(cfg.hidden_size, kv_out)
        params[f"layer_{i}"] = {
            "input_layernorm": rnd(cfg.hidden_size) + 1.0,
            "post_attention_layernorm": rnd(cfg.hidden_size) + 1.0,
            "pre_feedforward_layernorm": rnd(cfg.hidden_size) + 1.0,
            "post_feedforward_layernorm": rnd(cfg.hidden_size) + 1.0,
            "layer_scalar": jnp.asarray(0.6, dtype=jnp.float32),
            "attn": {
                "q_proj": rnd(cfg.hidden_size, q_out),
                "k_proj": k,
                "v_proj": k,                      # attention_k_eq_v
                "o_proj": rnd(q_out, cfg.hidden_size),
                "q_norm": rnd(h_dim) + 1.0,
                "k_norm": rnd(h_dim) + 1.0,
            },
            "mlp": {
                "gate_proj": rnd(cfg.hidden_size, cfg.intermediate_size),
                "up_proj": rnd(cfg.hidden_size, cfg.intermediate_size),
                "down_proj": rnd(cfg.intermediate_size, cfg.hidden_size),
            },
        }
    return params


@pytest.mark.parametrize("B,S,pad", [(1, 8, 0), (1, 8, 3), (2, 8, 2), (3, 16, 5)])
def test_logits_at_matches_full_sequence_bitwise(B, S, pad):
    """model(logits_at=p) equals model(...)[:, p, :], exactly."""
    cfg = _tiny_config()
    model = Gemma4EModelJAX(cfg)
    params = _random_params(cfg)

    rng = np.random.default_rng(0)
    ids = jnp.asarray(rng.integers(0, cfg.vocab_size, size=(B, S)), dtype=jnp.int32)
    valid = jnp.asarray(
        [[j < (S - pad - i % 2) for j in range(S)] for i in range(B)], dtype=bool)
    lens = valid.sum(axis=1).astype(jnp.int32)
    pos = jnp.arange(S, dtype=jnp.int32)[None, :].repeat(B, axis=0)
    mask = make_prefill_causal_mask(valid)
    sliding = make_prefill_causal_mask(valid, window=cfg.sliding_window)

    full = model(ids, params, pos, attention_mask=mask, quant_mode="fp16",
                 sliding_attention_mask=sliding)
    sliced = model(ids, params, pos, attention_mask=mask, quant_mode="fp16",
                   sliding_attention_mask=sliding, logits_at=lens - 1)

    assert sliced.shape == (B, 1, cfg.vocab_size)
    expected = jnp.take_along_axis(full, (lens - 1)[:, None, None], axis=1)
    np.testing.assert_array_equal(np.asarray(sliced), np.asarray(expected))


def test_prefill_last_logits_unchanged_bitwise():
    """The public prefill entry point returns the same last_logits as before.

    The pre-optimization behaviour is reconstructed here by calling the model
    without logits_at and slicing afterwards — i.e. exactly the code this change
    replaced.
    """
    cfg = _tiny_config()
    model = Gemma4EModelJAX(cfg)
    params = _random_params(cfg, seed=3)

    B, S = 2, 16
    rng = np.random.default_rng(7)
    ids = jnp.asarray(rng.integers(0, cfg.vocab_size, size=(B, S)), dtype=jnp.int32)
    valid = jnp.asarray([[j < 13 for j in range(S)],
                         [j < 9 for j in range(S)]], dtype=bool)

    last_logits, caches, out_valid = prefill_with_kv_cache(
        model, ids, valid, params, max_new_tokens=4,
        quant_mode="fp16", cache_dtype=jnp.float32, window_kv=False,
    )

    pos = jnp.arange(S, dtype=jnp.int32)[None, :].repeat(B, axis=0)
    mask = make_prefill_causal_mask(valid)
    sliding = make_prefill_causal_mask(valid, window=cfg.sliding_window)
    full = model(ids, params, pos, attention_mask=mask, quant_mode="fp16",
                 sliding_attention_mask=sliding)
    lens = valid.sum(axis=1).astype(jnp.int32)
    reference = jnp.take_along_axis(full, (lens - 1)[:, None, None], axis=1)[:, 0, :]

    assert last_logits.shape == (B, cfg.vocab_size)
    np.testing.assert_array_equal(np.asarray(last_logits), np.asarray(reference))
    assert out_valid.shape == (B, S + 4)
    assert len(caches) == cfg.num_hidden_layers


def test_full_sequence_logits_still_available_by_default():
    """Callers that do not pass logits_at must still get every position."""
    cfg = _tiny_config()
    model = Gemma4EModelJAX(cfg)
    params = _random_params(cfg, seed=11)
    B, S = 1, 8
    ids = jnp.zeros((B, S), dtype=jnp.int32)
    pos = jnp.arange(S, dtype=jnp.int32)[None, :]
    out = model(ids, params, pos, quant_mode="fp16")
    assert out.shape == (B, S, cfg.vocab_size)
