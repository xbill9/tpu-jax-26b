"""Parity for the gemma-4-26B-A4B sparse MoE block against the HF reference.

The 26B A4B is the first mixture-of-experts checkpoint this engine runs, and the
MoE block is the only part of it that is genuinely new — attention, proportional
RoPE and attention_k_eq_v are shared with the 31B, which is already covered.

What makes this worth a dedicated test rather than a smoke run: a mis-wired MoE
block still produces fluent text. The reference feeds the experts the
*post-attention residual*, before ``pre_feedforward_layernorm``, while the dense
MLP beside them gets the normalized version. Feeding both the same tensor — the
obvious reading, and what a careless port does — still routes, still runs, and
still decodes English. So the test compares against
``transformers.models.gemma4`` directly instead of against intuition.

Run in float32 on both sides. BF16 would put a 1e-2 noise floor under everything
and a genuine wiring error is perfectly capable of hiding beneath it.

Requires torch + transformers (dev-only); skipped when they are absent.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", False)

# Import torch and transformers INSIDE the tests, never at module scope.
#
# pytest imports every test module during collection, before running anything,
# so a module-level `import torch` here would put PyTorch in sys.modules for the
# whole session — and tests/test_jax_engine.py asserts that loading the engine
# pulls in no PyTorch at all. That assertion is the repo's central claim ("ZERO
# PyTorch in the path"), so this file defers its imports rather than weakening
# it. find_spec resolves availability without importing.
_MISSING = [m for m in ("torch", "transformers")
            if importlib.util.find_spec(m) is None]
pytestmark = pytest.mark.skipif(
    bool(_MISSING),
    reason=f"needs {' + '.join(_MISSING)} (dev-only reference implementation)")


def _ref():
    """Lazily import the HF reference. See the note above."""
    import torch
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4TextExperts, Gemma4TextModel, Gemma4TextRouter)
    return (torch, Gemma4TextConfig, Gemma4TextExperts, Gemma4TextModel,
            Gemma4TextRouter)


from ports.gemma4.jax_e_loader import convert_safetensors_to_jax_params
from ports.gemma4.jax_e_model import (
    Gemma4EConfig,
    Gemma4EModelJAX,
    make_prefill_causal_mask,
    moe_experts_dense,
    moe_experts_gather,
    moe_router_jax,
)

# Deliberately not powers of two everywhere: H=64 and E=8 would let a transposed
# or mis-broadcast tensor line up by accident. The input dims stay multiples of
# the Q4_0 group size of 32 because the packing requires it.
H, E, K, MOE_I, INTER, L = 96, 8, 3, 32, 64, 2
HEADS, KV, HD, V = 4, 2, 32, 128
ATOL = 2e-4


def _hf_config():
    _, Gemma4TextConfig, _, _, _ = _ref()
    return Gemma4TextConfig(
        vocab_size=V, hidden_size=H, intermediate_size=INTER, num_hidden_layers=L,
        num_attention_heads=HEADS, num_key_value_heads=KV, head_dim=HD,
        num_global_key_value_heads=KV, global_head_dim=HD,
        num_kv_shared_layers=0, hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=V, sliding_window=8,
        layer_types=["sliding_attention", "full_attention"],
        enable_moe_block=True, num_experts=E, top_k_experts=K,
        moe_intermediate_size=MOE_I, attention_k_eq_v=False,
        use_double_wide_mlp=False, tie_word_embeddings=True,
        final_logit_softcapping=30.0,
    )


def _jax_config():
    return Gemma4EConfig(
        vocab_size=V, hidden_size=H, intermediate_size=INTER, num_hidden_layers=L,
        num_attention_heads=HEADS, num_key_value_heads=KV, head_dim=HD,
        num_global_key_value_heads=KV, global_head_dim=HD,
        num_kv_shared_layers=0, hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=V, sliding_window=8,
        layer_types=["sliding_attention", "full_attention"],
        enable_moe_block=True, num_experts=E, top_k_experts=K,
        moe_intermediate_size=MOE_I, attention_k_eq_v=False,
        use_double_wide_mlp=False, logit_softcapping=30.0,
    )


def _to_raw(state_dict, prefix="model."):
    """HF state dict -> the flat {name: array} the JAX loader consumes."""
    return {f"{prefix}{k}": jnp.asarray(v.detach().float().numpy(), jnp.float32)
            for k, v in state_dict.items()}


def _causal(batch, seq):
    """Additive causal mask.

    The engine adds this to the attention scores, so it must be 0 / -1e9 —
    a bool mask would add 1.0 to every allowed position and 0.0 to every
    forbidden one, which is not masking at all and is exactly the bug that
    made the first version of this test fail against a correct engine.
    """
    return make_prefill_causal_mask(jnp.ones((batch, seq), bool))


def _relerr(a, b):
    scale = max(float(np.abs(a).max()), 1e-12)
    return float(np.abs(a - b).max() / scale)


# --------------------------------------------------------------------- router

def test_router_matches_reference():
    """Scale-less norm, learned per-channel scale, 1/sqrt(H), softmax, top-k,
    renormalize, per-expert scale — in that order."""
    torch, _, _, _, Gemma4TextRouter = _ref()
    torch.manual_seed(0)
    cfg_hf, cfg_jax = _hf_config(), _jax_config()
    ref = Gemma4TextRouter(cfg_hf).float().eval()
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0, 0.5)

    x = torch.randn(7, H, dtype=torch.float32)
    with torch.no_grad():
        _, ref_w, ref_i = ref(x)

    params = {
        "router_proj": jnp.asarray(ref.proj.weight.detach().numpy().T, jnp.float32),
        "router_scale": jnp.asarray(ref.scale.detach().numpy(), jnp.float32),
        "router_per_expert_scale": jnp.asarray(
            ref.per_expert_scale.detach().numpy(), jnp.float32),
    }
    got_w, got_i = moe_router_jax(
        jnp.asarray(x.numpy(), jnp.float32), params, cfg_jax)

    # Indices must match exactly: they choose which experts run, so a difference
    # here is a different model, not a rounding difference.
    np.testing.assert_array_equal(np.asarray(got_i), ref_i.numpy())
    assert _relerr(ref_w.numpy(), np.asarray(got_w)) < ATOL


# -------------------------------------------------------------------- experts

@pytest.mark.parametrize("impl", ["gather", "dense"])
def test_experts_match_reference(impl):
    """Both expert paths must reproduce the reference's index_add over experts."""
    torch, _, Gemma4TextExperts, _, _ = _ref()
    torch.manual_seed(1)
    cfg_hf, cfg_jax = _hf_config(), _jax_config()
    ref = Gemma4TextExperts(cfg_hf).float().eval()
    with torch.no_grad():
        ref.gate_up_proj.normal_(0, 0.3)
        ref.down_proj.normal_(0, 0.3)

    T = 11
    x = torch.randn(T, H, dtype=torch.float32)
    top_i = torch.stack([torch.randperm(E)[:K] for _ in range(T)])
    top_w = torch.rand(T, K, dtype=torch.float32)
    top_w = top_w / top_w.sum(-1, keepdim=True)

    with torch.no_grad():
        ref_out = ref(x, top_i, top_w)

    params = {
        "experts_gate_up_proj": jnp.asarray(
            ref.gate_up_proj.detach().numpy(), jnp.float32),
        "experts_down_proj": jnp.asarray(
            ref.down_proj.detach().numpy(), jnp.float32),
    }
    fn = moe_experts_gather if impl == "gather" else moe_experts_dense
    got = fn(jnp.asarray(x.numpy(), jnp.float32),
             jnp.asarray(top_w.numpy(), jnp.float32),
             jnp.asarray(top_i.numpy(), jnp.int32),
             params, cfg_jax, quant_mode="fp16")

    assert _relerr(ref_out.numpy(), np.asarray(got)) < ATOL


def test_gather_and_dense_agree():
    """The two paths differ only in how much work they do, never in the answer.

    ``auto`` switches between them on token count, so a divergence here would
    make prefill and decode disagree about the same weights.
    """
    torch, *_ = _ref()
    torch.manual_seed(2)
    cfg_jax = _jax_config()
    rng = np.random.default_rng(2)
    params = {
        "experts_gate_up_proj": jnp.asarray(
            rng.normal(0, 0.3, (E, 2 * MOE_I, H)), jnp.float32),
        "experts_down_proj": jnp.asarray(
            rng.normal(0, 0.3, (E, H, MOE_I)), jnp.float32),
    }
    T = 20
    x = jnp.asarray(rng.normal(0, 1, (T, H)), jnp.float32)
    top_i = jnp.asarray(
        np.stack([rng.permutation(E)[:K] for _ in range(T)]), jnp.int32)
    w = rng.random((T, K))
    top_w = jnp.asarray(w / w.sum(-1, keepdims=True), jnp.float32)

    a = moe_experts_gather(x, top_w, top_i, params, cfg_jax, quant_mode="fp16")
    b = moe_experts_dense(x, top_w, top_i, params, cfg_jax, quant_mode="fp16")
    assert _relerr(np.asarray(a), np.asarray(b)) < 1e-5


# ----------------------------------------------------------------- full model

@pytest.mark.parametrize("impl", ["gather", "dense"])
def test_full_model_matches_reference(impl):
    """End-to-end: embeddings, attention, RoPE, the MoE sandwich, layer_scalar,
    final norm, tied lm_head and softcapping, against Gemma4TextModel.

    This is the test that would catch feeding the experts the wrong tensor.
    """
    torch, _, _, Gemma4TextModel, _ = _ref()
    torch.manual_seed(3)
    cfg_hf, cfg_jax = _hf_config(), _jax_config()
    ref = Gemma4TextModel(cfg_hf).float().eval()
    with torch.no_grad():
        for p in ref.parameters():
            if p.dim() >= 2:
                p.normal_(0, 0.05)
            else:
                # Norm weights sit near 1; Gemma's are learned and far from it,
                # so perturb rather than leave them at the init value.
                p.uniform_(0.5, 1.5)

    ids = torch.randint(0, V, (1, 8))
    with torch.no_grad():
        hidden = ref(input_ids=ids).last_hidden_state
        cap = cfg_hf.final_logit_softcapping
        ref_logits = torch.tanh(
            (hidden @ ref.embed_tokens.weight.T) / cap) * cap

    params = convert_safetensors_to_jax_params(
        _to_raw(ref.state_dict()), num_layers=L, first_kv_shared_idx=L,
        attention_k_eq_v=False, dtype=jnp.float32)

    model = Gemma4EModelJAX(cfg_jax, moe_impl=impl)
    S = ids.shape[1]
    got = model(
        jnp.asarray(ids.numpy(), jnp.int32), params,
        jnp.arange(S, dtype=jnp.int32)[None, :],
        attention_mask=_causal(1, S),
        quant_mode="fp16",
    )

    got = np.asarray(got, np.float32)
    exp = ref_logits.numpy()
    assert got.shape == exp.shape
    assert _relerr(exp, got) < 5e-3
    np.testing.assert_array_equal(got[0].argmax(-1), exp[0].argmax(-1))


def test_router_reads_the_raw_residual_not_the_normed_one():
    """Regression guard for the bug this file actually caught.

    ``moe_block_jax`` takes the router input and the expert input separately
    because the reference routes on the RAW residual and only normalizes the
    copy the experts see. Collapsing them to one tensor left every unit test
    here passing while the full model was off by 0.36 relative.

    So: routing on a normalized tensor must produce a *different* answer. If this
    ever passes trivially, the two inputs have been merged again.
    """
    rng = np.random.default_rng(11)
    cfg = _jax_config()
    params = {
        "experts_gate_up_proj": jnp.asarray(
            rng.normal(0, 0.3, (E, 2 * MOE_I, H)), jnp.float32),
        "experts_down_proj": jnp.asarray(
            rng.normal(0, 0.3, (E, H, MOE_I)), jnp.float32),
        "router_proj": jnp.asarray(rng.normal(0, 0.3, (H, E)), jnp.float32),
        "router_scale": jnp.asarray(rng.uniform(0.5, 1.5, H), jnp.float32),
        "router_per_expert_scale": jnp.asarray(
            rng.uniform(0.5, 1.5, E), jnp.float32),
    }
    from ports.gemma4.jax_e_model import moe_block_jax, rms_norm_jax

    T = 20
    residual = jnp.asarray(rng.normal(0, 2.0, (T, H)), jnp.float32)
    pre_norm_2 = jnp.asarray(rng.uniform(0.5, 1.5, H), jnp.float32)
    normed = rms_norm_jax(residual, pre_norm_2, eps=cfg.rms_norm_eps)

    correct = moe_block_jax(residual, normed, params, cfg, quant_mode="fp16")
    merged = moe_block_jax(normed, normed, params, cfg, quant_mode="fp16")

    assert _relerr(np.asarray(correct), np.asarray(merged)) > 1e-3, (
        "routing on the normalized residual gave the same answer as routing on "
        "the raw one — the router and expert inputs have been collapsed")


def test_moe_input_is_the_residual_not_the_normed_mlp_input():
    """Guard the specific wiring mistake this port was most likely to make.

    Re-runs the model with ``pre_feedforward_layernorm_2`` replaced by
    ``pre_feedforward_layernorm``. If the experts were (wrongly) reading the same
    tensor as the dense MLP, that substitution would be a no-op. It must not be.
    """
    torch, _, _, Gemma4TextModel, _ = _ref()
    torch.manual_seed(4)
    cfg_hf, cfg_jax = _hf_config(), _jax_config()
    ref = Gemma4TextModel(cfg_hf).float().eval()
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0, 0.05) if p.dim() >= 2 else p.uniform_(0.5, 1.5)

    raw = _to_raw(ref.state_dict())
    params = convert_safetensors_to_jax_params(
        raw, num_layers=L, first_kv_shared_idx=L, attention_k_eq_v=False,
        dtype=jnp.float32)

    ids = jnp.asarray(np.random.default_rng(4).integers(0, V, (1, 8)), jnp.int32)
    pos = jnp.arange(8, dtype=jnp.int32)[None, :]
    mask = _causal(1, 8)
    model = Gemma4EModelJAX(cfg_jax)
    base = np.asarray(model(ids, params, pos, attention_mask=mask,
                            quant_mode="fp16"), np.float32)

    swapped = {k: (dict(v) if isinstance(v, dict) else v)
               for k, v in params.items()}
    for i in range(L):
        swapped[f"layer_{i}"]["pre_feedforward_layernorm_2"] = \
            swapped[f"layer_{i}"]["pre_feedforward_layernorm"]
    other = np.asarray(model(ids, swapped, pos, attention_mask=mask,
                             quant_mode="fp16"), np.float32)

    assert _relerr(base, other) > 1e-3, (
        "swapping the MoE pre-norm changed nothing — the experts are not reading "
        "their own normalized residual")
