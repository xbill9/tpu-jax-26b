"""JAX PyTree Parameter Loader for Gemma 4 E2B QAT Checkpoints.

Loads Hugging Face safetensors checkpoints (e.g. google/gemma-4-E2B-it-qat-w4a16-ct or
google/gemma-4-E2B-it-qat-q4_0-unquantized) directly into JAX PyTree dictionaries.

Pure Python & JAX — ZERO PyTorch, transformers, or confidential repository dependencies.
"""

from typing import Dict, Any
import jax.numpy as jnp


# Gemma 4 E2B ships as a multimodal checkpoint: the text decoder lives under
# `model.language_model.` alongside `model.audio_tower.` / vision towers. Older
# text-only exports use a bare `model.` prefix. Detect rather than assume — the
# wrong prefix silently yields a parameter tree full of None.
_TEXT_PREFIXES = ("model.language_model.", "language_model.", "model.")


def detect_text_prefix(raw_weights: Dict[str, Any]) -> str:
    """Returns the key prefix the text decoder lives under.

    Raises ValueError listing the candidates if none matches, rather than
    letting every subsequent lookup miss and produce an all-None param tree.
    """
    for prefix in _TEXT_PREFIXES:
        if f"{prefix}embed_tokens.weight" in raw_weights:
            return prefix
    sample = sorted(k for k in raw_weights if "embed_tokens" in k)[:5]
    raise ValueError(
        "Could not locate the text decoder in this checkpoint. Tried prefixes "
        f"{_TEXT_PREFIXES}; keys containing 'embed_tokens': {sample or 'none'}"
    )


def convert_safetensors_to_jax_params(
    raw_weights: Dict[str, Any],
    num_layers: int = 35,
    first_kv_shared_idx: int = 15,
    prefix: str = None,
    attention_k_eq_v: bool = False,
    requantize_q4_0: bool = False,
    q4_0_row_chunk: int = 256,
    stats: Dict[str, Any] = None,
    dtype=jnp.bfloat16,
) -> Dict[str, Any]:
    """Converts a dict of safetensors numpy/jax arrays into Gemma4EModelJAX parameter PyTree.

    Handles key remapping for:
      - Primary and Per-Layer Embeddings (PLE)
      - Non-shared layers (0..14) with q/k/v/o attention projections
      - Shared layers (15..34) with q/o attention projections (no k/v)
      - MatFormer Double-Wide MLP weights
      - QAT packed W4A16 weights and scales (_packed, _scale suffix mapping)
      - attention_k_eq_v: layers that ship k_proj but no v_proj, because V is K
      - Sparse MoE router + stacked expert banks (gemma-4-26B-A4B)

    attention_k_eq_v: set from the checkpoint config. True on gemma-4-31B, where
      the full-attention layers omit v_proj entirely; False on E2B.

    requantize_q4_0: pack every projection into W4A16 as it is converted, for
      ``-qat-q4_0-unquantized`` checkpoints that ship BF16 weights already on a
      Q4_0 grid. Needed by the 26B A4B, which has no ``-w4a16-ct`` release and
      does not fit a v6e-1 as BF16 (51.6 GB vs 33.55 GB of HBM).

      This has to happen HERE, before the ``[out, in] -> [in, out]`` transpose
      that the dense path applies, because the Q4_0 groups run along ``in`` and
      W4A16 packs along the last axis. Packing a transposed weight would group
      across ``out``, where no grid exists, and silently requantize for real —
      it would still load, still run, and be a different model. See
      ports/gemma4/jax_q4_0.py.

    stats: optional dict, populated with requantization counters when packing.

    dtype: element type for dense weights. BF16 is what serving uses; float32
      exists so parity tests against the reference implementation measure the
      port's error rather than BF16's, which is 100x larger and would hide a
      genuine wiring mistake inside its own noise floor.
    """
    if prefix is None:
        prefix = detect_text_prefix(raw_weights)

    jax_params: Dict[str, Any] = {}
    missing: list = []
    counters = {"packed": 0, "failed_groups": 0, "bytes_dense": 0, "bytes_packed": 0}

    def get_arr(key: str, default_dtype=None):
        default_dtype = dtype if default_dtype is None else default_dtype
        if key in raw_weights:
            arr = raw_weights[key]
            if not isinstance(arr, jnp.ndarray):
                arr = jnp.array(arr)
            if jnp.issubdtype(arr.dtype, jnp.integer):
                return arr
            return arr.astype(default_dtype)
        return None

    def get_linear(key: str):
        """Convert HF ``[out, in]`` dense weights to JAX ``[in, out]``."""
        arr = get_arr(key)
        return None if arr is None else arr.T

    def get_quantized(prefix: str, destination: Dict[str, Any], name: str):
        packed_key = f"{prefix}.{name}.weight_packed"
        scale_key = f"{prefix}.{name}.weight_scale"
        if packed_key not in raw_weights:
            return False
        if scale_key not in raw_weights:
            raise ValueError(f"Missing W4A16 scale tensor: {scale_key}")
        packed = get_arr(packed_key)
        scale = get_arr(scale_key)
        if packed.dtype != jnp.int32:
            raise TypeError(
                f"{packed_key} must be compressed-tensors int32, got "
                f"{packed.dtype}"
            )
        if packed.ndim != 2 or scale.ndim != 2:
            raise ValueError(
                f"{packed_key}/{scale_key} must be rank 2, got "
                f"{packed.shape}/{scale.shape}"
            )
        expected_scale = (packed.shape[0], packed.shape[1] // 4)
        if scale.shape != expected_scale:
            raise ValueError(
                f"{scale_key} has shape {scale.shape}; expected "
                f"{expected_scale} for group_size=32"
            )
        destination[f"{name}_packed"] = packed
        destination[f"{name}_scale"] = scale.astype(jnp.bfloat16)
        return True

    def put_weight(destination: Dict[str, Any], name: str, hf_key: str,
                   stacked: bool = False) -> bool:
        """Store an HF ``[out, in]`` (or ``[E, out, in]``) weight under ``name``.

        Dense: transposed to ``[in, out]`` for rank 2, left alone for the stacked
        expert banks, which the MoE einsums contract on their last axis.
        Requantized: packed W4A16 in HF orientation under ``name_packed`` /
        ``name_scale``, which is the layout the engine's dequant already reads.
        """
        arr = get_arr(hf_key)
        if arr is None:
            return False
        if requantize_q4_0:
            from ports.gemma4.jax_q4_0 import quantize_q4_0

            packed, scale, failed = quantize_q4_0(arr, row_chunk=q4_0_row_chunk)
            destination[f"{name}_packed"] = packed
            destination[f"{name}_scale"] = scale
            counters["packed"] += 1
            counters["failed_groups"] += failed
            counters["bytes_dense"] += arr.size * arr.dtype.itemsize
            counters["bytes_packed"] += packed.size * 4 + scale.size * 2
        else:
            destination[name] = arr if stacked else arr.T
        return True

    # Global Embeddings & Norms
    jax_params["embed_tokens"] = get_arr(f"{prefix}embed_tokens.weight")
    jax_params["final_norm"] = get_arr(f"{prefix}norm.weight")

    if f"{prefix}embed_tokens_per_layer.weight" in raw_weights:
        jax_params["embed_tokens_per_layer"] = get_arr(f"{prefix}embed_tokens_per_layer.weight")
        if not get_quantized(f"{prefix.rstrip('.')}", jax_params, "per_layer_model_projection"):
            jax_params["per_layer_model_projection"] = get_linear(
                f"{prefix}per_layer_model_projection.weight")
        jax_params["per_layer_projection_norm"] = get_arr(f"{prefix}per_layer_projection_norm.weight")

    # Layer Weights Loop
    for i in range(num_layers):
        lp = f"{prefix}layers.{i}"
        is_shared = i >= first_kv_shared_idx
        layer_p: Dict[str, Any] = {}

        # Layernorms
        layer_p["input_layernorm"] = get_arr(f"{lp}.input_layernorm.weight")
        layer_p["post_attention_layernorm"] = get_arr(f"{lp}.post_attention_layernorm.weight")

        # PLE per-layer weights. Shipped QAT checkpoints store these W4A16-packed
        # (weight_packed/weight_scale), the same scheme vLLM's loader does not
        # implement for per_layer_model_projection — see tpu-inference #3225.
        for ple_name in ("per_layer_input_gate", "per_layer_projection"):
            if not get_quantized(lp, layer_p, ple_name):
                dense = f"{lp}.{ple_name}.weight"
                if dense in raw_weights:
                    layer_p[ple_name] = get_linear(dense)
        layer_p["post_per_layer_input_norm"] = get_arr(f"{lp}.post_per_layer_input_norm.weight")

        # Gemma sandwich norms around the feed-forward block.
        layer_p["pre_feedforward_layernorm"] = get_arr(f"{lp}.pre_feedforward_layernorm.weight")
        layer_p["post_feedforward_layernorm"] = get_arr(f"{lp}.post_feedforward_layernorm.weight")
        # Single learned scalar per layer. This IS applied — the decoder layer ends
        # with `h *= layer_scalar` over the whole residual stream (jax_e_model.py,
        # end of the layer loop). Values are far from 1.0 in both checkpoints
        # (E2B mean 0.532 over 35 layers, 12B mean 0.606 over 48), so dropping it
        # is not a no-op.
        if f"{lp}.layer_scalar" in raw_weights:
            layer_p["layer_scalar"] = get_arr(f"{lp}.layer_scalar")

        # Attention weights
        attn_p: Dict[str, Any] = {}
        for proj in ["q_proj", "o_proj"]:
            proj_prefix = f"{lp}.self_attn"
            if not get_quantized(proj_prefix, attn_p, proj):
                put_weight(attn_p, proj, f"{proj_prefix}.{proj}.weight")

        if f"{lp}.self_attn.q_norm.weight" in raw_weights:
            attn_p["q_norm"] = get_arr(f"{lp}.self_attn.q_norm.weight")

        if not is_shared:
            for proj in ["k_proj", "v_proj"]:
                proj_prefix = f"{lp}.self_attn"
                if not get_quantized(proj_prefix, attn_p, proj):
                    put_weight(attn_p, proj, f"{proj_prefix}.{proj}.weight")

            # attention_k_eq_v: the checkpoint ships no v_proj on these layers
            # because V *is* K — one projection feeds both. Verified on
            # gemma-4-31B, where every full-attention layer (i % 6 == 5) has
            # k_proj and k_norm but no v_proj at all, while sliding layers carry
            # both. E2B sets the flag False and ships v_proj everywhere.
            #
            # Aliasing rather than copying: these are the same arrays, so the
            # duplicate reference costs nothing. Measured on the 31B (v6e-1,
            # 2026-07-31): the parameter tree reports 19.357 GB while only
            # 19.297 GB is resident, and the 60 MB gap is exactly these 10
            # full-attention k_proj tensors counted twice and stored once.
            #
            # Storing K and V separately in the *cache* is then redundant but
            # correct, and that one is not cheap. `init_kv_cache` allocates both
            # buffers unconditionally, so on the 31B's 10 full-attention layers
            # it duplicates 40 KiB/token — 22% of total KV at 8K context and
            # ~46% (5.37 GB) at 128K. Collapsing it is the single largest
            # long-context memory win available here. An earlier version of this
            # comment put the saving at "~4.5%", which is wrong at any context
            # long enough to matter.
            if attention_k_eq_v and attn_p.get("v_proj") is None \
                    and "v_proj_packed" not in attn_p:
                if "k_proj_packed" in attn_p:
                    attn_p["v_proj_packed"] = attn_p["k_proj_packed"]
                    attn_p["v_proj_scale"] = attn_p["k_proj_scale"]
                elif attn_p.get("k_proj") is not None:
                    attn_p["v_proj"] = attn_p["k_proj"]

            if f"{lp}.self_attn.k_norm.weight" in raw_weights:
                attn_p["k_norm"] = get_arr(f"{lp}.self_attn.k_norm.weight")
            if f"{lp}.self_attn.v_norm.weight" in raw_weights:
                attn_p["v_norm"] = get_arr(f"{lp}.self_attn.v_norm.weight")

        layer_p["attn"] = attn_p

        # MLP weights
        mlp_p: Dict[str, Any] = {}
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            proj_prefix = f"{lp}.mlp"
            if not get_quantized(proj_prefix, mlp_p, proj):
                put_weight(mlp_p, proj, f"{proj_prefix}.{proj}.weight")

        layer_p["mlp"] = mlp_p

        # Sparse MoE block (gemma-4-26B-A4B). Present on every layer when
        # present at all; absent entirely on the dense sizes, so detect rather
        # than switch on a config flag the caller might not have set.
        moe_p: Dict[str, Any] = {}
        if f"{lp}.router.proj.weight" in raw_weights:
            # Router. `proj` is an ordinary [E, H] Linear and transposes like any
            # other; `scale` [H] and `per_expert_scale` [E] are rank-1 and do not.
            #
            # Deliberately left DENSE even when everything else is packed. It is
            # 0.72 MB — 0.005% of the layer — so quantizing it saves nothing, and
            # it is the one matmul whose output is not averaged into anything:
            # its 128 logits pick which experts run, so an error there swaps an
            # expert outright rather than perturbing a sum.
            moe_p["router_proj"] = get_linear(f"{lp}.router.proj.weight")
            moe_p["router_scale"] = get_arr(f"{lp}.router.scale")
            moe_p["router_per_expert_scale"] = get_arr(f"{lp}.router.per_expert_scale")

            # Stacked expert banks, [E, out, in]. Kept in HF orientation — NOT
            # transposed like the rank-2 projections — because the W4A16 nibble
            # packing groups along the last axis and the Q4_0 grid these weights
            # already sit on runs along `in`. The einsums in jax_e_model contract
            # `in` on the last axis and consume [E, out, in] directly.
            #
            # Note these ship WITHOUT a `.weight` suffix (they are nn.Parameter,
            # not nn.Linear), so the key is `experts.gate_up_proj` exactly.
            for name in ("gate_up_proj", "down_proj"):
                if not put_weight(moe_p, f"experts_{name}",
                                  f"{lp}.experts.{name}", stacked=True):
                    missing.append(f"layer {i}: experts.{name}")

            # The MoE half of the sandwich: its own pre-norm on the residual and
            # post-norms on each of the two branches.
            layer_p["pre_feedforward_layernorm_2"] = get_arr(
                f"{lp}.pre_feedforward_layernorm_2.weight")
            for n in ("post_feedforward_layernorm_1", "post_feedforward_layernorm_2"):
                layer_p[n] = get_arr(f"{lp}.{n}.weight")
                if layer_p[n] is None:
                    missing.append(f"layer {i}: {n}")

        if moe_p:
            layer_p["moe"] = moe_p
        if not mlp_p:
            missing.append(f"layer {i}: no MLP weights")
        if not attn_p:
            missing.append(f"layer {i}: no attention weights")
        jax_params[f"layer_{i}"] = layer_p

    # Fail loudly. A prefix or naming mismatch otherwise yields a tree of Nones
    # that loads "successfully" in seconds and holds zero bytes on device.
    for required in ("embed_tokens", "final_norm"):
        if jax_params.get(required) is None:
            missing.append(f"global tensor '{required}'")
    if "embed_tokens_per_layer" in jax_params and not (
        jax_params.get("per_layer_model_projection") is not None
        or "per_layer_model_projection_packed" in jax_params
    ):
        missing.append("per_layer_model_projection (dense or W4A16-packed)")
    if missing:
        raise ValueError(
            f"Checkpoint is missing required tensors under prefix '{prefix}': "
            + "; ".join(missing[:8])
            + (f" (+{len(missing) - 8} more)" if len(missing) > 8 else "")
        )

    if requantize_q4_0:
        # A nonzero count means groups that did not lie on a 4-bit grid at any
        # level, i.e. this checkpoint is not Q4_0-QAT. It still loaded, and it
        # will still generate text — from weights that were quantized for real
        # rather than repacked. Refuse instead of reporting it in a log line
        # nobody reads.
        if counters["failed_groups"]:
            raise ValueError(
                f"{counters['failed_groups']} weight groups are not on a Q4_0 "
                f"4-bit grid; this checkpoint is not a '-qat-q4_0-unquantized' "
                f"export and requantize_q4_0 would silently degrade it. Load it "
                f"dense, or use a '-qat-w4a16-ct' checkpoint."
            )
        if stats is not None:
            stats.update(counters)

    return jax_params
