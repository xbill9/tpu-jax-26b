"""Q4_0 -> W4A16 requantization for Gemma 4 ``-qat-q4_0-unquantized`` checkpoints.

Why this module exists
----------------------
The 26B A4B is the one Gemma 4 size that ships **no** ``-qat-w4a16-ct``
checkpoint (verified against the Hub 2026-07-31: E2B, E4B, 12B and 31B all have
one; 26B has only ``-qat-q4_0-gguf`` and ``-qat-q4_0-unquantized``). So there is
no pre-packed int4 tensor to load, and the only non-GGUF option is the
"unquantized" export: 51.61 GB of BF16 against a v6e-1's 33.55 GB of HBM.

"Unquantized" is a statement about the *container*, not the values. These are QAT
weights that have already been through simulated Q4_0 quantization and then
written out in BF16, so every 32-wide group along the input axis already lies on
a 16-level grid. Re-deriving that grid and storing it packed is therefore mostly
a change of container rather than a second quantization — and it takes the text
decoder from 51.6 GB to 14.19 GB, which fits with 19.4 GB to spare.

Measured on the real checkpoint before any of this was written (see
``benchmarks/runs/2026-07-31-gemma4-26b-v6e1/REPORT.md``): every one of 256
sampled groups of 32 lies exactly on a 4-bit grid, for expert, attention, MLP,
router and embedding tensors alike. Group 64 fails (3/128), which is what pins
the group size at 32 rather than leaving it an assumption.

How close to lossless, exactly
------------------------------
Not bitwise, and the gap is worth stating precisely rather than rounding to
"lossless". Round-tripped against the checkpoint bytes for ten tensors spanning
every kind in the model (expert banks, attention, MLP, router, embeddings, both
a sliding and a full-attention layer):

    0 groups unrepresentable out of 52,096
    89-93% of individual values reconstruct BIT-IDENTICALLY
    worst-case relative error 6.4e-3, i.e. ~1.6 BF16 ULP

The residue is scale precision, not level assignment: the integer levels are
recovered exactly (verified — refining the step does not move a single level),
but Q4_0 blocks carry an fp16 step and this format stores BF16, which is three
mantissa bits shorter. That is the same precision regime the shipped
``-w4a16-ct`` checkpoints already run in on this engine, so it is a known
quantity rather than a new risk — but it is a real difference from the BF16
checkpoint and end-to-end parity should be measured, not assumed.

The one trap
------------
The textbook Q4_0 rule ``d = amax / -8`` is **wrong on this checkpoint**, and
wrong in a way that still runs. It assumes the largest-magnitude value in a group
sits at level +-8. When it does not — when a group's peak is at level 5, say — the
derived step is 5/8 of the true step, ``round(x/d)`` lands between grid points,
and the group is re-quantized onto a grid that does not contain its own values.
The result is a ~5% median relative error per group (measured: 4.92e-2 on
``experts.gate_up_proj``) that no exception reports and no shape check catches.

So the step is *recovered* rather than assumed: the true step is ``amax / m`` for
whichever integer level ``m`` in [1, 8] the group's peak actually occupies. Try
each in turn and keep the first that reproduces every value in the group to
within BF16 rounding. ``quantize_q4_0`` returns the count of groups where none of
the eight candidates worked, and callers should treat a nonzero count as a
checkpoint that is not what this module thinks it is.

Output format is exactly what ``jax_e_model.qat_w4a16_unpack_dequant_jax``
already decodes, so the engine's existing W4A16 path consumes it unchanged:

    packed  int32   [..., out, in // 8]    nibble i of word j -> input col 8j+i
    scale   bfloat16 [..., out, in // 32]   one step per 32-wide input group

with each nibble holding ``q + 8`` for ``q`` in [-8, 7].
"""

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp

GROUP_SIZE = 32
_NIBBLES_PER_WORD = 8

# A group is accepted at level m when every value is reproduced to within this
# relative slack. BF16 carries 8 mantissa bits, so a correctly recovered step
# still leaves up to ~2^-9 of relative rounding in the stored value; 2^-7 gives
# margin over that while staying far below the ~2^-4 error that the *next*
# coarser wrong level would produce. The two failure modes are separated by a
# factor of 8, so this threshold is not delicate.
_REL_TOL = 2.0 ** -7


def _recover_step(groups: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Recover the Q4_0 step for each group of 32.

    ``groups`` is [..., G, 32] float32. Returns ``(d, ok)`` where ``d`` is
    [..., G, 1] and ``ok`` is [..., G, 1] bool marking groups that landed on a
    valid 4-bit grid at some level.

    Candidate levels are tried coarsest-first (m = 1 upward, i.e. largest step
    first). Any level that reproduces the group is by definition lossless, so
    "first accepted" is as good as "true", and the ascending order keeps |q|
    bounded by m <= 8 automatically.
    """
    amax = jnp.max(jnp.abs(groups), axis=-1, keepdims=True)
    tol = amax * _REL_TOL
    nonzero = amax > 0

    # Seed with the textbook Q4_0 rule. This is the WRONG step for any group whose
    # peak is not at level 8 (see module docstring), but it is a valid quantizer,
    # so a group that survives none of the candidates below degrades to ordinary
    # lossy Q4_0 instead of being silently zeroed. An earlier version of this
    # function seeded with 0, which would have written all-zero weights for every
    # unrecoverable group and reported it only through the `ok` count.
    d = jnp.where(nonzero, amax / 8.0, 0.0)
    ok = ~nonzero  # all-zero groups are trivially exact at any step

    for m in range(1, 9):
        cand = amax / m
        safe = jnp.where(cand > 0, cand, 1.0)
        q = jnp.clip(jnp.round(groups / safe), -8.0, 7.0)
        err = jnp.max(jnp.abs(groups - q * safe), axis=-1, keepdims=True)
        good = (err <= tol) & nonzero & ~ok
        d = jnp.where(good, cand, d)
        ok = ok | good

    # Refine by least squares over the whole group. `amax / m` divides a single
    # value that was itself already rounded to BF16 on the way into the
    # checkpoint, so for m not a power of two it inherits that rounding; fitting
    # the step against all 32 values averages the error down instead.
    #
    # Measured on the real checkpoint (2026-07-31): this lifts exact BF16
    # reconstruction from 78.7% to 93.1% of values on experts.gate_up_proj, and
    # the integer levels do not move at all — only the step precision improves.
    # An fp16 scale would reach 97.0% at the same two bytes per group, but the
    # engine's dequant multiplies in BF16, so realizing that would mean an f32
    # multiply in the decode hot path. Not taken; revisit only with a benchmark.
    safe_d = jnp.where(d > 0, d, 1.0)
    q = jnp.clip(jnp.round(groups / safe_d), -8.0, 7.0)
    den = jnp.sum(q * q, axis=-1, keepdims=True)
    num = jnp.sum(q * groups, axis=-1, keepdims=True)
    d = jnp.where(den > 0, num / jnp.where(den > 0, den, 1.0), d)

    return d, ok


def _pack_nibbles(q: jax.Array) -> jax.Array:
    """[..., out, in] int levels in [-8, 7] -> int32 [..., out, in // 8].

    Nibble i of word j holds ``q + 8`` for input column ``8j + i`` — the layout
    ``qat_w4a16_unpack_dequant_jax`` reads with ``(word >> 4i) & 0xF``.
    """
    lead = q.shape[:-1]
    in_features = q.shape[-1]
    u = (q + 8).astype(jnp.uint32)
    u = u.reshape(*lead, in_features // _NIBBLES_PER_WORD, _NIBBLES_PER_WORD)
    shifts = (jnp.arange(_NIBBLES_PER_WORD, dtype=jnp.uint32) * 4)
    packed = jnp.sum(u << shifts, axis=-1, dtype=jnp.uint32)
    # Nibble 7 sets bit 31, which overflows a signed cast; reinterpret instead.
    return jax.lax.bitcast_convert_type(packed, jnp.int32)


@jax.jit
def _quantize_block(w: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Core: [..., out, in] BF16/F32 -> (packed, scale, n_failed)."""
    lead = w.shape[:-1]
    in_features = w.shape[-1]
    groups = w.astype(jnp.float32).reshape(
        *lead, in_features // GROUP_SIZE, GROUP_SIZE)

    d, ok = _recover_step(groups)

    # Derive the levels against the scale that will actually be STORED, not the
    # float32 one that was recovered. The scale ships as BF16 (matching the
    # shipped -w4a16-ct checkpoints and what the engine's dequant reads), so
    # quantizing against the f32 step and reconstructing with the BF16 one
    # leaves an avoidable half-ULP of drift on every group. Rounding first
    # makes the pair self-consistent.
    #
    # This cannot move a value across a level boundary: x is ~k*d with |k| <= 8
    # and BF16 perturbs d by at most 2^-9 relative, so x/d_bf16 sits within
    # 8*2^-9 = 0.016 of an integer — far from the 0.5 that would reround it.
    d_stored = d.astype(jnp.bfloat16).astype(jnp.float32)
    safe = jnp.where(d_stored > 0, d_stored, 1.0)
    q = jnp.clip(jnp.round(groups / safe), -8.0, 7.0)

    packed = _pack_nibbles(q.reshape(*lead, in_features).astype(jnp.int32))
    scale = d_stored.squeeze(-1).astype(jnp.bfloat16)
    return packed, scale, jnp.sum(~ok)


def quantize_q4_0(
    w: jax.Array,
    row_chunk: int = 256,
) -> Tuple[jax.Array, jax.Array, int]:
    """Requantize a BF16 Q4_0-QAT weight into packed W4A16.

    ``w`` is in Hugging Face orientation ``[..., out, in]`` — the same shape the
    checkpoint stores, *before* the ``[in, out]`` transpose the dense path
    applies. Both rank-2 (a plain linear) and rank-3 (``[E, out, in]`` stacked
    experts) are accepted.

    Returns ``(packed, scale, n_unrepresentable)``. The last is the number of
    32-wide groups that did not land on a 4-bit grid at any level; on a genuine
    ``-qat-q4_0-unquantized`` checkpoint it is 0, and anything else means the
    input is not Q4_0-QAT and the "lossless" claim above does not hold for it.

    ``row_chunk`` bounds peak memory. The step search holds several float32
    copies of the working slice, and the largest tensor here is a 507M-element
    expert bank, so the whole thing at once would need ~8 GB of scratch for a
    1 GB tensor. Chunking over the flattened row axis keeps that bounded without
    changing the result — groups never span rows.
    """
    if w.ndim < 2:
        raise ValueError(f"expected at least rank 2 [..., out, in], got {w.shape}")
    in_features = w.shape[-1]
    if in_features % GROUP_SIZE:
        raise ValueError(
            f"input dim {in_features} is not a multiple of the Q4_0 group size "
            f"{GROUP_SIZE}; this checkpoint is not Q4_0-quantizable as stored")

    lead = w.shape[:-1]
    rows = int(jnp.prod(jnp.array(lead))) if lead else 1
    flat = w.reshape(rows, in_features)

    packed_parts, scale_parts = [], []
    failed = 0
    for start in range(0, rows, row_chunk):
        part = flat[start:start + row_chunk]
        p, s, nf = _quantize_block(part)
        packed_parts.append(p)
        scale_parts.append(s)
        failed += int(nf)

    packed = jnp.concatenate(packed_parts, axis=0).reshape(
        *lead, in_features // _NIBBLES_PER_WORD)
    scale = jnp.concatenate(scale_parts, axis=0).reshape(
        *lead, in_features // GROUP_SIZE)
    return packed, scale, failed


def dequantize_q4_0(packed: jax.Array, scale: jax.Array) -> jax.Array:
    """Inverse of :func:`quantize_q4_0`, for any rank. Returns BF16 [..., out, in].

    ``jax_e_model.qat_w4a16_unpack_dequant_jax`` is the rank-2 version the
    decode path uses and is deliberately left alone; this one exists because the
    stacked expert banks are rank 3 and because the round-trip test needs an
    inverse that does not care about rank.
    """
    lead = packed.shape[:-1]
    packed_k = packed.shape[-1]
    in_features = packed_k * _NIBBLES_PER_WORD

    shifts = (jnp.arange(_NIBBLES_PER_WORD, dtype=jnp.int32) * 4)
    q = ((packed[..., None] >> shifts) & jnp.int32(0xF)).reshape(*lead, in_features)
    q = q.astype(jnp.bfloat16) - jnp.bfloat16(8)

    grouped = q.reshape(*lead, in_features // GROUP_SIZE, GROUP_SIZE)
    scaled = grouped * scale.astype(jnp.bfloat16)[..., None]
    return scaled.reshape(*lead, in_features)


def quantized_bytes(shape: Tuple[int, ...]) -> int:
    """Resident bytes for a ``[..., out, in]`` tensor once packed (4.5 bits/value)."""
    n = 1
    for d in shape:
        n *= d
    return n // 2 + (n // GROUP_SIZE) * 2


def requantize_tree(
    params: Dict[str, Any],
    skip: Tuple[str, ...] = (),
    row_chunk: int = 256,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Walk a loaded parameter tree and pack every dense 2-D/3-D weight in place.

    A leaf named ``x`` becomes ``x_packed`` / ``x_scale`` and the dense array is
    dropped, which is the naming the engine's W4A16 path already looks for.
    Leaves whose name appears in ``skip``, and every rank-1 leaf (norms, scalars,
    router biases), are left alone: norms are not on a 4-bit grid and are three
    orders of magnitude too small to be worth packing.

    Returns the rewritten tree and a stats dict.
    """
    stats = {"packed": 0, "skipped": 0, "failed_groups": 0,
             "bytes_before": 0, "bytes_after": 0}

    def walk(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        out: Dict[str, Any] = {}
        for key, val in node.items():
            if isinstance(val, dict):
                out[key] = walk(val)
                continue
            if val is None or key in skip or getattr(val, "ndim", 0) < 2 \
                    or jnp.issubdtype(val.dtype, jnp.integer):
                out[key] = val
                if val is not None:
                    stats["skipped"] += 1
                    stats["bytes_before"] += val.size * val.dtype.itemsize
                    stats["bytes_after"] += val.size * val.dtype.itemsize
                continue
            packed, scale, failed = quantize_q4_0(val, row_chunk=row_chunk)
            out[f"{key}_packed"] = packed
            out[f"{key}_scale"] = scale
            stats["packed"] += 1
            stats["failed_groups"] += failed
            stats["bytes_before"] += val.size * val.dtype.itemsize
            stats["bytes_after"] += (packed.size * 4 + scale.size * 2)
        return out

    return walk(params), stats
