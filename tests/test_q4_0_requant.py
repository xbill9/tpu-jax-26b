"""Q4_0 -> W4A16 requantization, the thing that makes the 26B A4B fit at all.

The 26B ships only ``-qat-q4_0-unquantized`` (51.6 GB BF16, no ``-w4a16-ct``
release exists), so it reaches HBM only if the already-Q4_0-quantized values can
be recovered and repacked. These tests pin the two properties that has to have:
the recovered grid must be the ORIGINAL grid, and the packed output must be
byte-compatible with the W4A16 decoder the engine already runs.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp

from ports.gemma4.jax_e_model import qat_w4a16_unpack_dequant_jax
from ports.gemma4.jax_q4_0 import (
    GROUP_SIZE,
    dequantize_q4_0,
    quantize_q4_0,
    quantized_bytes,
)


def _q4_0_weights(shape, seed=0, peak_level=8):
    """Weights that lie exactly on a Q4_0 grid, as the real checkpoint's do.

    ``peak_level`` is the level the largest magnitude in each group occupies.
    The textbook ``d = amax / 8`` rule is only correct when that is 8, and the
    real checkpoint is full of groups where it is not — which is the whole
    reason the step is searched for rather than assumed.
    """
    rng = np.random.default_rng(seed)
    n = int(np.prod(shape))
    assert shape[-1] % GROUP_SIZE == 0
    g = n // GROUP_SIZE
    # fp16 step, matching how Q4_0 stores its block scale
    d = (rng.uniform(0.005, 0.05, (g, 1))).astype(np.float16).astype(np.float32)
    # Q4_0 levels are [-8, 7] — asymmetric. Generating +8 would put the data off
    # the grid the decoder can represent and make this fixture, not the code
    # under test, the thing that fails.
    q = rng.integers(-peak_level, min(peak_level, 7) + 1,
                     (g, GROUP_SIZE)).astype(np.float32)
    q[:, 0] = -peak_level          # guarantee the peak sits at the stated level
    return jnp.asarray((q * d).reshape(shape), dtype=jnp.bfloat16)


@pytest.mark.parametrize("peak_level", [1, 2, 3, 5, 7, 8])
def test_recovers_grid_at_any_peak_level(peak_level):
    """d = amax/8 is wrong unless the peak is at level 8. Search must handle all."""
    w = _q4_0_weights((64, 256), seed=peak_level, peak_level=peak_level)
    packed, scale, failed = quantize_q4_0(w, row_chunk=32)
    assert failed == 0

    back = dequantize_q4_0(packed, scale)
    a = np.asarray(w, np.float32)
    b = np.asarray(back, np.float32)
    rel = np.abs(a - b).max() / np.abs(a).max()
    # Only BF16 rounding of the stored scale should survive.
    assert rel < 2 ** -6, f"peak_level={peak_level} rel={rel:.3e}"


def test_naive_amax_over_8_would_fail_this():
    """Pin the failure mode, so the guard above cannot be weakened by accident.

    Quantizing a level-3-peaked group with the textbook step lands its values
    between grid points. This asserts that error is real and large — if it ever
    stops being, the test above has stopped proving anything.
    """
    w = np.asarray(_q4_0_weights((8, 64), seed=1, peak_level=3), np.float32)
    g = w.reshape(-1, GROUP_SIZE)
    amax = np.abs(g).max(-1, keepdims=True)
    d_naive = amax / 8.0
    naive = np.clip(np.rint(g / d_naive), -8, 7) * d_naive
    assert np.abs(g - naive).max() / np.abs(g).max() > 0.02


def test_packed_is_readable_by_the_engine_decoder():
    """The packing must match the layout the W4A16 path already decodes.

    ``qat_w4a16_unpack_dequant_jax`` is what every dense model in this repo runs;
    if the 26B's repacked weights are not bit-compatible with it, the model
    loads and produces garbage rather than failing.
    """
    w = _q4_0_weights((32, 128), seed=7)
    packed, scale, failed = quantize_q4_0(w, row_chunk=16)
    assert failed == 0
    assert packed.dtype == jnp.int32
    assert packed.shape == (32, 128 // 8)
    assert scale.shape == (32, 128 // GROUP_SIZE)
    assert scale.dtype == jnp.bfloat16

    engine = np.asarray(qat_w4a16_unpack_dequant_jax(packed, scale), np.float32)
    ours = np.asarray(dequantize_q4_0(packed, scale), np.float32)
    np.testing.assert_array_equal(engine, ours)


def test_stacked_expert_rank3():
    """Expert banks are [E, out, in] and must pack per-expert along `in`."""
    w = _q4_0_weights((4, 16, 64), seed=3)
    packed, scale, failed = quantize_q4_0(w, row_chunk=8)
    assert failed == 0
    assert packed.shape == (4, 16, 8)
    assert scale.shape == (4, 16, 2)

    back = np.asarray(dequantize_q4_0(packed, scale), np.float32)
    a = np.asarray(w, np.float32)
    assert np.abs(a - back).max() / np.abs(a).max() < 2 ** -6

    # Per-expert independence: perturbing expert 0 must not move expert 3.
    w2 = np.asarray(w, np.float32).copy()
    w2[0] *= 4.0
    p2, s2, _ = quantize_q4_0(jnp.asarray(w2, jnp.bfloat16), row_chunk=8)
    np.testing.assert_array_equal(np.asarray(packed)[3], np.asarray(p2)[3])


def test_row_chunking_does_not_change_the_result():
    """Chunking is a memory bound, not a numerical choice."""
    w = _q4_0_weights((96, 128), seed=5)
    a = quantize_q4_0(w, row_chunk=7)
    b = quantize_q4_0(w, row_chunk=96)
    np.testing.assert_array_equal(np.asarray(a[0]), np.asarray(b[0]))
    np.testing.assert_array_equal(np.asarray(a[1]), np.asarray(b[1]))


def test_non_q4_0_input_is_reported_not_silently_accepted():
    """Plain Gaussian weights are not on a 4-bit grid; that must be visible.

    The loader turns a nonzero count into a hard error. If this returned 0, a
    non-QAT checkpoint would be quantized for real and reported as repacked.
    """
    rng = np.random.default_rng(9)
    w = jnp.asarray(rng.normal(0, 0.05, (32, 128)), jnp.bfloat16)
    _, _, failed = quantize_q4_0(w, row_chunk=16)
    assert failed > 0


def test_zero_groups_survive():
    """An all-zero group has no peak to derive a step from; it must not NaN."""
    w = np.asarray(_q4_0_weights((4, 64), seed=2), np.float32)
    w[1, :GROUP_SIZE] = 0.0
    packed, scale, failed = quantize_q4_0(jnp.asarray(w, jnp.bfloat16), row_chunk=4)
    assert failed == 0
    back = np.asarray(dequantize_q4_0(packed, scale), np.float32)
    assert np.isfinite(back).all()
    np.testing.assert_array_equal(back[1, :GROUP_SIZE], np.zeros(GROUP_SIZE))


def test_quantized_bytes_matches_reality():
    """The size projection that justified 'it fits' has to be the real size."""
    w = _q4_0_weights((64, 256), seed=4)
    packed, scale, _ = quantize_q4_0(w, row_chunk=32)
    actual = packed.size * 4 + scale.size * 2
    assert quantized_bytes((64, 256)) == actual
    # 4 bits/value + one BF16 scale per 32 = 4.5 bits/value = 3.56x off BF16
    assert 3.5 < (w.size * 2) / actual < 3.6
