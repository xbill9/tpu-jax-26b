# Gemma 4: architecture quirks, verified against the reference

Every entry below is checked against `transformers.models.gemma4.modeling_gemma4`
(transformers 5.12+). That module is readable locally — **no torch install, no TPU,
no checkpoint download needed** — and reading it is faster and more reliable than
inferring architecture from tensor shapes. Five of the bugs below were found the
slow way first; the rest were found by diffing.

Sections 1–14 are E2B-first (with 31B differences called out where they exist).
Sections 15–20 cover **gemma-4-26B-A4B**, which is the odd one out twice over: it
is the only sparse checkpoint in the family and the only size with no
`-w4a16-ct` release.

Status legend: **✅ verified** against the reference · **⚠️ inferred** from shapes or
measurement only.

---

## 1. The checkpoint is multimodal ✅

The text decoder lives under `model.language_model.`, alongside `model.audio_tower.`
(751 tensors) and vision towers (659). A loader that assumes a bare `model.` prefix
finds **nothing** — and `get_arr()` returning `None` for every key produces a
parameter tree that "loads" in seconds and holds zero bytes.

`jax_e_loader.detect_text_prefix()` tries `model.language_model.`, `language_model.`,
`model.` and raises listing the candidates if none matches.

## 2. Sandwich norms ✅

`post_attention_layernorm` normalizes the attention **output**, before the residual
add — it is *not* a pre-norm for the MLP. The feed-forward block is wrapped on both
sides:

```
residual = h
h = attn(input_layernorm(h))
h = residual + post_attention_layernorm(h)

residual = h
h = mlp(pre_feedforward_layernorm(h))
h = residual + post_feedforward_layernorm(h)
```

Getting this wrong still runs and still emits fluent-looking tokens. They are just
the wrong tokens.

## 3. `layer_scalar` scales the whole residual stream ✅

Each decoder layer ends with `hidden_states *= self.layer_scalar` — a single learned
scalar per layer, applied to the **entire** hidden state after every residual add
(including the PLE injection), not to the layer's delta.

It is the counterweight to this checkpoint's large RMSNorm weights (`final_norm`
mean ≈ 14, max 118). Without it the residual stream grows layer over layer and the
output logits pin against the ±30 softcap. Observed values on E2B run 0.02–0.79.

## 4. RMSNorm is `x_normed * weight` ✅

**Not** the `x_normed * (1 + weight)` convention used by earlier Gemma generations.
The weights here are not zero-centred (means of +5 to +19 are normal), which is the
quick way to tell the two conventions apart.

`v_norm` is constructed `with_scale=False` — it normalizes with no weight at all, and
the checkpoint ships no `v_norm` tensor. Passing `weight=None` to an RMSNorm that
multiplies only when a weight is present reproduces this.

## 5. Attention: no score softcap, and `scaling = 1.0` ✅

- The config declares `final_logit_softcapping: 30.0` and **no**
  `attn_logit_softcapping`. Gemma 3+ dropped softcapping of attention *scores*;
  applying it saturates `tanh` and destroys the attention distribution.
- The reference sets `self.scaling = 1.0` — **not** `head_dim ** -0.5`. `q_norm` and
  `k_norm` already normalize query and key before the dot product, so the usual
  `1/sqrt(d)` is not applied on top. (I "fixed" this once by adding it. That was the
  regression, not the fix.)
- `q_norm`/`k_norm` are applied **before** RoPE.

## 6. RoPE: concatenated frequency layout, partial rotary by masking ✅

`rotate_half` pairs channel *i* with *i + d/2*, so the frequency table must be
`cat(freqs, freqs)`. Building it with `repeat_interleave` pairs every channel against
the wrong frequency — the model still generates, but it echoes and repeats.

Per-layer-type RoPE comes from a nested `rope_parameters` dict:

| layer type | `rope_theta` | `rope_type` | `partial_rotary_factor` |
| --- | ---: | --- | ---: |
| `sliding_attention` | 10,000 | `default` | 1.0 (absent) |
| `full_attention` | 1,000,000 | `proportional` | 0.25 |

`proportional` keeps the **full** `head_dim` and zeroes the inverse frequencies past
the factor, rather than slicing channels: with `global_head_dim` 512 and factor 0.25,
the first 64 frequency pairs are rotated and the remaining 192 get `inv_freq = 0`
(i.e. `cos = 1`, `sin = 0`, identity). Masking the frequency table is equivalent and
keeps the `cat(freqs, freqs)` pairing intact.

For text-only inputs `apply_multidimensional_rope` falls through to standard
`apply_rotary_pos_emb` — the "multidimensional" path is for image/audio position dims.

## 7. KV sharing is keyed by layer *type* ✅

`first_kv_shared_layer_idx = num_hidden_layers - num_kv_shared_layers` (35 − 20 = 15
on E2B). Layers at or above it carry **no** `k_proj`, `v_proj` or `k_norm` — the
omission upstream's loader mishandles ([tpu-inference #3225](https://github.com/vllm-project/tpu-inference/issues/3225)).

The reference stores `shared_kv_states[layer_type] = (k, v)` and every non-shared
layer overwrites its type's entry, so a shared layer reads the **last non-shared
layer of its own type**. Since E2B interleaves sliding and full attention, there are
two independent sources. `Gemma4EConfig.kv_share_map()` computes exactly this.

## 8. Per-Layer Embeddings ✅

Two components, combined:

```
token identity  = embed_tokens_per_layer[ids] * sqrt(D_ple)     -> [B,S,L,D_ple]
context         = per_layer_model_projection(inputs_embeds) * hidden_size**-0.5
                  reshaped to [B,S,L,D_ple], then per_layer_projection_norm
per_layer_input = (context + token_identity) * 2**-0.5
```

Verified constants: `embed_scale = hidden_size**0.5`, PLE `embed_scale =
hidden_size_per_layer_input**0.5`, `per_layer_model_projection_scale =
hidden_size**-0.5`, `per_layer_input_scale = 2**-0.5`.

The trap: `per_layer_projection_norm` has shape **`[D_ple]`**, so the norm applies to
the reshaped `[B,S,L,D_ple]` tensor per layer-slice — *not* across the flattened
`[L*D_ple]` projection.

In the QAT checkpoint all three PLE projections (`per_layer_model_projection`,
`per_layer_input_gate`, `per_layer_projection`) ship **W4A16-packed**, not dense. The
global one is the exact tensor #3225 reports as unimplemented in vLLM's TPU loader.

## 9. Double-wide MLP on shared layers ✅

`use_double_wide_mlp=True` means KV-shared layers use `2 * intermediate_size`
(12288 = 2 × 6144 on E2B). Confirmed in the checkpoint: layer 20's `gate_proj` is
`[12288, 192]` packed.

## 10. The tokenizer does not add BOS ⚠️

`tok("hello")` → `[23391]`, no `<bos>`. Gemma expects one, and without it the model
**echoes the prompt** instead of answering. `JaxGemmaEngine` now prepends it.
(Measured behaviour, not read from the reference.)

## 11. Config values that differ from intuition ✅

| field | value | note |
| --- | ---: | --- |
| `hidden_size` | 1536 | not 2048 |
| `num_key_value_heads` | 1 | a single 256-dim KV head |
| `num_global_key_value_heads` | `null` | falls back to 1 |
| `sliding_window` | 512 | 32 of 35 layers |
| `tie_word_embeddings` | true | `lm_head.weight` is a materialized copy |
| `layer_types` | explicit list | full attention at 4, 9, 14 … (`i % 5 == 4`) |

## 12. `attention_k_eq_v`: V *is* K, and the checkpoint omits `v_proj` ✅

Set `True` on **gemma-4-31B**, `False` on E2B. Where it is set, the affected layers
ship no `v_proj` at all — one projection feeds both K and V.

Verified by reading the 31B checkpoint on a CPU box: all ten **full-attention**
layers (`i % 6 == 5` in its 60-layer `[s,s,s,s,s,f]` pattern) carry `q_proj`,
`k_proj`, `k_norm`, `o_proj` and **no `v_proj`**, while every sliding layer carries
both. Shapes at layer 5 confirm the geometry — `k_proj` is `[2048, 672]` packed,
i.e. `num_global_key_value_heads` (4) × `global_head_dim` (512), and `k_norm` is
`[512]`.

Loading the 31B without handling this yields exactly ten missing tensors and, in a
loader that tolerates `None`, a silently broken model. `jax_e_loader` aliases V to
K (the same arrays, not copies) when `Gemma4EConfig.attention_k_eq_v` is set.

The KV cache still stores K and V separately, which is redundant but correct.
Collapsing it would save one of the two planes on those layers — worth ~4.5% of the
31B's KV, since sliding layers dominate its budget.

**This does not explain E2B's KV-bytes discrepancy below.** E2B sets the flag
`False` and ships `v_proj` on all fifteen non-shared layers, checked key by key.

## 13. `use_bidirectional_attention` is a vision setting ✅

`"vision"` on the 31B, absent/`null` on E2B. It selects bidirectional attention for
image tokens; text decoding is unaffected, so the causal-only text path is correct
for both. `store_full_length_kv` is **not present in either config** — it is a
reference-implementation concept, not a checkpoint field.

## 14. KV is 18.0 KiB/token, and full-attention layers really are 512-dim ✅

Settled by reading the checkpoint. The three full-attention layers among the fifteen
that own KV (`i % 5 == 4`, so L4/L9/L14) carry a **512-wide** K projection, and their
`k_norm` is `[512]` to match:

| layer type | count | `k_proj` out | `k_norm` | KV head dim |
|---|---:|---:|---:|---:|
| `sliding_attention` | 12 | 256 | `[256]` | 256 |
| `full_attention` | 3 | **512** | **`[512]`** | **512** |

So `global_head_dim` is the KV head dim on those layers, not merely the query head
dim. `init_kv_cache` allocates **18.00 KiB/token** at every context length, matching
`12 × 1 × 256 × 2 + 3 × 1 × 512 × 2 = 9,216` elements × 2 B exactly.

An earlier note recorded a 15.0 KiB/token reading and concluded our estimates were
"~20% pessimistic". That was backwards. 15.0 KiB is precisely what a **uniform
256-dim** assumption produces (`15 × 1 × 256 × 2 × 2 B`), and the checkpoint
contradicts it. Our figure is the correct one; nothing needs adjusting.

Worth checking upstream: an allocator that sizes E2B's KV uniformly at `head_dim`
would under-provision the three full-attention layers by half. The provenance of
that 15.0 KiB reading is not recorded here, so this is a lead rather than a report.

---

# gemma-4-26B-A4B (sparse)

## 15. The MoE block runs *alongside* the dense MLP, not instead of it ✅

`enable_moe_block: true` does not swap the feed-forward block for an expert bank.
Every layer keeps its ordinary `mlp` **and** gains a 128-expert bank, and the two
outputs are summed before the shared post-norm. That is why the 26B carries
`mlp.gate_proj` / `up_proj` / `down_proj` at full `intermediate_size` (2112) on
top of 128 experts of `moe_intermediate_size` (704).

It also gains three extra norms, for five in the feed-forward block total:

| tensor | normalizes |
| --- | --- |
| `pre_feedforward_layernorm` | input to the **dense MLP** |
| `pre_feedforward_layernorm_2` | input to the **experts** |
| `post_feedforward_layernorm_1` | output of the **dense MLP** |
| `post_feedforward_layernorm_2` | output of the **experts** |
| `post_feedforward_layernorm` | the **sum**, before the residual add |

```
residual = h
dense = post_feedforward_layernorm_1( mlp( pre_feedforward_layernorm(h) ) )
moe   = post_feedforward_layernorm_2( experts( pre_feedforward_layernorm_2(h) ) )
h     = residual + post_feedforward_layernorm(dense + moe)
h    *= layer_scalar
```

Note both branches read `h` — the post-attention residual — not each other's
output. The dense branch is §2's sandwich, unchanged; the MoE branch is a second
sandwich in parallel with it.

## 16. The router reads the RAW residual; the experts read a normalized copy ✅

The single most dangerous line in this architecture:

```python
hidden_states_flat = residual.reshape(-1, residual.shape[-1])
_, top_k_weights, top_k_index = self.router(hidden_states_flat)      # RAW
hidden_states_2 = self.pre_feedforward_layernorm_2(hidden_states_flat)
hidden_states_2 = self.experts(hidden_states_2, top_k_index, top_k_weights)  # NORMED
```

Two different tensors. Passing the normalized one to both is the obvious
simplification and it is wrong: the router opens with its own scale-less RMSNorm,
so composing it with a *learned-weight* RMSNorm first does not cancel — it
reweights the channels the router sees and changes which experts fire.

**Measured cost of getting this wrong: 0.36 relative error on a 1-layer model,
with every unit test still green.** Router parity passed. Expert parity passed.
Only end-to-end comparison against `Gemma4TextModel` caught it. Guarded now by
`test_router_reads_the_raw_residual_not_the_normed_one` and
`test_moe_input_is_the_residual_not_the_normed_mlp_input`.

`moe_block_jax` therefore takes `router_in` and `expert_in` as separate
arguments, which looks redundant until you know why.

## 17. Router internals ✅

```
x -> RMSNorm(with_scale=False)          # no weight
  -> * router.scale                     # [H], learned, separate from the norm
  -> * hidden_size ** -0.5              # fixed
  -> router.proj                        # [E, H] -> E logits
  -> softmax over ALL E
  -> top_k                              # k = top_k_experts = 8
  -> renormalize so the k weights sum to 1
  -> * router.per_expert_scale[top_k_index]
```

Order matters at the tail: the per-expert scale is applied **after**
renormalization, so the final weights do *not* sum to 1. Renormalizing afterwards
"to fix" that changes the model.

Softmax-then-top-k-then-renormalize is mathematically a softmax over just the
top-k logits, so either is fine — but only if the per-expert scale stays last.

`jax_e_model.moe_router_jax` computes this in **float32** even when activations
are BF16. The router picks *which experts run at all*, so a tie resolved
differently at BF16 resolution swaps an expert and moves the output
discontinuously, unlike a matmul where BF16 noise stays proportional. 128 logits
per token is nothing to compute in f32.

## 18. Expert tensors ship without a `.weight` suffix, stacked and fused ✅

They are `nn.Parameter`, not `nn.Linear`, so the keys have no `.weight`:

| key | shape | meaning |
| --- | --- | --- |
| `layers.N.experts.gate_up_proj` | `[128, 1408, 2816]` | `[E, 2*moe_inter, hidden]` |
| `layers.N.experts.down_proj` | `[128, 2816, 704]` | `[E, hidden, moe_inter]` |

Two traps:

- **Gate and up are fused** into one `[2I, H]` tensor per expert; the first `I`
  rows are gate, the last `I` are up (`.chunk(2, dim=-1)` on the projection
  output). Splitting on the wrong axis or the wrong half silently swaps them —
  and `gelu(up) * gate` is a plausible-looking model that is not this one.
- **They stay in `[E, out, in]` orientation.** Unlike every rank-2 projection,
  which the loader transposes to `[in, out]`, these are consumed as-is by
  einsums that contract `in` on the last axis. See §20 for why transposing them
  would be actively harmful.

## 19. There is no `-w4a16-ct` checkpoint for this size ✅

Enumerated from the Hub, 2026-07-31 — do not assume the suffix set is uniform:

| size | `-w4a16-ct` | `-q4_0-unquantized` | `-q4_0-gguf` | mobile |
| :--- | :---: | :---: | :---: | :---: |
| E2B, E4B | ✅ | ✅ | ✅ | ✅ |
| 12B, 31B | ✅ | ✅ | ✅ | — |
| **26B A4B** | **❌** | ✅ | ✅ | — |

`google/gemma-4-26B-A4B-it-qat-w4a16-ct` does not exist. GGUF targets llama.cpp,
so the only usable export is `-q4_0-unquantized`: 51.61 GB of BF16 against a
v6e-1's 33.55 GB.

It fits anyway because **"unquantized" describes the container, not the values**.
These are QAT weights already sitting on a Q4_0 grid — measured by range-reading
the shards, all 256 sampled groups of 32 lie exactly on a 4-bit grid, for expert,
attention, MLP, router and embedding tensors alike. Group size **64** fails the
same test (3/128), which is what pins the group at 32 rather than leaving it
assumed. Repacking to W4A16 at load gives 15.27 GB.

## 20. Two ways to destroy those weights while "just repacking" them ✅

**(a) `d = amax / 8` is the wrong step.** The textbook Q4_0 rule assumes each
block's largest magnitude sits at level ±8. Many blocks here peak lower. When
they do, the derived step is a fraction of the true one, `round(x/d)` lands
between grid points, and the block is requantized onto a grid that does not
contain its own values:

| step rule | median relative error, `experts.gate_up_proj` |
| :--- | ---: |
| `d = amax/8` | **4.9e-2** |
| `d = amax/m`, m searched over 1..8 | 0 for 78.7% of values |
| ...plus least-squares refinement over the group | **0 for 93.1% of values** |

Nothing raises in the first case. `ports/gemma4/jax_q4_0.py` searches for the
level the peak actually occupies and returns a count of groups it could not
place; the loader raises on any nonzero count.

**(b) Packing after the transpose.** W4A16 packs nibbles along the **last** axis,
and the Q4_0 grid runs along `in`. The loader transposes dense rank-2 weights
`[out, in] -> [in, out]`, so packing a *transposed* weight groups across `out`,
where no grid exists — a real requantization dressed up as a repack. Packing must
happen in the loader, before the transpose, which is why `requantize_q4_0` is a
loader argument rather than a post-processing pass over the parameter tree.

Residual after doing both correctly: 89–93% of values bit-identical, worst case
~1.6 BF16 ULP. That gap is scale precision — Q4_0 carries an fp16 block scale and
this format stores BF16 — not level assignment; refining the step moves zero
levels. Same regime as the shipped `-w4a16-ct` checkpoints.

## 21. 26B config values that differ from the dense sizes ✅

| field | 26B A4B | note |
| --- | ---: | --- |
| `num_hidden_layers` | 30 | 25 sliding / 5 full, full at `i % 6 == 5` |
| `hidden_size` | 2816 | |
| `intermediate_size` | 2112 | the dense MLP, which still exists |
| `num_experts` / `top_k_experts` | 128 / 8 | ~4B active of 26.5B total |
| `moe_intermediate_size` | 704 | per expert |
| `num_attention_heads` | 16 | |
| `num_key_value_heads` | 8 | sliding layers |
| `num_global_key_value_heads` | **2** | 31B uses 4 — do not carry it over |
| `global_head_dim` | 512 | vs `head_dim` 256 |
| `num_kv_shared_layers` | **0** | no KV sharing (§7 does not apply) |
| `hidden_size_per_layer_input` | **0** | no PLE (§8 does not apply) |
| `use_double_wide_mlp` | false | §9 does not apply |
| `attention_k_eq_v` | **true** | as the 31B — the 5 full layers ship no `v_proj` |
| `sliding_window` | 1024 | vs E2B's 512 |

Consequence worth noting: because the full-attention layers carry only 2 KV heads
and the sliding layers window at 1024, KV on this model is unusually cheap —
110 KiB/token at int8 unwindowed, and 0.15 GB at 4K context with `window_kv`. On
the 26B, KV is not the constraint; prefill temporaries are.

## 22. Unresolved ⚠️

- **`store_full_length_kv` behaviour.** The reference marks the last non-shared layer
  of each type as storing full-length KV. Our windowed-KV ring windows every sliding
  layer including the source. Self-consistent (windowed and full-length outputs match
  in `tests/test_windowed_kv.py`), but whether it matches Gemma's intent for shared
  sliding layers is unverified.
- **26B on hardware.** Everything in §15–21 is verified against the reference or
  against the checkpoint bytes on CPU. Nothing is measured on a TPU yet — the
  15.27 GB resident figure is arithmetic, and the prefill ceiling is unknown. See
  `benchmarks/runs/2026-07-31-gemma4-26b-v6e1/REPORT.md`.

## How to check the next one

```python
import transformers.models.gemma4.modeling_gemma4 as m; print(m.__file__)
```

Read it before inferring anything from tensor shapes. Every fix in sections 2–6 came
from that file after hours of guessing produced nothing but plausible-looking garbage.

And reading it is necessary but not sufficient. §16 was found by *diffing against a
running reference model*, not by reading — the wiring it describes is two adjacent
lines that look interchangeable, and every unit test of the parts in isolation
passed while the whole was wrong. For anything with more than one input tensor,
build the parity harness before trusting the read.
