"""Port + verify + benchmark gemma-4-26B-A4B on the E-series JAX engine.

One process, several stages, because the load is by far the expensive part at
this size and running each check as its own script would pay it every time.
Every stage prints as it completes and appends to --json-out, so a flex-start VM
that expires mid-sweep still leaves the earlier stages on disk.

What makes the 26B different from every size ported before it
-------------------------------------------------------------
Two things, and only two — the rest is shared with the 31B and already covered.

1. IT IS SPARSE. 128 experts, top-8 per token, moe_intermediate_size 704. The
   expert bank runs *alongside* the dense MLP on every layer and the two are
   summed; it does not replace it. The router reads the raw post-attention
   residual while the experts read that residual after their own norm, which is
   the single easiest thing to get wrong here and is checked by
   tests/test_moe_parity.py against the HF reference rather than by eye.

2. THERE IS NO ``-w4a16-ct`` CHECKPOINT. Verified against the Hub 2026-07-31:
   E2B, E4B, 12B and 31B all publish one; 26B A4B publishes only
   ``-qat-q4_0-gguf`` and ``-qat-q4_0-unquantized``. GGUF targets llama.cpp, so
   the only usable export is 51.61 GB of BF16 against a v6e-1's 33.55 GB.

   It fits because "unquantized" describes the container, not the values: these
   are QAT weights already sitting on a Q4_0 grid, so they can be repacked to
   W4A16 at load for 14.19 GB. See ports/gemma4/jax_q4_0.py for the measurement
   that establishes that, and for the trap in the textbook Q4_0 scale rule.

Everything else — proportional RoPE on the full-attention layers,
attention_k_eq_v (5 of 30 layers ship no v_proj), global_head_dim 512,
logit softcapping, no PLE, no KV sharing — the 31B already exercised.

    python3.13 ports/gemma4/jax_26b_port.py --stages config,load,parity,sweep
"""

import argparse
import gc
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax
import jax.numpy as jnp

from jax_engine import JaxGemmaEngine, config_from_hf
from ports.gemma4.jax_e_model import (
    MOE_GATHER_MAX_TOKENS,
    Gemma4EModelJAX,
    make_cached_decode_step,
    pad_to_tpu_v6e_bucket,
    prefill_with_kv_cache,
    set_w4a16_impl,
)
from ports.gemma4.jax_q4_0 import quantized_bytes

DEFAULT_MODEL = "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized"
PARITY_PROMPT = "What is the capital of France?"

# Fields worth asserting rather than eyeballing: each is a place where a
# silently-wrong default would still load and still generate, just incorrectly.
EXPECTED_CONFIG = {
    "hidden_size": 2816,
    "intermediate_size": 2112,
    "num_hidden_layers": 30,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "num_global_key_value_heads": 2,
    "head_dim": 256,
    "global_head_dim": 512,
    "sliding_window": 1024,
    "num_kv_shared_layers": 0,
    "hidden_size_per_layer_input": 0,
    "attention_k_eq_v": True,
    "logit_softcapping": 30.0,
    "rope_theta": 10000.0,
    "global_rope_theta": 1000000.0,
    # the MoE block
    "enable_moe_block": True,
    "num_experts": 128,
    "top_k_experts": 8,
    "moe_intermediate_size": 704,
}


def _write(path: str, payload: Dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


# --------------------------------------------------------------------- stages


def stage_config(model_id: str, out: Dict[str, Any]) -> Any:
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(model_id, "config.json")) as fh:
        cfg = config_from_hf(json.load(fh))

    print("\n--- resolved config -------------------------------------------")
    mismatches = []
    for field, expected in EXPECTED_CONFIG.items():
        got = getattr(cfg, field)
        if got != expected:
            mismatches.append(f"{field}: got {got!r}, expected {expected!r}")
        print(f"  {field:30s} {str(got):>12s}   "
              f"{'ok' if got == expected else 'MISMATCH'}")

    sliding = cfg.layer_types.count("sliding_attention")
    full = cfg.layer_types.count("full_attention")
    print(f"  {'layer_types':30s} {sliding:>4d} sliding / {full} full")

    # Projected resident size, so the load stage has something to be checked
    # against rather than merely reported.
    E, MI, H, IM = cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size, cfg.intermediate_size
    per_expert_bank = quantized_bytes((E, 2 * MI, H)) + quantized_bytes((E, H, MI))
    dense_mlp = 2 * quantized_bytes((IM, H)) + quantized_bytes((H, IM))
    attn_sliding = (quantized_bytes((cfg.num_attention_heads * cfg.head_dim, H))
                    + 2 * quantized_bytes((cfg.num_key_value_heads * cfg.head_dim, H))
                    + quantized_bytes((H, cfg.num_attention_heads * cfg.head_dim)))
    attn_full = (quantized_bytes((cfg.num_attention_heads * cfg.global_head_dim, H))
                 + quantized_bytes((cfg.num_global_key_value_heads * cfg.global_head_dim, H))
                 + quantized_bytes((H, cfg.num_attention_heads * cfg.global_head_dim)))
    embed = cfg.vocab_size * H * 2      # left BF16: it is a gather, not a matmul
    projected = (sliding * (per_expert_bank + dense_mlp + attn_sliding)
                 + full * (per_expert_bank + dense_mlp + attn_full) + embed)

    print(f"\n  expert bank / layer          {per_expert_bank / 1e9:8.3f} GB packed")
    print(f"  dense MLP / layer            {dense_mlp / 1e9:8.3f} GB packed")
    print(f"  embed_tokens (BF16, unpacked){embed / 1e9:8.3f} GB")
    print(f"  PROJECTED RESIDENT           {projected / 1e9:8.3f} GB "
          f"(vs 51.61 GB as shipped)")

    out["config"] = {
        "resolved": {f: getattr(cfg, f) for f in EXPECTED_CONFIG},
        "sliding_layers": sliding, "full_layers": full,
        "projected_weight_gb": round(projected / 1e9, 3),
        "mismatches": mismatches,
    }
    if mismatches:
        print("  !! config resolution disagrees with the checkpoint:")
        for m in mismatches:
            print(f"     {m}")
    return cfg


def stage_load(args, out: Dict[str, Any]) -> JaxGemmaEngine:
    print("\n--- load ------------------------------------------------------", flush=True)
    print("  streaming shards through host RAM and packing Q4_0 -> W4A16 per "
          "tensor;\n  the BF16 checkpoint is 51.61 GB and never lands on device "
          "whole.", flush=True)
    eng = JaxGemmaEngine(
        model_id=args.model,
        kv_cache_dtype=args.kv_cache_dtype,
        quant_mode="w4a16",
        max_model_len=args.max_model_len,
        w4a16_impl=args.w4a16_impl,
        requantize_q4_0=True,
        moe_impl=args.moe_impl,
    )
    t0 = time.perf_counter()
    eng.load()
    t_load = time.perf_counter() - t0

    st = eng.requantize_stats
    mem = eng.memory_stats()
    limit = mem["hbm_bytes_limit"] or 33.55e9
    print(f"  load time            {t_load:8.1f} s")
    if st:
        ratio = st["bytes_dense"] / max(st["bytes_packed"], 1)
        print(f"  tensors packed       {st['packed']:8d}")
        print(f"  unrepresentable grps {st['failed_groups']:8d}   "
              f"(must be 0; the loader raises otherwise)")
        print(f"  packed               {st['bytes_dense'] / 1e9:8.2f} GB -> "
              f"{st['bytes_packed'] / 1e9:.2f} GB  ({ratio:.2f}x)")
    print(f"  weight bytes         {mem['weight_bytes'] / 1e9:8.2f} GB")
    print(f"  HBM in use           {mem['hbm_bytes_in_use'] / 1e9:8.2f} GB")
    print(f"  HBM limit            {limit / 1e9:8.2f} GB")
    print(f"  headroom             {(limit - mem['hbm_bytes_in_use']) / 1e9:8.2f} GB "
          f"({100.0 * (limit - mem['hbm_bytes_in_use']) / limit:.1f}%)")

    projected = out.get("config", {}).get("projected_weight_gb")
    if projected:
        delta = mem["weight_bytes"] / 1e9 - projected
        print(f"  vs projection        {delta:+8.3f} GB")

    out["load"] = {
        "load_s": round(t_load, 1),
        "weight_gb": round(mem["weight_bytes"] / 1e9, 3),
        "hbm_in_use_gb": round(mem["hbm_bytes_in_use"] / 1e9, 3),
        "hbm_limit_gb": round(limit / 1e9, 3),
        "headroom_gb": round((limit - mem["hbm_bytes_in_use"]) / 1e9, 3),
        "requantize": st,
        "w4a16_impl": args.w4a16_impl,
        "moe_impl": args.moe_impl,
    }
    return eng


def stage_parity(eng, args, out: Dict[str, Any]) -> None:
    """Greedy-decode the chat-templated parity prompt.

    The 12B port established that this family's IT QAT checkpoints emit digit
    strings for a bare prompt on BOTH the HF PyTorch reference and this engine —
    the chat template plus <bos> is a correctness requirement, not a nicety. So
    the prompt goes through apply_chat_template, with an unformatted control
    alongside: matching digits on the control is expected, and is what tells us a
    wrong answer here would be formatting rather than engine math.
    """
    from transformers import AutoTokenizer

    print("\n--- parity ----------------------------------------------------", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    eng.bos_token_id = tok.bos_token_id

    formatted = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    print(f"  formatted prompt: {formatted!r}")

    eos_ids = {i for i in (tok.eos_token_id,) if i is not None}
    for name in ("<turn|>", "<end_of_turn>", "<eos>"):
        tid = tok.convert_tokens_to_ids(name)
        if tid is not None and tid != tok.unk_token_id:
            eos_ids.add(tid)
    print(f"  stop tokens: {sorted(eos_ids)}")

    results = {}
    for label, text in (("templated", formatted), ("bare", args.prompt)):
        ids = tok(text, add_special_tokens=False,
                  return_tensors="np")["input_ids"][0].tolist()
        t0 = time.perf_counter()
        toks, stats = eng.generate(ids, max_new_tokens=args.parity_tokens,
                                   temperature=0.0, eos_token_ids=sorted(eos_ids))
        wall = time.perf_counter() - t0
        gen = tok.decode(toks)
        print(f"  [{label:9s}] {len(ids):4d} prompt tok -> {gen!r}")
        print(f"  {'':11s} {wall:.1f}s wall (incl. compile), "
              f"prefill {getattr(stats, 'prefill_ms', float('nan')):.1f} ms, "
              f"decode {stats.decode_tok_per_s:.1f} tok/s")
        results[label] = {
            "prompt_tokens": len(ids),
            "output": gen,
            "finish_reason": getattr(stats, "finish_reason", None),
            "prefill_ms": round(getattr(stats, "prefill_ms", 0.0), 1),
            "decode_tok_s": round(stats.decode_tok_per_s, 2),
            "wall_s": round(wall, 1),
        }

    out["parity"] = results


def stage_moe(eng, args, out: Dict[str, Any]) -> None:
    """Measure the gather/dense crossover instead of trusting the arithmetic.

    ``moe_block_jax`` picks gather below MOE_GATHER_MAX_TOKENS tokens and dense
    above, on the reasoning that gathering reads T*K expert banks while the dense
    path reads all E once, so they break even at T*K == E (T=16 here). That is a
    statement about bytes moved, and this engine has been wrong before about
    bytes moved predicting time — see the fused W4A16 Pallas kernel, which was a
    0.59x "optimization" because decode was not bandwidth-bound.

    So: time both paths across token counts and record where they actually
    cross. If it is nowhere near 16, MOE_GATHER_MAX_TOKENS is wrong.
    """
    print("\n--- moe path crossover ----------------------------------------", flush=True)
    print(f"  threshold in code: T <= {MOE_GATHER_MAX_TOKENS} uses gather")
    rows = []
    for T in [int(x) for x in args.moe_tokens.split(",")]:
        row: Dict[str, Any] = {"tokens": T}
        for impl in ("gather", "dense"):
            model = Gemma4EModelJAX(eng.config, moe_impl=impl)
            ids = jnp.ones((1, T), dtype=jnp.int32)
            padded, valid = pad_to_tpu_v6e_bucket(ids)
            fn = jax.jit(
                prefill_with_kv_cache,
                static_argnames=("model", "max_new_tokens", "quant_mode",
                                 "cache_dtype", "window_kv"))

            def run():
                return fn(model=model, prompt_ids=padded, prompt_valid=valid,
                          params=eng.params, max_new_tokens=1,
                          quant_mode="w4a16", window_kv=True)

            try:
                row[impl] = round(_time_median_ms(run, repeats=args.repeats), 2)
            except Exception as exc:
                row[impl] = None
                row[f"{impl}_error"] = str(exc).split("\n")[0][:70]
            gc.collect()
        g, d = row.get("gather"), row.get("dense")
        winner = "-" if (g is None or d is None) else ("gather" if g < d else "dense")
        row["faster"] = winner
        print(f"  T={T:5d}  gather {str(g):>9s} ms | dense {str(d):>9s} ms "
              f"-> {winner}")
        rows.append(row)
    out["moe_crossover"] = rows


def _time_median_ms(fn, repeats: int = 3, warmup: int = 1) -> float:
    for _ in range(warmup):
        jax.block_until_ready(fn())
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def _bench_cell(model, params, B: int, S: int, decode_steps: int,
                repeats: int) -> Dict[str, Any]:
    raw_ids = jnp.ones((B, S), dtype=jnp.int32)
    padded_ids, valid_mask = pad_to_tpu_v6e_bucket(raw_ids)
    bucket_s = int(padded_ids.shape[1])

    jit_prefill = jax.jit(
        prefill_with_kv_cache,
        static_argnames=("model", "max_new_tokens", "quant_mode", "cache_dtype",
                         "window_kv"),
    )

    def run_prefill():
        return jit_prefill(
            model=model, prompt_ids=padded_ids, prompt_valid=valid_mask,
            params=params, max_new_tokens=decode_steps, quant_mode="w4a16",
            window_kv=True,
        )

    prefill_ms = _time_median_ms(run_prefill, repeats=repeats)

    last_logits, caches, valid = jax.block_until_ready(run_prefill())
    # No donation here: run_step is called repeatedly with the same caches, which
    # a donated buffer would invalidate after the first call.
    step = jax.jit(make_cached_decode_step(model, quant_mode="w4a16",
                                           window_kv=True))
    prompt_lens = valid_mask.sum(axis=1).astype(jnp.int32)
    tok = jnp.argmax(last_logits, axis=-1, keepdims=True)

    def run_step():
        return step(params, caches, valid, tok, prompt_lens, jnp.int32(bucket_s))

    step_ms = _time_median_ms(run_step, repeats=repeats)

    del last_logits, caches, valid
    gc.collect()

    return {
        "B": B, "S": S, "bucket_S": bucket_s,
        "prefill_ms": round(prefill_ms, 2),
        "decode_step_ms": round(step_ms, 3),
        "agg_decode_tok_s": round(B * 1000.0 / step_ms, 1),
        "per_user_tok_s": round(1000.0 / step_ms, 1),
        "status": "OK",
    }


def stage_sweep(eng, args, out: Dict[str, Any], json_out: str) -> None:
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    contexts = [int(x) for x in args.contexts.split(",")]

    print("\n--- sweep -----------------------------------------------------", flush=True)
    results: List[Dict[str, Any]] = []
    for B in batch_sizes:
        for S in contexts:
            label = f"{S // 1024}K" if S >= 1024 else str(S)
            print(f"  B={B:2d} | ctx={label:>5s} ... ", end="", flush=True)
            try:
                cell = _bench_cell(eng.model, eng.params, B, S,
                                   decode_steps=args.decode_steps,
                                   repeats=args.repeats)
                print(f"prefill {cell['prefill_ms']:9.1f} ms | "
                      f"step {cell['decode_step_ms']:8.3f} ms | "
                      f"agg {cell['agg_decode_tok_s']:7.1f} tok/s | "
                      f"per-user {cell['per_user_tok_s']:6.1f} tok/s")
            except Exception as exc:
                msg = str(exc).split("\n")[0][:70]
                print(f"FAILED: {msg}")
                cell = {"B": B, "S": S, "status": "FAILED", "error": msg}
                gc.collect()
            results.append(cell)
            out["sweep"] = results
            _write(json_out, out)     # checkpoint after every cell

    print("\n| Users (B) | Context (S) | Prefill (TTFT) | Decode Step | Aggregate | Per-User |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        label = f"{r['S'] // 1024}K" if r["S"] >= 1024 else str(r["S"])
        if r["status"] == "OK":
            print(f"| {r['B']} | {label} | {r['prefill_ms']:.1f} ms | "
                  f"{r['decode_step_ms']:.2f} ms | "
                  f"{r['agg_decode_tok_s']:.1f} tok/s | "
                  f"{r['per_user_tok_s']:.1f} tok/s |")
        else:
            print(f"| {r['B']} | {label} | {r['status']} | — | — | — |")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--stages", default="config,load,parity,moe,sweep",
                    help="comma-separated: config,load,parity,moe,sweep")
    ap.add_argument("--json-out", default=os.path.join(
        _REPO_ROOT, "benchmarks/runs/2026-07-31-gemma4-26b-v6e1/results/"
                    "gemma4_26b_port.json"))
    ap.add_argument("--kv-cache-dtype", default="int8")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--w4a16-impl", default="reference")
    ap.add_argument("--moe-impl", default="auto",
                    choices=["auto", "gather", "dense"])
    ap.add_argument("--prompt", default=PARITY_PROMPT)
    ap.add_argument("--parity-tokens", type=int, default=48)
    ap.add_argument("--moe-tokens", default="1,4,8,16,32,64,256")
    ap.add_argument("--batch-sizes", default="1,4")
    ap.add_argument("--contexts", default="512,1024,2048")
    ap.add_argument("--decode-steps", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    out: Dict[str, Any] = {
        "model": args.model,
        "device": str(jax.devices()[0]),
        "stages": stages,
    }
    print(f"gemma-4-26B-A4B port | {args.model}")
    print(f"device: {jax.devices()}")

    set_w4a16_impl(args.w4a16_impl)

    if "config" in stages:
        stage_config(args.model, out)
        _write(args.json_out, out)

    eng = None
    if {"load", "parity", "moe", "sweep"} & set(stages):
        eng = stage_load(args, out)
        _write(args.json_out, out)

    if "parity" in stages and eng is not None:
        stage_parity(eng, args, out)
        _write(args.json_out, out)

    if "moe" in stages and eng is not None:
        stage_moe(eng, args, out)
        _write(args.json_out, out)

    if "sweep" in stages and eng is not None:
        stage_sweep(eng, args, out, args.json_out)
        _write(args.json_out, out)

    print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
