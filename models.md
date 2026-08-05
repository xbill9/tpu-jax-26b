# 🎛️ Gemma 4 Model Flags & Configuration Guide (`models.md`)

This guide provides a comprehensive reference for all model flags, CLI arguments, data types, quantization schemes, and supported checkpoints when deploying **Gemma 4** on **Google Cloud TPUs** using **JAX**.

---

## 📌 Supported Gemma 4 Model Checkpoints

| Model ID | Effective Parameters | Quantization Scheme | Format | Target TPU Hardware | Recommended Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`google/gemma-4-E2B-it-qat-w4a16-ct`** | **2.3B** (5.1B total) | QAT W4A16 | `compressed-tensors` | TPU v6e-1 (32 GB HBM) | High-throughput, lowest disk/load footprint (**Default**) |
| **`google/gemma-4-E2B-it-qat-q4_0-unquantized`** | **2.3B** (5.1B total) | QAT Q4_0 Baseline | Unquantized `bfloat16` | TPU v6e-1 (32 GB HBM) | Maximum QAT generation quality |
| **`google/gemma-4-E2B-it-qat-mobile-transformers`** | **2.3B** (5.1B total) | QAT Mixed (W4/W2, A8, KV8) | `transformers` | Edge / Mobile / TPU | Low-power mobile & edge deployment |
| **`google/gemma-4-E2B-it`** | **2.3B** (5.1B total) | Unquantized | Full `bfloat16` | TPU v6e-1 (32 GB HBM) | Baseline non-QAT model |
| **`google/gemma-4-E4B-it`** | **4.5B** (8.0B total) | Unquantized / QAT | `bfloat16` / `W4A16` | TPU v6e-1 / v6e-4 | Mid-sized multimodal serving |
| **`google/gemma-4-26B-A4B-it-qat-q4_0-unquantized`** | **26.5B total / 4B active** | QAT Q4_0, repacked to W4A16 at load | Unquantized `bfloat16` (51.61 GB as shipped) | TPU v6e-1 (32 GB HBM) | **Sparse MoE** — 128 experts, top-8. 15.25 GB resident after load-time packing. **No `-w4a16-ct` exists for this size** — see the note below |
| **`google/gemma-4-31B-it-qat-w4a16-ct`** | **31.0B** | QAT W4A16 | `compressed-tensors` | TPU v6e-1 (32 GB HBM) | Largest **dense** checkpoint that fits **one** v6e chip — 19.30 GB resident, 14.25 GB free (measured 2026-07-31) |
| **`google/gemma-4-31B-it-qat-q4_0-unquantized`** | **31.0B** | QAT Q4_0 Baseline | Unquantized `bfloat16` | TPU v6e-4 / v6e-8 | 62.6 GB of weights — **will not fit a single chip**; needs sharding the JAX engine does not yet do |
| **`google/gemma-4-31B-it`** | **31.0B** | Unquantized | `bfloat16` | TPU v6e-4 / v6e-8 | 62.6 GB; same multi-chip constraint as above |

### ⚠️ The 26B A4B has no `compressed-tensors` checkpoint

Verified against the Hub on 2026-07-31. `-qat-w4a16-ct` exists for **E2B, E4B,
12B and 31B** — and for **no other size**. The 26B A4B publishes exactly two QAT
variants, `-qat-q4_0-gguf` (llama.cpp) and `-qat-q4_0-unquantized`. There is no
`google/gemma-4-26B-A4B-it-qat-w4a16-ct` to load; requesting one 404s.

So the 26B path is: load the `-q4_0-unquantized` export and pack it to W4A16
during load. "Unquantized" describes the container, not the values — these are
QAT weights already on a Q4_0 grid, so repacking recovers that grid rather than
imposing a new one. 51.61 GB → 15.25 GB, which fits a v6e-1 with ~18 GB to
spare. It is enabled automatically for any model id containing
`q4_0-unquantized`; see `ports/gemma4/jax_q4_0.py` for the measurement behind
the "already quantized" claim and the trap in the textbook Q4_0 scale rule.

Two consequences worth knowing before serving it:

* **Weights are not bit-identical to the BF16 checkpoint.** 89–93% of values
  reconstruct exactly; the worst case is ~1.6 BF16 ULP, from storing the block
  scale in BF16 where Q4_0 used fp16. Same precision regime as the shipped
  `-w4a16-ct` checkpoints.
* **Prefill does 16× the necessary expert FLOPs.** The dense expert path
  evaluates all 128 experts per token when only 8 contribute. Decode is
  unaffected — it gathers just the selected 8. See `moe_experts_dense`.

---

## 🎛️ Complete Model & CLI Flags Reference

### 1. Model Selection & Precision Flags

| Flag | Type | Default Value | Valid Options | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`--model`** | `str` | `google/gemma-4-E2B-it-qat-w4a16-ct` | Any HF Repo ID or local path | Specifies the model checkpoint to load into JAX TPU memory. |
| **`--kv-cache-dtype`** | `str` | `fp8` | `fp8` (`fp8_e4m3fn`), `bfloat16`, `int8` | Data type for Key-Value attention cache in TPU HBM. `fp8` cuts KV memory usage by 50%. |
| **`--torch-dtype`** / **`--dtype`** | `str` | `bfloat16` | `bfloat16`, `float32` | Precision for intermediate activations and MXU matrix multiplications. |
| **`--max-new-tokens`** | `int` | `128` | Positive integers (`1` to `8192`) | Maximum number of new tokens generated per inference request. |
| **`--temperature`** | `float` | `0.7` | `0.0` to `2.0` | Sampling temperature. `0.0` enables deterministic greedy decoding. |
| **`--top-p`** | `float` | `0.95` | `0.0` to `1.0` | Nucleus sampling probability threshold. |
| **`--top-k`** | `int` | `50` | Positive integers (e.g. `40`, `50`) | Top-K sampling token candidate pool size. |
| **`--requantize-q4-0`** | `flag` | auto | on / `--no-requantize-q4-0` | Pack BF16 Q4_0-QAT weights to W4A16 during load. Auto-enabled for `-qat-q4_0-unquantized` model ids; required for the 26B A4B to fit a v6e-1. |
| **`--moe-impl`** | `str` | `auto` | `auto`, `gather`, `dense` | Expert dispatch for MoE checkpoints. `auto` gathers the selected experts at ≤16 tokens (decode) and runs the full bank above that (prefill). Pin only for A/B testing. |

---

### 2. Server & Network Execution Flags

| Flag | Type | Default Value | Example Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`--host`** | `str` | `0.0.0.0` | `0.0.0.0`, `127.0.0.1` | Network interface IP address for the FastAPI / Uvicorn server. |
| **`--port`** | `int` | `8000` | `8000`, `8080` | HTTP port for OpenAI API (`/v1/chat/completions`) & Prometheus (`/metrics`). |
| **`--prompt`** | `str` | *"Explain why TPUs excel..."* | `"Custom user prompt"` | Input prompt string for single-shot testing with [`jax_gemma4_e2b.py`](jax_gemma4_e2b.py). |

---

## 🔬 Quantization Precision Matrix

```
+---------------------------------------------------------------------------------------+
|                                    Gemma 4 QAT Model                                  |
+-------------------+-------------------+-------------------+---------------------------+
| Layer Component   | Storage Precision | Execution Dtype   | Scaling & Metadata        |
+-------------------+-------------------+-------------------+---------------------------+
| Model Weights     | INT4 (4-bit)      | bfloat16          | group_size = 32           |
| Layer Scale/Zero  | bfloat16          | bfloat16          | Per-channel / per-group   |
| Layer Biases      | bfloat16          | bfloat16          | Unquantized float16       |
| Activations       | bfloat16          | bfloat16          | MXU native execution      |
| KV Cache          | FP8 (e4m3fn)      | FP8 / bfloat16    | 50% HBM reduction         |
+-------------------+-------------------+-------------------+---------------------------+
```

---

## 🚀 Execution Examples

### 1. Launch OpenAI-Compatible API Server with W4A16 QAT & FP8 KV Cache
```bash
python3 jax_openai_server.py \
  --model google/gemma-4-E2B-it-qat-w4a16-ct \
  --kv-cache-dtype fp8 \
  --host 0.0.0.0 \
  --port 8000
```

### 2. Run CLI Benchmark Test
```bash
python3 jax_gemma4_e2b.py \
  --model google/gemma-4-E2B-it-qat-w4a16-ct \
  --prompt "Explain why TPUs excel at JAX workloads in two sentences." \
  --max-new-tokens 128
```

### 3. Serve the 26B A4B (sparse MoE) on a single v6e-1
```bash
# --requantize-q4-0 is inferred from the model id; shown here for clarity.
python3 jax_openai_server.py \
  --model google/gemma-4-26B-A4B-it-qat-q4_0-unquantized \
  --requantize-q4-0 \
  --kv-cache-dtype int8 \
  --host 0.0.0.0 --port 8000
```

### 4. Port / verify / benchmark the 26B end to end
```bash
python3 ports/gemma4/jax_26b_port.py --stages config,load,parity,moe,sweep
```
`config` runs anywhere — it only reads `config.json` and prints the projected
resident size. The rest needs a TPU.
