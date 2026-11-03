> **Moved.** This repo now lives at [xbill9/jax-dev/tpu-jax-v6e1-26b-q4_0](https://github.com/xbill9/jax-dev/tree/main/tpu-jax-v6e1-26b-q4_0), with its full history. This copy is archived and read-only.

# Gemma 4 E2B QAT on a Single TPU with Pure JAX

This repository is an experimental, inspectable inference path for
`google/gemma-4-E2B-it-qat-w4a16-ct` on one Cloud TPU v6e-1. It loads the QAT
checkpoint directly from safetensors, executes Gemma 4 in JAX without PyTorch,
and exposes an OpenAI-compatible HTTP/SSE server.

The project began with a practical incompatibility: the tested vLLM TPU stack
could not load this QAT export. Building the missing path uncovered a broader
measurement story about KV-cache accounting, JAX buffer donation, quantization,
static-shape batching, and the gap between a fast kernel and a useful server.

## What is verified

- The QAT checkpoint loads without PyTorch in the path.
- Cached decode matches full-sequence re-forward within float32 tolerance.
- Padding to 128-aligned TPU buckets does not change model output.
- INT8 KV attention matches a dequantize-first reference.
- Chunked prefill is token-exact against one-shot prefill.
- Greedy SSE and non-streaming HTTP responses produce identical text.
- CPU tests use a tiny synthetic checkpoint; TPU claims were measured on
  `ct6e-standard-1t` with 32 GB HBM3.

## Corrected v6e-1 results

The final 2026-07-29 revalidation used checkpoint-shaped static programs,
isolated processes, warmup, and 15 timed samples.

| Finding | Measured result |
| :--- | :--- |
| Buffer donation | **1.60–1.62×** faster with BF16 KV |
| INT8 KV | **1.17–1.19×** faster than donated BF16 |
| INT8 KV capacity | **1.82–1.98×** donated BF16 capacity |
| INT4 PLE | Parameter tree **53% smaller**; no meaningful throughput gain |
| Best static decode kernel | **2,888 aggregate tok/s**, B=32, context 8,192 |
| Real-checkpoint HTTP server | Approximately **139–141 aggregate tok/s** |

The largest optimization was buffer donation. Without `donate_argnums`, a
single-token `dynamic_update_slice` could leave two full KV caches live during
decode. Donation removed that copy, increased throughput, and nearly doubled
the resident-token ceiling.

INT8 KV then reduced bandwidth and approximately doubled capacity. Its real
checkpoint quality check measured 28.41 versus 28.73 perplexity for BF16, with
97.08% greedy-token agreement over 583 teacher-forced steps. Error did not grow
through a separate 968-step continuous decode.

Read the correction history before quoting an absolute capacity number:
[`benchmarks/runs/2026-07-29-kv-quant-v6e1/REPORT.md`](benchmarks/runs/2026-07-29-kv-quant-v6e1/REPORT.md).
Earlier results were withdrawn after finding an artificial power-of-two capacity
invariant, an undonated cache copy, and an incorrect roofline accounting of the
PLE gather.

## Kernel speed is not serving speed

The 2,888 tok/s point is a static-shape decode-kernel measurement using
architecture-shaped synthetic parameter values. Static values do not change the
compiled work, but this is not an HTTP benchmark or a direct vLLM comparison.

The real checkpoint served through `jax_openai_server.py` measured:

| Prompt tokens | Prefill | Decode |
| ---: | ---: | ---: |
| 506 | 9.3 ms | 141.1 tok/s |
| 2,045 | 32.0 ms | 140.5 tok/s |
| 7,679 | 318.6 ms | 138.5 tok/s |

At HTTP concurrency 2/4/8, aggregate throughput remained
128.7/133.6/143.3 tok/s while median request latency rose
497/952/1,775 ms. Every request succeeded, but none formed a device batch.

The current server runs independent B=1 executions. Reaching the batched-kernel
rate requires a request batcher, batched KV ownership, continuous admission, and
prefix reuse. Until then, this is a validated experimental engine—not a vLLM
replacement. Full results:
[`benchmarks/runs/2026-07-29-real-http-v6e1/REPORT.md`](benchmarks/runs/2026-07-29-real-http-v6e1/REPORT.md).

## Repository map

| Path | Purpose |
| :--- | :--- |
| `ports/gemma4/jax_e_model.py` | Gemma 4 E2B forward, cached decode, quantized KV, and Pallas experiments |
| `ports/gemma4/jax_e_loader.py` | Torch-free safetensors/QAT loader |
| `ports/gemma4/jax_e_benchmark_sweep_v2.py` | Corrected prefill and cached-decode benchmark |
| `jax_engine.py` | Stateful generation engine |
| `jax_openai_server.py` | OpenAI-compatible completions, chat, SSE, health, and metrics |
| `tests/` | CPU correctness and API regression suite |
| `benchmarks/runs/` | Raw logs, JSON, scripts, reports, and correction notes |
| `benchmarks/queued/` | Hardware profiling and queued benchmark utilities |

## Run the engine

Use a TPU VM with a current `jax[tpu]` installation and authenticated access to
the gated Gemma checkpoint. Install the HTTP and checkpoint dependencies, then:

```bash
python3 -m pip install fastapi uvicorn pydantic huggingface_hub safetensors \
  flax transformers sentencepiece 'jinja2>=3.1'

python3 jax_openai_server.py \
  --model google/gemma-4-E2B-it-qat-w4a16-ct \
  --kv-cache-dtype int8 \
  --quant-mode w4a16 \
  --max-model-len 8192 \
  --port 8000
```

To serve the sparse **26B A4B** on the same single chip, point `--model` at the
Q4_0 checkpoint — it is the one Gemma 4 size with no `-w4a16-ct` release, so the
server packs its BF16 weights to W4A16 during load (51.61 GB → 15.25 GB). That is
inferred from the model id; no extra flag needed:

```bash
python3 jax_openai_server.py \
  --model google/gemma-4-26B-A4B-it-qat-q4_0-unquantized \
  --kv-cache-dtype int8 --max-model-len 8192 --port 8000
```

See `models.md` for the checkpoint matrix and the two caveats that come with it.

Query the streaming endpoint:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "google/gemma-4-E2B-it-qat-w4a16-ct",
    "messages": [{"role": "user", "content": "Explain TPU buffer donation."}],
    "stream": true
  }'
```

Run the corrected kernel sweep:

```bash
python3 ports/gemma4/jax_e_benchmark_sweep_v2.py \
  --batch-sizes 1,2,4,8,16,32,64 \
  --contexts 8,128,512,2048 \
  --json-out results.json
```

Run CPU correctness tests:

```bash
python3 -m unittest discover -s tests
```

CPU is suitable for numerical correctness, scheduling, endpoint, and SSE tests.
It cannot validate TPU throughput, HBM capacity, compilation timing, or
Pallas/Mosaic performance.

## Current limitations

- No continuous batching or prefix cache.
- Static `(batch, sequence)` shapes can trigger first-touch compilation.
- Chunked prefill improves admission but does not yet compose with ring-buffer
  windowing.
- Greedy decoding is implemented; production grammar-constrained tool output is
  not.
- The Pallas W4A16 experiment regressed performance and is not the recommended
  path.
- Results describe this implementation and workload, not the v6e hardware limit.

## Supporting TPU infrastructure

The repository also retains the `tpu-management` skill and `tpu-devops` MCP
server used to provision flex-start TPU VMs, verify JAX devices, inspect logs,
run vLLM baselines, and clean up capacity:

```bash
./project-setup.sh /path/to/project --project <gcp-project-id>
make skill
make skill-install
```

See [SKILL.md](.claude/skills/tpu-management/SKILL.md) for the infrastructure
tool catalog. Root sources (`server.py`, `project-setup.sh`, and `tpu.md`) remain
authoritative; generated skill snapshots are refreshed with `make skill`.

## Results and writing

- [Corrected TPU report](benchmarks/runs/2026-07-29-kv-quant-v6e1/REPORT.md)
- [Real-weight HTTP report](benchmarks/runs/2026-07-29-real-http-v6e1/REPORT.md)
- [vLLM baseline](benchmarks/reports/2026-07-21-gemma4-e2b-v6e1.json)
- [Long-form article draft](devto-jax-gemma4-e2b.md)
- [Hugging Face benchmark dataset](https://huggingface.co/datasets/xbill9/gemma4-e2b-tpu-v6e-benchmarks)

The Hugging Face dataset is currently private. It contains reports, raw JSON,
benchmark scripts, and the article, but no model weights or credentials.

## Security

Never commit Hugging Face tokens, GCP credentials, model caches, or generated
server logs. Use scoped credentials and Google Secret Manager for remote TPU
deployments. The upstream Gemma checkpoint remains governed by its own access
and license terms.
