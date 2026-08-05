# 🚀 Deploying Gemma 4 E2B QAT on Cloud TPU v6e-1 using JAX

This document outlines the complete deployment architecture, precision specifications, memory layouts, API endpoints, auto-start `systemd` configuration, and reusable GCE image staging for running **Gemma 4 E2B QAT** models on **Google Cloud TPU v6e-1** via **JAX**.

---

## 📌 Overview & Key Highlights

- **Hardware Target**: Cloud TPU v6e single-chip VM (`ct6e-standard-1t`, 32 GB HBM3).
- **Backend Stack**: Pure JAX `0.11.0`, `libtpu-0.0.44`, `flax`, `compressed-tensors 0.17.1`, `google-deepmind/gemma 4.1.0`.
- **Model Checkpoints Verified**:
  1. `google/gemma-4-E2B-it-qat-q4_0-unquantized` (10.1 tok/s)
  2. `google/gemma-4-E2B-it-qat-w4a16-ct` (8.1 tok/s)
- **API Server**: Pure JAX FastAPI + Uvicorn server exposing OpenAI-compatible endpoints + Prometheus `/metrics`.
- **Auto-Start & Staging**: `systemd` daemon enabled + GCE Image `gemma4-jax-v6e1-image` registered for instant 15-second boot deployments.
- **Model Reference Guide**: See [models.md](models.md) for the complete reference of all model checkpoints, flags, and quantization options.

---

## 🎛️ Model & CLI Server Flags

For a detailed walkthrough of all model flags and checkpoint options, refer to [models.md](models.md).

The JAX TPU server [`jax_openai_server.py`](jax_openai_server.py) and CLI runner [`jax_gemma4_e2b.py`](jax_gemma4_e2b.py) support the following command-line flags and parameters:

### 1. Model & Precision Flags

| Flag | Type | Default Value | Valid Options / Examples | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`--model`** | `str` | `google/gemma-4-E2B-it-qat-w4a16-ct` | `google/gemma-4-E2B-it-qat-w4a16-ct`<br>`google/gemma-4-E2B-it-qat-q4_0-unquantized`<br>`google/gemma-4-E2B-it` | Specifies the Hugging Face model checkpoint repository or path to load onto the TPU. |
| **`--kv-cache-dtype`** | `str` | `fp8` | `fp8` (`fp8_e4m3fn`), `bfloat16`, `int8` | Sets the precision for Key-Value attention cache memory in TPU HBM. `fp8` reduces KV memory by 50%. |
| **`--torch-dtype`** / **`--dtype`** | `str` | `bfloat16` | `bfloat16`, `float32` | Data type for intermediate activations and MXU matrix multiplications. `bfloat16` is native on TPU. |
| **`--max-new-tokens`** | `int` | `128` | Positive integers (e.g. `128`, `512`, `2048`) | Maximum number of new tokens generated per completion request. |
| **`--temperature`** | `float` | `0.7` | `0.0` (greedy) to `1.0` | Controls sampling randomness. `0.0` uses deterministic greedy decoding. |

### 2. Network & Server Flags

| Flag | Type | Default Value | Example Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`--host`** | `str` | `0.0.0.0` | `0.0.0.0`, `127.0.0.1` | Network interface address for the FastAPI Uvicorn web server. |
| **`--port`** | `int` | `8000` | `8000`, `8080` | HTTP port on which the OpenAI REST API & `/metrics` endpoints listen. |
| **`--prompt`** | `str` | *"Explain why TPUs excel..."* | `"Any text prompt"` | Input prompt for single-shot CLI testing via `jax_gemma4_e2b.py`. |

---

## 🎯 Model Precision & Quantization Architecture

| Component | Precision / Scheme | Description |
| :--- | :--- | :--- |
| **Model Weights (`W4`)** | **INT4 (4-bit integer)** | QAT compressed-tensors with `group_size=32` (~3.52 GB compressed disk size) |
| **Activations (`A16`)** | **BF16 (`bfloat16`)** | 16-bit floating point for full precision matrix multiplies on TPU MXU |
| **KV Cache (`KV8`)** | **FP8 (`fp8_e4m3fn`)** | 8-bit floating point Key-Value cache (50% memory saving) |

---

## 🧠 TPU v6e-1 HBM Memory Breakdown (32 GB HBM3)

Queries executed via `jax.devices()[0].memory_stats()` on TPU v6e-1 (`bytes_limit: 33,546,042,880 bytes`):

```
+-------------------------------------------------------------------------+
|                  TPU v6e-1 HBM Total Capacity: 31.24 GiB                |
+-------------------------------------------------------------------------+
| [1] Model Weights (bfloat16):           10.21 GB  (31.2% of HBM)        |
| [2] KV-Cache Allocation (128K ctx):      2.40 GB   (7.4% of HBM)        |
| [3] MXU Activation & XLA Buffers:        1.50 GB   (4.6% of HBM)        |
| [4] Libtpu & XLA Runtime Reserves:      2.00 GB   (6.1% of HBM)        |
| [5] Available / Unallocated HBM:        15.13 GB  (50.7% Free HBM)      |
+-------------------------------------------------------------------------+
```

---

## 🌐 Server Endpoints & Usage (`jax_openai_server.py`)

The server script [`jax_openai_server.py`](jax_openai_server.py) runs on port `8000`.

### Exposed REST Endpoints

- **`GET /health`**: Health & precision status check.
- **`GET /metrics`**: Prometheus formatted metrics (request count, token count, tok/s, HBM bytes used).
- **`GET /v1/models`**: List active served models.
- **`POST /v1/chat/completions`**: OpenAI Chat Completions API.
- **`POST /v1/completions`**: OpenAI Text Completions API.

### Sample cURL Request

```bash
curl http://34.70.83.5:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-E2B-it-qat-w4a16-ct",
    "messages": [
      {"role": "user", "content": "Explain why TPUs excel at JAX workloads."}
    ],
    "max_tokens": 64
  }'
```

### Sample Prometheus Metrics Output (`/metrics`)

```prometheus
# HELP tpu_jax_requests_total Total HTTP requests processed by JAX TPU server
# TYPE tpu_jax_requests_total counter
tpu_jax_requests_total{model="google/gemma-4-E2B-it-qat-w4a16-ct",status="success"} 1

# HELP tpu_jax_tokens_per_second Current generation throughput in tokens per second
# TYPE tpu_jax_tokens_per_second gauge
tpu_jax_tokens_per_second{model="google/gemma-4-E2B-it-qat-w4a16-ct"} 8.1

# HELP tpu_jax_hbm_used_bytes High Bandwidth Memory used in bytes
# TYPE tpu_jax_hbm_used_bytes gauge
tpu_jax_hbm_used_bytes{device="TPU_0(process=0,(0,0,0,0))"} 2765312
```

---

## ⚙️ Systemd Auto-Start Configuration

The server runs under `systemd` to ensure automatic startup whenever the VM boots up.

Service file at `/etc/systemd/system/jax-openai.service`:

```ini
[Unit]
Description=JAX Gemma 4 OpenAI API Server
After=network.target

[Service]
Type=simple
User=xbill
WorkingDirectory=/home/xbill
ExecStart=/home/xbill/venv_jax312/bin/python3 /home/xbill/jax_openai_server.py --model google/gemma-4-E2B-it-qat-w4a16-ct --kv-cache-dtype fp8 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### Managing the Service
```bash
sudo systemctl status jax-openai
sudo systemctl restart jax-openai
sudo systemctl stop jax-openai
```

---

## 📦 Instant Boot via Reusable GCE Image

A reusable GCE image containing the complete environment and cached model weights has been created:

- **Snapshot**: `gemma4-jax-v6e1-snapshot`
- **GCE Image Name**: `gemma4-jax-v6e1-image`
- **GCP Project**: `aisprint-491218`

### Booting a New TPU VM in ~15 Seconds
To launch a new TPU VM pre-loaded with this server in any zone with TPU quota:

```bash
gcloud compute instances create my-tpu-vm \
    --project=aisprint-491218 \
    --zone=europe-west4-a \
    --machine-type=ct6e-standard-1t \
    --provisioning-model=FLEX_START \
    --image=gemma4-jax-v6e1-image
```
