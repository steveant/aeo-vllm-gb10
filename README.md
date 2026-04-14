# aeo-vllm-gb10

Deploy and serve large language models on NVIDIA GB10 and GB200 (Blackwell, unified memory).

## What This Is

A vLLM deployment orchestrator with a CLI (`bootstrap-vllm`), pre-calibrated model profiles, and a calibration lab with reusable load-test tooling. Each profile bundles the exact engine flags, quantization settings, and memory budget validated through multi-run calibration on real hardware. Ships with the first profile: Qwen3-Coder-Next 79.7B MoE (NVFP4).

## Prerequisites

- NVIDIA GB10 or GB200 (Blackwell GPU, unified memory, ARM64)
- Docker with NVIDIA Container Runtime
- Python 3.13+ and [uv](https://docs.astral.sh/uv/)
- HuggingFace account (for gated model download)

## Quick Start

```bash
git clone https://github.com/steveant/aeo-vllm-gb10.git
cd aeo-vllm-gb10

# Configure
cp .env.example .env
# Edit .env: set HF_TOKEN, adjust host paths
cat profiles/qwen3-coder-nvfp4/profile.env >> .env

# Start
uv sync
uv run bootstrap-vllm up
```

The first run downloads the model (~43 GiB). Subsequent starts use the cached weights.

## What to Expect

| Stage | Time | What's happening |
|-------|------|-----------------|
| First run: model download | ~10-30 min | Downloads ~43 GiB of model weights from HuggingFace |
| Container start → healthy | ~5 min | vLLM loads weights into GPU memory and allocates KV cache |
| First request | ~35-50 sec | One-time JIT compilation of CUDA kernels (TTFT) |
| Subsequent requests | 1-4 sec | Normal inference latency (TTFT) |

**`bootstrap-vllm up` waits for the server to become healthy** before returning. You'll see a spinner while the model loads.

**First request will be slow:** The very first chat completion triggers JIT kernel compilation (~35-50 seconds). This is a one-time cost per container start. All subsequent requests are fast.

## API Usage

The server exposes an OpenAI-compatible API on port 8000.

### curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer none" \
  -d '{
    "model": "saricles/Qwen3-Coder-Next-NVFP4-GB10",
    "messages": [{"role": "user", "content": "Explain unified memory in two sentences."}],
    "max_tokens": 256
  }'
```

### Python (streaming)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

stream = client.chat.completions.create(
    model="saricles/Qwen3-Coder-Next-NVFP4-GB10",
    messages=[{"role": "user", "content": "Write a Python fibonacci generator."}],
    max_tokens=512,
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

The `api_key` can be any non-empty string. The model name must match the value of `VLLM_MODEL` in your `.env`.

## CLI Reference

| Command | Description |
|---------|-------------|
| `bootstrap-vllm up [--force]` | Start the vLLM server (waits until healthy) |
| `bootstrap-vllm down` | Stop the server and remove the container |
| `bootstrap-vllm status` | Show server health, container state, and loaded model |
| `bootstrap-vllm logs [--follow] [--tail N]` | Stream container logs |
| `bootstrap-vllm model current` | Show the currently configured model |
| `bootstrap-vllm model list` | List locally cached models |
| `bootstrap-vllm model download <model-id>` | Download a model from HuggingFace |
| `bootstrap-vllm model switch <model-id>` | Switch to a different model (updates .env, restarts if running) |

All commands read configuration from `.env` in the project root.

## Profiles

| Profile | Model | Hardware | Status |
|---------|-------|----------|--------|
| `qwen3-coder-nvfp4` | Qwen3-Coder-Next 79.7B MoE | GB10/GB200 128 GiB | Calibrated, 4x48K PASS |

Each profile lives under `profiles/<name>/` and contains a `profile.env` with all engine and quantization settings. Append it to your `.env` to activate.

## Project Structure

```
aeo-vllm-gb10/
├── src/bootstrap_vllm/   # CLI package (bootstrap-vllm entrypoint)
│   ├── commands/          # up, down, status, logs
│   ├── core/              # config, docker, validation
│   └── utils/             # output formatting, process helpers
├── docker/                # docker-compose.yml
├── profiles/              # Per-model calibration profiles
│   └── qwen3-coder-nvfp4/ 
│       ├── profile.env    # Engine configuration
│       ├── docs/          # Resource utilization notes, retros
│       ├── drivers/       # Load-test scripts
│       └── runs/          # Calibration run artifacts
├── tools/                 # Reusable load-test infrastructure
├── tests/                 # Unit tests
└── pyproject.toml         # Project metadata and dependencies
```

## Calibration Lab

Each profile includes calibration run artifacts, load-test drivers, and documentation under `profiles/<name>/`. The `tools/` directory provides reusable infrastructure -- a shared `run_lib.py` for building load-test drivers, and a transcript viewer for reviewing run results. See the profile `docs/` and `runs/` directories for methodology and raw data.

## Troubleshooting

**`bootstrap-vllm up` succeeds but the API doesn't respond**

The model takes ~5 minutes to load after the container starts. Run `bootstrap-vllm status` — if health shows "starting", the model is still loading. Use `bootstrap-vllm logs --follow` to watch progress.

**First request hangs for 30-50 seconds**

This is expected. The first request triggers JIT compilation of CUDA kernels. Subsequent requests are fast (1-4s TTFT).

**`No GPU detected` or `nvidia-smi` not found**

Install NVIDIA drivers and ensure `nvidia-smi` works. The NVIDIA Container Runtime must also be installed for Docker GPU access.

**`Marlin NVFP4 backend requires SM 12.1`**

This profile requires a Blackwell GPU (GB10/GB200 with compute capability 12.1). It won't work on older GPUs.

**`Port 8000 already in use`**

Another process is using port 8000. Either stop it or change `VLLM_PORT` in your `.env`.

**`Image not pulled locally`**

Run `docker pull <image>` with the image shown in the error. If the pull fails, check your network connection and Docker login.

**Container exits immediately or OOM kill**

Check `bootstrap-vllm logs --tail 100` for the error. Common causes:
- `VLLM_GPU_MEMORY_UTILIZATION` too high for available memory
- Another process consuming GPU memory
- Model requires more memory than available (128 GiB needed for this profile)

**Model download fails or stalls**

Verify your `HF_TOKEN` is valid at https://huggingface.co/settings/tokens. The model is gated — you must accept the license on the model page first. Ensure you have ~50 GiB free disk space.

## License

[Apache 2.0](LICENSE)
