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
| `bootstrap-vllm up [--force]` | Start the vLLM server (pulls image on first run) |
| `bootstrap-vllm down` | Stop the server and remove the container |
| `bootstrap-vllm status` | Show server health and container state |
| `bootstrap-vllm logs [--tail N]` | Stream container logs |

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

## License

[Apache 2.0](LICENSE)
