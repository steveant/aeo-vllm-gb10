# CLAUDE.md

Project instructions for AI coding assistants working in aeo-vllm-gb10.

## Repository Structure

```
aeo-vllm-gb10/
├── src/bootstrap_vllm/    # CLI: up, down, status, logs, model commands
├── docker/                # Docker Compose config (parameterized via .env)
├── tools/                 # Reusable load-test infra and transcript viewer
├── profiles/              # Model+hardware profiles (first: qwen3-coder-nvfp4)
│   └── <profile>/
│       ├── profile.env    # Calibrated model config
│       ├── docs/          # Calibration docs and retros
│       ├── runs/          # Run artifacts (CSV, JSONL, HTML, XLSX)
│       └── drivers/       # Load-test driver scripts
├── scripts/               # Build helpers (make_*.py)
├── tests/                 # pytest tests
└── models/                # HuggingFace model cache (gitignored)
```

## Key Commands

- `uv sync` -- install dependencies
- `uv run bootstrap-vllm up` -- start vLLM server
- `uv run bootstrap-vllm down` -- stop server
- `uv run bootstrap-vllm status` -- check status
- `uv run pytest` -- run tests
- `make build` -- build standalone binary

## Configuration

- `.env` is gitignored -- user-specific config (HF token, paths)
- `.env.example` has user-specific template
- `profiles/<name>/profile.env` has model-specific calibrated values
- `docker/docker-compose.yml` reads all config from .env -- do NOT hardcode values in compose

## Conventions

- Profile convention: each model+hardware combo gets `profiles/<name>/`
- Tools in `tools/` are reusable across all profiles (stdlib-only, no side effects at import)
- Run artifacts go under the profile's `runs/` directory
- Drivers import shared infra from `tools/` via sys.path

## Hardware

- Target: NVIDIA GB10/GB200 (Blackwell, ARM64, unified memory)
- Use `free -m` for memory measurement, NOT `nvidia-smi` (unified memory)
- GPU-only inference -- no CPU fallback

## Directives

- Do not edit .env (gitignored) -- edit .env.example or profile.env
- Do not hardcode env vars in docker-compose.yml
- Validate before declaring root cause
- Minimal complexity -- prefer existing tools over custom code
