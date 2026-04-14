# E2E Test Plan — aeo-vllm-gb10

## Overview

End-to-end validation strategy for the full user lifecycle on NVIDIA GB10. Covers setup through shutdown, mixing automated scripts, existing calibration drivers, and agentic observation.

**Philosophy:** No mocks, no emulators. Real Docker, real GPU, real model, real API. Things that can't be codified into assertions are evaluated by agentic observation.

## Validation Matrix

| Stage | What | Validation | Tool | Status |
|-------|------|-----------|------|--------|
| Setup | `.env` assembled correctly | Manual + agent verify | Agent observation | Validated |
| Start + health | `bootstrap-vllm up` (waits for healthy) | Automated | `e2e_lifecycle.py` Phase 2 | PASS (271s) |
| Model discovery | `/v1/models` returns model ID | Automated | `e2e_lifecycle.py` Phase 3 | PASS |
| First request + TTFT | Streaming chat completion | Automated | `e2e_lifecycle.py` Phase 4 | PASS (47s TTFT) |
| Status reporting | `bootstrap-vllm status` | Automated | `e2e_lifecycle.py` Phase 5 | PASS |
| Graceful shutdown | `bootstrap-vllm down` | Automated | `e2e_lifecycle.py` Phase 6 | PASS |
| Verify stopped | Status shows not running | Automated | `e2e_lifecycle.py` Phase 7 | PASS |
| Multi-turn streaming | 4 sessions x 9 turns, ramped | Automated | `run_002d_enriched.py` | PASS (prior run) |
| Concurrency + memory | 4 concurrent x 48K context | Automated | `run_002c_ramped.py` | PASS (prior run) |
| Memory safety | Peak < 88 GiB, swap < 1.5 GiB | Automated | 002-series 6-criterion eval | PASS (prior run) |
| UX clarity | Output readable, timing expectations set | Agentic | Agent observation | Gaps found |

## Lifecycle Test Script

**Location:** `tests/e2e_lifecycle.py`
**Run:** `uv run python tests/e2e_lifecycle.py`
**Duration:** ~6 minutes (cold start with cached model)

Exercises a 7-phase lifecycle: prereq check, start (includes health wait), confirm healthy + discover model, chat completion, status, stop, verify stopped. Uses `tools/run_lib.py` for health polling, model discovery, and streaming chat.

## Existing Calibration Drivers

Located in `profiles/qwen3-coder-nvfp4/drivers/`. These assume the container is already running.

| Driver | What it proves | Duration |
|--------|---------------|----------|
| `run_002b_sequential.py` | JIT warmup characterization, TTFT baseline | ~5 min |
| `run_002c_ramped.py` | Ramped concurrent 4-session load, memory peaks | ~30 min |
| `run_002d_enriched.py` | 4x9 multi-turn with enriched content, 6-criterion eval | ~45 min |

All runs have artifacts in `profiles/qwen3-coder-nvfp4/runs/`.

## Measured Baselines (2026-04-13)

| Metric | Value | Notes |
|--------|-------|-------|
| Cold boot to healthy | 271s (~4.5 min) | Model cached, no download |
| First-request TTFT | 47s | JIT compilation, one-time cost |
| Subsequent TTFT | ~2-4s | After JIT warmup |
| Peak steady-state memory | ~75 GiB | At 4 concurrent sessions |
| Container healthcheck start_period | 600s | Configured in docker-compose.yml |

## UX Gaps Found During Testing

### P0 — First-time user will think it's broken

1. ~~**`up` returns immediately but server isn't ready for ~5 min.**~~ **FIXED:** `up` now waits for health with a spinner before returning.

2. ~~**First request hangs for ~47 seconds.**~~ **MITIGATED:** README now documents first-request JIT latency in the "What to Expect" section.

### P1 — Missing information

3. ~~**README doesn't explain boot timeline.**~~ **FIXED:** "What to Expect" section added with timing table.

4. ~~**`model` subcommands undocumented.**~~ **FIXED:** All 4 model subcommands added to CLI Reference.

5. ~~**No troubleshooting section.**~~ **FIXED:** 8 common failure modes documented.

### P2 — Polish

6. ~~**`up` could suggest `status` or `logs --follow`.**~~ **SUPERSEDED:** `up` now waits for health with a spinner — no need to suggest manual checks.

7. ~~**`up` could optionally wait for health.**~~ **FIXED:** `up` always waits. No flag needed.

## Improvement Backlog

All items from the original backlog have been addressed:

| # | Item | Status |
|---|------|--------|
| 1 | README: "What to Expect" section | Done |
| 2 | README: Troubleshooting section | Done |
| 3 | README: Document `model` subcommands | Done |
| 4 | CLI: `up` waits for health with spinner | Done |
| 5 | CLI: Prereq short-circuit on first failure | Done |
