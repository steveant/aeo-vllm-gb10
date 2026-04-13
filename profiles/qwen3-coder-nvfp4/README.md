# Qwen3-Coder-Next NVFP4 -- GB10 Profile

## Model

**Qwen3-Coder-Next 79.7B MoE** -- hybrid DeltaNet + attention architecture,
512 experts with 10 active per token (3B active parameters/token).

Quantized to **NVFP4** via llm-compressor, stored as compressed-tensors.
Weights on disk: ~43 GiB across 10 safetensors shards.

HuggingFace: [saricles/Qwen3-Coder-Next-NVFP4-GB10](https://huggingface.co/saricles/Qwen3-Coder-Next-NVFP4-GB10)

## Hardware Requirements

| Requirement | Value |
|-------------|-------|
| GPU | NVIDIA GB10 or GB200 (Blackwell, SM 12.x) |
| Memory | 128 GiB unified memory |
| Architecture | ARM64 (Grace) |
| Docker image | `avarok/dgx-vllm-nvfp4-kernel:v23` (vLLM 0.16.0rc2, CUDA 13.0) |
| KV cache dtype | FP8 |
| Attention backend | FlashInfer |
| Prefix caching | Enabled (production parity) |

## Calibrated Operating Point

| Parameter | Value |
|-----------|-------|
| Concurrent sessions | 4 |
| Context window | 48K tokens |
| Peak memory | ~75 GiB / 128 GiB |
| Headroom | ~13 GiB |
| `peak_steady` | 61.5 +/- 3.0 GiB |

The operating point was established across four calibration runs with prefix
caching enabled, matching the production configuration.

## Key Findings

- **Prefix cache is the KV-fill mask.** kv_pct scales linearly with
  concurrency (6.2% at 1 session, 21.8% at 4 sessions). The prefix cache
  deduplicates turn-over-turn history within each session, capping KV
  utilization at ~25% for this configuration regardless of content.

- **Context utilization reached 80--87%.** All four sessions accumulated
  34--39K prompt tokens against the 48K window before the context guard
  trimmed history.

- **Pool size drift spans 3.24 GiB.** Six cold starts showed pool sizes from
  8.93 to 12.17 GiB, with non-monotonic behavior ruling out persistent upward
  drift. Leading hypothesis: GPU-resident state at startup controls the
  util-budget split.

- **kv_pct at 4 concurrent: 21.8%.** Each session holds independent KV memory;
  the ratio running=4 / running=1 is 3.5x.

## Calibration Runs

| Run | Verdict | Description |
|-----|---------|-------------|
| 002 | KILLED | First concurrent-batch transient spiked to 80.79 GiB; kill switch tripped |
| 002b | PASS (8/8) | Sequential characterization; ruled out cold-engine/JIT as spike cause |
| 002c | PASS (64/64) | Ramped 1-2-3-4 concurrency; spike did not reproduce; prefix cache identified as KV-fill mask |
| 002d | PASS (36/36) | Enriched content with real code seeds; context guard validated; kv_pct reached 21.8% |

## Documentation

- [docs/resource-utilization.md](docs/resource-utilization.md) -- Running calibration log with per-run memory data, coefficient tables, and capacity projections
- [docs/run-002-series-retro.md](docs/run-002-series-retro.md) -- Full retrospective covering timeline, findings, and process improvements

## Run Artifacts

Each subdirectory under `runs/` contains the artifacts for one calibration run:

| File type | Description |
|-----------|-------------|
| `*_memtrail.csv` | Memory utilization time series (sampled during run) |
| `*_turns.csv` / `*_per_turn.csv` | Per-turn metrics (latency, tokens, kv_pct) |
| `*_driver.txt` | Driver stdout/stderr log |
| `*_transcript.jsonl` | Full conversation transcript (structured, machine-readable) |
| `*_transcript.html` | Browser-viewable transcript with GFM markdown rendering |
| `*_analysis.xlsx` | Excel workbook with charts for memory, concurrency, and cross-run comparison |
