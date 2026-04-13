# Resource Utilization — Qwen3-Coder-Next NVFP4 on GB10

Working document. Captures the memory model, the config matrix, and a calibration log that grows as we run experiments. **Append new observations here after each run** rather than starting over.

---

## Host

| Field | Value |
|---|---|
| Machine | sfspark1 (NVIDIA GB10, Grace Blackwell ARM64) |
| Total memory | 128 GiB nominal LPDDR5x (unified, no separate VRAM) |
| Usable memory (`free -h`) | ~121 GiB — kernel/firmware reserve the delta |
| CUDA / driver | 13.0 / 580.126.09 |
| Compute capability | SM 12.1 |
| Ambient non-vLLM processes | ~8.5 GiB (OS + GNOME + **STT service PID 1076583 holding 3.5 GiB — do not touch**) |
| Pre-existing swap usage | ~5.7 GiB (not caused by us) |
| **Hard cap on projected peak** | **80 GiB** (raised from 70 GiB mid-Run-001 on 2026-04-12 — see calibration log) |

`nvidia-smi` reports `Memory-Usage: Not Supported` on GB10 because the GPU and CPU share the same physical memory. Use `free -m` for peak measurement, not `nvidia-smi --query-gpu=memory.used`.

## Model & stack

| Field | Value |
|---|---|
| Model ID | `saricles/Qwen3-Coder-Next-NVFP4-GB10` |
| Base | Qwen3-Next 79.7B MoE (hybrid **DeltaNet + attention**; 10 active experts of 512; 3B active params/token) |
| Quantization | NVFP4 via llm-compressor, stored as `compressed-tensors` |
| Weights on disk | 43 GiB (10 safetensors shards) |
| Weights loaded in CUDA | 44 GiB (observed) |
| Native max context | 262144 tokens |
| Image | `avarok/dgx-vllm-nvfp4-kernel:v23` — vLLM `0.16.0rc2.dev236+g3b30e6150.d20260221`, CUDA 13.0 |
| KV cache dtype | `fp8` (via `--kv-cache-dtype fp8`) |
| Attention backend | Flashinfer (via `--attention-backend flashinfer`) |
| NVFP4 GEMM backend | Marlin (via `VLLM_NVFP4_GEMM_BACKEND=marlin`) |
| Required container env | `VLLM_NVFP4_GEMM_BACKEND=marlin`, `VLLM_TEST_FORCE_FP8_MARLIN=1`, `VLLM_USE_FLASHINFER_MOE_FP4=0`, `VLLM_MARLIN_USE_ATOMIC_ADD=1` |

## Memory model (the math used for planning)

> **SUPERSEDED 2026-04-12.** This section is the *pre-Run-001* planning model. Run 001 measurements falsified its specific coefficients (it underestimated overhead by ~7 GiB). Run 002 then exposed an entirely new operating point (a first-concurrent-batch transient) that no version of this static model captures. **For current numbers, see "Calibrated model (Run 001 corrected) and post-Run-002 caveats" further down.** The section is preserved as a historical artifact — it is what we believed *before* taking measurements, and the gap between belief and measurement is itself useful data.

**Components of peak system memory:**

```
Peak = Ambient + vLLM_process_footprint
     = Ambient + Weights + Python/Framework_overhead + Activations/Workspace + KV_pool
```

Measured / estimated constants:

| Component | Value | Source |
|---|---|---|
| Weights resident in CUDA | 43 GiB | observed via `nvidia-smi` during previous run |
| Python/framework overhead | ~2 GiB | gap between weights-in-CUDA and container RSS |
| Activations/workspace (MoE + Flashinfer + 8K prefill buffer) | ~4 GiB | estimated, **not yet empirically pinned** |
| Fixed vLLM overhead | **~49 GiB** | 43 + 2 + 4 |
| Ambient non-vLLM | 8.5 GiB | measured on sfspark1 |
| **Baseline before KV** | **~57.5 GiB** | fixed + ambient |

**KV cache rate:** `48 KiB/token` with `--kv-cache-dtype fp8`. Derived from the HF README's reported `61.7 GiB / 1,346,432 tokens = 49,203 bytes/token ≈ 48 KiB`. This number already amortizes the DeltaNet Mamba recurrent state and vLLM's block rounding.

**vLLM block size quirk:** on Qwen3-Next, vLLM enforces `attention_block_size >= mamba_page_size` and sets attention block size to **1072 tokens** regardless of `--block-size`. Every session pays a minimum of 1 block × 1072 × 48 KiB ≈ **52 MB KV floor**. Ignorable at ≥10K context; dominant at <1K.

**Formulas (for any (seqs, ctx) cell):**

```
total_K_tokens      = max_num_seqs × (max_model_len / 1024)
KV_GiB              = total_K_tokens × 0.046875          # 48 KiB/tok → 48/1024 GiB per 1K tok
Required util (min) = (49 + KV_GiB) / 128                # fits workload exactly
Chosen util         = required + small slack             # typically +0.005 to +0.02
Projected peak      = chosen_util × 128 + 8.5            # vLLM footprint + ambient
Fits cap            = Projected peak ≤ 70 GiB  ⟺  chosen_util ≤ 0.480
```

Note: `gpu_memory_utilization` is the vLLM process's ceiling, not a target. Setting it higher than required gives vLLM more KV pool (prefix cache slack), which raises actual peak. Tune it tight.

## The peak-memory matrix (projected, at minimum util)

> **SUPERSEDED 2026-04-12.** This matrix uses the pre-Run-001 model and the old 70 GiB cap. See "Calibrated model" section below for the post-Run-001 matrix at the 80 GiB cap, and the Run 002 caveat about the first-concurrent-batch transient that may make even the corrected matrix optimistic.

Cell value = projected peak GiB at `util = required_min`. Legend: ✅ comfortable (≥4 GiB under cap) · ⚠ RISKY (<4 GiB headroom — within overhead slop) · ❌ over cap.

| seqs ↓ / ctx → | **32K** | **48K** | **64K** | **96K** | **128K** | **192K** | **262K** |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 59.0 ✅ | 59.75 ✅ | 60.5 ✅ | 62.0 ✅ | 63.5 ✅ | 66.5 ✅ | 69.79 ⚠ |
| **2** | 60.5 ✅ | 62.0 ✅ | 63.5 ✅ | 66.5 ✅ | 69.5 ⚠ | 75.5 ❌ | 82.08 ❌ |
| **3** | 62.0 ✅ | 64.25 ✅ | 66.5 ✅ | 71.0 ❌ | 75.5 ❌ | — ❌ | — ❌ |
| **4** | 63.5 ✅ | 66.5 ✅ | 69.5 ⚠ | 75.5 ❌ | 81.5 ❌ | — ❌ | — ❌ |
| **5** | 65.0 ✅ | 68.75 ⚠ | 72.5 ❌ | — ❌ | — ❌ | — ❌ | — ❌ |
| **6** | 66.5 ✅ | 71.0 ❌ | — ❌ | — ❌ | — ❌ | — ❌ | — ❌ |

### Minimum required `gpu_memory_utilization`

| seqs ↓ / ctx → | 32K | 48K | 64K | 96K | 128K | 192K | 262K |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 0.395 | 0.401 | 0.406 | 0.418 | 0.430 | 0.453 | 0.479 |
| **2** | 0.406 | 0.418 | 0.430 | 0.453 | 0.477 | — | — |
| **3** | 0.418 | 0.436 | 0.453 | — | — | — | — |
| **4** | 0.430 | 0.453 | 0.477 | — | — | — | — |
| **5** | 0.441 | 0.471 | — | — | — | — | — |
| **6** | 0.453 | — | — | — | — | — | — |

### Sensitivity — what breaks if overhead is 2 GiB higher than estimated

| Config | Peak @ 6 GiB overhead | Peak @ 8 GiB overhead | Verdict |
|---|---:|---:|---|
| 4 × 64K | 69.5 | **71.5 ❌** | risky config fails |
| 2 × 128K | 69.5 | **71.5 ❌** | risky config fails |
| 5 × 48K | 68.75 | **70.75 ❌** | risky config fails |
| 1 × 262K | 69.79 | **71.79 ❌** | risky config fails |
| 4 × 48K | 66.5 | 68.5 ✅ | safe under slop |
| 3 × 64K | 66.5 | 68.5 ✅ | safe under slop |
| 1 × 128K | 63.5 | 65.5 ✅ | safe under slop |

Until we pin the actual overhead empirically, **prefer ✅ configs**. The point of the calibration runs below is to shrink this uncertainty band.

## Usage goals (what we are actually trying to serve)

- **Coding workloads** — context matters, 32K is too small to be useful.
- **Concurrency floor:** ≥ 4 simultaneous sessions (user-explicit minimum).
- **Context goal:** 128K per session if achievable while keeping the concurrency floor.
- **Priority order:** large context > high concurrency.

The matrix above shows that **no cell simultaneously satisfies ≥4 sessions AND ≥64K context under the ✅ (safe) band**. The only cell meeting both floors is `4 × 64K`, which is ⚠ risky. This is the core tension we are measuring our way out of.

## Calibration log

Append one entry per run. Each entry should include the config, the observed peak, and the deviation from the model's projection. These numbers are what let us tighten the overhead constant and eventually unlock the ⚠ configs with confidence.

### Run 001 — 4 × 48K baseline

| Field | Value |
|---|---|
| Date | 2026-04-12 |
| Image | `avarok/dgx-vllm-nvfp4-kernel:v23` |
| `max_num_seqs` | 4 |
| `max_model_len` | 49152 |
| Chosen `gpu_memory_utilization` | 0.46 |
| Projected peak (pre-Run-001 model, at 0.46) | 67.38 GiB |
| **Observed Available KV cache memory** | **8.93 GiB** (vLLM `gpu_worker.py` log line) |
| **Observed peak at health-pass** | **66.3 GiB** |
| **Observed steady serving (post-profile)** | **69.9 GiB** |
| **Observed profile-pass transient (~20 s)** | **72.9 GiB** ← binding constraint |
| Burst test (4-parallel, 2000 tokens each) | 88 s wall, ~91 tok/s aggregate, 100 % success |
| Burst test KV pool peak | **2.8 %** |
| Cap raised mid-session | **70 → 80 GiB** (because pre-Run-001 model under-projected by ~7 GiB and 70 was tighter than necessary on real numbers) |
| Status | **PASS — but limited.** Smoke test and 4-parallel burst both succeeded. KV pool was driven to only 2.8 %, so high-fill behavior was *not* characterized. That gap is what Run 002 was designed to fill. |
| Notes | The pre-Run-001 model was wrong by ~7 GiB. The corrected three-coefficient model (peak_health / peak_steady / peak_profile = 57.4 / 61.0 / 64.0 + pool) was derived from this run and is documented in "Calibrated model" below. |

### Calibrated model (Run 001 corrected) and post-Run-002 caveats

**This is the model we use for planning now.** It supersedes the static `Ambient + Weights + Workspace + KV` decomposition above.

```
pool_GiB     = max_num_seqs × max_model_len × 48 KiB / 1024²

peak_health  ≈ 57.4 + pool_GiB    # the moment health check first passes
peak_steady  ≈ 61.0 + pool_GiB    # post profile-pass quiescent serving
peak_profile ≈ 64.0 + pool_GiB    # the ~20 s transient during the post-startup profile pass
```

All three constants were validated against `free -m` measurements during Run 001 to within 0.1 GiB. The relationship `peak_steady - peak_health ≈ 3.6 GiB` is the cuda-cache pinning that survives the profile pass; `peak_profile - peak_steady ≈ 3 GiB` is the worst-case batch the profile pass pushes through the model on every container start.

> **POST-RUN-002 CAVEAT (SOFTENED by Run 002c).** Run 002 measured an operating point that none of the three coefficients describes: a first-concurrent-batch initialization transient of roughly +5 to +7 GiB above `peak_steady`, with `running=0`. **Run 002c was specifically designed to re-provoke this via a ramped 1→2→3→4 concurrency transition under the same config — and failed to.** Peak during each ramp transition was within 0.04 GiB of the solo-serving floor (see Run 002c section). So the +5–7 GiB tax is **not** a universal property of this config; Run 002's spike was either situational to that specific startup state (pool 9.91 GiB, unknown allocation ordering) or required a codepath the Run 002c engine did not hit. Going forward: treat this as a **bounded excursion risk** (one observation, three non-observations) rather than a binding constraint. The 88 GiB kill switch is still the right safety, but the "add 5–7 GiB" adjustment is no longer mandatory in planning arithmetic.

**Calibrated peak-memory matrix at 80 GiB cap (uses corrected `peak_profile = 64 + pool`).** Cell value = projected GiB. Legend: ✅ ≥3 GiB headroom · ⚠ 1–3 GiB · ❌ over cap. **None of these cells include the first-concurrent-batch tax** — see caveat above.

| seqs ↓ / ctx → | 32K | 48K | 64K | 96K | 128K | 192K | 262K |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 65.5 ✅ | 66.3 ✅ | 67.0 ✅ | 68.5 ✅ | 70.0 ✅ | 73.0 ✅ | 76.3 ✅ |
| 2 | 67.0 ✅ | 68.5 ✅ | 70.0 ✅ | 73.0 ✅ | 76.0 ✅ | 82.0 ❌ | 88.6 ❌ |
| 3 | 68.5 ✅ | 70.75 ✅ | 73.0 ✅ | 77.5 ⚠ | 82.0 ❌ | — | — |
| **4** | 70.0 ✅ | **73.0 ✅** *(Run 001)* | **76.0 ✅** *(was Run 003 target)* | 82.0 ❌ | 88.0 ❌ | — | — |
| 5 | 71.5 ✅ | 75.25 ⚠ | 79.0 ⚠ | — | — | — | — |
| 6 | 73.0 ✅ | 77.5 ⚠ | 82.0 ❌ | — | — | — | — |

**With the Run 002 caveat applied** (add ~7 GiB for any cell that hasn't been exercised at concurrency on a warm engine), several cells previously called ✅ become ⚠ or ❌:

| Cell | Without caveat | With +7 GiB cold-start tax | New verdict |
|---|---:|---:|:---:|
| 4 × 48K | 73.0 | 80.0 | ⚠ at the cap |
| 4 × 64K | 76.0 | 83.0 | ❌ |
| 2 × 96K | 73.0 | 80.0 | ⚠ at the cap |
| 1 × 128K | 70.0 | 77.0 | ⚠ |

This is the central finding from Run 002: **the model that promised 4 GiB of headroom for the original Run 003 target (4 × 64K) was structurally optimistic**, because it didn't know about a transient operating point that lives above its highest coefficient.

### Run 002 — 4 × 48K under load (KILLED)

| Field | Value |
|---|---|
| Date | 2026-04-12 |
| Container state | Restarted from cold (Run 001 container had been removed) |
| Config | Identical to Run 001 — `max_num_seqs=4`, `max_model_len=49152`, `gpu_memory_utilization=0.46`, image `avarok/dgx-vllm-nvfp4-kernel:v23` |
| **Observed Available KV cache memory** | **9.91 GiB** (Run 001 was 8.93 — 1 GiB more pool, presumably 1 GiB less compute workspace) |
| Plan | Three escalating phases A→B→C in one container lifetime (4×4×1500, 4×10×2000, 4×16×2500 max_tokens), full observability, kill switch at 78 GiB / +1.5 swap / 95 % KV |
| Driver | `/tmp/run_002_load_test.py` (stdlib only, 4 worker threads + 2 s monitor + kill switch) |
| **Verdict** | **KILLED on criterion #2** — peak host used 80.79 GiB > 78 GiB |
| **Turns completed** | 4 of 120 (Phase A turn 0 only; phases B and C never started) |
| Per-turn errors | **0** — every request that ran completed successfully (~57.9 s wall each, 1500 tokens, KV peak 2.5 %) |
| Peak host used | **80.79 GiB** |
| Peak swap | 10.00 GiB (+0.17 vs driver baseline of 9.83) — criterion #3 **PASS** |
| Peak KV pool | 2.5 % — criterion #5 **PASS** |
| Latency drift | insufficient data — criterion #4 N/A |
| Observed vs projection | Δ +9.88 GiB above `peak_steady` — criterion #6 **FAIL** |

**The smoking-gun memtrail** (`/tmp/run_002_memtrail.csv`, abbreviated):

```
21:32:02  used=75.24 GiB  swap=9.81  kv=0.0%  run=0    ← driver baseline
21:32:05  used=76.58       9.75      0.0      0
21:32:07  used=79.70       9.68      0.0      0    ← oscillation, no requests
21:32:09  used=80.29       9.85      0.0      0
21:32:11  used=75.31       10.00     0.0      0
21:32:13  used=75.43       9.71      0.0      0    ← container_rss jumps 280→646 MiB
21:32:17  used=80.79       9.67      0.0      0    ← PEAK, still no requests
21:32:19  used=80.26       9.64      0.0      0
21:32:23  used=76.07       9.64      0.0      0
21:32:25  used=76.06       9.63      0.0      0
21:32:29  used=77.43       9.63      2.0      4    ← workers active at last
21:32:31  used=79.16       9.62      2.0      4
21:32:42  used=68.34       9.39      2.0      4    ← drops 13 GiB mid-flight
21:32:45  used=66.06       9.29      2.0      4
21:33:00  used=65.86       9.26      2.5      4    ← steady state under 4-concurrent inference
```

**The peak occurred while `running=0`** — *before any worker had successfully sent a request*. Memory oscillated wildly between 75 and 81 GiB during the 24 s the workers were inside `urlopen()` waiting for vLLM to schedule the requests. By the time `running=4` finally appeared, the spike was already over and memory was trending down. Once the engine was actually serving 4 concurrent requests, memory dropped to ~66 GiB and stayed there.

Container RSS jumped 280 → 646 → 556 MiB across the same window, confirming the allocation activity was happening *inside* the vLLM process before it picked up the request batch.

**Hypothesis (`[inferred]`, not yet verified):** vLLM (flashinfer JIT? marlin kernel autotune? cudagraph re-capture for new batch shapes?) does ~5–7 GiB of allocation/release work the *first time* it sees N concurrent requests where N exceeds the largest batch it has serviced this session. The dry-run that preceded the main run only sent `running=1`, so the `running=4` codepath was cold. Run 001's burst test didn't catch this either — that test ran on a freshly-warmed engine where the post-startup profile pass had already touched batch sizes including 4 (per `cudagraph_capture_sizes: [1, 2, 4, 8]`).

**Other notable measurements from Run 002:**

- **Active inference at low fill (4 concurrent reqs, 2.5 % KV) used ~66 GiB** — *lower* than the corrected `peak_steady` projection of 70.91 GiB. Once warm, the engine is actually more memory-efficient than the model says.
- The driver's own baseline at run start (75.06 GiB) was already 7 GiB above the dry-run baseline (67.66 GiB), captured ~75 s earlier. Whatever consumed those 7 GiB happened in that gap, with no driver activity. Either vLLM's idle background work allocated, or the kernel's unified-memory accounting has lag.
- `Available KV cache memory` was 9.91 GiB (vs Run 001's 8.93 GiB) under identical config. That's a +1 GiB shift in pool size between two cold starts of the same image. We don't know what GPU resident state changed between the two runs to explain it.

**Artifacts:** `/tmp/run_002_load_test.py`, `/tmp/run_002_driver.log`, `/tmp/run_002_turns.csv` (4 rows), `/tmp/run_002_memtrail.csv` (26 samples).

### Run 002b — Sequential warmup characterization (PASS, outcome refined)

| Field | Value |
|---|---|
| Date | 2026-04-12 |
| Container state | Cold start (fresh `docker compose up -d` after Run 002 cleanup) |
| Config | Identical to Runs 001/002 — `max_num_seqs=4`, `max_model_len=49152`, `gpu_memory_utilization=0.46`, image v23 |
| Pool size | **9.72 GiB** (Run 001: 8.93, Run 002: 9.91, Run 002b: 9.72 — keeps drifting cold-start to cold-start) |
| Test shape | 4 sequential single-threaded requests, distinct topics, max_tokens=1500, 30 s idle gaps, streaming so we capture TTFT |
| Cap raised | 80 → 90 GiB for this run; kill switch 88 GiB |
| Driver | `vLLM/scripts/run_002b_sequential.py` (committed `630ec8a`) |
| Verdict | **PASS — outcome D in the static framework, but the TTFT progression refines the diagnosis** |

**The headline finding — TTFT progression:**

| # | Topic | TTFT (s) | Wall (s) | Tokens out | Peak GiB during request |
|---|---|---:|---:|---:|---:|
| 1 | Rust | **33.19** | 57.21 | 1500 | 68.02 |
| 2 | Postgres | **0.31** | 23.95 | 1500 | 67.93 |
| 3 | Linear algebra | **0.13** | 23.89 | 1500 | 67.96 |
| 4 | Distributed systems | **0.10** | 23.72 | 1500 | 67.96 |

Request 1 paid **33 seconds of first-token latency**. Requests 2–4 paid **~0.1–0.3 s**. That is a 100–330× speedup, and it is rock-solid evidence that vLLM does **once-per-engine-lifetime JIT-compile work** on the first real request — *separately* from the post-startup profile pass that happens during `docker compose up -d`. The four prompts were on completely different topics specifically to defeat the prefix cache, so the speedup cannot be explained by prefix-cache hits.

**The memory finding — flat as a board:**

| Phase | Used (GiB) |
|---|---:|
| Cold idle (60 s before first request) | 67.24 |
| Peak during request 1 | 68.02 |
| Peak during requests 2–4 | 67.93–67.96 |
| Warm tail (60 s after last request) | 67.97 |
| **Total memory swing across the entire 7-minute run** | **~1.1 GiB** |

The kill switch never approached its 88 GiB threshold (~20 GiB margin throughout). Compare to Run 002, which oscillated between 75 and 81 GiB over a 25-second window with `running=0` and tripped the kill switch *before any worker had successfully sent a request*.

**Together these two findings reframe the Run 002 spike entirely.** Run 002b proves that:

1. **JIT compile work exists** (the 33 s TTFT on req 1) but **does not allocate significant resident memory**. Whatever vLLM JITs on the first request fits inside the workspace already pre-allocated at startup. So a "pre-warm with one dummy request" strategy fixes TTFT for the first real user but does **not** fix the Run 002 memory spike — they're different problems.
2. **The Run 002 memory spike is therefore concurrency-specific**, not coldness-specific. Sequential cold requests don't trigger it; 4 simultaneous requests do. The likely cause is something on the chunked-prefill / parallel scheduler / multi-batch-attention codepath that allocates buffers only when the batch arrives in parallel.
3. **The corrected `peak_steady` coefficient is itself slightly too high.** Run 001 measured `peak_steady = 69.9 GiB` at pool 8.93 → const ≈ 61.0. Run 002b measured `peak_steady = 68.02 GiB` at pool 9.72 → const ≈ 58.3. That's a **~3 GiB downward revision** of the model's static coefficient. We'd informally read it as `peak_steady ≈ 58.3 + pool_GiB` going forward, but only one data point is not enough to commit a new constant. Track it, re-test on the next run.

**What this does NOT prove**

Run 002b sent requests one at a time, so it never exercised any concurrency codepath. It has therefore not measured what happens on a *concurrent* batch — only that *sequential* batches are well-behaved. The Run 002 spike is still not explained, only narrowed: it lives somewhere in the concurrent codepath, not in the cold-engine codepath.

**Updated outcome interpretation**

The plan classified outcomes as A/B/C (per-spike framework) or D (no spike). Run 002b lands in D *for the sequential test* — but D is not really "no spike" in the universal sense, it is "no spike on the sequential codepath." The right name for this outcome is:

> **Outcome E — sequential is fine, the spike is concurrency-bound.**

**Artifacts** (under `vLLM/runs/run_002b/`):
- `run_002b_driver.txt` — full stdout
- `run_002b_per_request.csv` — 9 rows: 4 boundary markers + 4 request rows + 1 warm tail
- `run_002b_memtrail.csv` — 463 samples at ~0.5 s cadence over the full ~7 min window

### Run 002c — Ramped Multi-Turn High-Fill Characterization (PASS)

| Field | Value |
|---|---|
| Date | 2026-04-13 |
| Container state | Cold start (fresh `docker compose up -d` after Run 002b cleanup) |
| Config | Identical to Runs 001/002/002b — `max_num_seqs=4`, `max_model_len=49152`, `gpu_memory_utilization=0.46`, image v23 |
| Pool size | **11.41 GiB** (001: 8.93, 002: 9.91, 002b: 9.72, 002c: 11.41 — drift *widening*, span now 2.48 GiB across 4 cold starts) |
| Test shape | 4 sessions × 16 turns × 2500 max_tokens, **ramped start** (S1 alone → S2 joins after S1.t1 → S3 after S2.t1 → S4 after S3.t1), then async finish, no post-ramp synchronization |
| Cap / kill switch | 90 GiB / 88 GiB (same as Run 002b) |
| Driver | `vLLM/scripts/run_002c_ramped.py` |
| **Verdict** | **PASS — all six criteria.** 64/64 turns ok, 0 errors, 0 kill trips |

**Headline finding — the Run 002 concurrency spike did not reproduce.** Ramp transition peaks:

| Transition | Peak used (GiB) | kv_max in window |
|---|---:|---:|
| 1 → 2 | 70.79 | 1.9 % |
| 2 → 3 | 70.76 | 2.5 % |
| 3 → 4 | 70.78 | 4.0 % |

Essentially flat — within 0.04 GiB of each other, and only ~0.4 GiB above the solo-serving floor. Run 002's 80.79 GiB spike at `running=0` was either situational to that cold-start state or required a specific allocation path that this engine instance did not traverse. The "+5–7 GiB first-concurrent-batch tax" previously posited as a universal caveat is **not supported** as a general rule under this config.

**Per-concurrency-level steady-state stats** (from memtrail segmentation by `running` value):

| running | samples | max used (GiB) | mean used (GiB) | max kv |
|---:|---:|---:|---:|---:|
| 0 | 29 | 70.15 | 70.12 | 0.0 % |
| 1 | 134 | 70.72 | 70.33 | 3.7 % |
| 2 | 122 | 70.79 | 70.43 | 7.2 % |
| 3 | 228 | 70.85 | 70.40 | 10.3 % |
| 4 | 1072 | 70.86 | 70.66 | 13.5 % |

**4× concurrency added ~0.33 GiB mean, ~0.14 GiB peak** over solo serving. Concurrency cost under this config is effectively nil.

**Per-session summary:**

| S | Topic | done | TTFT₁ | TTFT₁₆ | wall₁ | wall₁₆ | p95 first-4 | p95 last-4 | drift ratio | final prompt tok |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 | Rust | 16/16 | **32.94** | 0.48 | 69.30 | 26.32 | 72.43 | 85.71 | **1.18** | 32717 |
| S2 | Postgres | 16/16 | 0.28 | 0.34 | 51.95 | 32.67 | 74.88 | 77.75 | 1.04 | 33468 |
| S3 | Linear algebra | 16/16 | 0.30 | 0.48 | 68.36 | 42.68 | 74.85 | 40.86 | 0.55 | 29697 |
| S4 | Distributed systems | 16/16 | 0.53 | 0.38 | 53.51 | 45.04 | 78.40 | 43.30 | 0.55 | 30662 |

- **TTFT proof**: S1 first-turn TTFT was **32.94 s**; all subsequent first-turn TTFTs were sub-second. This matches Run 002b's 33.19 s to within 0.25 s — the JIT cost on request 1 is reproducible, memory-free, and a true once-per-engine-lifetime event.
- **Latency drift**: three sessions *decreased* (S3/S4 p95 shrunk because their last-4 turns ran at concurrency 1–3 as other workers finished first). S1 grew 18 %, S2 grew 4 %. All well under the 2× threshold.
- Sessions finished in order S1 → S3 → S2 → S4 (temperature 0.7 produced unpredictable but roughly balanced pacing).

**Peak memory sources:**
- Baseline (driver start): **69.45 GiB** (lower than initial free -m at container-start because engine still settling)
- Peak used: **70.86 GiB**
- Total memory swing across the entire ~27 min run: **1.41 GiB**

The run is arguably the calmest the engine has ever looked under load — calmer than Run 002b (1.1 GiB swing, but sequential) and a totally different universe from Run 002 (5.7 GiB transient in 24 s with `running=0`).

**Revised `peak_steady` coefficient — three data points now:**

| Run | pool (GiB) | peak_steady (GiB) | implied const |
|---|---:|---:|---:|
| 001 | 8.93 | 69.93 | 61.0 |
| 002b | 9.72 | 68.02 | 58.3 |
| 002c | 11.41 | 70.86 | **59.45** |

Mean const ≈ **59.58 ± 1.4 GiB** (sample stdev across three runs). Treat 59.5 + pool as the working center estimate and carry ±1.5 GiB as known noise. The fit is not monotonic in pool size — pool 11.41 is the largest but produced a mid-range const — consistent with the variance being driven by something other than pool alone (likely allocation ordering at startup).

**What Run 002c does NOT prove — the prefix-caching fill-masking finding:**

The plan's intent was to drive KV toward 80–90 % fill. Observed peak `kv_pct` was only **13.5 %**. Post-run analysis:

- Each session's turn-N prompt is ~99 % prefix-cache hit against turn-(N−1)'s prompt + response. Only ~2600 new tokens per session per turn actually require fresh KV allocation (new user message + new assistant response).
- With `enable_prefix_caching=True`, vLLM's `GPU KV cache usage` metric appears to report **currently-live active blocks only** — blocks that are "held for reuse" by the prefix cache count as free pool, not as used KV. Measured growth in `free -m used` (+1.41 GiB) is consistent with ~15 % fill, matching the reported kv_pct.
- Net result: **this test characterized concurrent steady-state memory with prefix caching active, not raw KV fill.** To measure actual 80–90 % fill behavior, we need either:
  1. `--enable-prefix-caching=false` (or driver-level disable) and run the same shape, **or**
  2. Per-turn topic-divergent prompts that don't build on prior turns (so each turn is fresh KV), **or**
  3. Significantly more concurrent sessions with distinct histories.

None of these are blocking to the question we did answer (concurrency safety), but they're required to actually stress the pool.

**Artifacts** (under `vLLM/runs/run_002c/`):
- `run_002c_driver.txt` — full stdout summary
- `run_002c_per_turn.csv` — 64 rows (4 sessions × 16 turns, all status=ok)
- `run_002c_memtrail.csv` — 1635 samples at ~0.75 s effective cadence over the full ~20 min run window (docker-log scraping slowed the nominal 0.5 s cadence)

### Run 002d — Enriched Multi-Turn Content-Stress Characterization (PASS)

**Date:** 2026-04-13. **Verdict: PASS** (36/36 turns, all criteria met).

Initial run was KILLED (S1 turn 7 HTTP 400 — context overflow at ~42 K prompt tokens vs 48 K window). Retry added a context-window guard (`trim_history_to_budget` drops oldest user/assistant pairs when prompt exceeds 88 % of `max_model_len`), JSONL transcript format, and extracted shared infrastructure into `run_lib.py`.

**Objective:** enrich the per-session content — real code files as seeds, scenario-specific follow-ups, max_tokens=7000 — to determine whether content richness pushes the KV pool past 002c's 13.5 % kv_pct ceiling. Prefix caching stayed enabled (production parity). Additionally: first run with a full conversation transcript artifact.

**Shape:** 4 concurrent sessions (ramped start), 9 turns each (1 seed + 8 follow-ups), max_tokens=7000 (up from 2500). Per-session content modules with distinct workloads:
- S1: code refactor of `mesh/commands/smb.py` (20 KB seed, real code)
- S2: code refactor of `bootstrap_vllm/commands/model.py` (31 KB seed, reading package)
- S3: Redis OOMKill incident postmortem (synthetic scenario, 20 KB seed)
- S4: ADR/RFC for a centralized feature-flag platform (synthetic scenario, 16 KB seed)

| Criterion | Result |
|---|---|
| Completion rate | **36 / 36** ok, 0 errors → **PASS** |
| Peak host memory | **75.06 GiB** (< 88 GiB) → PASS |
| Swap growth | **+0.22 GiB** (< 1.5 GiB) → PASS |
| Latency drift | All sessions pass (max ratio 1.45× S2) |
| KV pool peak | **21.8 %** (vs 002c's 13.5 %, meaningful increase) |
| Peak vs projection | Observed 75.06 GiB vs projected 70.18 GiB (Δ +4.88) → implied `peak_steady` const = **65.88** |

**Per-session completion:**

| Session | Topic | Turns ok | kv max | Final prompt tokens | Notes |
|---|---|---:|---:|---:|---|
| S1 | Code refactor (smb.py) | 9 / 9 | 21.8 % | 38,780 | Context guard trimmed history on later turns. JIT TTFT 35.6 s. |
| S2 | Code refactor (bootstrap_vllm) | 9 / 9 | 21.8 % | 34,057 | |
| S3 | Incident postmortem | 9 / 9 | 21.8 % | 37,011 | |
| S4 | ADR evolution (flagforge) | 9 / 9 | 21.8 % | 38,932 | |

**Pool size (sixth cold-start data point):** 9.18 GiB (prior: 8.93, 9.91, 9.72, 11.41, 12.17). Drift span now **3.24 GiB** across 6 cold starts. The pool swung from 12.17 (run 5) back down to 9.18 (run 6), confirming the variance is non-monotonic.

**`running` × `kv_pct` correlation — scheduler-serialization vs prefix-cache-masking diagnostic:**

| running | samples | max kv | mean kv |
|---:|---:|---:|---:|
| 0 | 39 | 0.0 % | 0.0 % |
| 1 | 241 | 6.2 % | 4.8 % |
| 2 | 176 | 10.6 % | 7.1 % |
| 3 | 160 | 15.7 % | 9.9 % |
| 4 | 1257 | 21.8 % | 15.0 % |

**running=4 max / running=1 max = 21.8 / 6.2 = 3.5×** — pool fills linearly with concurrency, confirming the KILLED run's finding. **Prefix caching is the within-session KV mask**, not scheduler serialization. Each concurrent session holds independent KV memory; the prefix cache deduplicates turn-over-turn history within each session.

**Why kv_pct reached 21.8 % this time vs 15.4 % on the KILLED run:** smaller pool (9.18 vs 12.17 GiB). The same absolute KV allocation fills a larger fraction. kv_pct = allocated_blocks / pool_blocks, so a 25% smaller pool yields proportionally higher percentages. The actual KV memory consumed was similar (~2.0 GiB peak across 4 sessions).

**Artifacts** (under `vLLM/runs/run_002d/`):
- `run_002d_driver.txt` — full stdout summary
- `run_002d_per_turn.csv` — 36 rows (4 sessions × 9 turns, all ok)
- `run_002d_memtrail.csv` — 1872 samples at ~0.5–1 s cadence
- `run_002d_transcript.jsonl` — JSONL conversation transcript (740 KB). First record is metadata (system prompt, config); subsequent records are per-turn with full user/assistant content, status, metrics, and raw vLLM usage dict.

### Planned follow-up runs (updated post-Run-002d)

- **Run 003 candidate: `4 × 64K` at util ≈ 0.48** — **highest priority.** Pool size at 64K should be ~16–17 GiB (extrapolating pool drift). peak_steady ≈ 59.5 + 16.5 ≈ 76 GiB with ~12 GiB headroom to the 88 GiB kill switch. Run 002d confirmed prefix caching is the KV-fill mask, not scheduler serialization, so 64K just gives more room for history accumulation before context overflow. **Must add a context-window guard** (truncate early history at 90 % of max_model_len) to avoid the HTTP 400 that killed 002d.
- **Run 002d-retry candidate**: re-run 002d's enriched shape with the context-window guard, targeting a clean PASS. Low priority — 002d already proved the key finding (linear KV fill, prefix cache is the mask). A retry only adds a fourth `peak_steady` data point on this config; Run 003 at 64K is more useful.
- **Run 004 candidate: `1 × 262K` at util ≈ 0.48** — still a worthwhile isolated bound check. Low risk.
- **Run 002e candidate** (deferred): warm-engine concurrent pre-warm. Lower priority now.

## Known uncertainties (tracked across runs)

1. ~~**Overhead constant (~6 GiB)** is not empirically pinned on this host.~~ **Resolved by Run 001 (partially):** the static overhead splits into three operating points (`peak_health` / `peak_steady` / `peak_profile` = 57.4 / 61.0 / 64.0 + pool, validated to ±0.1 GiB at one config). **Reopened by Run 002**: a fourth operating point exists (first-concurrent-batch transient) that adds another ~5–7 GiB and is not explained by any of the three static coefficients. See item 6.
2. **KV rate (48 KiB/tok)** was derived from aggregate numbers in the model card, not a direct measurement on this host. Run 001 saw 8.93 GiB pool, Run 002 saw 9.91 GiB pool *under identical config* — a 1 GiB shift between two cold starts of the same image. The KV rate may not be the only thing varying; could also be how vLLM splits the util budget between weights, workspace, and pool depending on what other processes hold GPU resident state at startup.
3. **Prefix cache slack** — vLLM will grab extra KV pool within the util budget for prefix caching. If the workload repeatedly hits long shared prefixes (likely for coding), prefix cache can dominate. **Not yet measured at high fill** — Run 002 was killed before reaching the phase that would have exercised this.
4. **Swap interaction** — the host already has ~5.7 GiB of pre-existing baseline swap. Run 002 saw the driver-start swap baseline at 9.83 GiB (3.6 GiB above the dry-run baseline 50 s earlier — unclear what allocated). Peak swap during the run was +0.17 GiB above baseline, so the kill switch's swap criterion was never triggered. But the *baseline shifting* between runs is itself a flag.
5. **`--enforce-eager=true`** is currently set; disabling for throughput is a separate experiment. Don't conflate with resource planning.
6. **(REFINED by Run 002c) Concurrency-bound memory transient — NOT reproducible under current conditions.** Run 002b ruled out the cold-engine/JIT-compile hypothesis. Run 002c was specifically designed to re-provoke the Run 002 spike via a ramped 1→2→3→4 concurrency transition under the same config. **The spike did not reproduce.** Peak memory during each ramp transition was 70.76–70.79 GiB (within 0.04 GiB of each other and only ~0.4 GiB above the solo-serving floor). The "+5–7 GiB first-concurrent-batch tax" is therefore **not** a universal property of this config; Run 002's 80.79 GiB spike was either situational to that specific cold-start state (pool 9.91 GiB, different startup allocation ordering) or required an allocation codepath that Run 002c's engine instance did not traverse. **Open question downgraded**: the spike is now a known unknown with one observation and three non-observations, rather than a binding constraint on the config matrix. Until we can provoke it again deliberately, treat the +7 GiB caveat as an *upper-bound* excursion risk rather than an expected operating point.
7. **(NEW, opened by Run 002; sixth data point from Run 002d retry)** vLLM `Available KV cache memory` is **not deterministic across cold starts** of the same image with the same `.env`. Run 001 → 8.93 GiB, Run 002 → 9.91 GiB, Run 002b → 9.72 GiB, Run 002c → 11.41 GiB, Run 002d (killed) → 12.17 GiB, **Run 002d (retry) → 9.18 GiB**. Span is **3.24 GiB across 6 cold starts**. The pool swung from 12.17 back to 9.18 on consecutive runs, confirming the variance is non-monotonic and not a persistent upward drift. Leading hypothesis: GPU-resident state at startup (including the STT service's 3.5 GiB and kernel memory fragmentation) controls the util-budget split.
8. **(NEW, opened by Run 002b; fifth data point from Run 002d retry)** The `peak_steady` coefficient varies across runs. Five data points:

   | Run | pool (GiB) | peak_steady (GiB) | implied const |
   |---|---:|---:|---:|
   | 001 | 8.93 | 69.93 | 61.0 |
   | 002b | 9.72 | 68.02 | 58.3 |
   | 002c | 11.41 | 70.86 | 59.45 |
   | 002d (killed) | 12.17 | 74.92 | 62.75 |
   | 002d (retry, PASS) | 9.18 | 75.06 | 65.88 |

   Mean ≈ **61.48 GiB**, sample stdev ≈ **2.9 GiB**. The 002d retry has the highest const (65.88) despite the smallest pool (9.18). This is concerning — it may indicate that higher KV fill (21.8% vs prior runs' <15%) does increase the overhead constant, or that some other startup-dependent factor pushed host memory higher. Note the 002d retry peak of 75.06 GiB included a brief spike to 75.06 during the session wind-down phase (running dropping from 4→3→2→1). **Working model for planning**: `peak_steady ≈ 61.5 + pool_GiB ± 3.0 GiB` (widened again). The noise band is uncomfortably large; Run 003 at 64K is critical for determining whether the variance tightens at a different config or is inherent to this host.
9. **(NEW, opened by Run 002c; PARTIALLY CONFIRMED by Run 002d)** `enable_prefix_caching=True` **masks the `GPU KV cache usage` metric** from representing raw KV pool fill. Run 002c saw 13.5 % kv_pct; Run 002d (enriched content, max_tokens=7000) saw 15.4 % — a modest increase despite much richer content. **Run 002d's running × kv_pct correlation confirms the mechanism**: kv_pct scales roughly linearly with concurrency (running=1 max 4.4 %, running=4 max 15.4 %, ratio 3.5×). This proves: (a) prefix caching is the within-session mask (each session reuses its own history), NOT scheduler serialization, and (b) cross-session KV allocation is genuinely independent (no sharing). The reason kv_pct stays at ~15 % is that each session's actual KV footprint is only ~3–4 % of the pool after prefix deduplication. **Consequence for planning**: to push past 15–20 % kv_pct with prefix caching enabled, need either more concurrent sessions (>4) or longer context windows (64K+). The 80–90 % fill measurement originally planned for the Run 002 series is not achievable at 4×48K with prefix caching enabled.

## Next steps (post Run 002d)

Run 002d was KILLED by a context overflow but produced the most informative diagnostic finding of the series: the `running × kv_pct` correlation that confirms prefix caching — not scheduler serialization — is why kv_pct stays at 13–15 % under 4×48K. Consolidated state of knowledge:

- ✅ JIT compile work on the first real request is real (~33–36 s TTFT) but **memory-free**.
- ✅ Sequential and concurrent requests are well-behaved (four runs, no repeat of Run 002's spike).
- ✅ Four runs of steady-state data, giving a coefficient estimate of **60.5 + pool ± 2.0 GiB** for planning.
- ✅ **Prefix caching is the KV-fill mask, not scheduler serialization.** kv_pct scales 3.5× from running=1 to running=4 — linear with concurrency. Each session genuinely holds independent KV memory. The mask is within-session turn-history reuse.
- ✅ **Enriched content (max_tokens=7000, real code seeds) only pushes kv_pct from 13.5 % to 15.4 %.** Content richness is not the lever for high fill. The 48K context window + 4 sessions is simply too small to stress the 12 GiB pool.
- ❓ **Run 002's 80.79 GiB spike is still unexplained.** One observation, four non-observations.
- ❓ **KV pool fill at 80–90 % is still not characterized** and cannot be at 4×48K with prefix caching enabled. Need either 4×64K or more concurrent sessions.
- ❓ **Pool size drift now spans 3.24 GiB across 5 cold starts** and trends upward. Increasingly affects projection accuracy.
- ❓ **Context-window management is missing from the driver.** S1's prompt accumulated to ~42 K tokens and hit the 48 K limit. Future runs need a guard.

### Run 003 candidate: `4 × 64K` at util ≈ 0.48 — **highest priority**

The corrected-model projection is ~77 GiB at steady state (util 0.48, peak_steady ≈ 60.5 + ~16.5 pool at 64K). With the +7 GiB concurrency-tax caveat relaxed by 002c, and 002d confirming that KV fills linearly with concurrency (no hidden cross-session overhead), this cell should land under 80 GiB with ~8 GiB headroom to the kill switch. The 64K context window will also give more room for prompt accumulation before overflow. **Must include a context-window guard** (truncate early history when prompt exceeds 90 % of max_model_len) — this is the lesson from 002d's KILLED exit.

### Run 002d-retry candidate — clean PASS with context guard

Re-run 002d's enriched shape with the context-window guard to get a clean 36/36 PASS. Low priority: 002d already answered the key questions (linear KV fill, prefix cache is the mask). A retry only adds a cleaner `peak_steady` data point at the same 4×48K config.

### Run 002e candidate (deferred): log forensics on the Run 002 spike

If Run 003 or Run 002d re-provokes a concurrency-bound spike, read vLLM source / docker logs around the window for the subsystem doing the allocation. Candidates (unchanged from prior): `vllm/v1/engine/core.py`, `vllm/attention/backends/flashinfer.py`, chunked-prefill scheduler, marlin MoE first-call autotune. Deprioritized because we now have one full run that *doesn't* trip it; we should wait for a second provocation before spending time on log archaeology.

### Other observations to pursue

- **Pool size drift** (item 7): 5 observations now span 3.24 GiB and trend upward. The standalone 5-cold-start experiment is overdue — run it before Run 003 to get a tighter variance bound.
- **`peak_steady` coefficient** (item 8): four data points now, mean 60.5 ± 2.0 GiB. 002d's outlier (62.75) may be inflated by the KILLED exit. Run 003 at 64K will be the first data point at a different config — critical for knowing if the coefficient is config-dependent.
- **Context-window guard**: add to the driver template before Run 003. Simple approach: if `sum(prompt_tokens) > 0.9 * max_model_len`, truncate oldest assistant turns from history. This prevents HTTP 400 without discarding the system prompt or most recent context.
- **The 4 × 64K promotion**: 002d confirms no hidden cross-session overhead. Run 003 is the next step; if it passes, the config is promotable.

### Cap policy

The 90 GiB lifted ceiling with 88 GiB kill switch held through Run 002b and Run 002c without ever being touched (Run 002c peak 70.86 GiB — 17 GiB of unused headroom). The ceiling/kill-switch pair can stay at 90/88 for Run 003 and any other concurrent high-fill test. If Run 003 or 002d also passes cleanly, consider dropping back to 80/78 for steady operation, but retain 90/88 for new-config exploration.

### The doc itself

After Run 003 lands, this document should finally be restructured: the pre-Run-001 SUPERSEDED sections moved to `resource-utilization-history.md`, the calibration log promoted to the top, the matrix regenerated against the revised coefficient, and Known Uncertainties trimmed to just the still-open items. We now have enough data (4 runs) that the historical planning content is more distracting than useful, but one more run at a different (seqs, ctx) cell will make the restructure far more informative.

## Commands used for observation

```bash
# Memory watchdog (sample every 3s, print max over window)
while true; do free -m | awk 'NR==2 {print strftime("%T"), $3}'; sleep 3; done

# Or a bounded one-liner that prints running max
python3 -c 'import subprocess, time, re
peak=0
for _ in range(300):
    out=subprocess.check_output(["free","-m"]).decode()
    used=int(re.findall(r"\d+", out.split("\n")[1])[1])
    peak=max(peak,used)
    print(f"{time.strftime(\"%T\")}  used={used}MiB  peak={peak}MiB", flush=True)
    time.sleep(3)'

# Log tail for backend confirmation
uv run bootstrap-vllm logs --tail 400 | grep -E 'NvFp4|FLASHINFER|MARLIN|fp8|attention block size|max_num_seqs|KV cache'
```

## References

- `/tmp/vllm-nvfp4-continuation.md` — pre-Run-001 session handoff with original constant derivations
- `/tmp/vllm-run-002-continuation.md` — Run 002 session handoff (corrected coefficients, the four open questions, the locked-in decisions)
- `/home/steve/.claude/plans/zesty-giggling-crystal.md` — Run 001 plan (4 × 48K calibration)
- `/home/steve/.claude/plans/twinkly-giggling-eagle.md` — Run 002 plan (load test against 4 × 48K)
- `/home/steve/.claude/plans/greedy-crafting-mochi.md` — Run 002c plan (ramped multi-turn high-fill characterization; originally Run 002b plan, repurposed in-session after 002b completed)
- `/home/steve/.claude/plans/woolly-dreaming-wave.md` — original NVFP4 harness code-change plan
- `vLLM/scripts/run_002_load_test.py` — Run 002 driver (committed `630ec8a`)
- `vLLM/scripts/run_002b_sequential.py` — Run 002b driver (committed `630ec8a`)
- `vLLM/scripts/run_002c_ramped.py` — Run 002c driver
- `vLLM/runs/run_002/` — Run 002 artifacts (driver log, turns CSV, memtrail CSV)
- `vLLM/runs/run_002b/` — Run 002b artifacts (driver log, per-request CSV, memtrail CSV)
- `vLLM/runs/run_002c/` — Run 002c artifacts (driver log, per-turn CSV, memtrail CSV)
- `vLLM/scripts/run_002d_enriched.py` — Run 002d driver
- `vLLM/scripts/run_002d_prompts/` — Run 002d per-session content modules (4 files)
- `vLLM/runs/run_002d/` — Run 002d artifacts (driver log, per-turn CSV, memtrail CSV, transcript)
- `/home/steve/.claude/plans/kind-questing-bear.md` — Run 002d plan (enriched multi-turn characterization)
- HF model card: https://huggingface.co/saricles/Qwen3-Coder-Next-NVFP4-GB10
