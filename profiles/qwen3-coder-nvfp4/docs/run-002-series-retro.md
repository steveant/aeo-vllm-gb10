# Retrospective: Run 002 Series — 4×48K Multi-Turn Characterization

**Date:** 2026-04-13
**Host:** NVIDIA GB10 (Grace Blackwell ARM64, 128 GiB unified memory)
**Model:** Qwen3-Coder-Next 79.7B MoE, NVFP4, max_model_len=48576
**Series span:** Run 002 through Run 002d (4 runs + 1 retry), 2026-04-12 to 2026-04-13

---

## Goal

Characterize the memory and KV cache behavior of 4 concurrent multi-turn sessions at 48K context on the GB10 with prefix caching enabled. Determine whether the configuration is safe for production, measure the `peak_steady` coefficient for future capacity planning, and understand why `kv_pct` stayed unexpectedly low in early runs.

## Timeline

| Run | Verdict | Key event |
|-----|---------|-----------|
| 002 | KILLED | First concurrent-batch transient spiked to 80.79 GiB. Kill switch tripped. |
| 002b | PASS (8/8) | Sequential characterization. Ruled out cold-engine/JIT as the spike cause. |
| 002c | PASS (64/64) | Ramped 1→2→3→4 concurrency. Spike did NOT reproduce. kv_pct peaked at only 13.5% despite ~168K total accumulated history — prefix caching identified as the KV-fill mask. |
| 002d (attempt 1) | KILLED | Enriched content (real code seeds, max_tokens=7000). S1 hit HTTP 400 at ~42K prompt tokens — context overflow. No context guard. |
| 002d (attempt 2) | PASS (36/36) | Added `trim_history_to_budget` guard. Extracted `run_lib.py`. JSONL transcripts. kv_pct reached 21.8%. All criteria met. |

## What worked

**Ramped concurrency pattern.** Staggering session starts (S2 joins after S1 turn 1, etc.) let us observe kv_pct at each concurrency level and build the `running × kv_pct` correlation table. This was the single most diagnostic measurement of the series.

**Enriched per-session content.** Moving from generic follow-ups (002c) to scenario-specific workloads — real code review, incident postmortem, ADR evolution — pushed prompts to 34–39K tokens. This achieved 80–87% context utilization (the actual goal) and forced the context guard to activate, proving it works under load.

**JSONL transcript format.** Saving full user/assistant content alongside metrics means we can replay and analyze conversations after the run. The `raw_usage` passthrough and `extra` catch-all field make the format forwards-compatible without schema changes.

**Reusable infrastructure (`run_lib.py`).** Extracting CSVWriter, TranscriptWriter, Monitor, KillState, and API helpers into a shared module means Run 003 can focus on its own content and logic without reimplementing I/O, monitoring, or kill-switch plumbing.

**Cross-platform tooling (`transcript_viewer.py`).** The HTML viewer with embedded marked.js renders full GFM markdown from any JSONL transcript. Works on both Linux and Windows (with mapped drive support). This makes the transcripts usable by anyone with a browser.

## What didn't work

**Initial seed-token target (8–12K) was unrealistic.** The actual source files were 2–5K tokens. We wasted time trying to pad seeds before accepting that generation tokens (max_tokens=7000 × 9 turns × 4 sessions ≈ 252K) are what fill the context, not the seed. Lesson: do the math on total token budget before setting seed-size targets.

**No context-window guard in the first 002d attempt.** We knew prompts would accumulate aggressively with max_tokens=7000 and 9 turns, but didn't add a guard until after the HTTP 400 killed the run. The fix (`trim_history_to_budget` at 88% of max_model_len) took 15 minutes to implement and should have been in the original driver. Lesson: context guards are mandatory for multi-turn drivers, not optional.

**Markdown transcript format (002d attempt 1) was replaced by JSONL.** The initial markdown transcript made it easy to read but hard to programmatically analyze. Switching to JSONL with a viewer tool was the right call — structured data with a presentation layer on top.

**`Path.resolve()` on Windows.** The HTML viewer's file:// URL generation broke on Windows because `Path.resolve()` and `Path.absolute()` canonicalize mapped drives to UNC paths. Required three iterations to fix (resolve → absolute → `os.path.abspath`). Lesson: always test cross-platform path handling on the actual target machine via SSH, not by reasoning about API contracts.

## Key findings

### 1. Prefix caching is the KV-fill mask

The headline result. kv_pct scales linearly with concurrency:

| running | max kv_pct |
|--------:|-----------:|
| 1 | 6.2% |
| 2 | 10.6% |
| 3 | 15.7% |
| 4 | 21.8% |

Ratio running=4 / running=1 = 3.5×. Each session holds independent KV memory; the prefix cache deduplicates turn-over-turn history within each session. This is a compute optimization (avoids re-encoding shared prefixes), not a memory shortcut — the KV blocks are still allocated, just reused across the prompt/generation boundary.

**Consequence:** at 4×48K with prefix caching, kv_pct cannot exceed ~25% regardless of content. Pushing higher requires more concurrent sessions (>4) or longer context windows (64K+).

### 2. Context utilization reached 80–87%

All four sessions accumulated 34–39K prompt tokens against the 48K window before the context guard trimmed history. This was the actual target — stress the context window, not the KV pool percentage.

### 3. Pool size drift is non-monotonic

Six cold starts, same config: 8.93, 9.91, 9.72, 11.41, 12.17, 9.18 GiB. Span 3.24 GiB. The pool swung from 12.17 back to 9.18 on consecutive runs, ruling out persistent upward drift. Leading hypothesis: GPU-resident state at startup (ambient GPU-resident processes, kernel fragmentation) controls the util-budget split.

### 4. `peak_steady` coefficient: 61.5 ± 3.0 GiB

Five data points from four runs:

| Run | pool | peak_steady | implied const |
|-----|-----:|------------:|-------------:|
| 001 | 8.93 | 69.93 | 61.0 |
| 002b | 9.72 | 68.02 | 58.3 |
| 002c | 11.41 | 70.86 | 59.45 |
| 002d (killed) | 12.17 | 74.92 | 62.75 |
| 002d (PASS) | 9.18 | 75.06 | 65.88 |

Mean 61.5, stdev 2.9. The noise band is wider than ideal. Run 003 at 64K will show whether this tightens at a different config.

### 5. Run 002's 80.79 GiB spike remains unexplained

One observation, four non-observations. Downgraded from blocking constraint to upper-bound excursion risk. Not reproducible under controlled conditions.

## Process improvements adopted

1. **Context guard is now mandatory** — `trim_history_to_budget()` in `run_lib.py`, used by all future drivers.
2. **JSONL transcripts are standard** — `TranscriptWriter` in `run_lib.py` with IO-error tolerance, `raw_usage` passthrough, and `extra` catch-all.
3. **Shared infrastructure module** — `run_lib.py` (773 lines, stdlib-only) replaces copy-paste across drivers.
4. **Cross-platform HTML viewer** — `transcript_viewer.py` with embedded marked.js for GFM rendering.
5. **Excel workbook for analysis** — charts for per-turn metrics, memory timeline, concurrency vs KV, and cross-run comparison.

## Artifacts

| Path | Description |
|------|-------------|
| `vLLM/scripts/run_lib.py` | Shared load-test infrastructure |
| `vLLM/scripts/run_002d_enriched.py` | Run 002d driver (reference implementation) |
| `vLLM/scripts/run_002d_prompts/` | Per-session content modules (4 files) |
| `vLLM/scripts/transcript_viewer.py` | Cross-platform HTML viewer |
| `vLLM/scripts/vendor/marked.umd.js` | Markdown renderer (embedded in HTML output) |
| `vLLM/runs/run_002d/` | Run artifacts (driver log, CSVs, JSONL, HTML, Excel) |
| `vLLM/docs/resource-utilization.md` | Running calibration log (updated through Run 002d) |

## What's next

**Run 003: 4×64K at util ≈ 0.48** — highest priority. Projected peak ~77 GiB with ~11 GiB headroom. The 64K window gives more room for history before the context guard kicks in, and the larger pool (~16 GiB projected) may show different kv_pct behavior. First data point at a new config for the `peak_steady` coefficient.

**Pool size drift characterization** — 5 rapid cold starts to tighten the variance bound before committing to Run 003's projection.
