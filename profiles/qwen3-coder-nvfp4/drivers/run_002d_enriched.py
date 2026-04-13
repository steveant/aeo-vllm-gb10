#!/usr/bin/env python3
"""
Run 002d - enriched multi-turn content-stress characterization for the 4 x 48K
vLLM config.

Four independent multi-turn conversations, started in a staggered ramp:

    S1: [t1][t2][t3][t4][t5] ...            through turn 9
    S2:     [t1][t2][t3][t4] ...            through turn 9
    S3:         [t1][t2][t3] ...            through turn 9
    S4:             [t1][t2] ...            through turn 9
         ^    ^    ^
         |    |    |
         |    |    S3 joins (trigger: S2 finished turn 1)
         |    S2 joins (trigger: S1 finished turn 1)
         t=0  S1 alone - pays 0->1 JIT transition

After all four workers are running, there is no synchronization - each session
runs its own multi-turn loop independently and exits when it hits 9 turns. The
probabilistic temperature means sessions finish in unpredictable order.

Each session uses per-session enriched prompts imported from run_002d_prompts,
with a markdown transcript written alongside the CSV artifacts.

Stdlib only. Caller is responsible for `docker compose up -d` before and
`docker compose down` after.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))          # run_002d_prompts/
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))  # run_lib

from run_lib import (
    CSVWriter, TranscriptWriter, Monitor, KillState,
    Sample, TurnRecord, RampTransition, LevelStats,
    chat_completion_streaming, discover_model_id, wait_for_health,
    read_free_m, read_pool_gib, iso, pct,
    find_ramp_transitions, per_level_stats, per_session_summary,
    trim_history_to_budget,
    PER_TURN_HEADER, MEMTRAIL_HEADER,
    EXIT_OK, EXIT_FAIL, EXIT_KILLED, EXIT_CRASH,
)

from run_002d_prompts import (
    session_1_code_refactor,
    session_2_code_refactor,
    session_3_postmortem,
    session_4_design_doc,
)
SESSIONS = [
    session_1_code_refactor,
    session_2_code_refactor,
    session_3_postmortem,
    session_4_design_doc,
]

# ---------------------------------------------------------------------------
# Hard-coded config (per plan greedy-crafting-mochi.md - Run 002d)
# ---------------------------------------------------------------------------

VLLM_BASE_URL = "http://localhost:8000"
CHAT_ENDPOINT = f"{VLLM_BASE_URL}/v1/chat/completions"
MODELS_ENDPOINT = f"{VLLM_BASE_URL}/v1/models"
HEALTH_ENDPOINT = f"{VLLM_BASE_URL}/health"

CONTAINER_NAME = "docker-vllm-1"

PER_TURN_CSV_PATH = Path("/tmp/run_002d_per_turn.csv")
MEMTRAIL_CSV_PATH = Path("/tmp/run_002d_memtrail.csv")
TRANSCRIPT_PATH = Path("/tmp/run_002d_transcript.jsonl")

# Context-window guard: if accumulated prompt tokens exceed this fraction of
# the model's max context, drop oldest user/assistant pairs (keeping the system
# prompt and the most recent turns).  The model's max_model_len is 49152 for
# the 48K config; we use a conservative 43000 (~88%) to leave room for
# max_tokens on the response side.
CONTEXT_GUARD_TOKENS = 43000

MONITOR_INTERVAL_S = 0.5
HEARTBEAT_INTERVAL_S = 15.0
DOCKER_STATS_INTERVAL_S = 5.0
SAMPLE_RING_SIZE = 8000  # ~100 min at 0.5 s cadence

TURN_TIMEOUT_S = 900
WARM_TAIL_S = 30
TEMPERATURE = 0.7
MAX_TOKENS = 7000
TURNS_PER_SESSION = 9
NUM_SESSIONS = 4

# Cap raised to 90 GiB (matches Run 002b); kill switch 2 GiB under cap.
KILL_USED_GIB = 88.0
KILL_SWAP_DELTA_GIB = 1.5
KILL_KV_PCT = 95.0

# Ramp transition window (seconds after a running->running+1 transition) used
# for per-transition peak reporting in the summary.
RAMP_WINDOW_S = 60.0

# For the informational "observed vs projection" criterion. The constant is
# known stale (Run 001 measured 61.0, Run 002b measured 58.3). This run
# produces the third data point needed to commit a revision.
DEFAULT_POOL_GIB = 9.72
PEAK_STEADY_CONST = 61.0


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a senior engineer giving precise technical explanations with "
    "code or worked examples where relevant. Your answers should build on "
    "prior turns of this conversation."
)


# ---------------------------------------------------------------------------
# Ramped worker
# ---------------------------------------------------------------------------

class RampedWorker(threading.Thread):
    def __init__(
        self,
        session_id: int,
        seed_topic: str,
        seed_prompt: str,
        follow_ups: list[str],
        model_id: str,
        kill_state: KillState,
        monitor: Monitor,
        turns_writer: CSVWriter,
        turns_lock: threading.Lock,
        turn_records: list[TurnRecord],
        start_event: threading.Event,
        next_start_event: Optional[threading.Event],
        transcript_writer: TranscriptWriter,
    ):
        super().__init__(name=f"worker-s{session_id+1}", daemon=True)
        self.session_id = session_id
        self.seed_topic = seed_topic
        self.seed_prompt = seed_prompt
        self.follow_ups = follow_ups
        self.model_id = model_id
        self.kill_state = kill_state
        self.monitor = monitor
        self.turns_writer = turns_writer
        self.turns_lock = turns_lock
        self.turn_records = turn_records
        self.start_event = start_event
        self.next_start_event = next_start_event
        self.transcript_writer = transcript_writer
        self.history: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def _user_msg_for_turn(self, turn_index: int) -> str:
        if turn_index == 0:
            return self.seed_prompt
        return self.follow_ups[turn_index - 1]

    def _release_next(self, reason: str) -> None:
        if self.next_start_event is not None and not self.next_start_event.is_set():
            print(
                f"-- S{self.session_id+1} releasing next worker ({reason}) --",
                flush=True,
            )
            self.next_start_event.set()

    def run(self) -> None:
        try:
            print(f"-- S{self.session_id+1} waiting for start signal --", flush=True)
            self.start_event.wait()
            if self.kill_state.tripped:
                print(f"-- S{self.session_id+1} killed before start --", flush=True)
                return

            print(
                f"-- S{self.session_id+1} starting ({self.seed_topic}) --",
                flush=True,
            )

            for turn_index in range(TURNS_PER_SESSION):
                if self.kill_state.tripped:
                    return

                user_msg = self._user_msg_for_turn(turn_index)
                self.history.append({"role": "user", "content": user_msg})
                self.history = trim_history_to_budget(self.history, CONTEXT_GUARD_TOKENS)
                messages = list(self.history)

                start_ts = time.time()
                t0 = time.monotonic()
                content = ""
                usage: dict = {}
                try:
                    content, usage, ttft = chat_completion_streaming(
                        CHAT_ENDPOINT, self.model_id, messages, MAX_TOKENS,
                        TEMPERATURE, TURN_TIMEOUT_S,
                    )
                    wall = time.monotonic() - t0
                    self.history.append({"role": "assistant", "content": content})
                    status = "ok"
                    prompt_tokens = int(usage.get("prompt_tokens", 0))
                    completion_tokens = int(usage.get("completion_tokens", 0))
                    total_tokens = int(usage.get("total_tokens", 0))
                except Exception as e:
                    wall = time.monotonic() - t0
                    ttft = -1.0
                    self.history.pop()  # un-append the user msg since it didn't land
                    status = f"error:{type(e).__name__}:{str(e)[:120]}"
                    prompt_tokens = completion_tokens = total_tokens = 0
                end_ts = time.time()

                max_used, max_swap, max_kv, max_running, _ = self.monitor.window_max(
                    start_ts, end_ts
                )

                ramp_event = f"S{self.session_id+1} joined" if turn_index == 0 else ""

                record = TurnRecord(
                    session_id=self.session_id,
                    turn_index=turn_index,
                    status=status,
                    wall=wall,
                    ttft=ttft,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    mem_used_max_during=max_used,
                    kv_pct_max_during=max_kv,
                    running_max_during=max_running,
                    ramp_event=ramp_event,
                )
                with self.turns_lock:
                    self.turn_records.append(record)

                self.turns_writer.write(
                    {
                        "ts": iso(end_ts),
                        "session_id": self.session_id,
                        "turn_index": turn_index,
                        "status": status,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "wall_latency_s": f"{wall:.3f}",
                        "ttft_s": f"{ttft:.3f}",
                        "running_max_during": max_running if max_running is not None else "",
                        "kv_pct_max_during": f"{max_kv:.2f}" if max_kv is not None else "",
                        "mem_used_max_during": max_used if max_used is not None else "",
                        "swap_max_during": max_swap if max_swap is not None else "",
                        "ramp_event": ramp_event,
                    }
                )

                self.transcript_writer.write_turn(
                    session_id=self.session_id,
                    session_topic=self.seed_topic,
                    turn_index=turn_index,
                    user_msg=user_msg,
                    assistant_msg=content,
                    status=status,
                    wall=wall,
                    ttft=ttft,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    raw_usage=usage if usage else None,
                )

                peak_gib = (max_used / 1024.0) if max_used is not None else float("nan")
                kv_str = f"{max_kv:5.1f}%" if max_kv is not None else "  -- "
                run_str = f"{max_running}" if max_running is not None else "-"
                join_str = f"  [{ramp_event}]" if ramp_event else ""
                print(
                    f"  S{self.session_id+1} t{turn_index+1:02d}/{TURNS_PER_SESSION}: "
                    f"{status:<10} wall={wall:6.2f}s ttft={ttft:5.2f}s "
                    f"p={prompt_tokens:>5} c={completion_tokens:>5} "
                    f"peak={peak_gib:5.2f}GiB kv_max={kv_str} run_max={run_str}{join_str}",
                    flush=True,
                )

                if status != "ok":
                    self.kill_state.trip(
                        f"S{self.session_id+1} t{turn_index} {status}"
                    )
                    return

                if turn_index == 0:
                    self._release_next("after turn 1 ok")
        finally:
            self._release_next("finally")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_ramp_coordinator(
    model_id: str,
    monitor: Monitor,
    kill_state: KillState,
    turns_writer: CSVWriter,
    transcript_writer: TranscriptWriter,
) -> list[TurnRecord]:
    turns_lock = threading.Lock()
    turn_records: list[TurnRecord] = []

    # Linked chain of start_events: w0 pre-set, each worker holds next's event
    start_events = [threading.Event() for _ in range(NUM_SESSIONS)]

    workers: list[RampedWorker] = []
    for i in range(NUM_SESSIONS):
        session_mod = SESSIONS[i]
        next_event = start_events[i + 1] if i + 1 < NUM_SESSIONS else None
        workers.append(
            RampedWorker(
                session_id=i,
                seed_topic=session_mod.SEED_TOPIC,
                seed_prompt=session_mod.SEED_PROMPT,
                follow_ups=session_mod.FOLLOW_UPS,
                model_id=model_id,
                kill_state=kill_state,
                monitor=monitor,
                turns_writer=turns_writer,
                turns_lock=turns_lock,
                turn_records=turn_records,
                start_event=start_events[i],
                next_start_event=next_event,
                transcript_writer=transcript_writer,
            )
        )

    for w in workers:
        w.start()

    # Kick off the ramp
    print("-- pre-setting S1 start event (ramp begins) --", flush=True)
    start_events[0].set()

    for w in workers:
        w.join()

    return turn_records


# ---------------------------------------------------------------------------
# Summary / evaluation
# ---------------------------------------------------------------------------

def evaluate(
    records: list[TurnRecord],
    monitor: Monitor,
    kill_state: KillState,
    swap_baseline_mib: int,
    pool_gib: float,
) -> tuple[bool, int, list[str]]:
    """Six criteria; criterion #6 is informational (does not fail)."""
    lines: list[str] = []
    passed = True
    first_fail: Optional[int] = None

    def fail(num: int) -> None:
        nonlocal passed, first_fail
        passed = False
        if first_fail is None:
            first_fail = num

    expected_turns = NUM_SESSIONS * TURNS_PER_SESSION
    actual_turns = len(records)
    ok_turns = sum(1 for r in records if r.status == "ok")
    err_turns = actual_turns - ok_turns

    # #1 completion rate
    c1 = (actual_turns == expected_turns) and (err_turns == 0)
    lines.append(
        f"[{'PASS' if c1 else 'FAIL'}] #1 completion: "
        f"{ok_turns}/{expected_turns} ok, {err_turns} errors"
    )
    if not c1:
        fail(1)

    # #2 peak used
    peak_used_gib = monitor.peak_used_mib / 1024.0
    c2 = peak_used_gib < KILL_USED_GIB
    lines.append(
        f"[{'PASS' if c2 else 'FAIL'}] #2 peak used:    "
        f"{peak_used_gib:.2f} GiB (< {KILL_USED_GIB} GiB)"
    )
    if not c2:
        fail(2)

    # #3 swap growth
    swap_growth_gib = (monitor.peak_swap_mib - swap_baseline_mib) / 1024.0
    c3 = swap_growth_gib < KILL_SWAP_DELTA_GIB
    lines.append(
        f"[{'PASS' if c3 else 'FAIL'}] #3 swap growth:  "
        f"{swap_growth_gib:+.2f} GiB (< {KILL_SWAP_DELTA_GIB} GiB)"
    )
    if not c3:
        fail(3)

    # #4 per-session p95 drift (last 4 vs first 4)
    sess = per_session_summary(records)
    c4 = True
    drift_lines: list[str] = []
    for sid in sorted(sess.keys()):
        s = sess[sid]
        ratio = (s["p95_last4"] / s["p95_first4"]) if s["p95_first4"] > 0 else float("inf")
        ok = ratio < 2.0
        if not ok:
            c4 = False
        drift_lines.append(
            f"  S{sid+1}: p95_first4={s['p95_first4']:6.2f}s  "
            f"p95_last4={s['p95_last4']:6.2f}s  ratio={ratio:5.2f}  "
            f"{'ok' if ok else 'FAIL'}"
        )
    lines.append(
        f"[{'PASS' if c4 else 'FAIL'}] #4 latency drift (per-session p95 last4 < 2x first4)"
    )
    lines.extend(drift_lines)
    if not c4:
        fail(4)

    # #5 kv peak
    peak_kv = monitor.peak_kv_pct
    c5 = peak_kv < KILL_KV_PCT
    lines.append(
        f"[{'PASS' if c5 else 'FAIL'}] #5 kv peak:      {peak_kv:.1f}% (< {KILL_KV_PCT}%)"
    )
    if not c5:
        fail(5)

    # #6 informational only - peak vs projection
    projected = PEAK_STEADY_CONST + pool_gib
    delta = peak_used_gib - projected
    lines.append(
        f"[INFO] #6 peak vs proj: observed {peak_used_gib:.2f} GiB vs "
        f"stale proj {projected:.2f} GiB (delta {delta:+.2f} GiB) "
        f"- INFORMATIONAL ONLY, coefficient revision input"
    )
    implied_const = peak_used_gib - pool_gib
    lines.append(
        f"         implied peak_steady const = {implied_const:.2f} "
        f"(Run 001: 61.0, Run 002b: 58.3)"
    )

    return passed, (first_fail or 0), lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--health-timeout", type=int, default=900)
    ap.add_argument(
        "--pool-gib",
        type=float,
        default=None,
        help="Override the auto-detected KV pool size (read from container logs)",
    )
    args = ap.parse_args()

    print("== Run 002d driver (enriched multi-turn content-stress) ==", flush=True)

    pool_gib = args.pool_gib if args.pool_gib is not None else read_pool_gib(CONTAINER_NAME)
    if pool_gib is None:
        pool_gib = DEFAULT_POOL_GIB
        print(f"WARN: could not read pool from logs, using default {pool_gib} GiB", flush=True)
    else:
        print(f"detected pool: {pool_gib:.2f} GiB", flush=True)

    projected_peak_steady = PEAK_STEADY_CONST + pool_gib
    print(
        f"stale projection: peak_steady = {projected_peak_steady:.2f} GiB "
        f"(informational only)",
        flush=True,
    )

    baseline_used, baseline_avail, baseline_swap = read_free_m()
    print(
        f"baseline: used={baseline_used/1024:.2f}GiB "
        f"avail={baseline_avail/1024:.2f}GiB "
        f"swap={baseline_swap/1024:.2f}GiB",
        flush=True,
    )
    print(
        f"kill thresholds: used>{KILL_USED_GIB}GiB, "
        f"swap>{(baseline_swap/1024)+KILL_SWAP_DELTA_GIB:.2f}GiB, "
        f"kv>{KILL_KV_PCT}%",
        flush=True,
    )

    print(f"waiting for vLLM health (timeout {args.health_timeout}s)...", flush=True)
    wait_for_health(HEALTH_ENDPOINT, args.health_timeout)
    print("health OK", flush=True)

    model_id = discover_model_id(MODELS_ENDPOINT)
    print(f"model id: {model_id}", flush=True)

    per_turn_writer = CSVWriter(PER_TURN_CSV_PATH, PER_TURN_HEADER)
    memtrail_writer = CSVWriter(MEMTRAIL_CSV_PATH, MEMTRAIL_HEADER)
    transcript_writer = TranscriptWriter(
        TRANSCRIPT_PATH,
        SYSTEM_PROMPT,
        config={
            "turns_per_session": TURNS_PER_SESSION,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "num_sessions": NUM_SESSIONS,
        },
    )

    kill_state = KillState()
    monitor = Monitor(
        memtrail_writer,
        kill_state,
        baseline_swap,
        container_name=CONTAINER_NAME,
        monitor_interval_s=MONITOR_INTERVAL_S,
        heartbeat_interval_s=HEARTBEAT_INTERVAL_S,
        docker_stats_interval_s=DOCKER_STATS_INTERVAL_S,
        sample_ring_size=SAMPLE_RING_SIZE,
        kill_used_gib=KILL_USED_GIB,
        kill_swap_delta_gib=KILL_SWAP_DELTA_GIB,
        kill_kv_pct=KILL_KV_PCT,
    )
    monitor.start()
    time.sleep(MONITOR_INTERVAL_S * 2)  # seed last_sample

    crashed = False
    records: list[TurnRecord] = []

    try:
        records = run_ramp_coordinator(model_id, monitor, kill_state, per_turn_writer, transcript_writer)
        # Warm tail so the memtrail captures post-run decay
        if not kill_state.tripped:
            print(f"-- warm tail {WARM_TAIL_S}s --", flush=True)
            deadline = time.monotonic() + WARM_TAIL_S
            while time.monotonic() < deadline and not kill_state.tripped:
                time.sleep(min(0.5, deadline - time.monotonic()))
    except Exception as e:
        crashed = True
        print(f"!! driver crash: {e}", flush=True)
    finally:
        monitor.stop()
        monitor.join(timeout=5)
        per_turn_writer.close()
        memtrail_writer.close()
        transcript_writer.close()

    # ---- Summary ----
    samples_snapshot = monitor.all_samples_snapshot()

    print("", flush=True)
    print("=" * 72, flush=True)
    print("RUN 002d ENRICHED MULTI-TURN SUMMARY", flush=True)
    print("=" * 72, flush=True)
    print(
        f"baseline used: {baseline_used/1024:.2f} GiB  "
        f"baseline swap: {baseline_swap/1024:.2f} GiB  "
        f"pool: {pool_gib:.2f} GiB",
        flush=True,
    )
    print(
        f"peak used: {monitor.peak_used_mib/1024:.2f} GiB  "
        f"peak swap: {monitor.peak_swap_mib/1024:.2f} GiB  "
        f"peak kv: {monitor.peak_kv_pct:.1f}%",
        flush=True,
    )
    print("", flush=True)

    # Per-session summary
    sess = per_session_summary(records)
    print("Per-session:", flush=True)
    print(
        f"{'#':>3}  {'topic':<22} {'done':>6}  {'ttft1':>7} {'ttftN':>7}  "
        f"{'wall1':>7} {'wallN':>7}  {'p95_f4':>7} {'p95_l4':>7}  "
        f"{'prmptN':>7}  {'kvmax':>7}",
        flush=True,
    )
    for sid in sorted(sess.keys()):
        s = sess[sid]
        topic = SESSIONS[sid].SEED_TOPIC[:22]
        kv_str = f"{s['kv_max']:5.1f}%" if s["kv_max"] is not None else "  -- "
        print(
            f"S{sid+1:>2}  {topic:<22} "
            f"{s['ok_count']}/{TURNS_PER_SESSION:>2}  "
            f"{s['ttft_first']:>7.2f} {s['ttft_last']:>7.2f}  "
            f"{s['wall_first']:>7.2f} {s['wall_last']:>7.2f}  "
            f"{s['p95_first4']:>7.2f} {s['p95_last4']:>7.2f}  "
            f"{s['prompt_last']:>7}  {kv_str:>7}",
            flush=True,
        )
    print("", flush=True)

    # Ramp transitions
    transitions = find_ramp_transitions(samples_snapshot, RAMP_WINDOW_S)
    print("Ramp transitions (peak in %ds window after running->running+1):" % int(RAMP_WINDOW_S), flush=True)
    if transitions:
        for t in transitions:
            kv_str = f"{t.max_kv_pct:5.1f}%" if t.max_kv_pct is not None else "  -- "
            print(
                f"  {t.from_level}->{t.to_level}: "
                f"peak={t.peak_used_mib/1024:5.2f}GiB  "
                f"kv_max={kv_str}  samples={t.sample_count}",
                flush=True,
            )
    else:
        print("  (no transitions observed)", flush=True)
    print("", flush=True)

    # Per-running-level steady stats (from memtrail segmentation)
    level_stats = per_level_stats(samples_snapshot)
    print("Per-concurrency-level steady-state stats (from memtrail):", flush=True)
    print(
        f"  {'running':>8}  {'samples':>8}  {'max_used':>10}  "
        f"{'mean_used':>10}  {'max_kv':>8}",
        flush=True,
    )
    for level in sorted(level_stats.keys()):
        ls = level_stats[level]
        kv_str = f"{ls.max_kv_pct:5.1f}%" if ls.max_kv_pct is not None else "  -- "
        print(
            f"  {ls.level:>8}  {ls.sample_count:>8}  "
            f"{ls.max_used_mib/1024:>8.2f}GiB  "
            f"{ls.mean_used_mib/1024:>8.2f}GiB  "
            f"{kv_str:>8}",
            flush=True,
        )
    print("", flush=True)

    # Six-criterion evaluation
    passed, first_fail, lines = evaluate(
        records, monitor, kill_state, baseline_swap, pool_gib
    )
    print("Criteria:", flush=True)
    for line in lines:
        print(line, flush=True)
    print("", flush=True)

    if crashed:
        print("RUN 002d 4x48K ENRICHED: CRASH (driver exception)", flush=True)
        return EXIT_CRASH
    if kill_state.tripped:
        print(f"RUN 002d 4x48K ENRICHED: KILLED ({kill_state.reason})", flush=True)
        return EXIT_KILLED
    if passed:
        print("RUN 002d 4x48K ENRICHED: PASS", flush=True)
        return EXIT_OK
    print(f"RUN 002d 4x48K ENRICHED: FAIL (criterion #{first_fail})", flush=True)
    return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
