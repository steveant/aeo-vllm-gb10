#!/usr/bin/env python3
"""
Run 002c - ramped multi-turn high-fill characterization for the 4 x 48K vLLM
config.

Four independent multi-turn conversations, started in a staggered ramp:

    S1: [t1][t2][t3][t4][t5] ...            through turn 16
    S2:     [t1][t2][t3][t4] ...            through turn 16
    S3:         [t1][t2][t3] ...            through turn 16
    S4:             [t1][t2] ...            through turn 16
         ^    ^    ^
         |    |    |
         |    |    S3 joins (trigger: S2 finished turn 1)
         |    S2 joins (trigger: S1 finished turn 1)
         t=0  S1 alone - pays 0->1 JIT transition

After all four workers are running, there is no synchronization - each session
runs its own multi-turn loop independently and exits when it hits 16 turns. The
probabilistic temperature means sessions finish in unpredictable order.

Stdlib only. Caller is responsible for `docker compose up -d` before and
`docker compose down` after.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# ---------------------------------------------------------------------------
# Hard-coded config (per plan greedy-crafting-mochi.md - Run 002c)
# ---------------------------------------------------------------------------

VLLM_BASE_URL = "http://localhost:8000"
CHAT_ENDPOINT = f"{VLLM_BASE_URL}/v1/chat/completions"
MODELS_ENDPOINT = f"{VLLM_BASE_URL}/v1/models"
HEALTH_ENDPOINT = f"{VLLM_BASE_URL}/health"

CONTAINER_NAME = "docker-vllm-1"

PER_TURN_CSV_PATH = Path("/tmp/run_002c_per_turn.csv")
MEMTRAIL_CSV_PATH = Path("/tmp/run_002c_memtrail.csv")

MONITOR_INTERVAL_S = 0.5
HEARTBEAT_INTERVAL_S = 15.0
DOCKER_STATS_INTERVAL_S = 5.0
SAMPLE_RING_SIZE = 5000  # ~40 min at 0.5 s cadence

TURN_TIMEOUT_S = 900
WARM_TAIL_S = 30
TEMPERATURE = 0.7
MAX_TOKENS = 2500
TURNS_PER_SESSION = 16
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

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_KILLED = 2
EXIT_CRASH = 3


# ---------------------------------------------------------------------------
# Prompts - 4 topic-diverse seeds + 15 general-purpose follow-ups
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a senior engineer giving precise technical explanations with "
    "code or worked examples where relevant. Your answers should build on "
    "prior turns of this conversation."
)

SEED_PROMPTS: list[tuple[str, str]] = [
    (
        "Rust",
        "Explain the difference between Box<dyn Trait> and impl Trait in function "
        "returns. Cover the cases where each is appropriate, the implications for "
        "trait object safety, and what the compiler does differently in each case. "
        "Include code examples.",
    ),
    (
        "Postgres",
        "Walk me through how PostgreSQL's MVCC implementation handles a long-running "
        "SELECT against a table that is concurrently receiving heavy UPDATE traffic. "
        "Cover xmin/xmax, vacuum interaction, and the bloat consequences. Include "
        "the SQL views I'd query to inspect this.",
    ),
    (
        "Linear algebra",
        "Explain singular value decomposition geometrically. Why does every real "
        "matrix have one? What do the three components U, Sigma, V represent in "
        "terms of rotation, scaling, and reflection? Walk me through computing it "
        "for a 2x2 example by hand.",
    ),
    (
        "Distributed systems",
        "Describe the trade-offs between Raft and Paxos for a new control plane "
        "that needs to coordinate ~50 nodes with low write throughput but strong "
        "consistency. Cover leader election, log replication, and operational "
        "complexity for each.",
    ),
]

FOLLOW_UPS: list[str] = [
    "Deepen the most important insight from your previous answer. What's the "
    "failure mode that would bite someone who missed it?",
    "Challenge the two most load-bearing assumptions you just made and argue "
    "the counter-position for each.",
    "Translate the concept you just explained into production-quality code in "
    "whatever language or tooling is natural to the topic.",
    "Now apply this at 100x scale. What changes? What breaks first?",
    "Add observability: what would you instrument, and how would you know "
    "something is wrong before a user complains?",
    "Write the runbook entry an on-call engineer would need if this system "
    "failed at 3 AM on a holiday.",
    "Pick a less-common edge case you didn't address. Walk through how it "
    "interacts with what you've built so far.",
    "Refactor your previous response for clarity. Name what's redundant, "
    "what's missing, what's wrong.",
    "Add a testing strategy. What tests would verify the claims in your "
    "previous answer, and how would you structure them?",
    "Compare your approach to the obvious alternative. What are the "
    "trade-offs, and when would you pick the other one?",
    "Steelman the position of someone who disagrees with your previous "
    "answer, then respond to their strongest objection.",
    "Add error handling and failure modes. What can go wrong, and how does "
    "the system detect and respond?",
    "Operationalize this: what configuration, deployment, and ops concerns "
    "would a team inherit on day one?",
    "Write the one-paragraph executive summary. Your reader has 30 seconds.",
    "One thing you'd change about your previous answer if you had to defend "
    "it in a code review - what and why?",
]

assert len(FOLLOW_UPS) >= TURNS_PER_SESSION - 1, (
    f"need at least {TURNS_PER_SESSION - 1} follow-ups, have {len(FOLLOW_UPS)}"
)


# ---------------------------------------------------------------------------
# Sample / parsing utilities (verbatim from run_002b_sequential.py)
# ---------------------------------------------------------------------------

VLLM_LOG_PATTERN = re.compile(
    r"Avg prompt throughput:\s*([\d.]+)\s*tokens/s.*?"
    r"Avg generation throughput:\s*([\d.]+)\s*tokens/s.*?"
    r"Running:\s*(\d+)\s*reqs.*?"
    r"Waiting:\s*(\d+)\s*reqs.*?"
    r"GPU KV cache usage:\s*([\d.]+)%"
    r"(?:.*?Prefix cache hit rate:\s*([\d.]+)%)?",
    re.DOTALL,
)

POOL_GIB_PATTERN = re.compile(r"Available KV cache memory:\s*([\d.]+)\s*GiB")


def parse_free_m(text: str) -> tuple[int, int, int]:
    used = avail = swap_used = -1
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "Mem:" and len(parts) >= 7:
            used = int(parts[2])
            avail = int(parts[6])
        elif parts[0] == "Swap:" and len(parts) >= 4:
            swap_used = int(parts[2])
    if used < 0 or avail < 0 or swap_used < 0:
        raise RuntimeError(f"could not parse free -m output: {text!r}")
    return used, avail, swap_used


def read_free_m() -> tuple[int, int, int]:
    out = subprocess.run(
        ["free", "-m"], capture_output=True, text=True, check=True
    ).stdout
    return parse_free_m(out)


def read_docker_stats_rss_mib() -> Optional[int]:
    try:
        out = subprocess.run(
            [
                "docker",
                "stats",
                "--no-stream",
                "--format",
                "{{.MemUsage}}",
                CONTAINER_NAME,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None
    if not out:
        return None
    first = out.split("/")[0].strip()
    return parse_size_to_mib(first)


def parse_size_to_mib(s: str) -> Optional[int]:
    s = s.strip()
    m = re.match(r"^([\d.]+)\s*([KMGT]i?B)$", s)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2)
    factors = {
        "B": 1 / (1024 * 1024),
        "KB": 1 / 1024,
        "KiB": 1 / 1024,
        "MB": 1.0,
        "MiB": 1.0,
        "GB": 1024.0,
        "GiB": 1024.0,
        "TB": 1024.0 * 1024,
        "TiB": 1024.0 * 1024,
    }
    return int(val * factors.get(unit, 1.0))


def read_latest_vllm_log_match() -> Optional[dict]:
    try:
        out = subprocess.run(
            ["docker", "logs", "--tail", "300", CONTAINER_NAME],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    blob = (out.stdout or "") + "\n" + (out.stderr or "")
    for line in reversed(blob.splitlines()):
        m = VLLM_LOG_PATTERN.search(line)
        if m:
            return {
                "prompt_tps": float(m.group(1)),
                "gen_tps": float(m.group(2)),
                "running": int(m.group(3)),
                "waiting": int(m.group(4)),
                "kv_pct": float(m.group(5)),
                "prefix_hit_pct": float(m.group(6)) if m.group(6) else None,
            }
    return None


def read_pool_gib() -> Optional[float]:
    try:
        out = subprocess.run(
            ["docker", "logs", CONTAINER_NAME],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    blob = (out.stdout or "") + "\n" + (out.stderr or "")
    for line in blob.splitlines():
        m = POOL_GIB_PATTERN.search(line)
        if m:
            return float(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    ts: float
    used_mib: int
    available_mib: int
    swap_mib: int
    container_rss_mib: Optional[int]
    running: Optional[int]
    waiting: Optional[int]
    kv_pct: Optional[float]
    prefix_hit_pct: Optional[float]
    prompt_tps: Optional[float]
    gen_tps: Optional[float]


@dataclass
class KillState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    tripped: bool = False
    reason: str = ""

    def trip(self, reason: str) -> None:
        with self.lock:
            if not self.tripped:
                self.tripped = True
                self.reason = reason
                print(f"!! KILL SWITCH TRIPPED: {reason}", flush=True)


class CSVWriter:
    def __init__(self, path: Path, header: list[str]):
        self.path = path
        self.lock = threading.Lock()
        self.fh = path.open("w", newline="")
        self.writer = csv.DictWriter(self.fh, fieldnames=header)
        self.writer.writeheader()
        self.fh.flush()

    def write(self, row: dict) -> None:
        with self.lock:
            self.writer.writerow(row)
            self.fh.flush()

    def close(self) -> None:
        with self.lock:
            try:
                self.fh.flush()
                self.fh.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Monitor thread
# ---------------------------------------------------------------------------

class Monitor(threading.Thread):
    def __init__(
        self,
        csv_writer: CSVWriter,
        kill_state: KillState,
        swap_baseline_mib: int,
    ):
        super().__init__(name="monitor", daemon=True)
        self.csv_writer = csv_writer
        self.kill_state = kill_state
        self.swap_baseline_mib = swap_baseline_mib
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.last_sample: Optional[Sample] = None
        self.samples: list[Sample] = []
        self.peak_used_mib: int = 0
        self.peak_swap_mib: int = swap_baseline_mib
        self.peak_kv_pct: float = 0.0
        self.tick: int = 0
        self._last_heartbeat = 0.0
        self._last_docker_stats = 0.0
        self._cached_rss: Optional[int] = None

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self._sample_once()
            except Exception as e:
                print(f"!! monitor sample error: {e}", flush=True)
            self.stop_event.wait(MONITOR_INTERVAL_S)

    def _sample_once(self) -> None:
        now = time.time()
        used, available, swap = read_free_m()
        log = read_latest_vllm_log_match()

        if now - self._last_docker_stats >= DOCKER_STATS_INTERVAL_S:
            self._cached_rss = read_docker_stats_rss_mib()
            self._last_docker_stats = now

        sample = Sample(
            ts=now,
            used_mib=used,
            available_mib=available,
            swap_mib=swap,
            container_rss_mib=self._cached_rss,
            running=log.get("running") if log else None,
            waiting=log.get("waiting") if log else None,
            kv_pct=log.get("kv_pct") if log else None,
            prefix_hit_pct=log.get("prefix_hit_pct") if log else None,
            prompt_tps=log.get("prompt_tps") if log else None,
            gen_tps=log.get("gen_tps") if log else None,
        )

        with self.lock:
            self.last_sample = sample
            self.samples.append(sample)
            if len(self.samples) > SAMPLE_RING_SIZE:
                del self.samples[: len(self.samples) - SAMPLE_RING_SIZE]
            self.tick += 1
            if sample.used_mib > self.peak_used_mib:
                self.peak_used_mib = sample.used_mib
            if sample.swap_mib > self.peak_swap_mib:
                self.peak_swap_mib = sample.swap_mib
            if sample.kv_pct is not None and sample.kv_pct > self.peak_kv_pct:
                self.peak_kv_pct = sample.kv_pct

        self.csv_writer.write(
            {
                "ts": iso(sample.ts),
                "used_mib": sample.used_mib,
                "available_mib": sample.available_mib,
                "swap_mib": sample.swap_mib,
                "container_rss_mib": sample.container_rss_mib if sample.container_rss_mib is not None else "",
                "running": sample.running if sample.running is not None else "",
                "waiting": sample.waiting if sample.waiting is not None else "",
                "kv_pct": sample.kv_pct if sample.kv_pct is not None else "",
                "prefix_hit_pct": sample.prefix_hit_pct if sample.prefix_hit_pct is not None else "",
                "prompt_tps": sample.prompt_tps if sample.prompt_tps is not None else "",
                "gen_tps": sample.gen_tps if sample.gen_tps is not None else "",
            }
        )

        if now - self._last_heartbeat >= HEARTBEAT_INTERVAL_S:
            self._last_heartbeat = now
            kv_str = f"{sample.kv_pct:5.1f}%" if sample.kv_pct is not None else "  -- "
            run_str = f"{sample.running}" if sample.running is not None else "-"
            print(
                f"[mon {iso(sample.ts)}] used={sample.used_mib/1024:5.1f}GiB "
                f"avail={sample.available_mib/1024:5.1f}GiB "
                f"swap={sample.swap_mib/1024:5.2f}GiB "
                f"kv={kv_str} run={run_str}",
                flush=True,
            )

        # Kill switch
        reasons = []
        if sample.used_mib / 1024.0 > KILL_USED_GIB:
            reasons.append(f"used {sample.used_mib/1024:.1f}GiB > {KILL_USED_GIB}")
        if (sample.swap_mib - self.swap_baseline_mib) / 1024.0 > KILL_SWAP_DELTA_GIB:
            reasons.append(
                f"swap {sample.swap_mib/1024:.2f}GiB > baseline+{KILL_SWAP_DELTA_GIB}"
            )
        if sample.kv_pct is not None and sample.kv_pct > KILL_KV_PCT:
            reasons.append(f"kv {sample.kv_pct:.1f}% > {KILL_KV_PCT}")
        if reasons:
            self.kill_state.trip("; ".join(reasons))

    def window_samples(self, start_ts: float, end_ts: float) -> list[Sample]:
        with self.lock:
            return [s for s in self.samples if start_ts <= s.ts <= end_ts]

    def window_max(
        self, start_ts: float, end_ts: float
    ) -> tuple[Optional[int], Optional[int], Optional[float], Optional[int], int]:
        """Return (max_used_mib, max_swap_mib, max_kv_pct, max_running, n) for [start,end]."""
        window = self.window_samples(start_ts, end_ts)
        if not window:
            return None, None, None, None, 0
        max_used = max(s.used_mib for s in window)
        max_swap = max(s.swap_mib for s in window)
        kv_values = [s.kv_pct for s in window if s.kv_pct is not None]
        max_kv = max(kv_values) if kv_values else None
        running_values = [s.running for s in window if s.running is not None]
        max_running = max(running_values) if running_values else None
        return max_used, max_swap, max_kv, max_running, len(window)

    def all_samples_snapshot(self) -> list[Sample]:
        with self.lock:
            return list(self.samples)

    def stop(self) -> None:
        self.stop_event.set()


# ---------------------------------------------------------------------------
# Streaming chat completion (verbatim from run_002b_sequential.py)
# ---------------------------------------------------------------------------

def chat_completion_streaming(
    model_id: str, messages: list[dict], max_tokens: int
) -> tuple[str, dict, float]:
    body = json.dumps(
        {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": TEMPERATURE,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode("utf-8")
    req = Request(
        CHAT_ENDPOINT,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )

    t0 = time.monotonic()
    ttft: Optional[float] = None
    content_parts: list[str] = []
    usage: dict = {}

    with urlopen(req, timeout=TURN_TIMEOUT_S) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                evt = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if "usage" in evt and evt["usage"]:
                usage = evt["usage"]
            choices = evt.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content") or ""
            if piece:
                if ttft is None:
                    ttft = time.monotonic() - t0
                content_parts.append(piece)

    if ttft is None:
        ttft = time.monotonic() - t0
    return "".join(content_parts), usage, ttft


def discover_model_id() -> str:
    req = Request(MODELS_ENDPOINT, headers={"Accept": "application/json"})
    with urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    models = data.get("data") or []
    if not models:
        raise RuntimeError(f"no models served: {data}")
    return models[0]["id"]


def wait_for_health(timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urlopen(Request(HEALTH_ENDPOINT), timeout=5) as resp:
                if resp.status == 200:
                    return
        except (HTTPError, URLError, TimeoutError, OSError):
            pass
        time.sleep(2)
    raise RuntimeError(f"vLLM health check did not pass within {timeout_s}s")


# ---------------------------------------------------------------------------
# Ramped worker
# ---------------------------------------------------------------------------

def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="milliseconds")


PER_TURN_HEADER = [
    "ts",
    "session_id",
    "turn_index",
    "status",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "wall_latency_s",
    "ttft_s",
    "running_max_during",
    "kv_pct_max_during",
    "mem_used_max_during",
    "swap_max_during",
    "ramp_event",
]


@dataclass
class TurnRecord:
    session_id: int
    turn_index: int
    status: str
    wall: float
    ttft: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    start_ts: float
    end_ts: float
    mem_used_max_during: Optional[int]
    kv_pct_max_during: Optional[float]
    running_max_during: Optional[int]
    ramp_event: str


class RampedWorker(threading.Thread):
    def __init__(
        self,
        session_id: int,
        seed_topic: str,
        seed_prompt: str,
        model_id: str,
        kill_state: KillState,
        monitor: Monitor,
        turns_writer: CSVWriter,
        turns_lock: threading.Lock,
        turn_records: list[TurnRecord],
        start_event: threading.Event,
        next_start_event: Optional[threading.Event],
    ):
        super().__init__(name=f"worker-s{session_id+1}", daemon=True)
        self.session_id = session_id
        self.seed_topic = seed_topic
        self.seed_prompt = seed_prompt
        self.model_id = model_id
        self.kill_state = kill_state
        self.monitor = monitor
        self.turns_writer = turns_writer
        self.turns_lock = turns_lock
        self.turn_records = turn_records
        self.start_event = start_event
        self.next_start_event = next_start_event
        self.history: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def _user_msg_for_turn(self, turn_index: int) -> str:
        if turn_index == 0:
            return self.seed_prompt
        follow_idx = (self.session_id + turn_index - 1) % len(FOLLOW_UPS)
        return FOLLOW_UPS[follow_idx]

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
                messages = list(self.history)

                start_ts = time.time()
                t0 = time.monotonic()
                content = ""
                usage: dict = {}
                try:
                    content, usage, ttft = chat_completion_streaming(
                        self.model_id, messages, MAX_TOKENS
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
) -> list[TurnRecord]:
    turns_lock = threading.Lock()
    turn_records: list[TurnRecord] = []

    # Linked chain of start_events: w0 pre-set, each worker holds next's event
    start_events = [threading.Event() for _ in range(NUM_SESSIONS)]

    workers: list[RampedWorker] = []
    for i in range(NUM_SESSIONS):
        topic, prompt = SEED_PROMPTS[i % len(SEED_PROMPTS)]
        next_event = start_events[i + 1] if i + 1 < NUM_SESSIONS else None
        workers.append(
            RampedWorker(
                session_id=i,
                seed_topic=topic,
                seed_prompt=prompt,
                model_id=model_id,
                kill_state=kill_state,
                monitor=monitor,
                turns_writer=turns_writer,
                turns_lock=turns_lock,
                turn_records=turn_records,
                start_event=start_events[i],
                next_start_event=next_event,
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

def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


@dataclass
class RampTransition:
    from_level: int
    to_level: int
    ts: float
    peak_used_mib: int
    max_kv_pct: Optional[float]
    sample_count: int


def find_ramp_transitions(samples: list[Sample]) -> list[RampTransition]:
    """Walk samples in time order. For each 1->2, 2->3, 3->4 transition on the
    running counter, capture the peak used_mib in the RAMP_WINDOW_S window
    starting at that transition."""
    transitions: list[RampTransition] = []
    seen_levels: set[tuple[int, int]] = set()
    ordered = sorted(samples, key=lambda s: s.ts)

    prev_running: Optional[int] = None
    for i, s in enumerate(ordered):
        if s.running is None:
            continue
        if prev_running is not None and s.running == prev_running + 1 and 1 <= prev_running <= 3:
            key = (prev_running, s.running)
            if key not in seen_levels:
                seen_levels.add(key)
                start_ts = s.ts
                end_ts = start_ts + RAMP_WINDOW_S
                window = [w for w in ordered[i:] if w.ts <= end_ts]
                if window:
                    peak = max(w.used_mib for w in window)
                    kv_values = [w.kv_pct for w in window if w.kv_pct is not None]
                    max_kv = max(kv_values) if kv_values else None
                    transitions.append(
                        RampTransition(
                            from_level=prev_running,
                            to_level=s.running,
                            ts=start_ts,
                            peak_used_mib=peak,
                            max_kv_pct=max_kv,
                            sample_count=len(window),
                        )
                    )
        prev_running = s.running

    return transitions


@dataclass
class LevelStats:
    level: int
    sample_count: int
    max_used_mib: int
    mean_used_mib: float
    max_kv_pct: Optional[float]
    mean_kv_pct: Optional[float]


def per_level_stats(samples: list[Sample]) -> dict[int, LevelStats]:
    """Segment samples by running value and compute per-level stats."""
    by_level: dict[int, list[Sample]] = defaultdict(list)
    for s in samples:
        if s.running is not None:
            by_level[s.running].append(s)
    stats: dict[int, LevelStats] = {}
    for level, group in by_level.items():
        used_vals = [g.used_mib for g in group]
        kv_vals = [g.kv_pct for g in group if g.kv_pct is not None]
        stats[level] = LevelStats(
            level=level,
            sample_count=len(group),
            max_used_mib=max(used_vals),
            mean_used_mib=statistics.fmean(used_vals),
            max_kv_pct=max(kv_vals) if kv_vals else None,
            mean_kv_pct=statistics.fmean(kv_vals) if kv_vals else None,
        )
    return stats


def per_session_summary(records: list[TurnRecord]) -> dict[int, dict]:
    by_session: dict[int, list[TurnRecord]] = defaultdict(list)
    for r in records:
        by_session[r.session_id].append(r)
    out: dict[int, dict] = {}
    for sid, rs in by_session.items():
        rs_sorted = sorted(rs, key=lambda r: r.turn_index)
        ok_turns = [r for r in rs_sorted if r.status == "ok"]
        first_four = [r for r in ok_turns[:4]]
        last_four = [r for r in ok_turns[-4:]]
        out[sid] = {
            "count": len(rs_sorted),
            "ok_count": len(ok_turns),
            "ttft_first": ok_turns[0].ttft if ok_turns else float("nan"),
            "ttft_last": ok_turns[-1].ttft if ok_turns else float("nan"),
            "wall_first": ok_turns[0].wall if ok_turns else float("nan"),
            "wall_last": ok_turns[-1].wall if ok_turns else float("nan"),
            "p95_first4": pct([r.wall for r in first_four], 95) if first_four else 0.0,
            "p95_last4": pct([r.wall for r in last_four], 95) if last_four else 0.0,
            "prompt_last": ok_turns[-1].prompt_tokens if ok_turns else 0,
            "kv_max": max(
                (r.kv_pct_max_during for r in ok_turns if r.kv_pct_max_during is not None),
                default=None,
            ),
        }
    return out


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

    print("== Run 002c driver (ramped multi-turn high-fill) ==", flush=True)

    pool_gib = args.pool_gib if args.pool_gib is not None else read_pool_gib()
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
    wait_for_health(args.health_timeout)
    print("health OK", flush=True)

    model_id = discover_model_id()
    print(f"model id: {model_id}", flush=True)

    per_turn_writer = CSVWriter(PER_TURN_CSV_PATH, PER_TURN_HEADER)
    memtrail_writer = CSVWriter(
        MEMTRAIL_CSV_PATH,
        [
            "ts",
            "used_mib",
            "available_mib",
            "swap_mib",
            "container_rss_mib",
            "running",
            "waiting",
            "kv_pct",
            "prefix_hit_pct",
            "prompt_tps",
            "gen_tps",
        ],
    )

    kill_state = KillState()
    monitor = Monitor(memtrail_writer, kill_state, baseline_swap)
    monitor.start()
    time.sleep(MONITOR_INTERVAL_S * 2)  # seed last_sample

    crashed = False
    records: list[TurnRecord] = []

    try:
        records = run_ramp_coordinator(model_id, monitor, kill_state, per_turn_writer)
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

    # ---- Summary ----
    samples_snapshot = monitor.all_samples_snapshot()

    print("", flush=True)
    print("=" * 72, flush=True)
    print("RUN 002c RAMPED MULTI-TURN HIGH-FILL SUMMARY", flush=True)
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
        topic = SEED_PROMPTS[sid % len(SEED_PROMPTS)][0]
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
    transitions = find_ramp_transitions(samples_snapshot)
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
        print("RUN 002c 4x48K RAMPED: CRASH (driver exception)", flush=True)
        return EXIT_CRASH
    if kill_state.tripped:
        print(f"RUN 002c 4x48K RAMPED: KILLED ({kill_state.reason})", flush=True)
        return EXIT_KILLED
    if passed:
        print("RUN 002c 4x48K RAMPED: PASS", flush=True)
        return EXIT_OK
    print(f"RUN 002c 4x48K RAMPED: FAIL (criterion #{first_fail})", flush=True)
    return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
