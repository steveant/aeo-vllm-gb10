#!/usr/bin/env python3
"""
Run 002 - 4 x 48K vLLM load test driver.

Three escalating phases (A -> idle -> B -> idle -> C) in one container
lifetime, with 2 s memory monitor, kill switch, per-turn CSV, and final
6-criterion verdict. Phase parameters are hard-coded per plan; --phase
selects a subset for partial / dry runs.

Stdlib only.
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
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# ---------------------------------------------------------------------------
# Hard-coded config (per plan)
# ---------------------------------------------------------------------------

VLLM_BASE_URL = "http://localhost:8000"
CHAT_ENDPOINT = f"{VLLM_BASE_URL}/v1/chat/completions"
MODELS_ENDPOINT = f"{VLLM_BASE_URL}/v1/models"
HEALTH_ENDPOINT = f"{VLLM_BASE_URL}/health"

CONTAINER_NAME = "docker-vllm-1"

TURNS_CSV_PATH = Path("/tmp/run_002_turns.csv")
MEMTRAIL_CSV_PATH = Path("/tmp/run_002_memtrail.csv")

TURN_TIMEOUT_S = 900
MONITOR_INTERVAL_S = 2.0
HEARTBEAT_INTERVAL_S = 30.0
DOCKER_STATS_INTERVAL_S = 10.0
PHASE_IDLE_S = 60
TEMPERATURE = 0.7

KILL_USED_GIB = 78.0
KILL_SWAP_DELTA_GIB = 1.5
KILL_KV_PCT = 95.0

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_KILLED = 2
EXIT_CRASH = 3


@dataclass(frozen=True)
class PhaseConfig:
    name: str
    sessions: int
    turns: int
    max_tokens: int


PHASES: list[PhaseConfig] = [
    PhaseConfig("A", 4, 4, 1500),
    PhaseConfig("B", 4, 10, 2000),
    PhaseConfig("C", 4, 16, 2500),
]


# ---------------------------------------------------------------------------
# Worker topic seeds (4 distinct topics, prefix-cache hostile)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Topic:
    system: str
    first_user: str
    follow_ups: list[str]  # must contain >= 15 entries (Phase C uses 15)


TOPICS: list[Topic] = [
    Topic(
        system="You are a Rust expert helping debug a real async lifetime issue. Be precise and use complete code examples.",
        first_user=(
            "I have a struct that holds a tokio::sync::Mutex<Vec<MyData>>. I need to spawn a "
            "task that borrows it for ~30 s while the main loop also reads it. Walk me through "
            "the lifetime constraints I will hit and how to structure the spawn so the borrow "
            "checker accepts it. Use complete code examples."
        ),
        follow_ups=[
            "Show the version using Arc<Mutex<...>> instead. Explain every line of the .clone() and .lock().await sequence.",
            "Now I want the spawned task to also write back to the vec. Walk through the deadlock cases with the same Arc<Mutex>.",
            "Replace the Mutex with tokio::sync::RwLock and re-derive the example. Compare read/write contention end to end.",
            "Add a graceful shutdown signal using tokio::sync::watch. Show the full main + worker code.",
            "Refactor that example into a struct that owns the runtime and exposes a typed handle. Walk through the trait bounds.",
            "Add error propagation via anyhow::Result. Explain how the ? operator interacts with the spawn closure return type.",
            "Convert the example to use tracing for structured logging. Show the spans you add and why.",
            "Write a tokio test that exercises the deadlock case from earlier and asserts your fix prevents it.",
            "Profile this with tokio-console - what counters would tell me my structure is inefficient under load?",
            "Take everything we built and write a README.md explaining the architecture decisions.",
            "Add a criterion benchmark comparing Mutex vs RwLock under 1000 contended readers.",
            "Walk me through a worst-case scenario where the mutex is held across .await. Show the symptom and the fix.",
            "Refactor the worker to use a bounded mpsc channel instead of shared state. Explain the trade-offs.",
            "Add backpressure to that channel. Explain how the sender behaves when full.",
            "Convert the pipeline to a custom Future that polls the receiver directly. Show the impl Future block in detail.",
        ],
    ),
    Topic(
        system="You are a Go database internals expert teaching B-tree implementation. Production-quality code only.",
        first_user=(
            "Walk me through implementing a copy-on-write B-tree in Go from scratch. Start with "
            "the node layout in memory and the reasons each field exists. I want production-quality "
            "code, not a toy."
        ),
        follow_ups=[
            "Show the insert path with node split logic written out fully, including the parent update.",
            "Add deletion with merge logic. Walk through the underflow case and the rebalance.",
            "Convert it to be persistent on disk using mmap. Walk through the page header layout.",
            "Add a freelist so reclaimed pages get reused. Show the data structure and the allocation path.",
            "Add transactions: readers see a consistent snapshot, writers serialize. Walk through the meta-page swap.",
            "Layer a cursor API on top. Show Seek, Next, Prev, and how they interact with concurrent writers.",
            "Add a bucket abstraction for namespacing - like BoltDB's buckets. Show the layout.",
            "Add iterators over a key range. Explain the cursor invalidation rules.",
            "Implement bulk loading from a sorted iterator - much faster than one-at-a-time inserts.",
            "Add CRC checksums to pages so we detect torn writes. Walk through verification on read.",
            "Add a freelist garbage collection sweep - pending vs free, txid-based.",
            "Walk through the recovery path if the process crashes mid-commit. Which invariants make this safe?",
            "Add compression for leaf-page values using snappy. Show the encode/decode integration points.",
            "Add prefix compression for keys within a leaf. Show the format and the lookup penalty.",
            "Write a fuzz test using Go's native fuzzer that hammers insert/delete/get and asserts the invariants.",
        ],
    ),
    Topic(
        system="You are a Python parser engineer who specializes in SQL grammars.",
        first_user=(
            "I want to write a Python parser for a useful subset of PostgreSQL DDL: CREATE TABLE, "
            "ALTER TABLE, CREATE INDEX. Walk me through the tokenizer first, with full code, and "
            "explain why each token type exists."
        ),
        follow_ups=[
            "Now implement the recursive descent parser for CREATE TABLE including PRIMARY KEY, FOREIGN KEY, CHECK, UNIQUE. Full code.",
            "Add support for column-level constraints inline as well as table-level. Walk through how the AST captures both.",
            "Add ALTER TABLE ADD/DROP/RENAME COLUMN. Show how you reuse the column-definition parser.",
            "Add CREATE INDEX with multi-column, expression indexes, and partial indexes (WHERE clauses).",
            "Add support for schemas (qualified names like schema.table). Walk through how the lexer and parser handle the dot.",
            "Add quoted identifiers with the embedded-double-quote escape rule. Show test cases.",
            "Add dollar-quoted strings ($$...$$ and $tag$...$tag$). Show how the lexer tracks the open tag.",
            "Add a pretty-printer that takes the AST and emits canonical SQL. Walk through the indent rules.",
            "Add a diff engine that takes two ASTs of the same table and produces ALTER TABLE statements to migrate.",
            "Add unit tests using pytest parameterize for every constraint type, with both positive and negative cases.",
            "Add a fuzzer using hypothesis that generates random valid CREATE TABLE statements and round-trips parse->print->parse.",
            "Add error recovery: when the parser hits an unexpected token, sync to the next semicolon and continue.",
            "Add line/column tracking to every token and AST node. Show how errors get reported with caret diagnostics.",
            "Add support for partitioning syntax (PARTITION BY RANGE/LIST/HASH). Walk through the AST representation.",
            "Package it as a pip-installable module with type stubs and a CLI entry point.",
        ],
    ),
    Topic(
        system="You are a senior TypeScript architect doing code review on an event-sourcing module.",
        first_user=(
            "I have a TypeScript event sourcing module with Aggregate, Event, Command, and "
            "Repository classes. The Aggregate replays events from the start each time you load "
            "it from the Repository. Review the architecture critically: what are the failure modes?"
        ),
        follow_ups=[
            "Show the snapshot strategy you would add. Include snapshot interval policy and load path.",
            "Walk me through the optimistic concurrency check on save. What goes wrong if I get it slightly wrong?",
            "Add an outbox table for publishing integration events. Walk through the transactional guarantee.",
            "Describe the consumer side: how a downstream service idempotently processes outbox events.",
            "Show the projection rebuild path. Which invariants must projections satisfy for replay to be deterministic?",
            "Walk through schema evolution for events: additive changes vs renames vs deletes. Code examples.",
            "Add upcasters for old event versions. Show the version chain logic.",
            "Address the monolith->microservice split. How do you keep the event store coherent when you break out a bounded context?",
            "Walk through a test pyramid: unit on the aggregate, integration on the repo, acceptance on the projection.",
            "Add property-based tests using fast-check that verify aggregate invariants across arbitrary command sequences.",
            "Implement a small saga/process manager that listens to events and emits commands. Show the state machine.",
            "Add a poison-message handler for the saga: failures get parked, alerted, replay-able.",
            "Walk through observability: metrics, traces, logs you add to the command/event/projection pipeline.",
            "Add a load-shedding strategy at the command handler when the event store is slow.",
            "Write a runbook for the on-call engineer when the projection lag alarm fires.",
        ],
    ),
]


# ---------------------------------------------------------------------------
# Sample / parsing utilities
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


def parse_free_m(text: str) -> tuple[int, int, int]:
    """Return (mem_used_mib, mem_available_mib, swap_used_mib)."""
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
    # e.g. "12.5GiB / 124GiB"
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
    """Tail the container log and return the most recent metrics line, parsed."""
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
    # Walk lines from newest backwards
    lines = blob.splitlines()
    for line in reversed(lines):
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

        # Heartbeat
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
                f"swap {sample.swap_mib/1024:.2f}GiB > baseline+{KILL_SWAP_DELTA_GIB} "
                f"(baseline {self.swap_baseline_mib/1024:.2f})"
            )
        if sample.kv_pct is not None and sample.kv_pct > KILL_KV_PCT:
            reasons.append(f"kv {sample.kv_pct:.1f}% > {KILL_KV_PCT}")
        if reasons:
            self.kill_state.trip("; ".join(reasons))

    def stop(self) -> None:
        self.stop_event.set()


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    phase: str
    session_id: int
    turn_index: int
    status: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    wall_latency_s: float
    sample_before: Optional[Sample]
    sample_after: Optional[Sample]


class Worker(threading.Thread):
    def __init__(
        self,
        session_id: int,
        topic: Topic,
        phase: PhaseConfig,
        model_id: str,
        kill_state: KillState,
        monitor: Monitor,
        turns_writer: CSVWriter,
        results: list[TurnResult],
        results_lock: threading.Lock,
    ):
        super().__init__(name=f"worker-{session_id}", daemon=True)
        self.session_id = session_id
        self.topic = topic
        self.phase = phase
        self.model_id = model_id
        self.kill_state = kill_state
        self.monitor = monitor
        self.turns_writer = turns_writer
        self.results = results
        self.results_lock = results_lock
        self.history: list[dict] = [
            {"role": "system", "content": topic.system}
        ]

    def _next_user_msg(self, turn_index: int) -> str:
        if turn_index == 0:
            return self.topic.first_user
        idx = turn_index - 1
        if idx < len(self.topic.follow_ups):
            return self.topic.follow_ups[idx]
        # Defensive — should never trigger because Phase C uses 16 turns and follow_ups has 15
        return f"Continue elaborating on point {turn_index}."

    def run(self) -> None:
        for turn_index in range(self.phase.turns):
            if self.kill_state.tripped:
                return
            user_msg = self._next_user_msg(turn_index)
            self.history.append({"role": "user", "content": user_msg})

            sample_before = self.monitor.last_sample
            t0 = time.monotonic()
            try:
                content, usage = chat_completion(
                    self.model_id, self.history, self.phase.max_tokens
                )
                wall = time.monotonic() - t0
                self.history.append({"role": "assistant", "content": content})
                status = "ok"
                prompt_tokens = int(usage.get("prompt_tokens", 0))
                completion_tokens = int(usage.get("completion_tokens", 0))
                total_tokens = int(usage.get("total_tokens", 0))
            except Exception as e:
                wall = time.monotonic() - t0
                status = f"error:{type(e).__name__}:{str(e)[:120]}"
                prompt_tokens = completion_tokens = total_tokens = 0

            sample_after = self.monitor.last_sample

            result = TurnResult(
                phase=self.phase.name,
                session_id=self.session_id,
                turn_index=turn_index,
                status=status,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                wall_latency_s=wall,
                sample_before=sample_before,
                sample_after=sample_after,
            )
            with self.results_lock:
                self.results.append(result)
            self.turns_writer.write(turn_row(result))

            if status != "ok":
                self.kill_state.trip(
                    f"worker {self.session_id} turn {turn_index} {status}"
                )
                return


def turn_row(r: TurnResult) -> dict:
    def s_used(s: Optional[Sample]) -> str:
        return str(s.used_mib) if s else ""

    def s_swap(s: Optional[Sample]) -> str:
        return str(s.swap_mib) if s else ""

    def s_kv(s: Optional[Sample]) -> str:
        return f"{s.kv_pct}" if (s and s.kv_pct is not None) else ""

    return {
        "ts": iso(time.time()),
        "phase": r.phase,
        "session_id": r.session_id,
        "turn_index": r.turn_index,
        "status": r.status,
        "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.completion_tokens,
        "total_tokens": r.total_tokens,
        "wall_latency_s": f"{r.wall_latency_s:.3f}",
        "mem_used_mib_before": s_used(r.sample_before),
        "mem_used_mib_after": s_used(r.sample_after),
        "swap_mib_before": s_swap(r.sample_before),
        "swap_mib_after": s_swap(r.sample_after),
        "kv_pct_before": s_kv(r.sample_before),
        "kv_pct_after": s_kv(r.sample_after),
    }


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def http_get_json(url: str, timeout: int = 10) -> dict:
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def chat_completion(
    model_id: str, messages: list[dict], max_tokens: int
) -> tuple[str, dict]:
    body = json.dumps(
        {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": TEMPERATURE,
            "stream": False,
        }
    ).encode("utf-8")
    req = Request(
        CHAT_ENDPOINT,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=TURN_TIMEOUT_S) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"empty choices: {data}")
    content = choices[0].get("message", {}).get("content", "")
    usage = data.get("usage") or {}
    return content, usage


def discover_model_id() -> str:
    data = http_get_json(MODELS_ENDPOINT)
    models = data.get("data") or []
    if not models:
        raise RuntimeError(f"no models served: {data}")
    return models[0]["id"]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="milliseconds")


def wait_for_health(timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            req = Request(HEALTH_ENDPOINT)
            with urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return
        except (HTTPError, URLError, TimeoutError, OSError):
            pass
        time.sleep(2)
    raise RuntimeError(f"vLLM health check did not pass within {timeout_s}s")


def run_phase(
    phase: PhaseConfig,
    model_id: str,
    monitor: Monitor,
    kill_state: KillState,
    turns_writer: CSVWriter,
    all_results: list[TurnResult],
    results_lock: threading.Lock,
) -> None:
    print(
        f"== Phase {phase.name}: {phase.sessions} sessions x {phase.turns} turns "
        f"x {phase.max_tokens} max_tokens ==",
        flush=True,
    )
    workers = [
        Worker(
            session_id=i,
            topic=TOPICS[i % len(TOPICS)],
            phase=phase,
            model_id=model_id,
            kill_state=kill_state,
            monitor=monitor,
            turns_writer=turns_writer,
            results=all_results,
            results_lock=results_lock,
        )
        for i in range(phase.sessions)
    ]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    print(f"== Phase {phase.name} done ==", flush=True)


def evaluate(
    all_results: list[TurnResult],
    monitor: Monitor,
    kill_state: KillState,
    swap_baseline_mib: int,
    pool_gib: float,
    selected_phases: list[PhaseConfig],
) -> tuple[bool, list[str]]:
    """Apply the six success criteria. Returns (passed, list of human-readable lines)."""
    lines: list[str] = []
    passed = True

    expected_turns = sum(p.sessions * p.turns for p in selected_phases)
    actual_turns = len(all_results)
    ok_turns = sum(1 for r in all_results if r.status == "ok")
    err_turns = actual_turns - ok_turns

    # Criterion 1 — completion rate 100%
    c1 = (actual_turns == expected_turns) and (err_turns == 0)
    lines.append(
        f"[{'PASS' if c1 else 'FAIL'}] #1 completion: "
        f"{ok_turns}/{expected_turns} ok, {err_turns} errors"
    )
    if not c1:
        passed = False

    # Criterion 2 — peak host memory < 78 GiB
    peak_used_gib = monitor.peak_used_mib / 1024.0
    c2 = peak_used_gib < KILL_USED_GIB
    lines.append(
        f"[{'PASS' if c2 else 'FAIL'}] #2 peak used:    "
        f"{peak_used_gib:.2f} GiB (< {KILL_USED_GIB} GiB)"
    )
    if not c2:
        passed = False

    # Criterion 3 — swap growth < 1.5 GiB
    swap_growth_gib = (monitor.peak_swap_mib - swap_baseline_mib) / 1024.0
    c3 = swap_growth_gib < KILL_SWAP_DELTA_GIB
    lines.append(
        f"[{'PASS' if c3 else 'FAIL'}] #3 swap growth:  "
        f"{swap_growth_gib:+.2f} GiB (< {KILL_SWAP_DELTA_GIB} GiB)"
    )
    if not c3:
        passed = False

    # Criterion 4 — p95 last vs first turn (per phase, last turn p95 vs first turn p95)
    by_phase: dict[str, dict[int, list[float]]] = {}
    for r in all_results:
        if r.status != "ok":
            continue
        by_phase.setdefault(r.phase, {}).setdefault(r.turn_index, []).append(r.wall_latency_s)
    drift_lines: list[str] = []
    c4 = True
    for phase in selected_phases:
        per_turn = by_phase.get(phase.name) or {}
        if 0 not in per_turn or (phase.turns - 1) not in per_turn:
            drift_lines.append(f"  {phase.name}: insufficient data")
            c4 = False
            continue
        first_p95 = pct(per_turn[0], 95)
        last_p95 = pct(per_turn[phase.turns - 1], 95)
        ratio = (last_p95 / first_p95) if first_p95 > 0 else float("inf")
        ok = ratio < 2.0
        if not ok:
            c4 = False
        drift_lines.append(
            f"  {phase.name}: first_p95={first_p95:.2f}s last_p95={last_p95:.2f}s ratio={ratio:.2f} {'ok' if ok else 'FAIL'}"
        )
    lines.append(f"[{'PASS' if c4 else 'FAIL'}] #4 latency drift (p95 last < 2x p95 first)")
    lines.extend(drift_lines)
    if not c4:
        passed = False

    # Criterion 5 — KV peak < 95
    peak_kv = monitor.peak_kv_pct
    c5 = peak_kv < KILL_KV_PCT
    lines.append(
        f"[{'PASS' if c5 else 'FAIL'}] #5 kv peak:      {peak_kv:.1f}% (< {KILL_KV_PCT}%)"
    )
    if not c5:
        passed = False

    # Criterion 6 — observed peak vs projected within +/- 3 GiB
    projected_peak_steady = 61.0 + pool_gib  # corrected steady-state model
    delta = peak_used_gib - projected_peak_steady
    c6 = abs(delta) <= 3.0
    lines.append(
        f"[{'PASS' if c6 else 'FAIL'}] #6 peak vs proj: "
        f"observed {peak_used_gib:.2f} GiB vs proj {projected_peak_steady:.2f} GiB "
        f"(delta {delta:+.2f}, |delta| <= 3.0)"
    )
    if not c6:
        passed = False

    return passed, lines


def pct(values: list[float], p: int) -> float:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["A", "B", "C", "all"], default="all")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--pool-gib",
        type=float,
        default=8.93,
        help="vLLM available KV pool GiB (Run 001 measured 8.93)",
    )
    ap.add_argument(
        "--health-timeout",
        type=int,
        default=900,
        help="seconds to wait for /health to return 200 before bailing",
    )
    args = ap.parse_args()

    if args.phase == "all":
        selected = list(PHASES)
    else:
        selected = [p for p in PHASES if p.name == args.phase]

    print(f"== Run 002 driver ==", flush=True)
    print(f"selected phases: {[p.name for p in selected]}", flush=True)
    print(f"dry-run: {args.dry_run}", flush=True)

    # Baseline state
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

    # Health gate
    print(f"waiting for vLLM health (timeout {args.health_timeout}s)...", flush=True)
    wait_for_health(args.health_timeout)
    print(f"health OK", flush=True)

    model_id = discover_model_id()
    print(f"model id: {model_id}", flush=True)

    # Quick log scrape sanity
    log = read_latest_vllm_log_match()
    print(f"latest log scrape: {log}", flush=True)

    # CSV writers
    turns_writer = CSVWriter(
        TURNS_CSV_PATH,
        [
            "ts",
            "phase",
            "session_id",
            "turn_index",
            "status",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "wall_latency_s",
            "mem_used_mib_before",
            "mem_used_mib_after",
            "swap_mib_before",
            "swap_mib_after",
            "kv_pct_before",
            "kv_pct_after",
        ],
    )
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
    # let monitor get one sample before workers start
    time.sleep(MONITOR_INTERVAL_S + 0.5)

    if args.dry_run:
        print("== DRY RUN ==", flush=True)
        # Send one tiny request to confirm chat endpoint works
        try:
            content, usage = chat_completion(
                model_id,
                [
                    {"role": "system", "content": "Reply with the single word OK."},
                    {"role": "user", "content": "go"},
                ],
                max_tokens=8,
            )
            print(f"dry-run chat ok: usage={usage} content={content!r}", flush=True)
        except Exception as e:
            print(f"!! dry-run chat failed: {e}", flush=True)
            monitor.stop()
            monitor.join(timeout=5)
            turns_writer.close()
            memtrail_writer.close()
            return EXIT_CRASH
        # Let monitor capture a few samples
        time.sleep(6)
        print(f"dry-run last sample: {monitor.last_sample}", flush=True)
        monitor.stop()
        monitor.join(timeout=5)
        turns_writer.close()
        memtrail_writer.close()
        print("== dry run complete ==", flush=True)
        return EXIT_OK

    all_results: list[TurnResult] = []
    results_lock = threading.Lock()
    crashed = False

    try:
        for i, phase in enumerate(selected):
            if kill_state.tripped:
                print(f"!! kill switch tripped, skipping remaining phases", flush=True)
                break
            run_phase(
                phase,
                model_id,
                monitor,
                kill_state,
                turns_writer,
                all_results,
                results_lock,
            )
            if kill_state.tripped:
                continue
            if i < len(selected) - 1:
                print(f"-- idle {PHASE_IDLE_S}s --", flush=True)
                slept = 0
                while slept < PHASE_IDLE_S and not kill_state.tripped:
                    time.sleep(min(2, PHASE_IDLE_S - slept))
                    slept += 2
    except Exception as e:
        crashed = True
        print(f"!! driver crash: {e}", flush=True)
    finally:
        monitor.stop()
        monitor.join(timeout=5)
        turns_writer.close()
        memtrail_writer.close()

    # Verdict
    print("", flush=True)
    print("=" * 60, flush=True)
    print("RUN 002 4x48K SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(
        f"baseline used: {baseline_used/1024:.2f} GiB  "
        f"baseline swap: {baseline_swap/1024:.2f} GiB",
        flush=True,
    )
    print(
        f"peak used: {monitor.peak_used_mib/1024:.2f} GiB  "
        f"peak swap: {monitor.peak_swap_mib/1024:.2f} GiB  "
        f"peak kv: {monitor.peak_kv_pct:.1f}%",
        flush=True,
    )
    print(f"turn count: {len(all_results)}", flush=True)
    print("", flush=True)
    passed, lines = evaluate(
        all_results, monitor, kill_state, baseline_swap, args.pool_gib, selected
    )
    for line in lines:
        print(line, flush=True)

    print("", flush=True)
    if crashed:
        print("RUN 002 4x48K: CRASH (driver exception)", flush=True)
        return EXIT_CRASH
    if kill_state.tripped:
        print(f"RUN 002 4x48K: KILLED ({kill_state.reason})", flush=True)
        return EXIT_KILLED
    if passed:
        print("RUN 002 4x48K: PASS", flush=True)
        return EXIT_OK
    print("RUN 002 4x48K: FAIL", flush=True)
    return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
