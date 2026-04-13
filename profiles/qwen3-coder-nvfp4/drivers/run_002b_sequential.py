#!/usr/bin/env python3
"""
Run 002b - sequential warmup characterization for the 4 x 48K vLLM config.

Sends four single-threaded, non-overlapping chat requests against an already-
running vLLM container. Each request uses streaming so we can capture TTFT
(time to first token) as a proxy for JIT-compile events. A 0.5 s memory
monitor runs the entire window. Idle gaps between requests give the engine a
chance to release transient working set.

Goal: classify the first-batch warmup transient observed in Run 002 as one of

  A - once per engine lifetime  (spike before req 1 only)
  B - per request               (spike before every req)
  C - per batch shape           (big spike on req 1, smaller on later reqs)
  D - no spike                  (Run 002 spike was situational)

Stdlib only. Driver is intentionally agnostic about container lifecycle - the
caller is responsible for `docker compose up -d` before and `docker compose
down` after.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# ---------------------------------------------------------------------------
# Hard-coded config (per plan greedy-crafting-mochi.md)
# ---------------------------------------------------------------------------

VLLM_BASE_URL = "http://localhost:8000"
CHAT_ENDPOINT = f"{VLLM_BASE_URL}/v1/chat/completions"
MODELS_ENDPOINT = f"{VLLM_BASE_URL}/v1/models"
HEALTH_ENDPOINT = f"{VLLM_BASE_URL}/health"

CONTAINER_NAME = "docker-vllm-1"

PER_REQUEST_CSV_PATH = Path("/tmp/run_002b_per_request.csv")
MEMTRAIL_CSV_PATH = Path("/tmp/run_002b_memtrail.csv")

# Tighter sampling than Run 002 - we want sub-second resolution around request
# boundaries to distinguish "spike at submit" from "spike at first-token" from
# "spike at completion".
MONITOR_INTERVAL_S = 0.5
HEARTBEAT_INTERVAL_S = 10.0
DOCKER_STATS_INTERVAL_S = 5.0
SAMPLE_RING_SIZE = 1500  # ~12.5 min of 0.5 s samples

TURN_TIMEOUT_S = 600
COLD_IDLE_S = 60
IDLE_GAP_S = 30
WARM_TAIL_S = 60
TEMPERATURE = 0.7
MAX_TOKENS = 1500

# Cap raised to 90 GiB for this run; kill switch 2 GiB under cap.
KILL_USED_GIB = 88.0
KILL_SWAP_DELTA_GIB = 1.5
KILL_KV_PCT = 95.0

# For the spike classifier - peak_steady is the corrected Run 001 model.
# Pool is read from vLLM's startup log; if not found we fall back to 9.91 GiB.
DEFAULT_POOL_GIB = 9.91
PEAK_STEADY_CONST = 61.0
SPIKE_DELTA_GIB = 3.0  # how far above peak_steady counts as a "spike"

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_KILLED = 2
EXIT_CRASH = 3


# ---------------------------------------------------------------------------
# Prompt set (different topics, similar lengths, defeats prefix cache)
# ---------------------------------------------------------------------------

PROMPTS: list[tuple[str, str]] = [
    (
        "Rust",
        "Explain the difference between Box<dyn Trait> and impl Trait in function returns. "
        "Cover the cases where each is appropriate, the implications for trait object safety, "
        "and what the compiler does differently in each case. Include code examples.",
    ),
    (
        "Postgres",
        "Walk me through how PostgreSQL's MVCC implementation handles a long-running SELECT "
        "against a table that is concurrently receiving heavy UPDATE traffic. Cover xmin/xmax, "
        "vacuum interaction, and the bloat consequences. Include the SQL views I'd query to "
        "inspect this.",
    ),
    (
        "Linear algebra",
        "Explain singular value decomposition geometrically. Why does every real matrix have "
        "one? What do the three components U, Sigma, V represent in terms of rotation, scaling, "
        "and reflection? Walk me through computing it for a 2x2 example by hand.",
    ),
    (
        "Distributed systems",
        "Describe the trade-offs between Raft and Paxos for a new control plane that needs to "
        "coordinate ~50 nodes with low write throughput but strong consistency. Cover leader "
        "election, log replication, and operational complexity for each.",
    ),
]

SYSTEM_PROMPT = "You are a senior engineer giving precise technical explanations with code or worked examples where relevant."


# ---------------------------------------------------------------------------
# Sample / parsing utilities (verbatim from Run 002 driver where applicable)
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

POOL_GIB_PATTERN = re.compile(
    r"Available KV cache memory:\s*([\d.]+)\s*GiB"
)


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
            ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", CONTAINER_NAME],
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
        self.samples: list[Sample] = []  # bounded ring of recent samples
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

    def window_max(
        self, start_ts: float, end_ts: float
    ) -> tuple[Optional[int], Optional[int], Optional[float], int]:
        """Return (max_used_mib, max_swap_mib, max_kv_pct, sample_count) for ts in [start, end]."""
        with self.lock:
            window = [s for s in self.samples if start_ts <= s.ts <= end_ts]
        if not window:
            return None, None, None, 0
        max_used = max(s.used_mib for s in window)
        max_swap = max(s.swap_mib for s in window)
        kv_values = [s.kv_pct for s in window if s.kv_pct is not None]
        max_kv = max(kv_values) if kv_values else None
        return max_used, max_swap, max_kv, len(window)

    def stop(self) -> None:
        self.stop_event.set()


# ---------------------------------------------------------------------------
# Streaming chat completion (captures TTFT)
# ---------------------------------------------------------------------------

def chat_completion_streaming(
    model_id: str, messages: list[dict], max_tokens: int
) -> tuple[str, dict, float]:
    """POST a chat-completions request with stream=true.

    Returns (full_content, usage_dict, ttft_seconds).

    TTFT is measured as the time from immediately-before-send to the first SSE
    'data:' event whose delta contains non-empty content. Reasoning steps that
    yield empty content are skipped on purpose - we want first *visible* token.
    """
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
        # response had no visible content - record total wall as fallback
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
# Driver
# ---------------------------------------------------------------------------

def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="milliseconds")


PER_REQUEST_HEADER = [
    "ts",
    "row_kind",          # boundary | request
    "label",             # e.g. "cold idle start", "request 1", "idle gap 2 end"
    "request_index",     # 0-3 for requests, empty otherwise
    "topic",
    "status",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "wall_latency_s",
    "ttft_s",
    "mem_used_max_during",
    "swap_max_during",
    "kv_pct_max_during",
    "monitor_samples_during",
]


def write_boundary(
    writer: CSVWriter,
    label: str,
    monitor: Monitor,
    start_ts: float,
    end_ts: float,
) -> None:
    max_used, max_swap, max_kv, n = monitor.window_max(start_ts, end_ts)
    writer.write(
        {
            "ts": iso(end_ts),
            "row_kind": "boundary",
            "label": label,
            "request_index": "",
            "topic": "",
            "status": "",
            "prompt_tokens": "",
            "completion_tokens": "",
            "total_tokens": "",
            "wall_latency_s": f"{end_ts - start_ts:.3f}",
            "ttft_s": "",
            "mem_used_max_during": max_used if max_used is not None else "",
            "swap_max_during": max_swap if max_swap is not None else "",
            "kv_pct_max_during": max_kv if max_kv is not None else "",
            "monitor_samples_during": n,
        }
    )


def write_request(
    writer: CSVWriter,
    request_index: int,
    topic: str,
    status: str,
    usage: dict,
    wall: float,
    ttft: float,
    monitor: Monitor,
    start_ts: float,
    end_ts: float,
) -> None:
    max_used, max_swap, max_kv, n = monitor.window_max(start_ts, end_ts)
    writer.write(
        {
            "ts": iso(end_ts),
            "row_kind": "request",
            "label": f"request {request_index + 1}",
            "request_index": request_index,
            "topic": topic,
            "status": status,
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
            "wall_latency_s": f"{wall:.3f}",
            "ttft_s": f"{ttft:.3f}",
            "mem_used_max_during": max_used if max_used is not None else "",
            "swap_max_during": max_swap if max_swap is not None else "",
            "kv_pct_max_during": max_kv if max_kv is not None else "",
            "monitor_samples_during": n,
        }
    )


def quiet_sleep(label: str, seconds: float, kill_state: KillState) -> None:
    """Sleep, but bail early if the kill switch trips."""
    print(f"-- {label}: {int(seconds)}s --", flush=True)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if kill_state.tripped:
            return
        time.sleep(min(0.5, deadline - time.monotonic()))


def classify(per_request_peaks_gib: list[float], spike_threshold_gib: float) -> tuple[str, str]:
    """Return (verdict, human description)."""
    spikes = [
        i for i, peak in enumerate(per_request_peaks_gib)
        if peak > spike_threshold_gib
    ]
    n = len(spikes)
    if n == 0:
        return (
            "D",
            "no spike on any request - the Run 002 transient was situational",
        )
    if n == 1 and spikes[0] == 0:
        return (
            "A",
            "spike before request 1 only - warmup is once per engine lifetime",
        )
    if n >= 3:
        return (
            "B",
            "spike before every request - per-request setup work",
        )
    # n == 2 or weird ordering
    return (
        "C",
        f"spikes on requests {[i+1 for i in spikes]} - "
        "looks per-batch-shape; check whether magnitudes shrink",
    )


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

    print("== Run 002b driver ==", flush=True)

    pool_gib = args.pool_gib if args.pool_gib is not None else read_pool_gib()
    if pool_gib is None:
        pool_gib = DEFAULT_POOL_GIB
        print(f"WARN: could not read pool from logs, using default {pool_gib} GiB", flush=True)
    else:
        print(f"detected pool: {pool_gib:.2f} GiB", flush=True)

    peak_steady_proj = PEAK_STEADY_CONST + pool_gib
    spike_threshold_gib = peak_steady_proj + SPIKE_DELTA_GIB
    print(
        f"projected peak_steady = {peak_steady_proj:.2f} GiB, "
        f"spike threshold = {spike_threshold_gib:.2f} GiB",
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

    per_request_writer = CSVWriter(PER_REQUEST_CSV_PATH, PER_REQUEST_HEADER)
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
    time.sleep(MONITOR_INTERVAL_S * 2)  # let monitor seed last_sample

    crashed = False
    per_request_peaks_gib: list[float] = []
    request_results: list[dict] = []

    try:
        # ---- Cold idle window ----
        cold_idle_start = time.time()
        quiet_sleep("cold idle", COLD_IDLE_S, kill_state)
        cold_idle_end = time.time()
        write_boundary(per_request_writer, "cold idle", monitor, cold_idle_start, cold_idle_end)
        if kill_state.tripped:
            raise RuntimeError("killed during cold idle")

        # ---- Four sequential requests with idle gaps ----
        for i, (topic, prompt) in enumerate(PROMPTS):
            req_start = time.time()
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            t0 = time.monotonic()
            try:
                content, usage, ttft = chat_completion_streaming(
                    model_id, messages, MAX_TOKENS
                )
                wall = time.monotonic() - t0
                status = "ok"
            except Exception as e:
                wall = time.monotonic() - t0
                ttft = -1.0
                content = ""
                usage = {}
                status = f"error:{type(e).__name__}:{str(e)[:120]}"

            req_end = time.time()
            write_request(
                per_request_writer,
                request_index=i,
                topic=topic,
                status=status,
                usage=usage,
                wall=wall,
                ttft=ttft,
                monitor=monitor,
                start_ts=req_start,
                end_ts=req_end,
            )
            max_used_mib, _, _, _ = monitor.window_max(req_start, req_end)
            peak_gib = (max_used_mib / 1024.0) if max_used_mib else float("nan")
            per_request_peaks_gib.append(peak_gib)
            request_results.append(
                {
                    "topic": topic,
                    "status": status,
                    "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                    "completion_tokens": int(usage.get("completion_tokens", 0)),
                    "wall": wall,
                    "ttft": ttft,
                    "peak_gib": peak_gib,
                }
            )
            print(
                f"  req {i+1} ({topic}): status={status} "
                f"wall={wall:6.2f}s ttft={ttft:6.2f}s "
                f"peak={peak_gib:5.2f}GiB tokens={usage.get('completion_tokens', 0)}",
                flush=True,
            )

            if kill_state.tripped:
                break
            if status != "ok":
                kill_state.trip(f"request {i+1} {status}")
                break

            if i < len(PROMPTS) - 1:
                gap_start = time.time()
                quiet_sleep(f"idle gap {i+1}", IDLE_GAP_S, kill_state)
                gap_end = time.time()
                write_boundary(per_request_writer, f"idle gap {i+1}", monitor, gap_start, gap_end)
                if kill_state.tripped:
                    break

        # ---- Warm tail ----
        if not kill_state.tripped:
            tail_start = time.time()
            quiet_sleep("warm tail", WARM_TAIL_S, kill_state)
            tail_end = time.time()
            write_boundary(per_request_writer, "warm tail", monitor, tail_start, tail_end)

    except Exception as e:
        crashed = True
        print(f"!! driver crash: {e}", flush=True)
    finally:
        monitor.stop()
        monitor.join(timeout=5)
        per_request_writer.close()
        memtrail_writer.close()

    # ---- Summary ----
    print("", flush=True)
    print("=" * 60, flush=True)
    print("RUN 002b SEQUENTIAL CHARACTERIZATION SUMMARY", flush=True)
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
    print(
        f"projected peak_steady: {peak_steady_proj:.2f} GiB  "
        f"spike threshold: {spike_threshold_gib:.2f} GiB",
        flush=True,
    )
    print("", flush=True)

    print("Per-request memory peaks (during the request only):", flush=True)
    print(
        f"{'#':>2}  {'topic':<22} {'status':<10} "
        f"{'wall_s':>8} {'ttft_s':>8} {'peak_GiB':>10}  {'delta_vs_steady':>15}",
        flush=True,
    )
    for i, r in enumerate(request_results):
        delta = r["peak_gib"] - peak_steady_proj
        marker = " <-- SPIKE" if r["peak_gib"] > spike_threshold_gib else ""
        print(
            f"{i+1:>2}  {r['topic']:<22} {r['status']:<10} "
            f"{r['wall']:>8.2f} {r['ttft']:>8.2f} {r['peak_gib']:>10.2f}  "
            f"{delta:>+15.2f}{marker}",
            flush=True,
        )
    print("", flush=True)

    if crashed:
        print("RUN 002b: CRASH (driver exception)", flush=True)
        return EXIT_CRASH
    if kill_state.tripped:
        print(f"RUN 002b: KILLED ({kill_state.reason})", flush=True)
        return EXIT_KILLED
    if len(request_results) < len(PROMPTS):
        print(f"RUN 002b: INCOMPLETE ({len(request_results)}/{len(PROMPTS)} requests)", flush=True)
        return EXIT_FAIL

    verdict, description = classify(per_request_peaks_gib, spike_threshold_gib)
    print(f"RUN 002b: outcome {verdict} - {description}", flush=True)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
