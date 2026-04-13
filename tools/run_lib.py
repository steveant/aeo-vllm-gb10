"""
Reusable helpers for vLLM load-test drivers.

This module contains all the infrastructure shared across run scripts:
parsing utilities, dataclasses, thread-safe writers, monitoring, API helpers,
and analysis functions.

No module-level side effects — nothing runs at import time.
Stdlib only.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import subprocess
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
# Exit codes
# ---------------------------------------------------------------------------

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_KILLED = 2
EXIT_CRASH = 3


# ---------------------------------------------------------------------------
# Compiled regexes
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


# ---------------------------------------------------------------------------
# CSV headers
# ---------------------------------------------------------------------------

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

MEMTRAIL_HEADER = [
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
]


# ---------------------------------------------------------------------------
# Timestamp / percentile utilities
# ---------------------------------------------------------------------------

def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="milliseconds")


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


# ---------------------------------------------------------------------------
# Parsing utilities
# ---------------------------------------------------------------------------

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


def read_docker_stats_rss_mib(container_name: str) -> Optional[int]:
    try:
        out = subprocess.run(
            [
                "docker",
                "stats",
                "--no-stream",
                "--format",
                "{{.MemUsage}}",
                container_name,
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


def read_latest_vllm_log_match(container_name: str) -> Optional[dict]:
    try:
        out = subprocess.run(
            ["docker", "logs", "--tail", "300", container_name],
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


def read_pool_gib(container_name: str) -> Optional[float]:
    try:
        out = subprocess.run(
            ["docker", "logs", container_name],
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
# Context-window guard
# ---------------------------------------------------------------------------

def trim_history_to_budget(history: list[dict], budget_tokens: int) -> list[dict]:
    """Estimate token count as sum(len(m["content"]) // 4) and drop oldest
    user/assistant pairs (preserving index 0 = system prompt and the most
    recent pairs) until within budget."""
    def _estimate_tokens(msgs: list[dict]) -> int:
        return sum(len(m["content"]) // 4 for m in msgs)

    if _estimate_tokens(history) <= budget_tokens:
        return history

    # history[0] is the system prompt; pairs start at index 1
    # We drop from the oldest pairs (index 1,2 then 3,4 etc.)
    result = list(history)
    while _estimate_tokens(result) > budget_tokens and len(result) > 3:
        # Drop the oldest user/assistant pair after the system prompt
        # result[1] should be user, result[2] should be assistant
        del result[1:3]

    return result


# ---------------------------------------------------------------------------
# Dataclasses
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


@dataclass
class RampTransition:
    from_level: int
    to_level: int
    ts: float
    peak_used_mib: int
    max_kv_pct: Optional[float]
    sample_count: int


@dataclass
class LevelStats:
    level: int
    sample_count: int
    max_used_mib: int
    mean_used_mib: float
    max_kv_pct: Optional[float]
    mean_kv_pct: Optional[float]


# ---------------------------------------------------------------------------
# Thread-safe state
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Thread-safe writers
# ---------------------------------------------------------------------------

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


class TranscriptWriter:
    """Thread-safe JSONL transcript writer. One JSON object per line, flushed on every write.

    The first line is a metadata record with the system prompt (written once).
    Each subsequent line is a turn record: session_id, turn_index, user message,
    assistant response, status, and per-turn metrics.

    IO errors are caught and warned about -- they never propagate into the worker.
    CSV is the authoritative artifact; transcript is the conversation record.
    """
    def __init__(self, path: Path, system_prompt: str, config: Optional[dict] = None):
        self.path = path
        self.lock = threading.Lock()
        self._fh = path.open("w")
        # Write system prompt once as the first record
        metadata: dict = {
            "type": "metadata",
            "ts": iso(time.time()),
            "system_prompt": system_prompt,
        }
        if config is not None:
            metadata["config"] = config
        self._write_record(metadata)

    def _write_record(self, record: dict) -> None:
        try:
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._fh.flush()
        except Exception as e:
            print(f"!! transcript write error (non-fatal): {e}", flush=True)

    def write_turn(
        self,
        session_id: int,
        session_topic: str,
        turn_index: int,
        user_msg: str,
        assistant_msg: str,
        status: str,
        wall: float,
        ttft: float,
        prompt_tokens: int,
        completion_tokens: int,
        raw_usage: Optional[dict] = None,
        extra: Optional[dict] = None,
    ) -> None:
        with self.lock:
            record = {
                "type": "turn",
                "ts": iso(time.time()),
                "session_id": session_id,
                "session_topic": session_topic,
                "turn_index": turn_index,
                "user": user_msg,
                "assistant": assistant_msg if status == "ok" else None,
                "status": status,
                "metrics": {
                    "wall_s": round(wall, 3),
                    "ttft_s": round(ttft, 3),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                },
            }
            # Pass through the full usage dict from vLLM (may contain fields
            # beyond prompt_tokens/completion_tokens that we don't parse yet)
            if raw_usage:
                record["raw_usage"] = raw_usage
            # Catch-all for anything extra that callers want to attach
            if extra:
                record["extra"] = extra
            self._write_record(record)

    def close(self) -> None:
        with self.lock:
            try:
                self._fh.flush()
                self._fh.close()
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
        *,
        container_name: str,
        monitor_interval_s: float = 0.5,
        heartbeat_interval_s: float = 15.0,
        docker_stats_interval_s: float = 5.0,
        sample_ring_size: int = 8000,
        kill_used_gib: float = 88.0,
        kill_swap_delta_gib: float = 1.5,
        kill_kv_pct: float = 95.0,
    ):
        super().__init__(name="monitor", daemon=True)
        self.csv_writer = csv_writer
        self.kill_state = kill_state
        self.swap_baseline_mib = swap_baseline_mib
        self.container_name = container_name
        self.monitor_interval_s = monitor_interval_s
        self.heartbeat_interval_s = heartbeat_interval_s
        self.docker_stats_interval_s = docker_stats_interval_s
        self.sample_ring_size = sample_ring_size
        self.kill_used_gib = kill_used_gib
        self.kill_swap_delta_gib = kill_swap_delta_gib
        self.kill_kv_pct = kill_kv_pct
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
            self.stop_event.wait(self.monitor_interval_s)

    def _sample_once(self) -> None:
        now = time.time()
        used, available, swap = read_free_m()
        log = read_latest_vllm_log_match(self.container_name)

        if now - self._last_docker_stats >= self.docker_stats_interval_s:
            self._cached_rss = read_docker_stats_rss_mib(self.container_name)
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
            if len(self.samples) > self.sample_ring_size:
                del self.samples[: len(self.samples) - self.sample_ring_size]
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

        if now - self._last_heartbeat >= self.heartbeat_interval_s:
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
        if sample.used_mib / 1024.0 > self.kill_used_gib:
            reasons.append(f"used {sample.used_mib/1024:.1f}GiB > {self.kill_used_gib}")
        if (sample.swap_mib - self.swap_baseline_mib) / 1024.0 > self.kill_swap_delta_gib:
            reasons.append(
                f"swap {sample.swap_mib/1024:.2f}GiB > baseline+{self.kill_swap_delta_gib}"
            )
        if sample.kv_pct is not None and sample.kv_pct > self.kill_kv_pct:
            reasons.append(f"kv {sample.kv_pct:.1f}% > {self.kill_kv_pct}")
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
# API helpers
# ---------------------------------------------------------------------------

def chat_completion_streaming(
    endpoint: str,
    model_id: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    timeout_s: int,
) -> tuple[str, dict, float]:
    body = json.dumps(
        {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode("utf-8")
    req = Request(
        endpoint,
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

    with urlopen(req, timeout=timeout_s) as resp:
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


def discover_model_id(models_endpoint: str) -> str:
    req = Request(models_endpoint, headers={"Accept": "application/json"})
    with urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    models = data.get("data") or []
    if not models:
        raise RuntimeError(f"no models served: {data}")
    return models[0]["id"]


def wait_for_health(health_endpoint: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urlopen(Request(health_endpoint), timeout=5) as resp:
                if resp.status == 200:
                    return
        except (HTTPError, URLError, TimeoutError, OSError):
            pass
        time.sleep(2)
    raise RuntimeError(f"vLLM health check did not pass within {timeout_s}s")


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def find_ramp_transitions(samples: list[Sample], ramp_window_s: float) -> list[RampTransition]:
    """Walk samples in time order. For each 1->2, 2->3, 3->4 transition on the
    running counter, capture the peak used_mib in the ramp_window_s window
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
                end_ts = start_ts + ramp_window_s
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
