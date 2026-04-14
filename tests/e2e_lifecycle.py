#!/usr/bin/env python3
"""
E2E lifecycle test: up → request → status → down.

Exercises the full CLI lifecycle a first-time user would follow.
`bootstrap-vllm up` now waits for health, so this script starts
timing from the `up` call and verifies the server is ready after.

Run directly:  uv run python tests/e2e_lifecycle.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

# Project root is one level up from tests/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from run_lib import (
    chat_completion_streaming,
    discover_model_id,
    wait_for_health,
)

VLLM_BASE_URL = "http://localhost:8000"
HEALTH_ENDPOINT = f"{VLLM_BASE_URL}/health"
MODELS_ENDPOINT = f"{VLLM_BASE_URL}/v1/models"
CHAT_ENDPOINT = f"{VLLM_BASE_URL}/v1/chat/completions"


def cli(*args: str) -> subprocess.CompletedProcess:
    """Run a bootstrap-vllm CLI command from the project root."""
    cmd = ["uv", "run", "bootstrap-vllm", *args]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)


def phase(num: int, name: str) -> None:
    print(f"\n{'='*60}", flush=True)
    print(f"Phase {num}: {name}", flush=True)
    print(f"{'='*60}", flush=True)


def main() -> int:
    results: list[tuple[int, str, bool, str]] = []
    boot_time = 0.0
    ttft = 0.0

    def record(num: int, name: str, ok: bool, detail: str = "") -> bool:
        tag = "PASS" if ok else "FAIL"
        results.append((num, name, ok, detail))
        print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""), flush=True)
        return ok

    # Phase 1: Prereq check
    phase(1, "Prereq check")
    r = cli("up", "--help")
    if not record(1, "CLI installed", r.returncode == 0, f"rc={r.returncode}"):
        print("  bootstrap-vllm not found — run `uv sync` first", flush=True)
        return 1

    # Phase 2: Start and wait for healthy
    phase(2, "Start server (includes health wait)")
    t0 = time.monotonic()
    r = cli("up")
    boot_time = time.monotonic() - t0
    print(r.stdout, end="", flush=True)
    if r.stderr:
        print(r.stderr, end="", flush=True)
    if not record(2, "bootstrap-vllm up", r.returncode == 0, f"rc={r.returncode}, {boot_time:.1f}s"):
        return 1

    # Phase 3: Confirm healthy + discover model
    phase(3, "Confirm healthy and discover model")
    try:
        wait_for_health(HEALTH_ENDPOINT, 30)
        record(3, "Health confirmed", True)
    except Exception as e:
        record(3, "Health confirmed", False, str(e))
        cli("down")
        return 1

    try:
        model_id = discover_model_id(MODELS_ENDPOINT)
        record(3, "Model discovered", True, model_id)
    except Exception as e:
        record(3, "Model discovered", False, str(e))
        cli("down")
        return 1

    # Phase 4: Chat completion
    phase(4, "Chat completion (streaming)")
    try:
        content, usage, ttft = chat_completion_streaming(
            CHAT_ENDPOINT,
            model_id,
            [{"role": "user", "content": "Say hello in exactly 3 words."}],
            max_tokens=50,
            temperature=0.0,
            timeout_s=120,
        )
        record(
            4,
            "Chat completion",
            bool(content),
            f"ttft={ttft:.2f}s tokens={usage.get('total_tokens', '?')}",
        )
        print(f"  Response: {content.strip()[:200]}", flush=True)
    except Exception as e:
        record(4, "Chat completion", False, str(e))

    # Phase 5: Status
    phase(5, "Status check")
    r = cli("status")
    print(r.stdout, end="", flush=True)
    record(5, "bootstrap-vllm status", r.returncode == 0, f"rc={r.returncode}")

    # Phase 6: Stop
    phase(6, "Stop server")
    r = cli("down")
    print(r.stdout, end="", flush=True)
    record(6, "bootstrap-vllm down", r.returncode == 0, f"rc={r.returncode}")

    # Phase 7: Verify stopped
    phase(7, "Verify stopped")
    r = cli("status")
    record(7, "Server stopped", r.returncode != 0, f"rc={r.returncode} (expect non-zero)")

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    all_passed = True
    for num, name, ok, detail in results:
        tag = "PASS" if ok else "FAIL"
        print(f"  {num}. [{tag}] {name}" + (f" — {detail}" if detail else ""), flush=True)
        if not ok:
            all_passed = False

    print(flush=True)
    if all_passed:
        print(
            f"LIFECYCLE PASS  (boot: {boot_time:.1f}s, ttft: {ttft:.2f}s)",
            flush=True,
        )
        return 0
    else:
        print("LIFECYCLE FAIL", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
