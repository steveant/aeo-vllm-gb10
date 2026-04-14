"""Start vLLM server command."""

import time
from typing import Annotated

import httpx
import typer

from bootstrap_vllm.core import docker, validate
from bootstrap_vllm.core.config import get_settings
from bootstrap_vllm.utils.output import error, info, ok, status, warn

HEALTH_TIMEOUT_S = 1200


def up(
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Force recreation even if running."),
    ] = False,
) -> None:
    """Start vLLM server and wait for it to become healthy."""
    if not validate.validate_prerequisites():
        error("Prerequisites check failed")
        raise typer.Exit(1)

    if not docker.up(force=force):
        raise typer.Exit(1)

    _wait_for_health()


def _wait_for_health() -> None:
    """Poll the health endpoint until the server is ready."""
    settings = get_settings()
    url = f"http://localhost:{settings.port}/health"
    deadline = time.monotonic() + HEALTH_TIMEOUT_S

    with status("[bold]Loading model...[/bold]"):
        while time.monotonic() < deadline:
            try:
                response = httpx.get(url, timeout=5.0)
                if response.status_code == 200:
                    ok("Server is healthy and ready to serve requests")
                    return
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            time.sleep(5)

    warn(f"Server did not become healthy within {HEALTH_TIMEOUT_S}s")
    info("The model may still be loading. Check with: bootstrap-vllm status")
