"""Docker Compose operations."""

import json
import os
from pathlib import Path

from bootstrap_vllm.core.config import get_project_root, get_settings
from bootstrap_vllm.utils.output import error, info, ok
from bootstrap_vllm.utils.process import run, run_stream


def get_compose_file() -> Path:
    """Get the docker-compose.yml path."""
    return get_project_root() / "docker" / "docker-compose.yml"


def get_env_file() -> Path:
    """Get the .env file path."""
    return get_project_root() / ".env"


def compose_cmd(args: list[str]) -> list[str]:
    """Build a docker compose command with proper file paths."""
    compose_file = get_compose_file()
    env_file = get_env_file()

    cmd = ["docker", "compose", "-f", str(compose_file)]
    if env_file.exists():
        cmd.extend(["--env-file", str(env_file)])
    cmd.extend(args)
    return cmd


def is_running() -> bool:
    """Check if the vLLM container is running."""
    result = run(compose_cmd(["ps", "--format", "json"]))
    if not result.success or not result.stdout.strip():
        return False

    try:
        # docker compose ps --format json outputs one JSON object per line
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            container = json.loads(line)
            if container.get("State") == "running":
                return True
    except json.JSONDecodeError:
        pass
    return False


def get_container_status() -> dict | None:
    """Get container status information."""
    result = run(compose_cmd(["ps", "--format", "json"]))
    if not result.success or not result.stdout.strip():
        return None

    try:
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            return json.loads(line)
    except json.JSONDecodeError:
        return None


def up(force: bool = False) -> bool:
    """Start the vLLM container.

    Args:
        force: Force recreation even if running.

    Returns:
        True if successful.
    """
    env_file = get_env_file()
    if not env_file.exists():
        error(f"Configuration file not found: {env_file}")
        info("Copy .env.example to .env and configure it first")
        return False

    if is_running() and not force:
        ok("vLLM is already running")
        return True

    args = ["up", "-d"]
    if force:
        args.append("--force-recreate")

    settings = get_settings()
    env = os.environ.copy()
    env["VLLM_ENFORCE_EAGER_FLAG"] = "--enforce-eager" if settings.enforce_eager else ""
    env["VLLM_QUANTIZATION_FLAG"] = (
        f"--quantization {settings.quantization}" if settings.quantization else ""
    )

    info("Starting vLLM container...")
    returncode = run_stream(compose_cmd(args), env=env)

    if returncode == 0:
        ok("vLLM container started")
        info(f"API endpoint: http://localhost:{settings.port}")
        return True
    else:
        error("Failed to start vLLM")
        return False


def down() -> bool:
    """Stop and remove the vLLM container.

    Returns:
        True if successful.
    """
    if not is_running():
        ok("vLLM is not running")
        return True

    info("Stopping vLLM container...")
    returncode = run_stream(compose_cmd(["down"]))

    if returncode == 0:
        ok("vLLM stopped")
        return True
    else:
        error("Failed to stop vLLM")
        return False


def logs(follow: bool = True, tail: int | None = None) -> int:
    """Stream container logs.

    Args:
        follow: Follow log output.
        tail: Number of lines to show from end.

    Returns:
        Exit code.
    """
    args = ["logs"]
    if follow:
        args.append("-f")
    if tail is not None:
        args.extend(["--tail", str(tail)])

    return run_stream(compose_cmd(args))
