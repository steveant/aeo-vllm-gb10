"""System validation checks."""

import socket

import httpx

from bootstrap_vllm.core.config import get_settings
from bootstrap_vllm.utils.output import error, info, ok, warn
from bootstrap_vllm.utils.process import run


def check_docker() -> bool:
    """Check if Docker daemon is running."""
    result = run(["docker", "info"])
    if result.success:
        ok("Docker daemon is running")
        return True
    else:
        error("Docker daemon is not running")
        info("Start Docker with: sudo systemctl start docker")
        return False


def check_gpu() -> bool:
    """Check if GPU is available; gate Marlin NVFP4 on exact SM 12.1 match."""
    result = run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"])
    if not result.success:
        error("No GPU detected")
        info("Ensure NVIDIA drivers are installed: nvidia-smi")
        return False

    compute_caps: list[str] = []
    for line in result.stdout.strip().split("\n"):
        parts = line.split(",")
        if len(parts) >= 2:
            name = parts[0].strip()
            compute_cap = parts[1].strip()
            compute_caps.append(compute_cap)
            ok(f"GPU: {name} (compute capability {compute_cap})")

    has_sm121 = any(cc == "12.1" for cc in compute_caps)
    settings = get_settings()
    if settings.nvfp4_gemm_backend == "marlin" and not has_sm121:
        error("Marlin NVFP4 backend requires SM 12.1 (GB10); no matching GPU found")
        return False
    if not has_sm121:
        info("Non-SM12.1 GPU — not gated, but Marlin NVFP4 path is SM12.1-only")

    return True


def check_image() -> bool:
    """Check that the configured Docker image is pulled locally."""
    settings = get_settings()
    result = run(["docker", "image", "inspect", settings.image])
    if result.success:
        ok(f"Image pulled: {settings.image}")
        return True
    error(f"Image not pulled locally: {settings.image}")
    info(f"Run: docker pull {settings.image}")
    return False


def check_port() -> bool:
    """Check that the configured port is available."""
    settings = get_settings()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", settings.port))
        except OSError:
            error(f"Port {settings.port} already in use")
            return False
    ok(f"Port {settings.port} available")
    return True


def check_health() -> bool:
    """Check if vLLM API is healthy."""
    settings = get_settings()
    url = f"http://localhost:{settings.port}/health"

    try:
        response = httpx.get(url, timeout=5.0)
        if response.status_code == 200:
            ok("vLLM API is healthy")
            return True
        else:
            warn(f"vLLM API returned status {response.status_code}")
            return False
    except httpx.ConnectError:
        warn("vLLM API is not responding (container may still be starting)")
        return False
    except httpx.TimeoutException:
        warn("vLLM API request timed out")
        return False


def check_models_loaded() -> list[str]:
    """Get list of loaded models."""
    settings = get_settings()
    url = f"http://localhost:{settings.port}/v1/models"

    try:
        response = httpx.get(url, timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            return [m["id"] for m in data.get("data", [])]
    except (httpx.ConnectError, httpx.TimeoutException):
        pass
    return []


def validate_prerequisites() -> bool:
    """Run all prerequisite checks.

    Returns:
        True if all checks pass.
    """
    checks = [
        ("Docker", check_docker),
        ("GPU", check_gpu),
        ("Image", check_image),
        ("Port", check_port),
    ]

    all_passed = True
    for _name, check in checks:
        if not check():
            all_passed = False

    return all_passed
