"""Model management commands."""

import re
import shutil
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from bootstrap_vllm.core.config import get_hf_settings, get_project_root, get_settings
from bootstrap_vllm.core.docker import down, is_running, up
from bootstrap_vllm.utils.output import console, error, info, ok, warn
from bootstrap_vllm.utils.process import run_stream

app = typer.Typer(help="Model management commands.")


_NVFP4_EXTRA_ARGS = (
    "--kv-cache-dtype fp8 "
    "--attention-backend flashinfer "
    "--enable-prefix-caching "
    "--enable-chunked-prefill "
    "--max-num-batched-tokens 8192 "
    "--max-num-seqs 64 "
    "--enable-auto-tool-choice "
    "--tool-call-parser qwen3_coder"
)

KNOWN_MODELS: dict[str, dict[str, str]] = {
    "saricles/Qwen3-Coder-Next-NVFP4-GB10": {
        "VLLM_IMAGE": "avarok/dgx-vllm-nvfp4-kernel:v23",
        "VLLM_QUANTIZATION": "compressed-tensors",
        "VLLM_NVFP4_GEMM_BACKEND": "marlin",
        "VLLM_TEST_FORCE_FP8_MARLIN": "1",
        "VLLM_USE_FLASHINFER_MOE_FP4": "0",
        "VLLM_MARLIN_USE_ATOMIC_ADD": "1",
        "VLLM_MAX_MODEL_LEN": "262144",
        "VLLM_GPU_MEMORY_UTILIZATION": "0.90",
        "VLLM_EXTRA_ARGS": _NVFP4_EXTRA_ARGS,
    },
    "Qwen/Qwen2.5-72B-Instruct-AWQ": {
        "VLLM_IMAGE": "nvcr.io/nvidia/vllm:25.12-py3",
        "VLLM_QUANTIZATION": "",
        "VLLM_NVFP4_GEMM_BACKEND": "",
        "VLLM_TEST_FORCE_FP8_MARLIN": "",
        "VLLM_USE_FLASHINFER_MOE_FP4": "",
        "VLLM_MARLIN_USE_ATOMIC_ADD": "",
        "VLLM_MAX_MODEL_LEN": "32768",
        "VLLM_GPU_MEMORY_UTILIZATION": "0.75",
        "VLLM_EXTRA_ARGS": "",
    },
}


def _get_hub_dir() -> Path | None:
    """Get the HuggingFace hub cache directory."""
    settings = get_settings()
    cache_dir = settings.model_cache.absolute()

    if not cache_dir.exists():
        return None

    hub_dir = cache_dir / "hub"
    if hub_dir.exists():
        return hub_dir

    # Also check for direct huggingface cache structure
    hub_dir = cache_dir / "huggingface" / "hub"
    if hub_dir.exists():
        return hub_dir

    return None


def _get_cached_models() -> list[str]:
    """Get list of cached model IDs."""
    hub_dir = _get_hub_dir()
    if not hub_dir:
        return []

    models = []
    for d in hub_dir.iterdir():
        if d.is_dir() and d.name.startswith("models--"):
            parts = d.name.replace("models--", "").split("--", 1)
            if len(parts) == 2:
                model_id = f"{parts[0]}/{parts[1]}"
                models.append(model_id)
    return models


def _is_model_cached(model_id: str) -> bool:
    """Check if a model is already cached."""
    return model_id in _get_cached_models()


@app.command()
def download(
    model_id: Annotated[
        str,
        typer.Argument(help="HuggingFace model ID (e.g., Qwen/Qwen2.5-72B-Instruct-AWQ)."),
    ],
) -> None:
    """Download a model from HuggingFace."""
    if _is_model_cached(model_id):
        ok(f"Model already cached: {model_id}")
        return

    if not _download_model(model_id):
        raise typer.Exit(1)


@app.command(name="list")
def list_models() -> None:
    """List cached models."""
    settings = get_settings()
    current_model = settings.model
    models = _get_cached_models()

    if not models:
        warn("No models cached yet")
        info(f"Configured: {current_model}")
        return

    table = Table(title="Cached Models")
    table.add_column("Model ID")
    table.add_column("Status", justify="center")

    for model_id in sorted(models):
        if model_id == current_model:
            table.add_row(f"[cyan]{model_id}[/cyan]", "[green]active[/green]")
        else:
            table.add_row(model_id, "")

    console.print(table)

    # Show if configured model isn't cached
    if current_model not in models:
        warn(f"Configured model not cached: {current_model}")


def _find_hf_cli() -> str | None:
    """Return the name of an available HuggingFace CLI binary, or None.

    huggingface-hub 1.x replaces `huggingface-cli` with `hf`. Support both.
    """
    for binary in ("hf", "huggingface-cli"):
        if shutil.which(binary):
            return binary
    return None


def _download_model(model_id: str) -> bool:
    """Download a model. Returns True if successful."""
    cli = _find_hf_cli()
    if cli is None:
        error("Neither 'hf' nor 'huggingface-cli' found on PATH")
        info("Install with: pip install huggingface_hub")
        return False

    settings = get_settings()
    hf_settings = get_hf_settings()

    info(f"Downloading model: {model_id}")

    # Target the /hub subdirectory explicitly. huggingface-cli (legacy) would
    # auto-create hub/ under cache-dir, but `hf` (huggingface-hub 1.x) uses
    # cache-dir verbatim — so we must pass the /hub path directly for both
    # CLIs to land files where vLLM's HF_HOME expects them.
    cmd = [
        cli,
        "download",
        model_id,
        "--cache-dir",
        str((settings.model_cache / "hub").absolute()),
    ]

    if hf_settings.hf_token:
        cmd.extend(["--token", hf_settings.hf_token])

    exit_code = run_stream(cmd)

    if exit_code == 0:
        ok(f"Model downloaded: {model_id}")
        return True
    else:
        error("Model download failed")
        if not hf_settings.hf_token:
            warn("Some models require HF_TOKEN in .env for gated access")
        return False


@app.command()
def switch(
    model_id: Annotated[
        str,
        typer.Argument(help="Model ID to switch to (e.g., Qwen/Qwen2.5-72B-Instruct-AWQ)."),
    ],
    skip_download: Annotated[
        bool,
        typer.Option("--skip-download", help="Don't download if not cached."),
    ] = False,
) -> None:
    """Switch to a different model (downloads if needed, stops server, restarts)."""
    settings = get_settings()
    env_file = get_project_root() / ".env"

    if not env_file.exists():
        error(f"Configuration file not found: {env_file}")
        raise typer.Exit(1)

    current_model = settings.model
    if current_model == model_id:
        ok(f"Already configured for: {model_id}")
        return

    info(f"Switching from {current_model} to {model_id}")

    # Download if not cached
    if not _is_model_cached(model_id) and not skip_download:
        info("Model not cached, downloading...")
        if not _download_model(model_id):
            raise typer.Exit(1)

    # Stop if running
    was_running = is_running()
    if was_running:
        info("Stopping current server...")
        if not down():
            error("Failed to stop server")
            raise typer.Exit(1)

    # Read once; build the full override map (VLLM_MODEL + KNOWN_MODELS profile).
    original_content = env_file.read_text()
    overrides: dict[str, str] = {"VLLM_MODEL": model_id}
    if model_id in KNOWN_MODELS:
        overrides.update(KNOWN_MODELS[model_id])
    else:
        warn(
            "Model not in KNOWN_MODELS — configure VLLM_IMAGE / VLLM_QUANTIZATION / "
            "VLLM_MAX_MODEL_LEN / VLLM_EXTRA_ARGS manually in .env"
        )

    # Side-save backup before any rewrite (live .env holds HF_TOKEN).
    (env_file.parent / ".env.bak").write_text(original_content)

    # Apply all overrides in a single in-memory buffer, then single write.
    new_content = original_content
    for key, value in overrides.items():
        pattern = rf"^{re.escape(key)}=.*$"
        replacement = f"{key}={value}"
        if re.search(pattern, new_content, re.MULTILINE):
            new_content = re.sub(pattern, replacement, new_content, flags=re.MULTILINE)
        else:
            new_content = new_content.rstrip() + f"\n{replacement}\n"

    env_file.write_text(new_content)
    ok(f"Updated .env: VLLM_MODEL={model_id}")
    if model_id in KNOWN_MODELS:
        info(f"Applied profile: {len(overrides) - 1} overrides")

    # Restart if was running
    if was_running:
        info("Starting server with new model...")
        if not up():
            error("Failed to start server")
            raise typer.Exit(1)
        ok(f"Now serving: {model_id}")
    else:
        info("Server not running. Start with: uv run bootstrap-vllm up")


@app.command()
def current() -> None:
    """Show the currently configured model."""
    settings = get_settings()
    console.print(f"[cyan]{settings.model}[/cyan]")
