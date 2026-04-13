"""Run 002d — Session 2 prompts: multi-turn Python code refactor.

Scenario: A senior engineer is asked to review and incrementally refactor the
`bootstrap_vllm` CLI used to orchestrate vLLM containers on NVIDIA GB10. The
seed message embeds the primary module under review plus enough surrounding
infrastructure for the model to reason about cross-module call sites. Subsequent
turns iterate toward a production-grade refactor with tests, migration plan,
observability, hypothesis properties, merge-conflict resolution, and a PR.

Both embedded files are pulled verbatim from `vLLM/src/bootstrap_vllm/`:
    - Seed primary: `vLLM/src/bootstrap_vllm/commands/model.py`
      (plus supporting modules bundled as a reading package so the reviewer
      can trace call sites — they were NOT refactor targets, only context).
    - Follow-up #3 second file: `vLLM/src/bootstrap_vllm/core/validate.py`
      (realistic interaction: `model switch` should re-validate prerequisites
      after swapping image/quantization profile, and `validate_prerequisites`
      reads settings that `model switch` just rewrote).

All string literals are valid Python; none of the embedded sources contain
triple-double-quote sequences, so the r\"\"\"...\"\"\" form is safe throughout.

Token counts reported by `estimate_tokens()` use the convention
`len(text.split()) * 1.3` shared with sibling session modules.
"""

from __future__ import annotations

SEED_TOPIC: str = (
    "Python code refactor: vLLM/src/bootstrap_vllm/commands/model.py "
    "(with core/docker.py, core/config.py, utils/process.py, and CLI entrypoints "
    "as supporting context)"
)


# ---------------------------------------------------------------------------
# SEED PROMPT
# ---------------------------------------------------------------------------
#
# The seed embeds the primary refactor target (`commands/model.py`) alongside
# the modules it directly imports from: `core/config.py`, `core/docker.py`,
# `utils/process.py`, `utils/output.py`. We also include the CLI registration
# file (`cli.py`), the thin `commands/{up,down,logs,status}.py` wrappers, and
# `commands/status.py` so the reviewer can see every public call site of the
# functions defined in `commands/model.py`.
#
# The single file under active refactor is `commands/model.py`. The rest is
# "reading package" context so the reviewer does not guess at the interfaces.
# This matches how a real senior engineer would be briefed: "Here is the file
# I want you to refactor, and here is every file that imports from it or that
# it imports from. Flag the five biggest problems."
# ---------------------------------------------------------------------------

SEED_PROMPT: str = r"""You are reviewing this module as a senior engineer. First, read it carefully and identify the five most concerning code quality, correctness, or architectural issues. Cite specific line ranges and function names. For each issue, explain what could go wrong in production and propose the fix direction without writing the fix yet.

The primary file under review is `bootstrap_vllm/commands/model.py`. It is the Typer subcommand group that handles model lifecycle for a vLLM deployment on an NVIDIA GB10 host: listing cached HuggingFace models, downloading them, and switching the running container between models by rewriting the `.env` file, restarting the docker-compose stack, and applying a hard-coded per-model profile of environment overrides. It is the most state-ful file in the codebase — it touches the filesystem (`.env`, HuggingFace cache directory), shells out to `hf` / `huggingface-cli`, and coordinates a stop/restart of the running vLLM container. That is a lot of implicit coupling for one 278-line module, which is part of why we are reviewing it.

Because the fixes will need to preserve behavior of every call site, I am also including the surrounding modules as a reading package. You should NOT produce review findings for these supporting files, but you SHOULD use them to reason about the refactor target's public surface, exception semantics, side-effect ordering, and what callers currently rely on.

Reading package layout:

1. `bootstrap_vllm/commands/model.py`    <-- PRIMARY refactor target (review this)
2. `bootstrap_vllm/core/config.py`       <-- pydantic-settings Settings + HFSettings + get_project_root()
3. `bootstrap_vllm/core/docker.py`       <-- `is_running`, `up`, `down` used by `switch`
4. `bootstrap_vllm/utils/process.py`     <-- `run_stream` used to invoke the HF CLI
5. `bootstrap_vllm/utils/output.py`      <-- `console`, `info`, `ok`, `warn`, `error` helpers
6. `bootstrap_vllm/cli.py`               <-- top-level Typer app; shows how `model.app` is mounted
7. `bootstrap_vllm/commands/up.py`       <-- start command — relevant because `model switch` calls `up()` through docker.up
8. `bootstrap_vllm/commands/down.py`     <-- stop command — same reason
9. `bootstrap_vllm/commands/logs.py`     <-- log-streaming command
10. `bootstrap_vllm/commands/status.py`  <-- health/status command; calls into docker.get_container_status and validate.check_*

Specific things I want you to scrutinize in `commands/model.py`, because they are the places I suspect we have latent bugs that have not fired yet:

- The `KNOWN_MODELS` dict: it is effectively a hard-coded registry of per-model configuration that cross-cuts with the `.env` file and with the `VLLM_*` namespace in `core/config.py`. Is this the right abstraction? Does it risk drifting out of sync with the Settings schema? What happens when a key exists in `KNOWN_MODELS` that Settings does not understand, or vice versa?
- The `switch` command's `.env` rewrite logic (roughly lines 205-258): it reads the whole file, mutates it in-memory, writes a `.env.bak` sidecar, then writes the new `.env`. Is that transactional? What happens on a partial failure between the backup and the final write? What happens if `.env` has comments, blank lines, duplicated keys, or values containing `=`? What happens if the user has a variable that is a multi-line shell export?
- The regex `rf"^{re.escape(key)}=.*$"` and the `re.sub` call with `re.MULTILINE`: consider what this does if a value contains a `$`, `\n`, or CRLF. Consider what happens if the user has `VLLM_EXTRA_ARGS="--foo --bar"` and the replacement contains spaces. Is the replacement shell-safe?
- `_find_hf_cli` returning the *name* of a CLI binary and then reusing that name unquoted as `cmd[0]`: OK for subprocess call, but note the comment explaining that `hf` (huggingface-hub 1.x) and `huggingface-cli` (legacy) behave differently with respect to where they place files. Does the current code actually handle both behaviors correctly given the `--cache-dir (settings.model_cache / "hub")` argument? Is the invariant "files land where vLLM's HF_HOME expects them" actually preserved for both CLIs, or is one of them broken?
- `_is_model_cached` and `_get_cached_models`: the cache layout detection walks `hub_dir`. Is this robust to partially-downloaded models (e.g., an `incomplete` marker file, a lockfile, a snapshots/blobs/refs directory scheme that varies between huggingface-hub versions)? A model directory can exist without any snapshot being complete — should the check be "is the model cached" or "is the model fully downloaded"?
- The order of operations inside `switch`: (1) download if needed, (2) stop if running, (3) rewrite `.env`, (4) restart if was running. Think about every pair of adjacent steps: is there a state where the user can end up with a stopped container and an inconsistent `.env`? With a started container pointing at the old model? With a `.env.bak` that does not match reality? Is the `.env.bak` single-slot, and what happens if the user runs `switch` twice in a row?
- The lack of typing around the `KNOWN_MODELS` values: they are `dict[str, str]`, but conceptually some of these are booleans ("1"), some are paths, some are numeric, some are free-form CLI arg strings. Is this type erasure a latent footgun for anyone adding a new model profile?
- The `warn(...)` fallthrough when a model is not in `KNOWN_MODELS`: the code writes `VLLM_MODEL=...` without clearing stale values for the other VLLM_* profile keys. If the previous model was NVFP4 and the new model is not, the old NVFP4 env vars will stick around and be applied to a non-NVFP4 image, which will almost certainly fail at runtime or, worse, half-succeed. Is that the intended semantics? (I suspect no.)
- Global singletons from `get_settings()` / `get_hf_settings()`: are these cached, or re-read every time? If `switch` rewrites `.env` and then calls `is_running()` or `up()`, which in turn re-call `get_settings()`, will the callees see the old cached Settings or the new file? (Check `core/config.py`.)
- `run_stream(cmd)` exit code handling: `_download_model` treats 0 as success and anything else as failure. But `run_stream` returns the raw subprocess return code, which for signal termination is negative on POSIX. Is that handled?
- Error path on `down()` failure during `switch`: the code calls `raise typer.Exit(1)` but the `.env` has not been rewritten yet. Good. But what about the `up()` failure at the end? At that point `.env` has been rewritten AND the old container is stopped. The user is left with a stopped stack and a new config. Is the error message actionable?

Produce the five most concerning findings. Rank them by severity (highest first). For each:

- A short headline (one sentence).
- Line range and function name(s) involved.
- What can go wrong in production — concrete failure modes, not abstract worries.
- Fix direction in 2-4 sentences. Do not write the fix yet.

Below is the reading package. The primary file (`commands/model.py`) is first; the rest are in the order listed above.

=== FILE 1 OF 10: bootstrap_vllm/commands/model.py (PRIMARY) ===

```python
\"\"\"Model management commands.\"\"\"

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
    \"\"\"Get the HuggingFace hub cache directory.\"\"\"
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
    \"\"\"Get list of cached model IDs.\"\"\"
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
    \"\"\"Check if a model is already cached.\"\"\"
    return model_id in _get_cached_models()


@app.command()
def download(
    model_id: Annotated[
        str,
        typer.Argument(help="HuggingFace model ID (e.g., Qwen/Qwen2.5-72B-Instruct-AWQ)."),
    ],
) -> None:
    \"\"\"Download a model from HuggingFace.\"\"\"
    if _is_model_cached(model_id):
        ok(f"Model already cached: {model_id}")
        return

    if not _download_model(model_id):
        raise typer.Exit(1)


@app.command(name="list")
def list_models() -> None:
    \"\"\"List cached models.\"\"\"
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
    \"\"\"Return the name of an available HuggingFace CLI binary, or None.

    huggingface-hub 1.x replaces `huggingface-cli` with `hf`. Support both.
    \"\"\"
    for binary in ("hf", "huggingface-cli"):
        if shutil.which(binary):
            return binary
    return None


def _download_model(model_id: str) -> bool:
    \"\"\"Download a model. Returns True if successful.\"\"\"
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
    \"\"\"Switch to a different model (downloads if needed, stops server, restarts).\"\"\"
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
    \"\"\"Show the currently configured model.\"\"\"
    settings = get_settings()
    console.print(f"[cyan]{settings.model}[/cyan]")
```

=== FILE 2 OF 10: bootstrap_vllm/core/config.py (CONTEXT) ===

```python
\"\"\"Configuration management using pydantic-settings.\"\"\"

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def find_env_file() -> Path | None:
    \"\"\"Find .env file in current directory or parent directories.\"\"\"
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        env_file = parent / ".env"
        if env_file.exists():
            return env_file
    return None


class Settings(BaseSettings):
    \"\"\"vLLM deployment configuration.\"\"\"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="VLLM_",
        extra="ignore",
    )

    # Model configuration
    model: str = "Qwen/Qwen2.5-72B-Instruct-AWQ"

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    max_model_len: int = 32768

    # GPU configuration
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    enforce_eager: bool = True

    # Quantization / backend
    quantization: str | None = None
    nvfp4_gemm_backend: str | None = None
    extra_args: str = ""

    # Paths (defaults are overridden by .env values when present)
    model_cache: Path = Path("./models")

    # Docker
    image: str = "nvcr.io/nvidia/vllm:25.12-py3"


class HFSettings(BaseSettings):
    \"\"\"HuggingFace-specific settings (no prefix).\"\"\"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    hf_token: str = ""


def get_settings() -> Settings:
    \"\"\"Load settings from .env file.\"\"\"
    env_file = find_env_file()
    if env_file:
        return Settings(_env_file=env_file)  # type: ignore[call-arg]
    return Settings()


def get_hf_settings() -> HFSettings:
    \"\"\"Load HuggingFace settings from .env file.\"\"\"
    env_file = find_env_file()
    if env_file:
        return HFSettings(_env_file=env_file)  # type: ignore[call-arg]
    return HFSettings()


def get_project_root() -> Path:
    \"\"\"Get the project root directory (where .env or docker-compose.yml lives).\"\"\"
    env_file = find_env_file()
    if env_file:
        return env_file.parent

    # Fall back to looking for docker-compose.yml
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if (parent / "docker" / "docker-compose.yml").exists():
            return parent

    return cwd
```

=== FILE 3 OF 10: bootstrap_vllm/core/docker.py (CONTEXT) ===

```python
\"\"\"Docker Compose operations.\"\"\"

import json
import os
from pathlib import Path

from bootstrap_vllm.core.config import get_project_root, get_settings
from bootstrap_vllm.utils.output import error, info, ok
from bootstrap_vllm.utils.process import run, run_stream


def get_compose_file() -> Path:
    \"\"\"Get the docker-compose.yml path.\"\"\"
    return get_project_root() / "docker" / "docker-compose.yml"


def get_env_file() -> Path:
    \"\"\"Get the .env file path.\"\"\"
    return get_project_root() / ".env"


def compose_cmd(args: list[str]) -> list[str]:
    \"\"\"Build a docker compose command with proper file paths.\"\"\"
    compose_file = get_compose_file()
    env_file = get_env_file()

    cmd = ["docker", "compose", "-f", str(compose_file)]
    if env_file.exists():
        cmd.extend(["--env-file", str(env_file)])
    cmd.extend(args)
    return cmd


def is_running() -> bool:
    \"\"\"Check if the vLLM container is running.\"\"\"
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
    \"\"\"Get container status information.\"\"\"
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
    \"\"\"Start the vLLM container.

    Args:
        force: Force recreation even if running.

    Returns:
        True if successful.
    \"\"\"
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
        ok("vLLM started successfully")
        info(f"API endpoint: http://localhost:{settings.port}")
        return True
    else:
        error("Failed to start vLLM")
        return False


def down() -> bool:
    \"\"\"Stop and remove the vLLM container.

    Returns:
        True if successful.
    \"\"\"
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
    \"\"\"Stream container logs.

    Args:
        follow: Follow log output.
        tail: Number of lines to show from end.

    Returns:
        Exit code.
    \"\"\"
    args = ["logs"]
    if follow:
        args.append("-f")
    if tail is not None:
        args.extend(["--tail", str(tail)])

    return run_stream(compose_cmd(args))
```

=== FILE 4 OF 10: bootstrap_vllm/utils/process.py (CONTEXT) ===

```python
\"\"\"Subprocess execution helpers.\"\"\"

import subprocess
from dataclasses import dataclass


@dataclass
class Result:
    \"\"\"Result of a subprocess execution.\"\"\"

    returncode: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        return self.returncode == 0


def run(
    cmd: list[str],
    *,
    capture: bool = True,
    check: bool = False,
    cwd: str | None = None,
) -> Result:
    \"\"\"Run a command and return the result.

    Args:
        cmd: Command and arguments to run.
        capture: Whether to capture stdout/stderr.
        check: Whether to raise on non-zero exit.
        cwd: Working directory for the command.

    Returns:
        Result with returncode, stdout, and stderr.

    Raises:
        subprocess.CalledProcessError: If check=True and command fails.
    \"\"\"
    result = subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        cwd=cwd,
    )

    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

    return Result(
        returncode=result.returncode,
        stdout=result.stdout or "",
        stderr=result.stderr or "",
    )


def run_stream(
    cmd: list[str],
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> int:
    \"\"\"Run a command with output streaming to terminal.

    Args:
        cmd: Command and arguments to run.
        cwd: Working directory for the command.
        env: Environment dict to pass to subprocess; inherits parent env when None.

    Returns:
        Exit code of the command.
    \"\"\"
    result = subprocess.run(cmd, cwd=cwd, env=env)
    return result.returncode
```

=== FILE 5 OF 10: bootstrap_vllm/utils/output.py (CONTEXT) ===

```python
\"\"\"Rich console output helpers.\"\"\"

from rich.console import Console

console = Console()


def info(message: str) -> None:
    \"\"\"Print an informational message.\"\"\"
    console.print(f"[blue]i[/blue] {message}")


def ok(message: str) -> None:
    \"\"\"Print a success message.\"\"\"
    console.print(f"[green]+[/green] {message}")


def warn(message: str) -> None:
    \"\"\"Print a warning message.\"\"\"
    console.print(f"[yellow]![/yellow] {message}")


def error(message: str) -> None:
    \"\"\"Print an error message.\"\"\"
    console.print(f"[red]x[/red] {message}")


def status(message: str):
    \"\"\"Create a status spinner context manager.\"\"\"
    return console.status(message)
```

=== FILE 6 OF 10: bootstrap_vllm/cli.py (CONTEXT) ===

```python
\"\"\"Main CLI application.\"\"\"

from typing import Annotated

import typer

from bootstrap_vllm import __version__
from bootstrap_vllm.commands import down, logs, model, status, up

app = typer.Typer(
    name="bootstrap-vllm",
    help="vLLM deployment orchestrator for NVIDIA GB10",
    no_args_is_help=True,
)


def version_callback(value: bool) -> None:
    if value:
        print(f"bootstrap-vllm {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-V",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = None,
) -> None:
    \"\"\"vLLM deployment orchestrator for NVIDIA GB10.\"\"\"
    pass


# Register commands
app.command()(up.up)
app.command()(down.down)
app.command()(status.status)
app.command()(logs.logs)

# Register model subcommand group
app.add_typer(model.app, name="model")
```

=== FILE 7 OF 10: bootstrap_vllm/commands/up.py (CONTEXT) ===

```python
\"\"\"Start vLLM server command.\"\"\"

from typing import Annotated

import typer

from bootstrap_vllm.core import docker, validate
from bootstrap_vllm.utils.output import error


def up(
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Force recreation even if running."),
    ] = False,
) -> None:
    \"\"\"Start vLLM server.\"\"\"
    if not validate.validate_prerequisites():
        error("Prerequisites check failed")
        raise typer.Exit(1)

    if not docker.up(force=force):
        raise typer.Exit(1)
```

=== FILE 8 OF 10: bootstrap_vllm/commands/down.py (CONTEXT) ===

```python
\"\"\"Stop vLLM server command.\"\"\"

import typer

from bootstrap_vllm.core import docker


def down() -> None:
    \"\"\"Stop and remove vLLM containers.\"\"\"
    if not docker.down():
        raise typer.Exit(1)
```

=== FILE 9 OF 10: bootstrap_vllm/commands/logs.py (CONTEXT) ===

```python
\"\"\"Stream container logs command.\"\"\"

from typing import Annotated

import typer

from bootstrap_vllm.core import docker


def logs(
    follow: Annotated[
        bool,
        typer.Option("--follow", "-f", help="Follow log output."),
    ] = True,
    tail: Annotated[
        int | None,
        typer.Option("--tail", "-n", help="Number of lines to show from end."),
    ] = None,
) -> None:
    \"\"\"Stream container logs.\"\"\"
    exit_code = docker.logs(follow=follow, tail=tail)
    if exit_code != 0:
        raise typer.Exit(exit_code)
```

=== FILE 10 OF 10: bootstrap_vllm/commands/status.py (CONTEXT) ===

```python
\"\"\"Show service status command.\"\"\"

import typer
from rich.table import Table

from bootstrap_vllm.core import docker, validate
from bootstrap_vllm.core.config import get_settings
from bootstrap_vllm.utils.output import console, info, ok, warn


def status() -> None:
    \"\"\"Show service health and status.\"\"\"
    settings = get_settings()

    # Container status
    container = docker.get_container_status()

    if container is None:
        warn("vLLM container is not running")
        info("Start with: bootstrap-vllm up")
        raise typer.Exit(1)

    state = container.get("State", "unknown")
    health = container.get("Health", "unknown")
    name = container.get("Name", "vllm")

    # Build status table
    table = Table(title="vLLM Status", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("Container", name)
    table.add_row(
        "State", f"[green]{state}[/green]" if state == "running" else f"[red]{state}[/red]"
    )
    table.add_row(
        "Health",
        f"[green]{health}[/green]" if health == "healthy" else f"[yellow]{health}[/yellow]",
    )
    table.add_row("Model", settings.model)
    table.add_row("Endpoint", f"http://localhost:{settings.port}")

    console.print(table)

    # API health check
    if state == "running":
        if validate.check_health():
            models = validate.check_models_loaded()
            if models:
                ok(f"Model loaded: {models[0]}")
        else:
            warn("API not yet responding (model may still be loading)")
```

=== END OF READING PACKAGE ===

Remember: your output is just the top-5 findings for `commands/model.py`, ranked by severity. Headline, line range, failure modes, fix direction. No code yet. I will ask for the refactored code in the next turn.

A few meta constraints for this review that I want to bake in up front, because they will shape which "fixes" are realistic when we get to the next turn:

- This CLI ships as a single pip-installable package and runs on one host at a time. You will not get to introduce a database, a message queue, or a background worker. Filesystem and subprocess are the only side-effect primitives on the table.
- The `.env` file is treated as the user's source of truth for configuration; it is edited by humans too. Any refactor must preserve human-readable semantics (comments, ordering, blank lines).
- `KNOWN_MODELS` is currently a hard-coded dict but we expect it to grow to ~20 entries within two quarters. It is worth investing in a better abstraction now, as long as that abstraction does not require a new dependency beyond what is already in the project: typer, pydantic, pydantic-settings, rich, httpx.
- Backwards compatibility matters: the `bootstrap_vllm` CLI is invoked from scripts in other repos, and command names / exit codes / stdout framing are part of the contract. Do not break `model switch`, `model list`, `model download`, or `model current`.
- The target platform is NVIDIA GB10 (ARM64 + Blackwell). The code must remain ARM64-clean. No x86-only syscalls, no assumptions about glibc versions, no architecture-gated imports.
- There is no test suite for this module today. Your refactor will be the first test target. That is a feature, not a bug — it means you get to establish the test conventions for the rest of the subsystem.
- Production usage looks like: one operator on one GB10 machine, occasionally running `bootstrap-vllm model switch <id>` between experiments, occasionally running `bootstrap-vllm model download <id>` ahead of time, very frequently running `bootstrap-vllm status`. Latency is not a concern; correctness and "does not leave the system in a half-configured state" are the concerns.

Go.
"""


# ---------------------------------------------------------------------------
# FOLLOW-UPS
# ---------------------------------------------------------------------------
#
# 8-step progression mirrors Session 1 (mesh-side sibling): refactor, tests,
# cross-file interaction fix with a SECOND embedded real file, migration plan,
# observability, hypothesis property tests, merge-conflict resolution, and
# polished PR description.
# ---------------------------------------------------------------------------

FOLLOW_UPS: list[str] = [
    # --- Turn 2: Refactor issue #1 ---------------------------------------------------
    (
        "Take issue #1 from your previous answer and produce the complete refactored "
        "code. Full file rewrite, not a diff. Preserve the module's public interface "
        "so existing callers don't break — the Typer commands `download`, `list`, "
        "`switch`, `current` must keep the same argument names, the same exit codes, "
        "and the same stdout framing that the CLI currently emits via `ok`, `info`, "
        "`warn`, and `error`. Keep the module-level `app = typer.Typer(...)` object "
        "intact because `cli.py` does `app.add_typer(model.app, name=\"model\")` and "
        "any other public module-level symbols that are imported by other files in "
        "`bootstrap_vllm/` must stay importable under their current names. Inside the "
        "function bodies you may restructure freely. Explain each non-obvious change "
        "inline as a brief comment — at most one line of justification per change, "
        "but every non-obvious edit should have one. If you introduce any new helper "
        "functions, give them a private underscore prefix and colocate them in the "
        "same module rather than creating new files. The goal of this turn is a "
        "drop-in replacement for `bootstrap_vllm/commands/model.py` that fixes issue "
        "#1 and nothing else — do not attempt to fix the other four findings yet, we "
        "will address them in later turns."
    ),

    # --- Turn 3: Tests for the refactored code ---------------------------------------
    (
        "Write a comprehensive unit test suite for the refactored code. Cover the "
        "happy path, the edge cases, and the error paths. Use pytest. Include "
        "fixtures and parametrized cases. Every public function from the refactored "
        "module should have at least two tests — one asserting correct behavior on a "
        "valid input, one asserting correct behavior on an adversarial or edge-case "
        "input. For the `switch` command specifically, I want to see tests for: "
        "(a) switching to a model already configured (should be a no-op with exit 0), "
        "(b) switching to an unknown model that is not in `KNOWN_MODELS`, "
        "(c) switching while the container is running vs while it is stopped, "
        "(d) `.env` containing comments and blank lines that must survive the rewrite, "
        "(e) a value in `.env` that contains `=`, `#`, or a trailing newline, "
        "(f) `--skip-download` when the model is not cached, "
        "(g) the backup file already existing from a previous run. "
        "For filesystem state use `tmp_path`. For subprocess calls use monkeypatch "
        "to replace `run_stream` with a spy that records the argv. For the container "
        "state use a fake `is_running`/`down`/`up` that can be flipped per test. Do "
        "not call the real docker compose or the real HuggingFace CLI from any test. "
        "Show me the full test file, placed under `vLLM/tests/bootstrap_vllm/commands/"
        "test_model.py` — include the pytest fixtures, the parametrize decorators, "
        "and the conftest changes if any are needed."
    ),

    # --- Turn 4: Cross-file interaction fix with SECOND real file injected ----------
    (
        "Here's a related module that interacts with the one you just refactored. "
        "It is the validation layer used by `bootstrap-vllm up` to gate startup on "
        "GPU/Docker/image/port checks. Specifically, `check_image()` reads "
        "`settings.image`, and `check_gpu()` reads `settings.nvfp4_gemm_backend` — "
        "both of which are keys that your refactored `model switch` command writes "
        "into `.env` via the `KNOWN_MODELS` profile. Walk through the interaction "
        "carefully, identify two bugs or mismatches that arise from your refactor, "
        "and show the fixes. Produce both files in their new state with all the "
        "fixes applied.\n\n"
        "Specifically, think about: does your refactored `switch` leave "
        "`settings.image` in a state that `check_image()` will subsequently be able "
        "to validate? If the pydantic Settings object is cached at module import "
        "time (inside a `@lru_cache`-style wrapper inside `get_settings`), what "
        "happens when `switch` rewrites `.env` and then calls `docker.up()`, which "
        "in turn calls `get_settings()` — does the second call see the fresh file "
        "or the stale cached object? If `switch` sets an NVFP4 profile but "
        "`check_gpu()` gates NVFP4 on exact SM 12.1, does the user get a clean "
        "error or a confusing late failure from inside the container?\n\n"
        "Here is `bootstrap_vllm/core/validate.py` verbatim:\n\n"
        "```python\n"
        "\"\"\"System validation checks.\"\"\"\n\n"
        "import socket\n\n"
        "import httpx\n\n"
        "from bootstrap_vllm.core.config import get_settings\n"
        "from bootstrap_vllm.utils.output import error, info, ok, warn\n"
        "from bootstrap_vllm.utils.process import run\n\n\n"
        "def check_docker() -> bool:\n"
        "    \"\"\"Check if Docker daemon is running.\"\"\"\n"
        "    result = run([\"docker\", \"info\"])\n"
        "    if result.success:\n"
        "        ok(\"Docker daemon is running\")\n"
        "        return True\n"
        "    else:\n"
        "        error(\"Docker daemon is not running\")\n"
        "        info(\"Start Docker with: sudo systemctl start docker\")\n"
        "        return False\n\n\n"
        "def check_gpu() -> bool:\n"
        "    \"\"\"Check if GPU is available; gate Marlin NVFP4 on exact SM 12.1 match.\"\"\"\n"
        "    result = run([\"nvidia-smi\", \"--query-gpu=name,compute_cap\", \"--format=csv,noheader\"])\n"
        "    if not result.success:\n"
        "        error(\"No GPU detected\")\n"
        "        info(\"Ensure NVIDIA drivers are installed: nvidia-smi\")\n"
        "        return False\n\n"
        "    compute_caps: list[str] = []\n"
        "    for line in result.stdout.strip().split(\"\\n\"):\n"
        "        parts = line.split(\",\")\n"
        "        if len(parts) >= 2:\n"
        "            name = parts[0].strip()\n"
        "            compute_cap = parts[1].strip()\n"
        "            compute_caps.append(compute_cap)\n"
        "            ok(f\"GPU: {name} (compute capability {compute_cap})\")\n\n"
        "    has_sm121 = any(cc == \"12.1\" for cc in compute_caps)\n"
        "    settings = get_settings()\n"
        "    if settings.nvfp4_gemm_backend == \"marlin\" and not has_sm121:\n"
        "        error(\"Marlin NVFP4 backend requires SM 12.1 (GB10); no matching GPU found\")\n"
        "        return False\n"
        "    if not has_sm121:\n"
        "        info(\"Non-SM12.1 GPU - not gated, but Marlin NVFP4 path is SM12.1-only\")\n\n"
        "    return True\n\n\n"
        "def check_image() -> bool:\n"
        "    \"\"\"Check that the configured Docker image is pulled locally.\"\"\"\n"
        "    settings = get_settings()\n"
        "    result = run([\"docker\", \"image\", \"inspect\", settings.image])\n"
        "    if result.success:\n"
        "        ok(f\"Image pulled: {settings.image}\")\n"
        "        return True\n"
        "    error(f\"Image not pulled locally: {settings.image}\")\n"
        "    info(f\"Run: docker pull {settings.image}\")\n"
        "    return False\n\n\n"
        "def check_port() -> bool:\n"
        "    \"\"\"Check that the configured port is available.\"\"\"\n"
        "    settings = get_settings()\n"
        "    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:\n"
        "        try:\n"
        "            sock.bind((\"127.0.0.1\", settings.port))\n"
        "        except OSError:\n"
        "            error(f\"Port {settings.port} already in use\")\n"
        "            return False\n"
        "    ok(f\"Port {settings.port} available\")\n"
        "    return True\n\n\n"
        "def check_health() -> bool:\n"
        "    \"\"\"Check if vLLM API is healthy.\"\"\"\n"
        "    settings = get_settings()\n"
        "    url = f\"http://localhost:{settings.port}/health\"\n\n"
        "    try:\n"
        "        response = httpx.get(url, timeout=5.0)\n"
        "        if response.status_code == 200:\n"
        "            ok(\"vLLM API is healthy\")\n"
        "            return True\n"
        "        else:\n"
        "            warn(f\"vLLM API returned status {response.status_code}\")\n"
        "            return False\n"
        "    except httpx.ConnectError:\n"
        "        warn(\"vLLM API is not responding (container may still be starting)\")\n"
        "        return False\n"
        "    except httpx.TimeoutException:\n"
        "        warn(\"vLLM API request timed out\")\n"
        "        return False\n\n\n"
        "def check_models_loaded() -> list[str]:\n"
        "    \"\"\"Get list of loaded models.\"\"\"\n"
        "    settings = get_settings()\n"
        "    url = f\"http://localhost:{settings.port}/v1/models\"\n\n"
        "    try:\n"
        "        response = httpx.get(url, timeout=5.0)\n"
        "        if response.status_code == 200:\n"
        "            data = response.json()\n"
        "            return [m[\"id\"] for m in data.get(\"data\", [])]\n"
        "    except (httpx.ConnectError, httpx.TimeoutException):\n"
        "        pass\n"
        "    return []\n\n\n"
        "def validate_prerequisites() -> bool:\n"
        "    \"\"\"Run all prerequisite checks.\n\n"
        "    Returns:\n"
        "        True if all checks pass.\n"
        "    \"\"\"\n"
        "    checks = [\n"
        "        (\"Docker\", check_docker),\n"
        "        (\"GPU\", check_gpu),\n"
        "        (\"Image\", check_image),\n"
        "        (\"Port\", check_port),\n"
        "    ]\n\n"
        "    all_passed = True\n"
        "    for _name, check in checks:\n"
        "        if not check():\n"
        "            all_passed = False\n\n"
        "    return all_passed\n"
        "```\n\n"
        "Now produce both files in their post-fix state. I want `commands/model.py` "
        "and `core/validate.py` fully rewritten, in the order in which I should apply "
        "them. Name the two bugs explicitly in a short paragraph between the two "
        "file listings so a reviewer can verify that each file's change targets a "
        "specific bug. Do not touch any other file in this turn."
    ),

    # --- Turn 5: Migration plan -------------------------------------------------------
    (
        "Write a migration plan: how do we roll this out across the existing "
        "deployments without breaking anything? Cover rollback, canary strategy, "
        "and the exact git workflow. Reference the specific function names from "
        "your refactor.\n\n"
        "The deployment topology is: one GB10 production host, two "
        "developer workstations that run the same `bootstrap-vllm` CLI against "
        "their own local docker, and a CI job that invokes `bootstrap-vllm model "
        "switch` inside a smoke test. We ship as a uv-installed package pinned "
        "to a git SHA in the consuming repo's `pyproject.toml`. We do not publish "
        "to a private PyPI; bumps happen by updating the git SHA and re-running "
        "`uv sync`.\n\n"
        "The plan should cover: "
        "(1) the exact sequence of commits and branches you would create, "
        "(2) which callers need to be touched in lockstep and which can lag, "
        "(3) how we smoke-test the refactor on one workstation before the prod "
        "host ever sees the change, "
        "(4) what the rollback procedure is if the refactor breaks `model switch` "
        "on prod mid-experiment — including the exact git commands to pin the "
        "consuming repo back to the previous SHA, "
        "(5) any schema-level changes to `.env` (new keys, removed keys, renamed "
        "keys) and how existing `.env` files get migrated forward without manual "
        "edits, "
        "(6) a deprecation window for any removed public symbols from "
        "`commands/model.py`, "
        "(7) how we verify that `KNOWN_MODELS` still resolves to the same set of "
        "env var overrides after the refactor — i.e., a golden-file check that "
        "the new code produces the same `.env` diff that the old code would have "
        "for each entry in `KNOWN_MODELS`. "
        "Put the whole plan in a single markdown document with sections. Include "
        "go/no-go criteria for each stage."
    ),

    # --- Turn 6: Observability -------------------------------------------------------
    (
        "Add observability (metrics, logs, traces) to BOTH refactored files — "
        "`bootstrap_vllm/commands/model.py` and `bootstrap_vllm/core/validate.py`. "
        "Use Python's `logging` module and add Prometheus client metrics where "
        "appropriate. Show the full updated source of both files with observability "
        "integrated. Describe the three SLOs and alerts you would set up on top of "
        "these metrics.\n\n"
        "A few constraints for the observability pass. "
        "The CLI is short-lived: `bootstrap-vllm model switch` runs once and "
        "exits, so push-style metrics (pushgateway, or a text file collector) "
        "make more sense than a long-lived HTTP exposition endpoint. Pick one, "
        "justify it, and wire it up — do not leave this as 'in a real system we "
        "would...'. "
        "The `logging` configuration should respect `LOG_LEVEL` from the "
        "environment, default to INFO, and use a structured format (JSON or "
        "logfmt, your call) so logs can be ingested downstream. "
        "Traces are optional if you judge that OpenTelemetry is overkill for a "
        "one-shot CLI — but if you skip them, say why in one paragraph. If you "
        "keep them, use the `opentelemetry-sdk` and `opentelemetry-exporter-otlp` "
        "stack and emit a single span per top-level Typer command. "
        "The three SLOs should be concrete — names, metrics, thresholds, alert "
        "conditions (e.g., 'switch_duration_seconds p95 < 120s measured over a "
        "rolling 7-day window; alert at 150s for 3 consecutive runs'). Do not "
        "write 'high reliability' — write the number. "
        "Do not break the existing stdout framing from `rich` (`ok`/`info`/`warn`/"
        "`error`) — the user-facing output is still the primary channel and logs "
        "are secondary. The two output channels should not duplicate each other "
        "verbatim; pick a convention and stick to it.\n\n"
        "Deliverable for this turn: full source of both files with observability "
        "integrated, followed by the three SLOs/alerts as a short bulleted list "
        "with metric names that match what you actually emit."
    ),

    # --- Turn 7: Hypothesis property tests -------------------------------------------
    (
        "Write property-based tests using `hypothesis` that exercise the invariants "
        "your refactor preserves. Make the invariants explicit — state them in "
        "English at the top of the test file — and then test them. At least four "
        "properties, each with a meaningful strategy.\n\n"
        "Some invariants that are candidates for this refactor: "
        "(A) For any `.env` input and any `KNOWN_MODELS` entry, running the "
        "`.env` rewrite produces a file where every key in `KNOWN_MODELS[model_id]` "
        "is present exactly once and set to the profile value — even if the key "
        "appeared zero, one, or N times in the original. "
        "(B) Comments and blank lines in the original `.env` that are not part of "
        "any rewritten key are preserved verbatim, in their original position "
        "relative to surrounding non-rewritten lines. "
        "(C) The `.env.bak` file, if it exists after a successful rewrite, is "
        "byte-for-byte equal to the original `.env` before the rewrite. "
        "(D) `_get_cached_models` applied to any directory layout that matches "
        "`models--<org>--<name>` yields model IDs such that round-tripping "
        "`_is_model_cached(id)` returns True for every yielded ID. "
        "(E) For any known `model_id`, `KNOWN_MODELS[model_id]` never contains a "
        "key that is not a valid `VLLM_*` field on the pydantic `Settings` class "
        "(or a known un-prefixed HF field). This is a refactor-level invariant — "
        "the test should fail-fast on any new profile key that the Settings "
        "schema does not understand.\n\n"
        "Pick at least four of these, or substitute equivalents you think are "
        "stronger. Write strategies that actually exercise the edge cases: "
        "hypothesis's `text()` with a custom alphabet for keys, `lists` and "
        "`dictionaries` for `.env` shapes, and `sampled_from(list(KNOWN_MODELS))` "
        "for model IDs. Use `@given(...)` on every property, not `@example(...)` "
        "alone. Use `@settings(max_examples=200, deadline=None)` at least on the "
        "slow ones. Put the full test file in "
        "`vLLM/tests/bootstrap_vllm/commands/test_model_properties.py` and show "
        "the complete source."
    ),

    # --- Turn 8: Merge-conflict resolution -------------------------------------------
    (
        "Imagine a pre-existing open PR introduces a conflicting change to the "
        "same module. Here is the plausible conflict: someone else opened PR #412 "
        "titled 'model: add `--dry-run` flag to `switch` and a `model rm` "
        "subcommand'. Their PR adds a `--dry-run` option to the `switch` Typer "
        "command that short-circuits before writing `.env`, logs the overrides it "
        "would have written, and exits 0. It also adds a new `model rm` subcommand "
        "that removes a cached model directory under `settings.model_cache / "
        "\"hub\"`. Their PR was opened two weeks before yours, has been reviewed, "
        "and is queued to merge tomorrow. Your refactor rewrites the `.env`-"
        "mutation path, the `KNOWN_MODELS` handling, and the Typer command "
        "registration inside `commands/model.py`.\n\n"
        "Draft a resolution: which parts of your refactor you keep, which you "
        "cede, how you rebase. Show the merge strategy in git commands and "
        "highlight the semantic conflicts beyond what git can auto-resolve.\n\n"
        "Your answer should cover: "
        "(1) which side should merge first — yours or PR #412 — and why. "
        "(2) the exact `git rebase` / `git merge` commands the losing side has "
        "to run to pick up the winning side. "
        "(3) the syntactic conflicts git will flag and how you would resolve each "
        "(give me the before/after lines). "
        "(4) the semantic conflicts git CANNOT detect. Example candidates: "
        "PR #412's `--dry-run` short-circuits the `.env` write path — does your "
        "refactored `.env` write path still have a single short-circuit point, "
        "or did you inline it such that `--dry-run` has to be re-plumbed in "
        "multiple places? PR #412's `model rm` calls `_get_hub_dir()` — did you "
        "rename or remove that helper? If yes, `model rm` will silently break "
        "without a test to catch it. "
        "(5) a short post-merge sanity checklist — not an exhaustive test run, "
        "just the three things a human reviewer should manually verify after "
        "applying the merge. "
        "(6) whether you would push back on PR #412 and ask them to rebase on "
        "your refactor first, or whether you would absorb the conflict yourself. "
        "Justify the choice in two sentences."
    ),

    # --- Turn 9: PR description ------------------------------------------------------
    (
        "Produce the complete PR description for your refactor: title, summary, "
        "motivation, design decisions, test plan, rollout plan, risk assessment "
        "table (with likelihood and blast radius for each risk), and a reviewer "
        "checklist. This should be the kind of PR description a staff engineer "
        "would write for a major cleanup — polished, specific, and reviewable.\n\n"
        "Format requirements: "
        "Title under 72 characters, imperative mood, no trailing period, starts "
        "with the conventional-commits-style scope prefix `model:` since the "
        "refactor is scoped to `bootstrap_vllm/commands/model.py` plus the "
        "`core/validate.py` follow-on fix. "
        "Summary is exactly three bullets. "
        "Motivation is a short paragraph (3-5 sentences) explaining why this "
        "refactor is worth merging now and not next quarter — reference the "
        "concrete bugs from earlier turns. "
        "Design decisions section enumerates at least four non-obvious choices "
        "you made, each with a one-sentence 'why this instead of the obvious "
        "alternative'. "
        "Test plan is a checklist of the test files added or touched plus any "
        "manual smoke tests the reviewer should run on a GB10 host. "
        "Rollout plan is the summarized form of the migration plan from the "
        "earlier turn, not a re-derivation — three bullets, not three paragraphs. "
        "Risk table has columns: risk, likelihood (low/med/high), blast radius "
        "(one user / all users / all users + data loss), mitigation. At least "
        "five rows. Order by likelihood * blast radius, descending. "
        "Reviewer checklist is an explicit `- [ ]` markdown list of at least six "
        "things the reviewer must verify — not 'code looks good', but specific "
        "things like '`.env.bak` is written before the new `.env` on every code "
        "path' or '`model switch` preserves comments in `.env`'.\n\n"
        "Output the whole PR description in one markdown block. No preamble, no "
        "postamble — just the PR body the way it would appear in the GitHub "
        "compose box. Assume the reviewer has not seen any of the previous turns "
        "in this thread; the PR description must stand on its own."
    ),
]


# ---------------------------------------------------------------------------
# estimate_tokens() helper — matches sibling modules
# ---------------------------------------------------------------------------

def estimate_tokens() -> None:
    """Print approximate token counts using len(text.split()) * 1.3."""
    def _tok(text: str) -> int:
        return int(len(text.split()) * 1.3)

    print(f"SEED_TOPIC            : {_tok(SEED_TOPIC):>6} tokens")
    print(f"SEED_PROMPT           : {_tok(SEED_PROMPT):>6} tokens")
    print(f"FOLLOW_UPS (count)    : {len(FOLLOW_UPS):>6}")
    total_followup = 0
    for i, fu in enumerate(FOLLOW_UPS, start=1):
        t = _tok(fu)
        total_followup += t
        print(f"  follow_up[{i}]         : {t:>6} tokens")
    print(f"FOLLOW_UPS total      : {total_followup:>6} tokens")
    print(f"SESSION total         : {_tok(SEED_PROMPT) + total_followup:>6} tokens")


if __name__ == "__main__":
    estimate_tokens()
