"""Configuration management using pydantic-settings."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def find_env_file() -> Path | None:
    """Find .env file in current directory or parent directories."""
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        env_file = parent / ".env"
        if env_file.exists():
            return env_file
    return None


class Settings(BaseSettings):
    """vLLM deployment configuration."""

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
    """HuggingFace-specific settings (no prefix)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    hf_token: str = ""


def get_settings() -> Settings:
    """Load settings from .env file."""
    env_file = find_env_file()
    if env_file:
        return Settings(_env_file=env_file)  # type: ignore[call-arg]
    return Settings()


def get_hf_settings() -> HFSettings:
    """Load HuggingFace settings from .env file."""
    env_file = find_env_file()
    if env_file:
        return HFSettings(_env_file=env_file)  # type: ignore[call-arg]
    return HFSettings()


def get_project_root() -> Path:
    """Get the project root directory (where .env or docker-compose.yml lives)."""
    env_file = find_env_file()
    if env_file:
        return env_file.parent

    # Fall back to looking for docker-compose.yml
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if (parent / "docker" / "docker-compose.yml").exists():
            return parent

    return cwd
