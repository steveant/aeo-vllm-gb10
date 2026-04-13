"""Tests for configuration module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from bootstrap_vllm.core.config import Settings, find_env_file, get_project_root


class TestFindEnvFile:
    """Tests for find_env_file function."""

    def test_finds_env_in_current_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Should find .env in current directory."""
        env_file = tmp_path / ".env"
        env_file.write_text("VLLM_PORT=9000")
        monkeypatch.chdir(tmp_path)

        result = find_env_file()

        assert result == env_file

    def test_finds_env_in_parent_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Should find .env in parent directory."""
        env_file = tmp_path / ".env"
        env_file.write_text("VLLM_PORT=9000")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        monkeypatch.chdir(subdir)

        result = find_env_file()

        assert result == env_file

    def test_returns_none_when_no_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Should return None when no .env exists."""
        monkeypatch.chdir(tmp_path)

        result = find_env_file()

        assert result is None


class TestSettings:
    """Tests for Settings class."""

    def test_default_values(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Should have sensible defaults when no .env or env vars override."""
        monkeypatch.chdir(tmp_path)
        for key in list(os.environ):
            if key.startswith("VLLM_"):
                monkeypatch.delenv(key, raising=False)

        settings = Settings()

        assert settings.model == "Qwen/Qwen2.5-72B-Instruct-AWQ"
        assert settings.port == 8000
        assert settings.host == "0.0.0.0"
        assert settings.max_model_len == 32768
        assert settings.gpu_memory_utilization == 0.90
        assert settings.tensor_parallel_size == 1
        assert settings.enforce_eager is True
        assert settings.quantization is None
        assert settings.nvfp4_gemm_backend is None
        assert settings.extra_args == ""
        assert settings.model_cache == Path("./models")
        assert settings.image == "nvcr.io/nvidia/vllm:25.12-py3"

    def test_nvfp4_fields_load_from_env(self, monkeypatch: pytest.MonkeyPatch):
        """Should load NVFP4 fields from environment variables."""
        monkeypatch.setenv("VLLM_QUANTIZATION", "compressed-tensors")
        monkeypatch.setenv("VLLM_NVFP4_GEMM_BACKEND", "marlin")
        monkeypatch.setenv("VLLM_EXTRA_ARGS", "--kv-cache-dtype fp8 --enable-prefix-caching")

        settings = Settings()

        assert settings.quantization == "compressed-tensors"
        assert settings.nvfp4_gemm_backend == "marlin"
        assert settings.extra_args == "--kv-cache-dtype fp8 --enable-prefix-caching"

    def test_loads_from_env_vars(self, monkeypatch: pytest.MonkeyPatch):
        """Should load values from environment variables."""
        monkeypatch.setenv("VLLM_PORT", "9000")
        monkeypatch.setenv("VLLM_MODEL", "test/model")
        monkeypatch.setenv("VLLM_GPU_MEMORY_UTILIZATION", "0.85")

        settings = Settings()

        assert settings.port == 9000
        assert settings.model == "test/model"
        assert settings.gpu_memory_utilization == 0.85

    def test_loads_from_env_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Should load values from .env file."""
        env_file = tmp_path / ".env"
        env_file.write_text("VLLM_PORT=8888\nVLLM_MODEL=custom/model\n")

        settings = Settings(_env_file=env_file)  # type: ignore[call-arg]

        assert settings.port == 8888
        assert settings.model == "custom/model"


class TestGetProjectRoot:
    """Tests for get_project_root function."""

    def test_returns_env_file_parent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Should return parent of .env file."""
        env_file = tmp_path / ".env"
        env_file.write_text("VLLM_PORT=8000")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        monkeypatch.chdir(subdir)

        result = get_project_root()

        assert result == tmp_path

    def test_falls_back_to_compose_location(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Should find project root via docker-compose.yml location."""
        docker_dir = tmp_path / "docker"
        docker_dir.mkdir()
        (docker_dir / "docker-compose.yml").write_text("services: {}")
        monkeypatch.chdir(tmp_path)

        result = get_project_root()

        assert result == tmp_path

    def test_returns_cwd_as_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Should return cwd when no markers found."""
        monkeypatch.chdir(tmp_path)

        result = get_project_root()

        assert result == tmp_path
