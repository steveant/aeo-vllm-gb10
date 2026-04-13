"""Tests for CLI module."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from bootstrap_vllm.cli import app
from bootstrap_vllm.commands.model import KNOWN_MODELS, switch

runner = CliRunner()


class TestCLI:
    """Tests for main CLI application."""

    def test_help_shows_commands(self):
        """Should show available commands in help."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "up" in result.output
        assert "down" in result.output
        assert "status" in result.output
        assert "logs" in result.output
        assert "model" in result.output

    def test_version_flag(self):
        """Should show version with --version flag."""
        result = runner.invoke(app, ["--version"])

        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_model_subcommand_help(self):
        """Should show model subcommand help."""
        result = runner.invoke(app, ["model", "--help"])

        assert result.exit_code == 0
        assert "download" in result.output
        assert "list" in result.output

    def test_up_help(self):
        """Should show up command options."""
        result = runner.invoke(app, ["up", "--help"])

        assert result.exit_code == 0
        assert "--force" in result.output
        assert "--model" not in result.output
        assert "--port" not in result.output


def _seed_env(tmp_path: Path) -> Path:
    """Write a minimum .env file and return its path."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "VLLM_MODEL=Qwen/Qwen2.5-72B-Instruct-AWQ\n"
        "VLLM_PORT=8000\n"
        "VLLM_MAX_MODEL_LEN=32768\n"
        "VLLM_GPU_MEMORY_UTILIZATION=0.75\n"
        "VLLM_IMAGE=nvcr.io/nvidia/vllm:25.12-py3\n"
        "HF_TOKEN=placeholder\n"
    )
    return env_file


def _patch_switch_deps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Patch the imported references inside bootstrap_vllm.commands.model."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("bootstrap_vllm.commands.model.is_running", lambda: False)
    monkeypatch.setattr("bootstrap_vllm.commands.model._is_model_cached", lambda _: True)


class TestModelSwitch:
    """Tests for the `model switch` profile application logic."""

    def test_switch_to_nvfp4_applies_profile(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Switching to the NVFP4 profile should write all override keys."""
        env_file = _seed_env(tmp_path)
        _patch_switch_deps(monkeypatch, tmp_path)

        switch("saricles/Qwen3-Coder-Next-NVFP4-GB10", skip_download=True)

        content = env_file.read_text()
        expected = KNOWN_MODELS["saricles/Qwen3-Coder-Next-NVFP4-GB10"]
        assert "VLLM_MODEL=saricles/Qwen3-Coder-Next-NVFP4-GB10" in content
        assert f"VLLM_IMAGE={expected['VLLM_IMAGE']}" in content
        assert f"VLLM_QUANTIZATION={expected['VLLM_QUANTIZATION']}" in content
        assert f"VLLM_NVFP4_GEMM_BACKEND={expected['VLLM_NVFP4_GEMM_BACKEND']}" in content
        assert f"VLLM_MAX_MODEL_LEN={expected['VLLM_MAX_MODEL_LEN']}" in content
        assert f"VLLM_GPU_MEMORY_UTILIZATION={expected['VLLM_GPU_MEMORY_UTILIZATION']}" in content
        assert f"VLLM_EXTRA_ARGS={expected['VLLM_EXTRA_ARGS']}" in content

        bak_file = tmp_path / ".env.bak"
        assert bak_file.exists()
        assert "Qwen/Qwen2.5-72B-Instruct-AWQ" in bak_file.read_text()

    def test_switch_to_awq_resets_profile(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Switching to AWQ from an NVFP4-seeded env should clear NVFP4 vars."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "VLLM_MODEL=saricles/Qwen3-Coder-Next-NVFP4-GB10\n"
            "VLLM_PORT=8000\n"
            "VLLM_QUANTIZATION=compressed-tensors\n"
            "VLLM_NVFP4_GEMM_BACKEND=marlin\n"
            "VLLM_MAX_MODEL_LEN=262144\n"
            "VLLM_GPU_MEMORY_UTILIZATION=0.90\n"
            "VLLM_EXTRA_ARGS=--kv-cache-dtype fp8\n"
            "VLLM_IMAGE=avarok/dgx-vllm-nvfp4-kernel:v23\n"
            "HF_TOKEN=placeholder\n"
        )
        _patch_switch_deps(monkeypatch, tmp_path)

        switch("Qwen/Qwen2.5-72B-Instruct-AWQ", skip_download=True)

        content = env_file.read_text()
        assert "VLLM_MODEL=Qwen/Qwen2.5-72B-Instruct-AWQ" in content
        assert "VLLM_IMAGE=nvcr.io/nvidia/vllm:25.12-py3" in content
        assert "VLLM_QUANTIZATION=\n" in content
        assert "VLLM_NVFP4_GEMM_BACKEND=\n" in content
        assert "VLLM_MAX_MODEL_LEN=32768" in content
        assert "VLLM_GPU_MEMORY_UTILIZATION=0.75" in content
        assert "VLLM_EXTRA_ARGS=\n" in content
