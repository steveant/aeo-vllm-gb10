"""Tests for process utility module."""

import subprocess

import pytest

from bootstrap_vllm.utils.process import Result, run, run_stream


class TestResult:
    """Tests for Result dataclass."""

    def test_success_true_on_zero_returncode(self):
        """Should report success when returncode is 0."""
        result = Result(returncode=0, stdout="output", stderr="")

        assert result.success is True

    def test_success_false_on_nonzero_returncode(self):
        """Should report failure when returncode is non-zero."""
        result = Result(returncode=1, stdout="", stderr="error")

        assert result.success is False


class TestRun:
    """Tests for run function."""

    def test_captures_stdout(self):
        """Should capture stdout from command."""
        result = run(["echo", "hello"])

        assert result.success is True
        assert "hello" in result.stdout

    def test_captures_stderr(self):
        """Should capture stderr from command."""
        result = run(["python", "-c", "import sys; sys.stderr.write('error')"])

        assert "error" in result.stderr

    def test_returns_nonzero_on_failure(self):
        """Should return non-zero returncode on command failure."""
        result = run(["false"])  # Unix command that always fails

        assert result.success is False
        assert result.returncode != 0

    def test_check_raises_on_failure(self):
        """Should raise CalledProcessError when check=True and command fails."""
        with pytest.raises(subprocess.CalledProcessError):
            run(["false"], check=True)

    def test_respects_cwd(self, tmp_path):
        """Should run command in specified directory."""
        result = run(["pwd"], cwd=str(tmp_path))

        assert str(tmp_path) in result.stdout


class TestRunStream:
    """Tests for run_stream function."""

    def test_returns_exit_code(self):
        """Should return command exit code."""
        exit_code = run_stream(["true"])  # Unix command that always succeeds

        assert exit_code == 0

    def test_returns_nonzero_on_failure(self):
        """Should return non-zero exit code on failure."""
        exit_code = run_stream(["false"])

        assert exit_code != 0
