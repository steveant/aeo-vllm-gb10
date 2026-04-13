"""Subprocess execution helpers."""

import subprocess
from dataclasses import dataclass


@dataclass
class Result:
    """Result of a subprocess execution."""

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
    """Run a command and return the result.

    Args:
        cmd: Command and arguments to run.
        capture: Whether to capture stdout/stderr.
        check: Whether to raise on non-zero exit.
        cwd: Working directory for the command.

    Returns:
        Result with returncode, stdout, and stderr.

    Raises:
        subprocess.CalledProcessError: If check=True and command fails.
    """
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
    """Run a command with output streaming to terminal.

    Args:
        cmd: Command and arguments to run.
        cwd: Working directory for the command.
        env: Environment dict to pass to subprocess; inherits parent env when None.

    Returns:
        Exit code of the command.
    """
    result = subprocess.run(cmd, cwd=cwd, env=env)
    return result.returncode
