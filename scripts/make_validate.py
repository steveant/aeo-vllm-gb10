#!/usr/bin/env python3
"""Run linting and type checking."""

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).parent.parent
    src = root / "src"

    # Run ruff format check
    print("Running ruff format check...")
    result = subprocess.run(
        ["ruff", "format", "--check", str(src)],
        cwd=root,
    )
    if result.returncode != 0:
        print("Format check failed. Run 'ruff format src/' to fix.")
        return result.returncode

    # Run ruff lint
    print("Running ruff lint...")
    result = subprocess.run(
        ["ruff", "check", str(src)],
        cwd=root,
    )
    if result.returncode != 0:
        print("Lint check failed.")
        return result.returncode

    # Run ty type checker
    print("Running ty type check...")
    result = subprocess.run(
        ["uvx", "ty", "check", str(src)],
        cwd=root,
    )
    if result.returncode != 0:
        print("Type check failed.")
        return result.returncode

    print("All checks passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
