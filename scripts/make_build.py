#!/usr/bin/env python3
"""Build standalone executable with PyInstaller."""

import subprocess
import sys
from pathlib import Path

HIDDEN_IMPORTS = [
    "bootstrap_vllm.commands.up",
    "bootstrap_vllm.commands.down",
    "bootstrap_vllm.commands.status",
    "bootstrap_vllm.commands.logs",
    "bootstrap_vllm.commands.model",
    "bootstrap_vllm.core.config",
    "bootstrap_vllm.core.docker",
    "bootstrap_vllm.core.validate",
    "bootstrap_vllm.utils.output",
    "bootstrap_vllm.utils.process",
]


def main() -> int:
    root = Path(__file__).parent.parent
    entry_point = root / "src" / "bootstrap_vllm" / "__main__.py"

    # Build hidden imports args
    hidden_args = []
    for mod in HIDDEN_IMPORTS:
        hidden_args.extend(["--hidden-import", mod])

    cmd = [
        "pyinstaller",
        "--onefile",
        "--name", "bootstrap-vllm",
        "--paths", str(root / "src"),
        "--collect-data", "rich",
        "--collect-submodules", "rich._unicode_data",
        *hidden_args,
        str(entry_point),
    ]

    print(f"Building: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=root)

    if result.returncode == 0:
        print(f"\nBuild complete: {root / 'dist' / 'bootstrap-vllm'}")

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
