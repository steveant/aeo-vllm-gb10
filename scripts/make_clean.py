#!/usr/bin/env python3
"""Clean build artifacts and caches."""

import shutil
from pathlib import Path


def main() -> None:
    root = Path(__file__).parent.parent

    dirs_to_remove = [
        root / "dist",
        root / "build",
        root / ".ruff_cache",
    ]

    # Find and remove __pycache__ directories
    for pycache in root.rglob("__pycache__"):
        dirs_to_remove.append(pycache)

    # Find and remove .egg-info directories
    for egg_info in root.rglob("*.egg-info"):
        dirs_to_remove.append(egg_info)

    removed = []
    for d in dirs_to_remove:
        if d.exists():
            shutil.rmtree(d)
            removed.append(d.relative_to(root))

    if removed:
        print(f"Removed: {', '.join(str(p) for p in removed)}")
    else:
        print("Nothing to clean")


if __name__ == "__main__":
    main()
