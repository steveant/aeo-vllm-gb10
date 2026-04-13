"""Stop vLLM server command."""

import typer

from bootstrap_vllm.core import docker


def down() -> None:
    """Stop and remove vLLM containers."""
    if not docker.down():
        raise typer.Exit(1)
