"""Start vLLM server command."""

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
    """Start vLLM server."""
    if not validate.validate_prerequisites():
        error("Prerequisites check failed")
        raise typer.Exit(1)

    if not docker.up(force=force):
        raise typer.Exit(1)
