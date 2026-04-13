"""Stream container logs command."""

from typing import Annotated

import typer

from bootstrap_vllm.core import docker


def logs(
    follow: Annotated[
        bool,
        typer.Option("--follow", "-f", help="Follow log output."),
    ] = True,
    tail: Annotated[
        int | None,
        typer.Option("--tail", "-n", help="Number of lines to show from end."),
    ] = None,
) -> None:
    """Stream container logs."""
    exit_code = docker.logs(follow=follow, tail=tail)
    if exit_code != 0:
        raise typer.Exit(exit_code)
