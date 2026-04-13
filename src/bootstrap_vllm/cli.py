"""Main CLI application."""

from typing import Annotated

import typer

from bootstrap_vllm import __version__
from bootstrap_vllm.commands import down, logs, model, status, up

app = typer.Typer(
    name="bootstrap-vllm",
    help="vLLM deployment orchestrator for NVIDIA GB10",
    no_args_is_help=True,
)


def version_callback(value: bool) -> None:
    if value:
        print(f"bootstrap-vllm {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-V",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = None,
) -> None:
    """vLLM deployment orchestrator for NVIDIA GB10."""
    pass


# Register commands
app.command()(up.up)
app.command()(down.down)
app.command()(status.status)
app.command()(logs.logs)

# Register model subcommand group
app.add_typer(model.app, name="model")
