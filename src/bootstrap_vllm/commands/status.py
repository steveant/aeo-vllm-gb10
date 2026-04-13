"""Show service status command."""

import typer
from rich.table import Table

from bootstrap_vllm.core import docker, validate
from bootstrap_vllm.core.config import get_settings
from bootstrap_vllm.utils.output import console, info, ok, warn


def status() -> None:
    """Show service health and status."""
    settings = get_settings()

    # Container status
    container = docker.get_container_status()

    if container is None:
        warn("vLLM container is not running")
        info("Start with: bootstrap-vllm up")
        raise typer.Exit(1)

    state = container.get("State", "unknown")
    health = container.get("Health", "unknown")
    name = container.get("Name", "vllm")

    # Build status table
    table = Table(title="vLLM Status", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("Container", name)
    table.add_row(
        "State", f"[green]{state}[/green]" if state == "running" else f"[red]{state}[/red]"
    )
    table.add_row(
        "Health",
        f"[green]{health}[/green]" if health == "healthy" else f"[yellow]{health}[/yellow]",
    )
    table.add_row("Model", settings.model)
    table.add_row("Endpoint", f"http://localhost:{settings.port}")

    console.print(table)

    # API health check
    if state == "running":
        if validate.check_health():
            models = validate.check_models_loaded()
            if models:
                ok(f"Model loaded: {models[0]}")
        else:
            warn("API not yet responding (model may still be loading)")
