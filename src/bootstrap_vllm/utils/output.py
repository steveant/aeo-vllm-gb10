"""Rich console output helpers."""

from rich.console import Console

console = Console()


def info(message: str) -> None:
    """Print an informational message."""
    console.print(f"[blue]ℹ[/blue] {message}")


def ok(message: str) -> None:
    """Print a success message."""
    console.print(f"[green]✓[/green] {message}")


def warn(message: str) -> None:
    """Print a warning message."""
    console.print(f"[yellow]⚠[/yellow] {message}")


def error(message: str) -> None:
    """Print an error message."""
    console.print(f"[red]✗[/red] {message}")


def status(message: str):
    """Create a status spinner context manager."""
    return console.status(message)
