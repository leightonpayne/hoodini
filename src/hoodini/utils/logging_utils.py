"""Centralized logging utilities."""

import logging
from rich.console import Console

# Shared Console instance
console = Console()

# Library logger; configuration should be done by the CLI entrypoint.
logger = logging.getLogger("hoodini")


def stage_header(title: str, emoji: str = "") -> None:
    """
    Print a horizontal rule with a bold, blue title and optional emoji.
    Example: stage_header("Initializing", "🚀") → ───🚀 Initializing ───
    """
    # Place emoji before the title, no extra space if no emoji
    text = f"{emoji} {title}" if emoji else title
    console.rule(f"[bold blue]{text}[/bold blue]", style="blue")


def stage_done(message: str) -> None:
    """
    Print a green checkmark panel with the given message.
    Example: stage_done("Done") → green panel with ✔️ Done
    """
    from rich.panel import Panel

    console.print(Panel.fit(f"[bold green]✔️  {message}[/bold green]", border_style="green"))


def run_with_spinner(title: str, func, *args, spinner_name: str = "dots", **kwargs):
    """
    Display a Rich spinner with `title` while running `func(*args, **kwargs)`.
    Returns whatever `func` returns.

    Example:
        df = run_with_spinner("Fetching IPG", parser.run)
    """
    # Instead of constructing Spinner(spinner_name), just pass spinner_name directly.
    with console.status(f"[bold cyan]{title}[/bold cyan]", spinner=spinner_name):
        return func(*args, **kwargs)
