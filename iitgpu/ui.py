# iitgpu/ui.py
import questionary
from questionary import Separator, Style
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.theme import Theme

_theme = Theme({
    "info": "cyan",
    "ok": "bold green",
    "warn": "bold yellow",
    "err": "bold red",
    "label": "bold magenta",
    "value": "white",
})

console = Console(theme=_theme)

# One shared prompt style for every screen in the tool — previously
# duplicated (near-identically) in menu.py, wizard.py, workspace.py,
# monitor.py, upload.py, models.py, and re-declared even narrower in
# admin.py/accounting.py/notebooks.py/files.py.
STYLE = Style([
    ("qmark", "fg:cyan bold"),
    ("question", "bold"),
    ("answer", "fg:magenta bold"),
    ("pointer", "fg:cyan bold"),
    ("highlighted", "fg:cyan bold"),
    ("selected", "fg:magenta"),
])

# The one spelling of "go back". The arrow is a navigation affordance, not
# decorative emoji — kept per the same rule that already spares questionary's
# own pointer glyph (see commit 8702b1e).
BACK = "← Back"
BACK_TO_MAIN = "← Back to main menu"


def header(text: str) -> None:
    console.rule(f"[bold cyan]{text}[/]")


def ok(text: str) -> None:
    console.print(f"[ok]OK  {text}[/]")


def warn(text: str) -> None:
    console.print(f"[warn]WARN  {text}[/]")


def err(text: str) -> None:
    console.print(f"[err]ERROR  {text}[/]")


def info(text: str) -> None:
    console.print(f"[info]{text}[/]")


def kv(key: str, value: str) -> None:
    console.print(f"[label]{key}:[/] [value]{value}[/]")


def panel(title: str, body: str) -> None:
    console.print(Panel(body, title=title, border_style="cyan"))


def screen(title: str, *, status: "RenderableType | str | None" = None) -> None:
    """Render a screen's entry header: a bordered Panel with the title, and
    an optional live status body (a one-line string or a Rich Table) — the
    same look the launch review hub already uses. Called once per menu-loop
    iteration in place of the old bare header() rule."""
    body = status if status is not None else ""
    console.print(Panel(body, title=f"[bold]{title}[/]", border_style="cyan"))


def select_menu(prompt: str, choices: list, *, back: str = BACK) -> str | None:
    """questionary.select wrapped with one always-present, always-last Back
    item. Returns None for Esc/Ctrl-C AND for picking Back — callers never
    need to know which spelling of "back" a screen used, because there's
    only ever one."""
    sel = questionary.select(
        prompt, choices=[*choices, Separator(), back], style=STYLE
    ).ask()
    return None if sel in (None, back) else sel
