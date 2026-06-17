import select
import sys
import time
from rich.align import Align
from rich.columns import Columns
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.console import Group
from rich import box
from iitgpu import __version__
from iitgpu.ui import console

try:
    import termios
    import tty
    _HAS_TERMIOS = True
except ImportError:
    _HAS_TERMIOS = False

_ART_IIT_GPU = r"""
  ___ ___ _____      ____ ____  _   _
 |_ _|_ _|_   _|    / ___|  _ \| | | |
  | | | |  | |     | |  _| |_) | | | |
  | | | |  | |     | |_| |  __/| |_| |
 |___|___| |_|      \____|_|    \___/
"""

_ART_MANAGER = r"""
  __  __
 |  \/  | __ _ _ __   __ _  __ _  ___ _ __
 | |\/| |/ _` | '_ \ / _` |/ _` |/ _ \ '__|
 | |  | | (_| | | | | (_| | (_| |  __/ |
 |_|  |_|\__,_|_| |_|\__,_|\__, |\___|_|
                            |___/
"""

_STATUS_REFRESH_SECS = 2.0
_STATUS_DISPLAY_SECS = 6.0
_SPINNERS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def show_splash(pause: float = 1.5) -> None:
    iit_gpu = Text(_ART_IIT_GPU.strip("\n"), style="bold bright_cyan")
    manager = Text(_ART_MANAGER.strip("\n"), style="cyan")

    tagline = Text(
        "GPU Cluster Job Manager  ·  SLURM 25.11.2  ·  RTX 5090",
        style="dim white",
    )
    sep = Text("─" * 46, style="dim cyan")

    footer = Text()
    footer.append(f"v{__version__}", style="bold cyan")
    footer.append("   ·   ", style="dim white")
    footer.append("By: IIT Research Team", style="italic white")

    content = Group(
        Text(""),
        Align.center(iit_gpu),
        Align.center(manager),
        Text(""),
        Align.center(tagline),
        Align.center(sep),
        Text(""),
        Align.center(footer),
        Text(""),
    )

    console.print(
        Panel(
            content,
            box=box.DOUBLE_EDGE,
            border_style="cyan",
            expand=True,
            padding=(0, 2),
        )
    )
    console.print()

    if pause > 0:
        time.sleep(pause)


def _hw_bar(pct: float, width: int = 18) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = round(pct / 100 * width)
    color = "red" if pct >= 90 else "yellow" if pct >= 70 else "green"
    return f"[{color}]{'█' * filled}[/][dim]{'░' * (width - filled)}[/]"


def _build_status_renderable(stats, jobs, username: str, spin: str) -> Panel:
    """Build the live status panel shown between splash and the wizard."""
    left_lines = [""]

    # ── User ──────────────────────────────────────────────────────────────────
    left_lines.append(f"  [bold cyan]User:[/bold cyan]  [bold white]{username}[/bold white]")
    left_lines.append("")

    # ── Active jobs ───────────────────────────────────────────────────────────
    if jobs is None:
        left_lines.append("  [bold cyan]Jobs:[/bold cyan]  [dim]fetching…[/dim]")
    else:
        running = [j for j in jobs if j.state == "RUNNING"]
        pending = [j for j in jobs if j.state == "PENDING"]
        if running or pending:
            parts = []
            if running:
                names = ", ".join(j.name for j in running[:2])
                if len(running) > 2:
                    names += f" +{len(running) - 2} more"
                parts.append(f"[green]{len(running)} running[/green] [dim]({names})[/dim]")
            if pending:
                parts.append(f"[yellow]{len(pending)} pending[/yellow]")
            left_lines.append("  [bold cyan]Jobs:[/bold cyan]  " + "  ".join(parts))
        else:
            left_lines.append("  [bold cyan]Jobs:[/bold cyan]  [dim]none[/dim]")

    left_lines.append("")

    # ── Hardware ──────────────────────────────────────────────────────────────
    if stats is None:
        left_lines.append("  [dim]Hardware stats unavailable[/dim]")
    elif stats.live_stats:
        gpu_pct  = float(stats.gpu_util)
        vram_pct = stats.gpu_mem_used_mb / stats.gpu_mem_total_mb * 100 if stats.gpu_mem_total_mb else 0
        cpu_pct  = float(stats.cpu_util)
        mem_pct  = stats.mem_used_mb / stats.mem_total_mb * 100 if stats.mem_total_mb else 0

        gpu_c = "red" if gpu_pct >= 90 else "yellow" if gpu_pct >= 70 else "green"
        cpu_c = "red" if cpu_pct >= 90 else "yellow" if cpu_pct >= 70 else "green"
        mem_c = "red" if mem_pct >= 90 else "yellow" if mem_pct >= 70 else "green"

        left_lines.append(
            f"  [bold cyan]GPU[/bold cyan]  {_hw_bar(gpu_pct)}"
            f"  [{gpu_c}]{gpu_pct:3.0f}%[/]"
            f"  [dim]{stats.gpu_mem_used_mb/1024:.1f}/{stats.gpu_mem_total_mb/1024:.0f} GB VRAM"
            f"  {stats.gpu_temp}°C  {stats.gpu_power_w:.0f}W[/dim]"
        )
        left_lines.append(
            f"  [bold cyan]CPU[/bold cyan]  {_hw_bar(cpu_pct)}"
            f"  [{cpu_c}]{cpu_pct:3.0f}%[/]"
            f"  [dim]load {stats.cpu_load:.2f} / {stats.cpu_load5:.2f}[/dim]"
        )
        left_lines.append(
            f"  [bold cyan]RAM[/bold cyan]  {_hw_bar(mem_pct)}"
            f"  [{mem_c}]{stats.mem_used_mb/1024:.1f}/{stats.mem_total_mb/1024:.0f} GB[/]"
            f"  [dim]({mem_pct:.0f}%)[/dim]"
        )
    else:
        left_lines.append(
            f"  [bold cyan]GPU[/bold cyan]  [yellow]{stats.gpu_alloc}/{stats.gpu_total} alloc[/]  "
            f"[bold cyan]CPU[/bold cyan] {stats.cpu_alloc}/{stats.cpu_total}  "
            f"[bold cyan]RAM[/bold cyan] {stats.mem_alloc_mb//1024}/{stats.mem_total_mb//1024} GB"
        )

    left_lines.append("")

    body = "\n".join(left_lines)
    return Panel(
        body,
        title=f"[bold]  {spin}  System Status  ·  press any key to continue[/bold]",
        border_style="blue",
        expand=True,
        padding=(0, 1),
    )


def show_status_block() -> None:
    """Live-refreshing GPU/CPU/RAM + jobs panel shown after splash, before the wizard.

    Runs for _STATUS_DISPLAY_SECS seconds or until the user presses any key.
    """
    import getpass

    try:
        username = getpass.getuser()
    except Exception:
        username = "?"

    from iitgpu.slurm import get_node_stats, queue as _queue

    stats_ref: list = [None]
    jobs_ref:  list = [None]
    last_ts:   list = [0.0]

    def _refresh() -> None:
        try:
            stats_ref[0] = get_node_stats()
        except Exception:
            stats_ref[0] = None
        try:
            jobs_ref[0] = _queue(user=username)
        except Exception:
            jobs_ref[0] = []
        last_ts[0] = time.monotonic()

    _refresh()

    old_settings = None
    if _HAS_TERMIOS and sys.stdin.isatty():
        try:
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except termios.error:
            old_settings = None

    deadline = time.monotonic() + _STATUS_DISPLAY_SECS

    try:
        with Live(console=console, refresh_per_second=4, screen=False) as live:
            while time.monotonic() < deadline:
                spin = _SPINNERS[int(time.monotonic() * 4) % len(_SPINNERS)]
                live.update(_build_status_renderable(
                    stats_ref[0], jobs_ref[0], username, spin
                ))

                # Check for any keypress to break early
                key_pressed = False
                if _HAS_TERMIOS:
                    try:
                        r, _, _ = select.select([sys.stdin], [], [], 0.1)
                        if r:
                            sys.stdin.read(1)
                            key_pressed = True
                    except (OSError, ValueError):
                        pass
                else:
                    time.sleep(0.1)

                if key_pressed:
                    break

                if time.monotonic() - last_ts[0] >= _STATUS_REFRESH_SECS:
                    _refresh()
    finally:
        if old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except termios.error:
                pass

    console.print()
