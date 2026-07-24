import select
import sys
import time
from rich.align import Align
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

_STATUS_REFRESH_SECS = 1.5
_STATUS_DISPLAY_SECS = 8.0
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


def _resource_seg(stats) -> str:
    """Free GPU/CPU/RAM + a plain-language verdict on whether another job can start now.

    The cluster schedules per-resource (select/cons_tres), not per-node, so
    "GPU busy" no longer means the node is unavailable — CPU-only jobs can
    still start while the single GPU is in use. Spell that out rather than
    showing one binary GPU flag.
    """
    if stats is None:
        return "[dim]cluster stats unavailable[/]"

    gpu_free = max(0, stats.gpu_total - stats.gpu_alloc)
    cpu_free = max(0, stats.cpu_total - stats.cpu_alloc)
    mem_free_gb = max(0.0, (stats.mem_total_mb - stats.mem_alloc_mb) / 1024)

    gpu_color = "green" if gpu_free > 0 else "yellow"
    counts = (
        f"[{gpu_color}]GPU {gpu_free}/{stats.gpu_total} free[/]  "
        f"[dim]CPU {cpu_free}/{stats.cpu_total} free · RAM {mem_free_gb:.0f}GB free[/]"
    )

    if gpu_free > 0:
        verdict = "[bold green]✓ OK to submit a GPU job now[/]"
    elif cpu_free > 0 and mem_free_gb > 0:
        verdict = "[bold yellow]⚠ GPU busy — CPU-only jobs can still run[/]"
    else:
        verdict = "[bold red]✗ Cluster full — wait for a job to finish[/]"

    return f"{counts}  [dim]·[/]  {verdict}"


def _build_status_line(jobs, stats, username: str, spin: str) -> Panel:
    """Single-line status panel: user · all running jobs (any user) · free resources + submit verdict."""
    user_seg = f"[bold cyan]User:[/] [bold white]{username}[/]"

    if jobs is None:
        job_seg = "[dim]fetching…[/]"
    else:
        running = [j for j in jobs if j.state == "RUNNING"]
        if running:
            parts = [
                f"[green]{j.name}[/] [dim]({j.user})[/] [cyan]{j.time_used}[/]"
                for j in running[:4]
            ]
            if len(running) > 4:
                parts.append(f"[dim]+{len(running) - 4} more[/]")
            job_seg = "  ".join(parts)
        else:
            job_seg = "[dim]no jobs running[/]"

    res_seg = _resource_seg(stats)

    line = f"  {user_seg}  [dim]·[/]  {job_seg}  [dim]·[/]  {res_seg}"
    return Panel(
        line,
        title=f"[bold] {spin} Cluster Status  ·  any key to continue [/bold]",
        border_style="blue",
        expand=True,
        padding=(0, 0),
    )


def show_status_block() -> None:
    """Live-refreshing single-line status panel between splash and the wizard."""
    import getpass

    try:
        username = getpass.getuser()
    except Exception:
        username = "?"

    from iitgpu.slurm import queue as _queue, get_node_stats as _get_node_stats

    jobs: list | None = None
    stats = None
    last_refresh = 0.0

    def _refresh() -> None:
        nonlocal jobs, stats, last_refresh
        try:
            jobs = _queue(all_users=True)
        except Exception:
            jobs = []
        try:
            stats = _get_node_stats()
        except Exception:
            stats = None
        last_refresh = time.monotonic()

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
        # auto_refresh=False + explicit live.refresh() forces each update to
        # render immediately and synchronously — prevents the display freezing.
        with Live(console=console, auto_refresh=False, screen=False) as live:
            while time.monotonic() < deadline:
                spin = _SPINNERS[int(time.monotonic() * 4) % len(_SPINNERS)]
                live.update(_build_status_line(jobs, stats, username, spin))
                live.refresh()

                if _HAS_TERMIOS:
                    try:
                        r, _, _ = select.select([sys.stdin], [], [], 0.25)
                        if r:
                            sys.stdin.read(1)
                            break
                    except (OSError, ValueError):
                        time.sleep(0.25)
                else:
                    time.sleep(0.25)

                if time.monotonic() - last_refresh >= _STATUS_REFRESH_SECS:
                    _refresh()
    finally:
        if old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except termios.error:
                pass

    console.print()
