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


def _build_status_line(stats, jobs, username: str, spin: str) -> Panel:
    """Single-line status panel: user · GPU/CPU/RAM % · active jobs."""
    segments = []

    # User
    segments.append(f"[bold cyan]User:[/] [bold white]{username}[/]")

    # Hardware
    if stats is None:
        segments.append("[dim]hardware unavailable[/]")
    elif stats.live_stats:
        gpu_c = "red" if stats.gpu_util >= 90 else "yellow" if stats.gpu_util >= 70 else "green"
        cpu_c = "red" if stats.cpu_util >= 90 else "yellow" if stats.cpu_util >= 70 else "green"
        mem_pct = int(stats.mem_used_mb / stats.mem_total_mb * 100) if stats.mem_total_mb else 0
        mem_c = "red" if mem_pct >= 90 else "yellow" if mem_pct >= 70 else "green"
        vram_gb = f"{stats.gpu_mem_used_mb/1024:.1f}/{stats.gpu_mem_total_mb/1024:.0f}GB"

        segments.append(
            f"[bold cyan]GPU:[/] [{gpu_c}]{stats.gpu_util}%[/]"
            f" [dim]{vram_gb} VRAM  {stats.gpu_temp}°C  {stats.gpu_power_w:.0f}W[/]"
        )
        segments.append(
            f"[bold cyan]CPU:[/] [{cpu_c}]{stats.cpu_util}%[/]"
            f" [dim]load {stats.cpu_load:.2f}[/]"
        )
        segments.append(
            f"[bold cyan]RAM:[/] [{mem_c}]{mem_pct}%[/]"
            f" [dim]{stats.mem_used_mb/1024:.0f}/{stats.mem_total_mb/1024:.0f}GB[/]"
        )
    else:
        gpu_c = "yellow" if stats.gpu_alloc > 0 else "green"
        segments.append(
            f"[bold cyan]GPU:[/] [{gpu_c}]{stats.gpu_alloc}/{stats.gpu_total} alloc[/]  "
            f"[bold cyan]CPU:[/] {stats.cpu_alloc}/{stats.cpu_total}  "
            f"[bold cyan]RAM:[/] {stats.mem_alloc_mb//1024}/{stats.mem_total_mb//1024}GB"
        )

    # Jobs — show current user's jobs with username
    if jobs is None:
        segments.append("[dim]jobs: fetching…[/]")
    else:
        my_jobs = [j for j in jobs if j.user == username]
        running = [j for j in my_jobs if j.state == "RUNNING"]
        pending = [j for j in my_jobs if j.state == "PENDING"]
        if running:
            job_parts = [f"{j.name} [{j.user}]" for j in running[:2]]
            if len(running) > 2:
                job_parts.append(f"+{len(running)-2} more")
            segments.append(f"[bold cyan]Jobs:[/] [green]{', '.join(job_parts)} RUNNING[/]")
        elif pending:
            job_parts = [f"{j.name} [{j.user}]" for j in pending[:2]]
            if len(pending) > 2:
                job_parts.append(f"+{len(pending)-2} more")
            segments.append(f"[bold cyan]Jobs:[/] [yellow]{', '.join(job_parts)} PENDING[/]")
        else:
            segments.append("[bold cyan]Jobs:[/] [dim]none[/]")

    line = "  " + "  ·  ".join(segments)
    return Panel(
        line,
        title=f"[bold] {spin} System Status  ·  any key to continue [/bold]",
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

    from iitgpu.slurm import get_node_stats, queue as _queue

    stats = None
    jobs: list | None = None
    last_refresh = 0.0

    def _refresh() -> None:
        nonlocal stats, jobs, last_refresh
        try:
            stats = get_node_stats()
        except Exception:
            stats = None
        try:
            # all_users=True so j.user is populated for the job filter
            jobs = _queue(all_users=True)
        except Exception:
            jobs = []
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
                live.update(_build_status_line(stats, jobs, username, spin))
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
