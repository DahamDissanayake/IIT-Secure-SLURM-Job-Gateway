# iitgpu/dashboard.py
from __future__ import annotations

import select
import sys
import time
from pathlib import Path

from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from iitgpu.config import load_config, jobs_dir
from iitgpu.slurm import (NodeStats, QueueEntry, cancel, extend_job_time, get_node_stats,
                          queue, filtered_history, _effective_user)
from iitgpu.ui import console, err, ok

try:
    import termios
    import tty
    _HAS_TERMIOS = True
except ImportError:
    _HAS_TERMIOS = False

_DATA_REFRESH_SECS = 2.0
_DISPLAY_FPS       = 4
_COMPLETED_HISTORY = 2
_SPINNERS          = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


# ── Time helpers ──────────────────────────────────────────────────────────────

def _slurm_time_to_secs(t: str) -> int | None:
    if not t or t in ("N/A", "UNLIMITED", "NOT_SET", "Partition_Limit", "-"):
        return None
    try:
        days = 0
        if "-" in t:
            d, t = t.split("-", 1)
            days = int(d)
        parts = t.split(":")
        if len(parts) == 3:
            return days * 86400 + int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        if len(parts) == 2:
            return days * 86400 + int(parts[0]) * 60 + int(parts[1])
        return None
    except (ValueError, IndexError):
        return None


def _fmt_duration(secs: int) -> str:
    m, s = divmod(abs(secs), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


# ── Bar helper (shared by cluster panel and hardware stats) ───────────────────

def _hw_bar(pct: float, width: int = 22) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = round(pct / 100 * width)
    color = "red" if pct >= 90 else "yellow" if pct >= 70 else "green"
    return f"[{color}]{'█' * filled}[/][dim]{'░' * (width - filled)}[/]"


# ── Log helpers ───────────────────────────────────────────────────────────────

def _get_log_tail(log_path: str, lines: int = 20) -> list[str]:
    p = Path(log_path)
    if not p.exists():
        return []
    try:
        all_lines = p.read_text(errors="replace").splitlines()
        return all_lines[-lines:]
    except OSError:
        return []


def _find_job_log(job_id: str, search_root: str) -> str | None:
    """search_root is a single user's job dir; the file is one level down at
    {search_root}/{job_folder}/slurm-<id>.out — no need to recurse further."""
    target = f"slurm-{job_id}.out"
    for p in Path(search_root).glob(f"*/{target}"):
        return str(p)
    return None


def _get_job_output(job_id: str, jdir: str, lines: int = 500) -> tuple[list[str], str | None]:
    log_path = _find_job_log(job_id, jdir)
    if log_path is None:
        return [], None
    out_lines = _get_log_tail(log_path, lines=lines)
    err_path = str(Path(log_path).with_suffix(".err"))
    err_lines = _get_log_tail(err_path, lines=100)
    if err_lines:
        combined = err_lines + (["", "[dim]── stdout ──[/dim]"] + out_lines if out_lines else [])
    else:
        combined = out_lines
    return combined, log_path


def _time_remaining(job: QueueEntry) -> int | None:
    """Seconds remaining for a running job with a time limit; None otherwise."""
    if job.state not in ("RUNNING", "COMPLETING"):
        return None
    limit = _slurm_time_to_secs(job.time_limit)
    used  = _slurm_time_to_secs(job.time_used)
    if limit is None or used is None:
        return None
    return max(0, limit - used)


def _is_jupyter_job(job_id: str, jdir: str) -> bool:
    """True when the job folder contains the .iit-jupyter marker file."""
    log_path = _find_job_log(job_id, jdir)
    if log_path is None:
        return False
    return (Path(log_path).parent / ".iit-jupyter").exists()


# ── Cluster panel (compact summary bar) ──────────────────────────────────────

def _build_cluster_panel(stats: NodeStats | None) -> Panel:
    if stats is None:
        body = "[dim]Cluster stats unavailable[/]"
    else:
        state = stats.state.split("+")[0]
        sc = "green" if "IDLE" in state else "yellow" if "ALLOC" in state else "red"

        if stats.live_stats:
            gpu_color = "red" if stats.gpu_util >= 90 else "yellow" if stats.gpu_util >= 70 else "green"
            gpu_str = (
                f"GPU [bold {gpu_color}]{stats.gpu_util}%[/] "
                f"[dim]{stats.gpu_mem_used_mb/1024:.1f}/{stats.gpu_mem_total_mb/1024:.0f}GB "
                f"{stats.gpu_temp}°C {stats.gpu_power_w:.0f}W[/]"
            )
            cpu_color = "red" if stats.cpu_util >= 90 else "yellow" if stats.cpu_util >= 70 else "green"
            cpu_str = f"CPU [bold {cpu_color}]{stats.cpu_util}%[/] [dim]load {stats.cpu_load:.2f}[/]"
            mem_pct = stats.mem_used_mb / stats.mem_total_mb * 100 if stats.mem_total_mb else 0
            mem_color = "red" if mem_pct >= 90 else "yellow" if mem_pct >= 70 else "green"
            mem_str = f"RAM [bold {mem_color}]{stats.mem_used_mb/1024:.0f}/{stats.mem_total_mb/1024:.0f} GB[/]"
        else:
            if stats.shard_total:
                gpu_color = "yellow" if stats.shard_alloc >= stats.shard_total else "green"
                gpu_str = (f"GPU [bold {gpu_color}]{stats.shard_alloc}/"
                           f"{stats.shard_total} slices[/]")
            else:
                gpu_color = "yellow" if stats.gpu_alloc > 0 else "green"
                gpu_str = f"GPU [bold {gpu_color}]{stats.gpu_alloc}/{stats.gpu_total} alloc[/]"
            cpu_str = f"CPU [bold]{stats.cpu_alloc}/{stats.cpu_total}[/] [dim]load {stats.cpu_load:.2f}[/]"
            mem_str = f"RAM [bold]{stats.mem_alloc_mb//1024}/{stats.mem_total_mb//1024} GB[/] [dim]alloc[/]"

        body = f"  iit-MS-7E06  [{sc}]{state}[/]  │  {gpu_str}  │  {cpu_str}  │  {mem_str}"

    return Panel(body, title="[bold]Cluster: iit[/bold]", border_style="blue", height=3)


# ── Jobs table ────────────────────────────────────────────────────────────────

def _build_jobs_table(jobs: list[QueueEntry], selected_idx: int, current_user: str) -> Table:
    table = Table(
        show_header=True, header_style="bold cyan",
        box=box.SIMPLE, expand=True, show_edge=False,
    )
    table.add_column("",        width=2)
    table.add_column("ID",      style="magenta", width=7,  no_wrap=True)
    table.add_column("User",    width=9,  no_wrap=True)
    table.add_column("Name",    width=21, no_wrap=True)
    table.add_column("State",   width=14, no_wrap=True)
    table.add_column("Time Left", width=9,  no_wrap=True)
    table.add_column("Part",    width=5,  no_wrap=True)

    spin = _SPINNERS[int(time.monotonic() * _DISPLAY_FPS) % len(_SPINNERS)]
    added_done_sep = False

    for i, j in enumerate(jobs):
        is_done = j.state in ("COMPLETED", "FAILED", "CANCELLED")
        is_selected = i == selected_idx

        if is_done and not added_done_sep:
            added_done_sep = True
            table.add_row(
                "", "[dim]──[/]", "[dim]──────[/]",
                "[dim]─── recent ──────────────[/]",
                "[dim]────────────[/]", "[dim]───────[/]", "[dim]───[/]",
            )

        prefix = "[bold cyan]❯[/]" if is_selected else " "

        if is_done:
            s_color = "cyan" if j.state == "COMPLETED" else ("yellow" if j.state == "CANCELLED" else "red")
            elapsed_secs = _slurm_time_to_secs(j.time_used) or 0
            elapsed_str = _fmt_duration(elapsed_secs) if elapsed_secs > 0 else j.time_used
            table.add_row(
                prefix,
                f"[dim strike]{j.job_id}[/]",
                f"[dim strike]{j.user[:7]}[/]",
                f"[dim strike]{j.name[:21]}[/]",
                f"[{s_color} strike]{j.state}[/]",
                f"[dim strike]{elapsed_str}[/]",
                f"[dim strike]{j.partition}[/]",
            )
        elif j.state in ("RUNNING", "COMPLETING"):
            label = "RUNNING" if j.state == "RUNNING" else "FINISHING"
            is_own = j.user == current_user
            run_color = "green" if is_own else "cyan"
            user_markup = f"[bold]{j.user[:8]}[/]" if is_own else f"[dim]{j.user[:8]}[/]"
            remaining = _time_remaining(j)
            if remaining is not None:
                t_color = "red" if remaining < 1800 else "yellow" if remaining < 3600 else "green"
                time_cell = f"[{t_color}]{_fmt_duration(remaining)}[/]"
            else:
                time_cell = f"[dim]∞ {j.time_used}[/]"
            table.add_row(
                prefix,
                j.job_id,
                user_markup,
                j.name[:20],
                f"[{run_color}]{spin} {label}[/]",
                time_cell,
                f"[dim]{j.partition}[/]",
            )
        elif j.state == "PENDING":
            is_own = j.user == current_user
            user_markup = f"[bold]{j.user[:8]}[/]" if is_own else f"[dim]{j.user[:8]}[/]"
            table.add_row(
                prefix,
                j.job_id,
                user_markup,
                j.name[:20],
                "[yellow]⋯ PENDING[/]",
                "[dim]─[/]",
                f"[dim]{j.partition}[/]",
            )
        else:
            is_own = j.user == current_user
            user_markup = f"[bold]{j.user[:8]}[/]" if is_own else f"[dim]{j.user[:8]}[/]"
            table.add_row(
                prefix,
                j.job_id,
                user_markup,
                j.name[:20],
                f"[dim]{j.state}[/]",
                f"[dim]{j.time_used}[/]",
                f"[dim]{j.partition}[/]",
            )

    return table


# ── Dashboard layout ──────────────────────────────────────────────────────────

def _build_layout(
    jobs: list[QueueEntry],
    selected_idx: int,
    log_lines: list[str],
    log_path: str | None,
    node_stats: NodeStats | None,
    current_user: str = "",
    is_jupyter: bool = False,
    log_scroll: int = 0,
    is_admin: bool = False,
) -> Layout:
    layout = Layout()
    jobs_height = min(len(jobs) + 6, 16)

    layout.split_column(
        Layout(name="cluster", size=3),
        Layout(name="jobs",    size=jobs_height),
        Layout(name="log"),
        Layout(name="footer",  size=1),
    )

    layout["cluster"].update(_build_cluster_panel(node_stats))

    if jobs:
        layout["jobs"].update(
            Panel(_build_jobs_table(jobs, selected_idx, current_user),
                  title="[bold]Job Queue[/bold]", border_style="cyan")
        )
    else:
        layout["jobs"].update(
            Panel("[dim]No jobs in queue or history.[/]",
                  title="[bold]Job Queue[/bold]", border_style="cyan")
        )

    selected_job = jobs[selected_idx] if jobs and selected_idx < len(jobs) else None
    is_own_job = selected_job is not None and selected_job.user == current_user
    can_view_log = is_own_job or is_admin
    log_title = f"Output: {log_path}" if log_path else "Output"
    if not selected_job:
        log_body = "[dim]No job selected.[/]"
    elif not can_view_log:
        log_body = f"[dim]{selected_job.user}'s job — output not shown.[/dim]"
    elif log_lines:
        panel_h = max(3, console.height - jobs_height - 6)
        total = len(log_lines)
        # Tentative slice assuming no scroll-position header. If that clips
        # content (start > 0), the header itself will consume one of the
        # panel's rows, so redo the slice with one fewer row available —
        # otherwise the header pushes the last line of content past the
        # panel's fixed height and it gets silently cropped.
        scroll = min(log_scroll, max(0, total - panel_h))
        start = max(0, total - panel_h - scroll)
        if start > 0:
            avail = max(1, panel_h - 1)
            scroll = min(log_scroll, max(0, total - avail))
            start = max(0, total - avail - scroll)
            visible = log_lines[start:start + avail]
            end_line = start + len(visible)
            log_body = (
                f"[dim]↑ {start} lines above  ·  {start+1}–{end_line}/{total}"
                f"  ·  ↑↓=line  PgUp/PgDn=jump[/dim]\n"
                + "\n".join(visible)
            )
        else:
            visible = log_lines[start:start + panel_h]
            log_body = "\n".join(visible)
    elif selected_job.state == "CANCELLED":
        log_body = "[yellow]Job was cancelled.[/]"
    elif selected_job.state == "FAILED":
        log_body = "[red]Job failed — output not yet visible. Press R to refresh.[/]"
    elif selected_job.state == "COMPLETED":
        log_body = "[dim]Job completed — output not yet visible. Press R to refresh.[/]"
    else:
        log_body = "[dim]Waiting for job to start...[/]"

    # no_wrap: a log line wider than the panel would otherwise wrap onto
    # extra terminal rows the fixed-height panel_h slice above didn't budget
    # for, silently cropping later lines (e.g. the SSH-tunnel command/link)
    # off the bottom. Crop with an ellipsis instead so 1 source line = 1 row.
    log_text = Text.from_markup(log_body, overflow="ellipsis")
    log_text.no_wrap = True
    layout["log"].update(Panel(log_text, title=log_title, border_style="cyan"))
    _active = selected_job and selected_job.state not in ("COMPLETED", "FAILED", "CANCELLED")
    can_cancel = bool(_active and (is_own_job or is_admin))
    can_extend = (is_own_job and is_jupyter
                  and selected_job and selected_job.state == "RUNNING")
    cancel_hint = "[bold]C=cancel[/bold]" if can_cancel else "[dim]C=─[/dim]"
    extend_hint = "[bold]E=+2h[/bold]"   if can_extend else "[dim]E=─[/dim]"
    admin_tag   = "  [dim](admin)[/dim]" if is_admin else ""
    layout["footer"].update(
        f"[dim]  Q=quit   S=switch   {cancel_hint}   {extend_hint}   R=refresh"
        f"   ↑↓=scroll  PgUp/PgDn=jump{admin_tag}[/dim]"
    )
    return layout


# ── Hardware stats view ───────────────────────────────────────────────────────

def _build_hw_panel(stats: NodeStats | None) -> Panel:
    lines = [""]

    if stats is None:
        lines.append("  [dim]SLURM node unavailable[/]")
    elif stats.live_stats:
        # ── GPU ───────────────────────────────────────────────────────────
        gpu_pct   = float(stats.gpu_util)
        vram_pct  = stats.gpu_mem_used_mb / stats.gpu_mem_total_mb * 100 if stats.gpu_mem_total_mb else 0
        lines.append("  [bold]GPU[/bold]")
        lines.append(
            f"  Utilization   {_hw_bar(gpu_pct)}"
            f"  [bold]{gpu_pct:3.0f}%[/bold]"
            f"   [dim]{stats.gpu_temp}°C  {stats.gpu_power_w:.0f} W[/dim]"
        )
        lines.append(
            f"  VRAM          {_hw_bar(vram_pct)}"
            f"  [bold]{stats.gpu_mem_used_mb/1024:.1f} / {stats.gpu_mem_total_mb/1024:.0f} GB[/bold]"
        )
        lines.append("")

        # ── CPU ───────────────────────────────────────────────────────────
        cpu_pct = float(stats.cpu_util)
        lines.append(f"  [bold]CPU[/bold]  [dim]({stats.cpu_total} cores)[/dim]")
        lines.append(
            f"  Utilization   {_hw_bar(cpu_pct)}"
            f"  [bold]{cpu_pct:3.0f}%[/bold]"
        )
        lines.append(
            f"  Load avg      [dim]{stats.cpu_load:.2f}  ·  {stats.cpu_load5:.2f}[/dim]"
            f"   [dim](1 / 5 min)[/dim]"
        )
        lines.append("")

        # ── RAM ───────────────────────────────────────────────────────────
        mem_pct = stats.mem_used_mb / stats.mem_total_mb * 100 if stats.mem_total_mb else 0
        lines.append("  [bold]RAM[/bold]")
        lines.append(
            f"  Used          {_hw_bar(mem_pct)}"
            f"  [bold]{stats.mem_used_mb/1024:.1f} / {stats.mem_total_mb/1024:.0f} GB[/bold]"
            f"  [dim]({mem_pct:.0f}%)[/dim]"
        )
    else:
        lines.append("  [yellow]Live stats unavailable[/yellow]  "
                     "[dim]— iit-gpu-stats-writer not running on iit-MS-7E06[/dim]")
        lines.append("")
        lines.append("  [dim]Start it with:  python3 /usr/local/bin/iit-gpu-stats-writer &[/dim]")

    lines.append("")

    # ── SLURM allocation ──────────────────────────────────────────────────────
    if stats:
        node_state = stats.state.split("+")[0]
        sc = "green" if "IDLE" in node_state else "yellow" if "ALLOC" in node_state else "red"
        alloc_parts = [
            f"GPU {stats.shard_alloc}/{stats.shard_total} slices"
            if stats.shard_total else f"GPU {stats.gpu_alloc}/{stats.gpu_total}",
            f"CPU {stats.cpu_alloc}/{stats.cpu_total}",
            f"RAM {stats.mem_alloc_mb//1024}/{stats.mem_total_mb//1024} GB",
        ]
        lines.append(
            f"  [bold]SLURM[/bold]   [{sc}]{node_state}[/]  ·  "
            + "  ·  ".join(alloc_parts)
        )

    lines.append("")
    return Panel("\n".join(lines), title="[bold]Hardware Stats: iit-MS-7E06[/bold]", border_style="blue")


def run_hardware_stats() -> None:
    """Live hardware utilization view: GPU, CPU, RAM, SLURM. Q to quit."""
    _stats:   list[NodeStats | None] = [None]
    _last_ts: list[float]            = [0.0]

    def _refresh() -> None:
        _stats[0]   = get_node_stats()
        _last_ts[0] = time.monotonic()

    _refresh()

    old_settings = None
    if _HAS_TERMIOS and sys.stdin.isatty():
        try:
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except termios.error:
            old_settings = None

    try:
        with Live(console=console, refresh_per_second=_DISPLAY_FPS, screen=True) as live:
            while True:
                live.update(_build_hw_panel(_stats[0]))

                key = None
                if _HAS_TERMIOS:
                    try:
                        r, _, _ = select.select([sys.stdin], [], [], 1.0 / _DISPLAY_FPS)
                        if r:
                            key = sys.stdin.read(1).lower()
                    except (OSError, ValueError):
                        pass
                else:
                    time.sleep(1.0 / _DISPLAY_FPS)

                if key == "q":
                    break

                if time.monotonic() - _last_ts[0] >= _DATA_REFRESH_SECS:
                    _refresh()
    finally:
        if old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except termios.error:
                pass


# ── Keyboard ──────────────────────────────────────────────────────────────────

def _wait_key(timeout: float) -> str | None:
    if not _HAS_TERMIOS:
        time.sleep(timeout)
        return None
    try:
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if r:
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                r2, _, _ = select.select([sys.stdin], [], [], 0.05)
                if r2:
                    ch2 = sys.stdin.read(1)
                    if ch2 == '[':
                        r3, _, _ = select.select([sys.stdin], [], [], 0.05)
                        if r3:
                            ch3 = sys.stdin.read(1)
                            if ch3 == 'A':
                                return 'up'
                            if ch3 == 'B':
                                return 'down'
                            # Page Up = ESC[5~  Page Down = ESC[6~
                            if ch3 in ('5', '6'):
                                r4, _, _ = select.select([sys.stdin], [], [], 0.05)
                                if r4:
                                    sys.stdin.read(1)  # consume trailing '~'
                                return 'pgup' if ch3 == '5' else 'pgdn'
                return None  # swallow unrecognised escape sequences
            return ch.lower()
    except (OSError, ValueError):
        pass
    return None


# ── Merged job list ───────────────────────────────────────────────────────────

def _merged_jobs(jdir: str) -> list[QueueEntry]:
    live = queue(all_users=True)
    live_ids = {j.job_id for j in live}
    # filtered_history prefers sacct's authoritative state (all_users=True) and
    # only falls back to the file-scan heuristic in recent_jobs() when sacct is
    # disabled. The file-scan heuristic can't tell a clean scancel/exit from a
    # crash for jobs (like notebooks) that routinely log to stderr, so recent
    # jobs would otherwise show as FAILED even when sacct says CANCELLED.
    done_all = filtered_history(jdir, limit=_COMPLETED_HISTORY, all_users=True)
    done = [j for j in done_all if j.job_id not in live_ids]
    return live + done


# ── Main dashboard ────────────────────────────────────────────────────────────

def run_dashboard(job_id: str | None = None) -> None:
    """Show the live job dashboard. If job_id given, start with that job selected."""
    cfg = load_config()
    jdir = jobs_dir(cfg)
    current_user = _effective_user()
    from iitgpu.config import is_admin as _is_admin_fn
    _admin = _is_admin_fn(cfg)

    jobs: list[QueueEntry] = _merged_jobs(jdir)
    selected_idx = 0
    pinned_job_id: str | None = job_id

    if job_id is not None:
        for i, j in enumerate(jobs):
            if j.job_id == job_id:
                selected_idx = i
                break

    _node_stats:   list[NodeStats | None] = [None]
    _log_lines:    list[list[str]]        = [[]]
    _log_path_ref: list[str | None]       = [None]
    _last_data_ts: list[float]            = [0.0]
    _is_jupyter:   list[bool]             = [False]
    _log_scroll:   list[int]              = [0]
    _warned_jobs:  set[str]               = set()    # job IDs already warned

    def _refresh_data() -> None:
        nonlocal jobs, selected_idx
        _node_stats[0] = get_node_stats()
        jobs = _merged_jobs(jdir)
        if jobs and selected_idx >= len(jobs):
            selected_idx = len(jobs) - 1
        sel = jobs[selected_idx] if jobs and selected_idx < len(jobs) else None
        is_own = sel and sel.user == current_user
        lookup_id = sel.job_id if (is_own or _admin) else None
        owner_user = sel.user if (lookup_id and sel) else None
        if lookup_id is None and pinned_job_id:
            lookup_id = pinned_job_id
            owner_user = next((j.user for j in jobs if j.job_id == pinned_job_id),
                               current_user)
        if lookup_id:
            # Scope the log search to the job owner's folder — rglob-ing the
            # whole jobs_dir (every user's history) on every 2s refresh tick
            # was hammering NFS and made the dashboard crawl on the shared LAN.
            lines, path = _get_job_output(lookup_id, str(Path(jdir) / owner_user))
        else:
            lines, path = [], None
        _log_lines[0]    = lines
        _log_path_ref[0] = path
        _last_data_ts[0] = time.monotonic()
        # detect jupyter + 30-min warning (own jobs only)
        if sel and is_own and sel.job_id:
            _is_jupyter[0] = _is_jupyter_job(sel.job_id, str(Path(jdir) / sel.user))
            remaining = _time_remaining(sel)
            if (remaining is not None
                    and remaining < 1800
                    and sel.job_id not in _warned_jobs
                    and sel.state == "RUNNING"):
                _warned_jobs.add(sel.job_id)
                try:
                    from iitgpu import daemonclient as _dc
                    from iitgpu import mailer as _mailer
                    _email = _dc.email_for(current_user)
                    if _email:
                        from iitgpu.config import load_config as _lcfg
                        _cfg2 = _lcfg()
                        _mailer.send_jupyter_warning(
                            _email, sel.job_id, sel.name,
                            max(1, remaining // 60),
                            _cfg2.gateway_host, int(_cfg2.gateway_port),
                        )
                except Exception:
                    pass
        else:
            _is_jupyter[0] = False

    _refresh_data()

    old_settings = None
    if _HAS_TERMIOS and sys.stdin.isatty():
        try:
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except termios.error:
            old_settings = None

    try:
        with Live(console=console, refresh_per_second=_DISPLAY_FPS, screen=True) as live:
            while True:
                live.update(_build_layout(
                    jobs, selected_idx,
                    _log_lines[0], _log_path_ref[0],
                    _node_stats[0],
                    current_user,
                    _is_jupyter[0],
                    _log_scroll[0],
                    _admin,
                ))

                key = _wait_key(1.0 / _DISPLAY_FPS)

                if key == "q":
                    break
                elif key == "up":
                    _log_scroll[0] += 1
                elif key == "down":
                    _log_scroll[0] = max(0, _log_scroll[0] - 1)
                elif key == "pgup":
                    _log_scroll[0] += 10
                elif key == "pgdn":
                    _log_scroll[0] = max(0, _log_scroll[0] - 10)
                elif key == "s" and jobs:
                    selected_idx = (selected_idx + 1) % len(jobs)
                    _log_scroll[0] = 0
                    _refresh_data()
                elif key == "c":
                    sel = jobs[selected_idx] if jobs and selected_idx < len(jobs) else None
                    if sel and sel.user != current_user and not _admin:
                        live.stop()
                        err(f"Job {sel.job_id} belongs to {sel.user} — you can only cancel your own jobs.")
                        import time as _t; _t.sleep(1.5)
                        live.start()
                    elif sel:
                        live.stop()
                        import questionary
                        from questionary import Style
                        _s = Style([("question", "bold"), ("answer", "fg:magenta bold")])
                        _owner_note = f" [{sel.user}]" if sel.user != current_user else ""
                        if questionary.confirm(
                            f"Cancel job {sel.job_id} ({sel.name}){_owner_note}?",
                            default=False, style=_s,
                        ).ask():
                            success, msg = cancel(sel.job_id)
                            (ok if success else err)(msg)
                            if success:
                                from iitgpu import auditclient as _audit
                                _detail = "dashboard_admin" if sel.user != current_user else "dashboard"
                                _audit.log("job_cancel", detail=_detail, job_id=sel.job_id)
                        live.start()
                elif key == "e":
                    sel = jobs[selected_idx] if jobs and selected_idx < len(jobs) else None
                    if sel and sel.user == current_user and _is_jupyter[0] and sel.state == "RUNNING":
                        live.stop()
                        success, msg = extend_job_time(sel.job_id, 2)
                        (ok if success else err)(msg)
                        if success:
                            try:
                                from iitgpu import daemonclient as _dc2
                                from iitgpu import mailer as _m2
                                _email2 = _dc2.email_for(current_user)
                                if _email2:
                                    from iitgpu.slurm import show_job as _sj
                                    _details = _sj(sel.job_id)
                                    _new_lim = "unknown"
                                    for _line in _details.splitlines():
                                        if "TimeLimit=" in _line:
                                            _new_lim = _line.split("TimeLimit=")[1].split()[0]
                                            break
                                    _m2.send_jupyter_extended(_email2, sel.job_id, sel.name, _new_lim, 2)
                            except Exception:
                                pass
                            from iitgpu import auditclient as _audit2
                            _audit2.log("jupyter_extend", detail="+2h", job_id=sel.job_id)
                        import time as _t2; _t2.sleep(0.8)
                        _refresh_data()
                        live.start()
                elif key == "r":
                    _log_scroll[0] = 0
                    _refresh_data()

                if time.monotonic() - _last_data_ts[0] >= _DATA_REFRESH_SECS:
                    _refresh_data()

    finally:
        if old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except termios.error:
                pass
