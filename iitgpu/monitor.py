# iitgpu/monitor.py
import getpass
import shlex
from pathlib import Path

import questionary
from rich.table import Table

from iitgpu import auditclient
from iitgpu.config import is_admin, load_config
from iitgpu.slurm import (cancel, hold, release, requeue, queue,
                          job_detail, job_efficiency, filtered_history)
from iitgpu.ui import (BACK_TO_MAIN, STYLE, console, err, info, kv, ok,
                       screen, select_menu, warn)
from iitgpu.validate import in_jail, safe_listdir

_STYLE = STYLE


def show_queue() -> None:
    screen("My Job Queue")
    entries = queue(user=getpass.getuser())
    if not entries:
        info("No jobs in queue.")
        return
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Job ID", style="magenta")
    table.add_column("Name")
    table.add_column("State", style="cyan")
    table.add_column("Partition")
    table.add_column("Time Used")
    table.add_column("Nodes")
    for e in entries:
        s = "green" if e.state == "RUNNING" else "yellow" if e.state == "PENDING" else "red"
        table.add_row(e.job_id, e.name, f"[{s}]{e.state}[/]", e.partition, e.time_used, str(e.nodes))
    console.print(table)


def cancel_job() -> None:
    """Back-compat alias — opens the full job-management menu."""
    manage_job()


def manage_job() -> None:
    """Admins (gpuadmins) see and manage every user's jobs here, matching
    the live dashboard's admin-cancel behaviour; everyone else is scoped to
    their own jobs only, as before."""
    screen("Manage Job")
    current_user = getpass.getuser()
    admin = is_admin(load_config())
    entries = queue(all_users=True) if admin else queue(user=current_user)
    if not entries:
        info("No active jobs.")
        return
    choices = [
        f"{e.job_id}  {e.name}  [{e.state}]" + (f"  ({e.user})" if admin else "")
        for e in entries
    ]
    choice = select_menu("Select a job:", choices)
    if choice is None:
        return
    job_id = choice.split()[0]
    sel = next((e for e in entries if e.job_id == job_id), None)
    is_own = sel is not None and sel.user == current_user

    # Defense in depth: non-admins only ever see their own jobs above, but
    # guard the mutating actions on ownership regardless.
    if not is_own and not admin:
        err(f"Job {job_id} belongs to {sel.user if sel else 'another user'} "
            "— you can only manage your own jobs.")
        return

    action = select_menu(
        f"Action for job {job_id}:",
        ["Cancel", "Hold", "Release", "Requeue", "Details + efficiency"])
    if action is None:
        return

    if action == "Details + efficiency":
        from iitgpu.ui import console
        screen(f"Job {job_id} detail")
        console.print(job_detail(job_id))
        console.print()
        console.print("[bold cyan]── Efficiency (seff) ──[/]")
        console.print(job_efficiency(job_id))
        questionary.press_any_key_to_continue("").ask()
        return

    _ACTIONS = {
        "Cancel":  (cancel,  "job_cancel"),
        "Hold":    (hold,    "job_hold"),
        "Release": (release, "job_release"),
        "Requeue": (requeue, "job_requeue"),
    }
    fn, audit_action = _ACTIONS[action]
    _owner_note = f" [{sel.user}]" if (sel and not is_own) else ""
    if action in ("Cancel", "Requeue") and not questionary.confirm(
        f"{action} job {job_id}{_owner_note}?", default=False, style=_STYLE
    ).ask():
        return
    _detail = "user_requested_admin" if not is_own else "user_requested"
    auditclient.log(audit_action, detail=_detail, job_id=job_id)
    success, msg = fn(job_id)
    (ok if success else err)(msg)


def tail_log(log_path: str, lines: int | None = None) -> None:
    """Display a job log.

    By default the FULL log is shown through a pager (less) so it can be
    scrolled and searched with `/` — important for analyzing failures that
    happen early in the run (e.g. an import traceback at the top, which a
    bottom-only tail would hide). Pass an int to show only the last N lines.
    """
    if not in_jail(log_path):
        err("Access denied: log path is outside allowed directories.")
        return
    p = Path(log_path)
    if not p.exists():
        warn(f"Log file not found: {log_path}")
        return
    try:
        text = p.read_text(errors="replace")
    except OSError as exc:
        err(f"Cannot read log: {exc}")
        return

    all_lines = text.splitlines()
    if lines is not None:
        screen(f"Log: {p.name}  (last {min(lines, len(all_lines))} of {len(all_lines)} lines)")
        for line in all_lines[-lines:]:
            console.print(line, markup=False, highlight=False)
        return

    # Full log via pager so the whole thing is scrollable + searchable.
    screen(f"Log: {p.name}  ({len(all_lines)} lines)  —  arrows/PgUp to scroll, '/' to search, 'q' to quit")
    with console.pager(styles=False):
        # markup/highlight off: log text (tracebacks, "[Errno 13]", etc.) must
        # render literally and not be interpreted as Rich markup.
        for line in all_lines:
            console.print(line, markup=False, highlight=False)


def browse_and_tail_log() -> None:
    screen("View Job Log")
    from iitgpu.config import load_config, jobs_dir
    cfg = load_config()
    user_dir = str(Path(jobs_dir(cfg)) / getpass.getuser())
    folders = safe_listdir(user_dir)
    if not folders:
        info("No job folders found.")
        return
    choice = select_menu("Select job folder:", sorted(folders, reverse=True))
    if choice is None:
        return
    job_folder = str(Path(user_dir) / choice)
    if not in_jail(job_folder):
        err("Access denied.")
        return
    logs = [f for f in safe_listdir(job_folder) if f.endswith(".out") or f.endswith(".err")]
    if not logs:
        info("No log files in that folder.")
        return
    log_choice = select_menu("Select log file:", logs)
    if log_choice is None:
        return
    tail_log(str(Path(job_folder) / log_choice))


def cluster_status() -> None:
    from iitgpu.slurm import get_partitions
    screen("Cluster Status")
    partitions = get_partitions()
    if not partitions:
        warn("Could not retrieve partition info.")
        return
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Partition", style="magenta")
    table.add_column("State", style="cyan")
    table.add_column("Nodes")
    table.add_column("GPUs/Node")
    for p in partitions:
        s = "green" if p.state == "up" else "red"
        table.add_row(p.name, f"[{s}]{p.state}[/]", str(p.nodes), str(p.gpus_per_node))
    console.print(table)


def monitor_menu() -> None:
    while True:
        screen("Monitor")
        choice = select_menu(
            "Monitor options:",
            [
                "View my queue",
                "Cancel a job",
                "View job log",
                "View hardware stats",
            ],
            back=BACK_TO_MAIN,
        )
        if choice is None:
            return
        if choice == "View my queue":
            show_queue()
        elif choice == "Cancel a job":
            cancel_job()
        elif choice == "View job log":
            browse_and_tail_log()
        elif choice == "View hardware stats":
            from iitgpu.dashboard import run_hardware_stats
            run_hardware_stats()


def follow_log() -> None:
    """Live-follow a running job's output (like tail -f). Ctrl-C to stop."""
    import time
    from iitgpu.config import load_config, jobs_dir
    cfg = load_config()
    user_dir = str(Path(jobs_dir(cfg)) / getpass.getuser())
    folders = safe_listdir(user_dir)
    if not folders:
        info("No job folders found.")
        return
    choice = select_menu("Follow which job folder?", sorted(folders, reverse=True))
    if choice is None:
        return
    folder = str(Path(user_dir) / choice)
    if not in_jail(folder):
        err("Access denied."); return
    logs = [f for f in safe_listdir(folder) if f.endswith(".out")]
    if not logs:
        info("No .out file yet."); return
    log_path = str(Path(folder) / sorted(logs)[0])
    screen(f"Following {Path(log_path).name}  (Ctrl-C to stop)")
    try:
        pos = 0
        for _ in range(100000):  # bounded so it can't run forever in a TUI
            try:
                with open(log_path, "r", errors="replace") as fh:
                    fh.seek(pos)
                    chunk = fh.read()
                    pos = fh.tell()
                if chunk:
                    console.print(chunk, end="")
            except OSError:
                pass
            time.sleep(1.0)
    except KeyboardInterrupt:
        console.print("\n[dim]— stopped following —[/]")


def show_history() -> None:
    """Completed-job history with state filter and user scope."""
    from iitgpu.config import load_config, jobs_dir, is_admin
    from rich.table import Table
    cfg = load_config()
    screen("Job History")
    state = questionary.select(
        "Filter by state:",
        choices=["All", "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"], style=_STYLE,
    ).ask()
    if state is None:
        return
    all_users = False
    if is_admin(cfg):
        all_users = questionary.confirm("Show ALL users (admin)?", default=False, style=_STYLE).ask()
    rows = filtered_history(jobs_dir(cfg), limit=50,
                            state=None if state == "All" else state,
                            all_users=all_users)
    if not rows:
        info("No matching history."); return
    table = Table(show_header=True, header_style="bold cyan")
    for col in ("Job ID", "User", "Name", "State", "Elapsed"):
        table.add_column(col)
    for e in rows:
        sc = "green" if e.state == "COMPLETED" else "red" if e.state in ("FAILED","TIMEOUT") else "yellow"
        table.add_row(e.job_id, e.user, e.name, f"[{sc}]{e.state}[/]", e.time_used)
    console.print(table)
def _parse_sbatch(sbatch_text: str) -> dict:
    """Parse key fields from an sbatch script into a dict for wizard prefill."""
    import re
    result: dict = {}

    # SBATCH directives
    for line in sbatch_text.splitlines():
        m = re.match(r"#SBATCH\s+--partition=(.+)", line)
        if m:
            result["partition"] = m.group(1).strip()
        # Jobs request GPU slices (shard:N); older scripts on disk still say
        # gpu:N, so accept both when re-running an archived job.
        m = re.match(r"#SBATCH\s+--gres=(?:shard|gpu):(\d+)", line)
        if m:
            result["gpu_shards"] = int(m.group(1))
        m = re.match(r"#SBATCH\s+--cpus-per-task=(\d+)", line)
        if m:
            result["cpus"] = int(m.group(1))
        m = re.match(r"#SBATCH\s+--mem=(\d+)G", line)
        if m:
            result["mem_gb"] = int(m.group(1))
        m = re.match(r"#SBATCH\s+--time=(.+)", line)
        if m:
            result["time_limit"] = m.group(1).strip()
        m = re.match(r"#SBATCH\s+--array=(.+)", line)
        if m:
            result["array"] = m.group(1).strip()
        m = re.match(r"#SBATCH\s+--dependency=(.+)", line)
        if m:
            result["dependency"] = m.group(1).strip()

    # conda activate <path>
    for line in sbatch_text.splitlines():
        m = re.match(r"\s*conda\s+activate\s+(\S+)", line)
        if m:
            result["conda_env"] = m.group(1).strip()
            break

    # apptainer exec ... <image.sif>
    for line in sbatch_text.splitlines():
        m = re.search(r"apptainer\s+exec\s+.*?(\S+\.sif)", line)
        if m:
            result["container_image"] = m.group(1).strip()
            break

    # export DATA_PATH=<path>
    for line in sbatch_text.splitlines():
        m = re.match(r"\s*export\s+DATA_PATH=(.+)", line)
        if m:
            result["data_path"] = m.group(1).strip()
            break

    # run_command: last non-comment, non-blank, non-export, non-source, non-cd line
    run_cmd = ""
    for line in sbatch_text.splitlines():
        stripped = line.strip()
        if (stripped
                and not stripped.startswith("#")
                and not stripped.startswith("export ")
                and not stripped.startswith("source ")
                and not stripped.startswith("cd ")
                and not stripped.startswith("conda ")
                and not stripped.startswith("module ")
                and not stripped.startswith("_conda_sh")
                and not stripped.startswith("[")
                and not stripped.startswith("echo ")
                and not stripped.startswith("JUPYTER")
                and not stripped.startswith("apptainer ")):
            run_cmd = stripped
    if run_cmd:
        result["run_command"] = run_cmd
        # Script path and arguments, quote-aware. The wizard shlex.quotes the
        # script path, so a notebook or script with a space in its name arrives
        # as `python3 '/a/my train.py' --lr 3`. Splitting on whitespace hands
        # back "'/a/my" — a path that does not exist, offered to the user as
        # the thing they are about to re-run.
        try:
            parts = shlex.split(run_cmd)
        except ValueError:          # unbalanced quotes: leave it unparsed
            parts = []
        if len(parts) >= 2 and parts[0] in ("python", "python3", "bash"):
            result["script_path"] = parts[1]
            if len(parts) > 2:
                result["extra_args"] = " ".join(shlex.quote(a) for a in parts[2:])

    return result


def rerun_job() -> None:
    """Pick a previous job folder, parse its sbatch, and relaunch via wizard."""
    from iitgpu.config import load_config, jobs_dir
    cfg = load_config()
    user_dir = str(Path(jobs_dir(cfg)) / getpass.getuser())
    folders = safe_listdir(user_dir)
    if not folders:
        info("No job folders found.")
        return
    choice = select_menu("Rerun which job?", sorted(folders, reverse=True))
    if choice is None:
        return

    job_folder = str(Path(user_dir) / choice)
    if not in_jail(job_folder):
        err("Access denied.")
        return

    sbatch_file = Path(job_folder) / "job.sbatch"
    if not sbatch_file.exists():
        warn(f"No job.sbatch found in {choice}")
        return

    try:
        sbatch_text = sbatch_file.read_text(errors="replace")
    except OSError as exc:
        err(f"Cannot read sbatch: {exc}")
        return

    prefill = _parse_sbatch(sbatch_text)
    auditclient.log("job_rerun", detail=choice)

    from iitgpu.wizard import run_wizard
    run_wizard(prefill=prefill)
