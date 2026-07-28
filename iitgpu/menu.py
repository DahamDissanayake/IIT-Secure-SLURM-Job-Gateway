# iitgpu/menu.py
import questionary

from iitgpu.config import load_config, jobs_dir
from iitgpu.ui import BACK_TO_MAIN, STYLE, info, screen, select_menu

_MAIN_ITEMS = [
    "1. New Job       (JupyterLab, script, or shell — pick, review, launch)",
    "2. My Workspace  (files, models, environments)",
    "3. Jobs          (queue, history, logs, rerun)",
    "4. Settings      (health check, shell, cluster status, hardware)",
]


def run_menu() -> None:
    from iitgpu.config import load_config as _lc, is_admin as _ia
    _admin = _ia(_lc())
    while True:
        # Maintenance banner
        _maint = None
        try:
            from iitgpu.admin import get_maintenance
            _maint = get_maintenance()
        except Exception:
            pass
        if _maint:
            from iitgpu.ui import console
            from rich.panel import Panel
            _body = (
                "[bold yellow]MAINTENANCE[/]  "
                + _maint.get("reason", "") + "\n"
                + "[dim]Set by " + _maint.get("set_by", "?") + " at "
                + _maint.get("since", "")[:19] + " UTC[/]"
            )
            console.print(Panel(_body, border_style="yellow", expand=False))
            if not _admin:
                screen("Main Menu")
                info("[dim]The cluster is currently unavailable. Please try again later.[/]")
                questionary.select(
                    "Select an option:", choices=["Quit"], style=STYLE
                ).ask()
                info("Goodbye.")
                return

        screen("Main Menu")
        _choices = list(_MAIN_ITEMS)
        if _admin:
            _choices.append("5. Admin         (cluster ops, users, audit)")
        choice = select_menu("Select an option:", _choices, back="Quit")

        if choice is None:
            info("Goodbye.")
            return

        elif choice.startswith("1."):
            from iitgpu.wizard import run_wizard
            run_wizard()

        elif choice.startswith("2."):
            from iitgpu.workspace import run_workspace
            run_workspace()

        elif choice.startswith("3."):
            _jobs_menu()

        elif choice.startswith("4."):
            _settings_menu()

        elif choice.startswith("5."):
            from iitgpu.admin import admin_menu
            admin_menu()


def _jobs_status() -> str:
    """My queued/running count + free GPU pods — the two numbers someone
    opening the Jobs menu actually wants to know before picking a screen."""
    try:
        from iitgpu.slurm import get_node_stats, queue
        import getpass
        mine = queue(user=getpass.getuser())
        running = sum(1 for e in mine if e.state == "RUNNING")
        pending = sum(1 for e in mine if e.state == "PENDING")
        stats = get_node_stats()
        free = max(0, stats.shard_total - stats.shard_alloc) if stats and stats.shard_total else None
        parts = [f"[bold]My jobs:[/] {running} running, {pending} queued"]
        if free is not None:
            parts.append(f"[dim]·[/] {free}/{stats.shard_total} GPU pods free")
        return "  ".join(parts)
    except Exception:
        return "[dim]status unavailable[/]"


def _jobs_menu() -> None:
    from iitgpu.dashboard import run_dashboard, run_hardware_stats
    from iitgpu.monitor import (show_queue, manage_job, browse_and_tail_log,
                                show_history, rerun_job)

    while True:
        screen("Jobs", status=_jobs_status())
        choice = select_menu(
            "Jobs options:",
            [
                "Live dashboard  (auto-refresh)",
                "View queue",
                "Manage a job  (cancel/hold/release/requeue/details)",
                "View job log",
                "Job history  (filters)",
                "Rerun a job",
                questionary.Separator("─────────────────────"),
                "Hardware stats",
                "Usage & accounting",
                "My running services",
                "Cluster status",
            ],
            back=BACK_TO_MAIN,
        )

        if choice is None:
            return
        elif "Live dashboard" in choice:
            run_dashboard()
        elif choice == "View queue":
            show_queue()
        elif choice.startswith("Manage a job"):
            manage_job()
        elif choice == "View job log":
            browse_and_tail_log()
        elif choice.startswith("Job history"):
            show_history()
        elif choice == "Rerun a job":
            rerun_job()
        elif choice == "Hardware stats":
            run_hardware_stats()
        elif choice == "Usage & accounting":
            from iitgpu.accounting import usage_menu
            usage_menu()
        elif choice == "My running services":
            from iitgpu.notebooks import services_menu
            services_menu()
        elif choice == "Cluster status":
            _show_cluster_status()


def _settings_status(cfg) -> str:
    return f"[dim]NFS root:[/] {cfg.nfs_root}"


def _settings_menu() -> None:
    from iitgpu.config import load_config as _lc
    _cfg = _lc()

    while True:
        screen("Settings", status=_settings_status(_cfg))
        choice = select_menu(
            "Settings options:",
            [
                "Cluster health check",
                "Build environment",
                "Install prebuilt environment",
                "Run smoke test",
                "Advanced SLURM shell",
            ],
            back=BACK_TO_MAIN,
        )

        if choice is None:
            return
        elif choice == "Cluster health check":
            from iitgpu.setup import check_cluster_health
            from iitgpu.ui import console
            result = check_cluster_health(_cfg)
            console.print(result)
        elif choice == "Build environment":
            from iitgpu.setup import _run_env_setup
            _run_env_setup(_cfg)
        elif choice == "Install prebuilt environment":
            from iitgpu.setup import _run_install_prebuilt
            _run_install_prebuilt(_cfg)
        elif choice == "Run smoke test":
            from iitgpu.setup import _run_smoke_test
            _run_smoke_test(_cfg)
        elif choice == "Advanced SLURM shell":
            from iitgpu.shell import run_shell
            run_shell()


def _show_cluster_status() -> None:
    from iitgpu.slurm import get_partitions
    from iitgpu.ui import console, warn
    from rich.table import Table

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
        table.add_row(
            p.name, f"[{s}]{p.state}[/]", str(p.nodes), str(p.gpus_per_node)
        )
    console.print(table)
