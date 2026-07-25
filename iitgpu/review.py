# iitgpu/review.py — the launch review hub. One screen, every field editable,
# availability shown where the decision is made (RunPod's pattern). Pure
# rendering is split from the loop so tests can capture it.
from __future__ import annotations

import questionary
from rich.markup import escape
from rich.panel import Panel

from iitgpu.jobs import gpu_share_note
from iitgpu.launchspec import (SIZES, LaunchSpec, apply_size, availability_line,
                               size_availability, size_label)
from iitgpu.slurm import get_node_stats
from iitgpu.ui import console, info, warn

_STYLE = None  # set by wizard at import-time hookup if desired; questionary default otherwise

_TIME_PRESETS = [("1h", "01:00:00"), ("2h", "02:00:00"),
                 ("4h", "04:00:00"), ("8h (cluster max)", "08:00:00")]


def _fmt_time(t: str) -> str:
    for label, val in _TIME_PRESETS:
        if val == t:
            return label.split()[0]
    return t


def _env_display(ls: LaunchSpec) -> str:
    return (ls.container_image or ls.conda_env or ls.venv_path
            or ("(none — system python)" if ls.env_kind == "none" else "(not set)"))


def render_hub(ls: LaunchSpec, stats) -> Panel:
    # Every value below that can contain user-supplied text (script name/path,
    # env paths, data/model, args) is wrapped in escape() — this is the one
    # screen whose job is to show exactly what will launch, so a filename or
    # --arg containing "[...]" must render literally, not as markup/color.
    rows = []
    if ls.intent == "batch":
        from pathlib import Path
        p = Path(ls.script)
        rows.append(("Script", f"{escape(p.name)}   ({escape(str(p.parent))})"
                     if ls.script else "(not set)"))
    rows += [
        ("Environment", escape(_env_display(ls))),
        ("Size", f"{size_label(ls)}   [dim]{size_availability(ls.gpu_shards, stats)}[/]"),
        ("Time limit", _fmt_time(ls.time_limit)),
        ("Data / model", escape(ls.data_path or ls.model_path or "(none)")),
    ]
    if ls.intent == "batch":   # nothing else has a command line to carry args
        rows.append(("Args", escape(ls.args or "(none)")))
    rows.append(
        ("Advanced", "on" if (ls.array or ls.dependency or not ls.mail) else "off"))
    body = "\n".join(f"  [bold]{k:<12}[/] {v}" for k, v in rows)
    share = gpu_share_note(ls.gpu_shards)
    vram = ("GPU memory is shared between jobs and not enforced — "
            "your fair share is about 8 GB of 32.")
    body += f"\n\n  [dim]{share}[/]\n  [dim]{vram}[/]"
    return Panel(body, title=f"[bold] Ready to launch ─ {availability_line(stats)} [/bold]",
                 border_style="cyan")


def _edit_size(ls: LaunchSpec, stats) -> None:
    choices, mapping = [], {}
    for key in ("standard", "small", "whole"):
        s = SIZES[key]
        probe = LaunchSpec(intent=ls.intent, gpu_shards=s.gpu_shards,
                           cpus=s.cpus, mem_gb=s.mem_gb)
        label = f"{size_label(probe)}  {size_availability(s.gpu_shards, stats)}"
        choices.append(label); mapping[label] = key
    sel = questionary.select("Size:", choices=choices).ask()
    if sel:
        apply_size(ls, mapping[sel])


def _edit_time(ls: LaunchSpec) -> None:
    labels = [l for l, _ in _TIME_PRESETS] + ["custom (HH:MM)"]
    sel = questionary.select("Time limit:", choices=labels).ask()
    if sel is None:
        return
    for label, val in _TIME_PRESETS:
        if sel == label:
            ls.time_limit = val
            return
    raw = questionary.text("Time limit (HH:MM):").ask() or ""
    import re
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", raw.strip())
    if not m:
        warn("Not HH:MM — keeping the current limit.")
        return
    hours, mins = int(m.group(1)), int(m.group(2))
    if mins >= 60 or (hours == 0 and mins == 0):
        warn("Not HH:MM — keeping the current limit.")
        return
    if hours > 8:
        warn("Max is 8h (cluster QOS limit) — keeping the current limit.")
        return
    ls.time_limit = f"{hours:02d}:{mins:02d}:00"


def _edit_env(ls: LaunchSpec, cfg) -> None:
    from pathlib import Path
    envs_dir = Path(getattr(cfg, "nfs_root", "/shared")) / "envs"
    prebuilt: list[str] = []
    if envs_dir.is_dir():
        try:
            prebuilt = sorted(str(p) for p in envs_dir.iterdir() if p.is_dir())
        except OSError:
            prebuilt = []   # permission denied etc. — menu still offers own-path/container/none
    choices = [f"prebuilt: {Path(p).name}" for p in prebuilt] + [
        "own conda env (path)", "own venv (path)", "container image (.sif)", "none (system python)"]
    sel = questionary.select("Environment:", choices=choices).ask()
    if sel is None:
        return
    ls.conda_env = ls.venv_path = ls.container_image = ""
    if sel.startswith("prebuilt: "):
        name = sel.split(": ", 1)[1]
        ls.env_kind, ls.conda_env = "prebuilt", str(envs_dir / name)
    elif sel.startswith("own conda"):
        ls.env_kind = "conda"
        ls.conda_env = questionary.text("Conda env path:").ask() or ""
    elif sel.startswith("own venv"):
        ls.env_kind = "venv"
        ls.venv_path = questionary.text("Venv path:").ask() or ""
    elif sel.startswith("container"):
        ls.env_kind = "container"
        ls.container_image = questionary.text("Full path to .sif image:").ask() or ""
    else:
        ls.env_kind = "none"


def _wants_deps(ls: LaunchSpec) -> bool:
    """Both notebook shapes install deps before the first cell runs: the live
    JupyterLab session, and a .ipynb submitted as a batch job."""
    return ls.intent == "notebook" or ls.script.endswith(".ipynb")


def _edit_data_model(ls: LaunchSpec, browse_data, deps_prompt) -> None:
    opts = ["data folder (browse)", "clear data", "model path (text)", "clear model"]
    if _wants_deps(ls) and deps_prompt is not None:
        opts.append("python packages to pre-install")
    sel = questionary.select("Data / model:", choices=opts + ["back"]).ask()
    if sel == "data folder (browse)":
        picked = browse_data()
        if picked:
            ls.data_path = picked
    elif sel == "clear data":
        ls.data_path = ""
    elif sel == "model path (text)":
        ls.model_path = questionary.text("Model path (or HF repo id):").ask() or ls.model_path
    elif sel == "clear model":
        ls.model_path = ""
    elif sel == "python packages to pre-install":
        req, pkgs = deps_prompt()
        ls.requirements, ls.packages = req, pkgs


def _edit_args(ls: LaunchSpec) -> None:
    from iitgpu.validate import clean_run_command
    raw = questionary.text("Extra arguments (blank = none):", default=ls.args).ask()
    if raw is not None:
        ls.args = clean_run_command(raw) if raw.strip() else ""


def _edit_advanced(ls: LaunchSpec) -> None:
    from iitgpu.validate import clean_array_spec, clean_dependency
    while True:
        sel = questionary.select("Advanced:", choices=[
            f"job array [{ls.array or 'off'}]",
            f"run after job [{ls.dependency or 'off'}]",
            f"email notifications [{'on' if ls.mail else 'off'}]",
            "view generated sbatch",
            "back",
        ]).ask()
        if sel is None or sel == "back":
            return
        if sel.startswith("job array"):
            raw = questionary.text("Array spec (e.g. 0-9 or 1-100%4, blank = off):",
                                   default=ls.array).ask() or ""
            ls.array = clean_array_spec(raw) or ""
        elif sel.startswith("run after"):
            raw = questionary.text("Parent job ID (blank = off):").ask() or ""
            ls.dependency = (clean_dependency(f"afterok:{raw.strip()}") or ""
                             if raw.strip().isdigit() else "")
        elif sel.startswith("email"):
            ls.mail = not ls.mail
        elif sel.startswith("view generated"):
            info("The final script is shown here after Launch builds it; "
                 "fields above fully determine it.")


_HUB_CHOICES = ["🚀 Launch", "Change script", "Change size", "Change time limit",
                "Change environment", "Change data / model", "Change args",
                "Advanced…", "Save as template", "Cancel"]


def run_hub(ls: LaunchSpec, cfg, user: str, *, browse_script, browse_data,
            deps_prompt=None) -> str | None:
    """Loop until Launch / Save as template / Cancel. Mutates ls in place."""
    while True:
        stats = None
        try:
            stats = get_node_stats()
        except Exception:
            pass
        console.print(render_hub(ls, stats))
        # Script and args only exist for a batch job — a JupyterLab session or a
        # shell has no command line to put them on, so offering the rows would
        # be offering a setting that silently does nothing.
        choices = [c for c in _HUB_CHOICES
                   if not (c in ("Change script", "Change args")
                           and ls.intent != "batch")]
        sel = questionary.select("Select:", choices=choices).ask()
        if sel is None or sel == "Cancel":
            return None
        if sel == "🚀 Launch":
            if ls.intent == "batch" and not ls.script:
                warn("Pick a script first (Change script).")
                continue
            return "launch"
        if sel == "Save as template":
            return "template"
        if sel == "Change script":
            picked = browse_script()
            if picked:
                ls.script = picked
        elif sel == "Change size":
            _edit_size(ls, stats)
        elif sel == "Change time limit":
            _edit_time(ls)
        elif sel == "Change environment":
            _edit_env(ls, cfg)
        elif sel == "Change data / model":
            _edit_data_model(ls, browse_data, deps_prompt)
        elif sel == "Change args":
            _edit_args(ls)
        elif sel == "Advanced…":
            _edit_advanced(ls)
