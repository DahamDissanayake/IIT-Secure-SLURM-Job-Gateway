# iitgpu/review.py — the launch review hub. One screen, every field editable,
# availability shown where the decision is made (RunPod's pattern). Pure
# rendering is split from the loop so tests can capture it.
from __future__ import annotations

import questionary
from rich.markup import escape
from rich.panel import Panel

from iitgpu.jobs import SHARDS_PER_GPU, gpu_share_note
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


# Fields the submit path does NOT consume for a given intent. A row that shows a
# value the renderer discards is a lie, and an editor for it is a setting that
# silently does nothing — so both the summary panel and the menu drop them.
#   shell    — build_interactive_cmd() takes partition/cpus/mem/gres/time only:
#              no environment, no data/model, no array/dependency/mail. Spec §1
#              says a shell is size + time.
#   notebook — render_notebook_sbatch() ignores data_path and model_path (it has
#              no run_command to export them for). What a session really needs
#              pre-installed is packages, which gets its own row below.
_NOOP_FIELDS: dict[str, set[str]] = {
    "shell":    {"Script", "Environment", "Data / model", "Args", "Advanced"},
    "notebook": {"Script", "Data / model", "Args"},
    "batch":    set(),
}

_CHOICE_FOR_FIELD = {
    "Script":       "Change script",
    "Environment":  "Change environment",
    "Data / model": "Change data / model",
    "Args":         "Change args",
    "Advanced":     "Advanced…",
}

_PKG_CHOICE = "Change python packages"


def _noop_fields(ls: LaunchSpec) -> set[str]:
    return _NOOP_FIELDS.get(ls.intent, set())


def _vram_note(ls: LaunchSpec, stats) -> str:
    """The VRAM caveat, with the actual per-shard share when the node reports it.

    This used to hardcode "about 8 GB of 32" no matter how much of the card the
    job had reserved, so a Whole-GPU job was told it owned the card and got an
    eighth of it in the same panel. The arithmetic mirrors `_vram_check()` in
    wizard.py, which has always done this correctly at submit time.
    """
    base = "GPU memory is shared between jobs and not enforced"
    total_mb = getattr(stats, "gpu_mem_total_mb", 0) if stats else 0
    if not (stats and getattr(stats, "live_stats", False) and total_mb):
        return f"{base}."
    total_gb = total_mb / 1024
    shards = max(1, min(ls.gpu_shards, SHARDS_PER_GPU))
    share_gb = total_gb * shards / SHARDS_PER_GPU
    return (f"{base} — your fair share is about "
            f"{share_gb:.0f} GB of {total_gb:.0f}.")


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
    if ls.intent == "notebook":   # the one dependency question a session has
        rows.append(("Packages", escape(ls.requirements or ls.packages or "(none)")))
    if ls.intent == "batch":   # nothing else has a command line to carry args
        rows.append(("Args", escape(ls.args or "(none)")))
    rows.append(
        ("Advanced", "on" if (ls.array or ls.dependency or not ls.mail) else "off"))
    hidden = _noop_fields(ls)
    rows = [(k, v) for k, v in rows if k not in hidden]
    body = "\n".join(f"  [bold]{k:<12}[/] {v}" for k, v in rows)
    share = gpu_share_note(ls.gpu_shards)
    body += f"\n\n  [dim]{share}[/]\n  [dim]{_vram_note(ls, stats)}[/]"
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
    if mins >= 60:
        warn("Not HH:MM — keeping the current limit.")
        return
    # The cap is on the total, not on the hours: QOS `normal` carries
    # MaxWallDurationPerJob=08:00:00 with Flags=DenyOnLimit, so "8:30" is 500
    # minutes and sbatch rejects it outright — after the job folder exists and
    # with no error text that points at the time limit.
    total_minutes = hours * 60 + mins
    if total_minutes == 0:
        warn("Not HH:MM — keeping the current limit.")
        return
    if total_minutes > 480:
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

    def _set(kind: str, *, conda: str = "", venv: str = "", image: str = "") -> None:
        # Only ever clear the other three once a choice has actually produced a
        # usable value — a rejected image must leave the environment as it was.
        ls.env_kind = kind
        ls.conda_env, ls.venv_path, ls.container_image = conda, venv, image

    if sel.startswith("prebuilt: "):
        name = sel.split(": ", 1)[1]
        _set("prebuilt", conda=str(envs_dir / name))
    elif sel.startswith("own conda"):
        _set("conda", conda=questionary.text("Conda env path:").ask() or "")
    elif sel.startswith("own venv"):
        _set("venv", venv=questionary.text("Venv path:").ask() or "")
    elif sel.startswith("container"):
        # The image path lands unquoted in the generated `apptainer exec` line,
        # so it gets the same check the old wizard applied: inside the jail and
        # actually a .sif.
        from iitgpu.containers import validate_image
        raw = (questionary.text("Full path to .sif image:").ask() or "").strip()
        if not validate_image(raw):
            warn("Image path is outside the allowed jail or not a .sif — "
                 "keeping the current environment.")
            return
        _set("container", image=raw)
    else:
        _set("none")


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
        raw = (questionary.text("Model path (or HF repo id):").ask() or "").strip()
        if not raw:
            return
        # A local path is exported verbatim into the job script (MODEL_PATH /
        # HF_HOME), so it faces the jail; an HF repo id ("org/name") is not a
        # path and passes through, exactly as the old wizard treated it.
        if raw.startswith("/"):
            from iitgpu.validate import in_jail
            if not in_jail(raw):
                warn("Path is outside the allowed jail — keeping the current model.")
                return
        ls.model_path = raw
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


def _edit_advanced(ls: LaunchSpec, preview=None) -> None:
    from iitgpu.validate import clean_array_spec, clean_dependency
    while True:
        opts = []
        # render_notebook_sbatch() emits neither --array nor --dependency: a
        # JupyterLab session is one interactive allocation, and both settings
        # would be accepted here and thrown away at submit.
        if ls.intent != "notebook":
            opts += [f"job array [{ls.array or 'off'}]",
                     f"run after job [{ls.dependency or 'off'}]"]
        opts += [f"email notifications [{'on' if ls.mail else 'off'}]",
                 "view generated sbatch", "back"]
        sel = questionary.select("Advanced:", choices=opts).ask()
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
            if preview is None:
                # No renderer was wired in for this launch (a shell allocation
                # has no sbatch at all) — say so instead of promising a script
                # that never appears.
                info("No sbatch preview is available here — this launch does "
                     "not generate a job script.")
                continue
            try:
                text = preview(ls)
            except Exception as exc:      # a preview must never sink the wizard
                warn(f"Could not build the preview: {exc}")
                continue
            console.print(Panel(escape(text), title="[bold] Generated sbatch "
                                "(preview — folder assigned at submit) [/bold]",
                                border_style="cyan"))


_HUB_CHOICES = ["🚀 Launch", "Change script", "Change size", "Change time limit",
                "Change environment", "Change data / model", "Change args",
                "Advanced…", "Save as template", "Cancel"]


def run_hub(ls: LaunchSpec, cfg, user: str, *, browse_script, browse_data,
            deps_prompt=None, preview=None) -> str | None:
    """Loop until Launch / Save as template / Cancel. Mutates ls in place.

    *preview* is an optional `LaunchSpec -> str` renderer for the Advanced menu's
    "view generated sbatch". It is injected rather than imported so review.py
    keeps knowing nothing about wizard.py (the import cycle Task 6 removed).
    """
    while True:
        stats = None
        try:
            stats = get_node_stats()
        except Exception:
            pass
        console.print(render_hub(ls, stats))
        # Same rule as the panel above: never offer an editor for a field this
        # intent's submit path throws away (see _NOOP_FIELDS).
        hidden = {_CHOICE_FOR_FIELD[f] for f in _noop_fields(ls)
                  if f in _CHOICE_FOR_FIELD}
        choices = [c for c in _HUB_CHOICES if c not in hidden]
        if ls.intent == "notebook" and deps_prompt is not None:
            # A session's data/model rows are gone, but what to pip-install
            # before the first cell runs is real — render_notebook_sbatch()
            # consumes it — so it gets its own row.
            choices.insert(choices.index("Change environment") + 1, _PKG_CHOICE)
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
        elif sel == _PKG_CHOICE:
            ls.requirements, ls.packages = deps_prompt()
        elif sel == "Change args":
            _edit_args(ls)
        elif sel == "Advanced…":
            _edit_advanced(ls, preview)
