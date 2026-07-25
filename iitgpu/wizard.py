# iitgpu/wizard.py
from __future__ import annotations

import getpass
import os
import re
import shlex
import shutil
from pathlib import Path

import questionary
from questionary import Style

from iitgpu import auditclient
from iitgpu.config import load_config, jobs_dir, user_dir
from iitgpu.jobs import (SHARDS_PER_GPU, JobSpec, build_interactive_cmd,
                         gpu_share_note, make_job_folder, pip_install_block,
                         render_notebook_sbatch, render_sbatch, resource_defaults)
from iitgpu.launchspec import (LaunchSpec, default_spec, from_rerun, from_template,
                               recent_scripts, to_job_spec)
from iitgpu.review import run_hub
from iitgpu.slurm import submit_job, get_node_stats
from iitgpu.ui import err, header, info, kv, ok, panel, warn
from iitgpu.validate import clean_run_command, in_jail, safe_listdir


_STYLE = Style([
    ("qmark", "fg:cyan bold"),
    ("question", "bold"),
    ("answer", "fg:magenta bold"),
    ("pointer", "fg:cyan bold"),
    ("highlighted", "fg:cyan bold"),
    ("selected", "fg:magenta"),
])

# The whole intake is three questions wide: what you are doing, what you are
# running it on, and — everything else — the review hub. The old seven
# task-type labels asked the user to classify their work before the tool would
# talk to them; the classification only ever picked a resource default, which
# the hub now shows and lets them change directly.
_INTENTS: list[tuple[str, str]] = [
    ("notebook", "Open JupyterLab            — interactive notebook on the GPU"),
    ("batch",    "Run a script or notebook   — batch job (.py or .ipynb)"),
    ("shell",    "Open a shell on the GPU node"),
]

_OTHER_CHOICE = "Other: my own .sbatch · templates"

# What the batch intake will accept, typed or browsed.
_BATCH_EXTS = (".py", ".sh", ".ipynb")

# Internal task_type recorded on the JobSpec (audit trail + resource archaeology).
_TASK_TYPE = {"notebook": "notebook", "shell": "interactive", "batch": "custom"}


def _browse_script(start_dir: str, jail=in_jail, exts=(".py", ".sh")) -> str | None:
    """Jailed file browser that only shows files with one of *exts* (plus dirs).

    `jail` is the navigation predicate: the global `in_jail` for admins, or the
    caller's per-user browse jail for regular users so they stay confined to
    their own shared/users/<user> area (plus shared read-only models/envs).
    `exts` selects which files are pickable (default scripts; (".ipynb",) for
    the notebook-as-batch-job flow).
    """
    current = start_dir
    while True:
        entries = safe_listdir(current)
        dirs = sorted(e for e in entries if Path(current, e).is_dir())
        files = sorted(
            e for e in entries
            if Path(current, e).is_file() and e.endswith(tuple(exts))
        )
        choices = ["[.. up]"] + [f"[dir] {d}" for d in dirs] + files + ["[cancel]"]
        choice = questionary.select(
            f"Browse ({current}):", choices=choices, style=_STYLE
        ).ask()
        if choice is None or choice == "[cancel]":
            return None
        if choice == "[.. up]":
            parent = str(Path(current).parent)
            if jail(parent):
                current = parent
            else:
                warn("Already at root of allowed paths.")
            continue
        if choice.startswith("[dir] "):
            candidate = str(Path(current) / choice[6:])
            if jail(candidate):
                current = candidate
            else:
                warn("Access denied.")
            continue
        chosen = str(Path(current) / choice)
        if jail(chosen):
            return chosen
        warn("Access denied.")
        return None


# A conservative pip package-spec matcher (name, optional extras, optional
# version pins). Anything with shell metacharacters or spaces is rejected so a
# package list can never inject into the generated job script.
_PKG_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?"
    r"([=<>!~]=?[A-Za-z0-9._*+-]+)*$"
)


def _valid_pkg_tokens(raw: str) -> list[str]:
    """Keep only tokens that look like safe pip package specs."""
    return [t for t in raw.split() if _PKG_RE.match(t)]


def _notebook_deps_prompt(notebook_path: str, browse_jail, start_dir: str,
                          question: str = "Install Python dependencies first?") -> tuple[str, str]:
    """Ask how to install Python deps before a notebook runs / a session starts.
    When *notebook_path* is given, auto-detects a requirements.txt next to it or in
    its project root. Returns (requirements_path, packages_str) — at most one set."""
    found = ""
    if notebook_path:
        nb = Path(notebook_path)
        candidates = [nb.parent / "requirements.txt", nb.parent.parent / "requirements.txt"]
        found = next((str(c) for c in candidates
                      if c.is_file() and browse_jail(str(c))), "")
    auto = f"Install from {found}  (auto-detected)" if found else None
    choices = ([auto] if auto else []) + [
        "Choose a requirements.txt file",
        "Type package names (e.g. tqdm wfdb h5py)",
        "Skip — my environment already has everything",
    ]
    sel = questionary.select(
        question,
        choices=choices, style=_STYLE,
    ).ask()
    if not sel or sel.startswith("Skip"):
        return "", ""
    if auto and sel == auto:
        return found, ""
    if sel.startswith("Choose"):
        picked = _browse_script(start_dir, browse_jail, exts=(".txt",))
        return (picked or ""), ""
    if sel.startswith("Type"):
        raw = questionary.text("Packages (space-separated):", style=_STYLE).ask() or ""
        toks = _valid_pkg_tokens(raw)
        if raw.strip() and not toks:
            warn("No valid package names recognised — skipping dependency install.")
        return "", " ".join(toks)
    return "", ""


def _browse_data_folder(start_dir: str, jail=in_jail) -> str | None:
    """Jailed folder browser (directories only, for picking a data directory).

    `jail` is the navigation predicate (see `_browse_script`): regular users are
    confined to their own area; admins get the full global jail.
    """
    current = start_dir
    while True:
        entries = safe_listdir(current)
        dirs = sorted(e for e in entries if Path(current, e).is_dir())
        choices = (["[.. up]"] + [f"[dir] {d}" for d in dirs]
                   + ["[select this folder]", "[cancel]"])
        choice = questionary.select(
            f"Browse ({current}):", choices=choices, style=_STYLE
        ).ask()
        if choice is None or choice == "[cancel]":
            return None
        if choice == "[select this folder]":
            if jail(current):
                return current
            warn("Access denied.")
            return None
        if choice == "[.. up]":
            parent = str(Path(current).parent)
            if jail(parent):
                current = parent
            else:
                warn("Already at root of allowed paths.")
            continue
        if choice.startswith("[dir] "):
            candidate = str(Path(current) / choice[6:])
            if jail(candidate):
                current = candidate
            else:
                warn("Access denied.")
            continue


def _validate_and_show_errors(script_text: str, username: str, cfg) -> bool:
    """Run validate_sbatch; print errors and return False if any found."""
    from iitgpu.validate import validate_sbatch
    from iitgpu.ui import err as _err
    errors = validate_sbatch(script_text, username, cfg)
    if errors:
        for e in errors:
            _err(f"  Script error: {e}")
        return False
    return True


def _tier3_own_script(username: str, cfg) -> str | None:
    """Browse to a user's .sbatch file; validate and return its content."""
    import questionary
    from iitgpu.validate import in_user_browse_jail, user_browse_roots, in_jail
    from iitgpu.ui import info, err as _err
    from iitgpu import auditclient

    info("Browse to your .sbatch file (must be inside your allowed directories).")
    roots = user_browse_roots(cfg.nfs_root, username)
    start = roots[0] if roots else cfg.nfs_root
    current = start

    while True:
        try:
            entries = [e for e in os.listdir(current)
                       if os.path.isdir(os.path.join(current, e))
                       or e.endswith(".sbatch")]
        except OSError:
            _err("Cannot list directory.")
            return None
        choices = ["[.. up]"] + \
                  [f"[dir] {e}" for e in sorted(entries) if os.path.isdir(os.path.join(current, e))] + \
                  [e for e in sorted(entries) if e.endswith(".sbatch")] + \
                  ["[cancel]"]
        pick = questionary.select(f"Browse ({current}):", choices=choices,
                                  style=_STYLE).ask()
        if pick is None or pick == "[cancel]":
            return None
        if pick == "[.. up]":
            parent = str(Path(current).parent)
            if in_jail(parent):
                current = parent
            continue
        if pick.startswith("[dir] "):
            candidate = str(Path(current) / pick[6:])
            if in_jail(candidate):
                current = candidate
            continue
        # selected a .sbatch file
        sbatch_path = str(Path(current) / pick)
        if not in_jail(sbatch_path):
            _err("File is outside the allowed directory.")
            return None
        try:
            text = Path(sbatch_path).read_text()
        except OSError as exc:
            _err(f"Cannot read file: {exc}")
            return None
        if not _validate_and_show_errors(text, username, cfg):
            import questionary
            if not questionary.confirm("Select a different file?", default=True,
                                       style=_STYLE).ask():
                return None
            continue
        auditclient.log("sbatch_own_script", meta={"path": sbatch_path})
        return text


def _vram_check() -> bool:
    """State the VRAM situation at submit time. Asks nothing, blocks nothing.

    This used to interrogate the user for an estimate and refuse the job when
    it exceeded free headroom. That gate was theatre: VRAM is shared between
    concurrent jobs and is not enforced by SLURM, so a number typed here bound
    nobody — least of all the job that OOMs you thirty seconds later. The
    review hub now states the same fact where the sizing decision is actually
    made, so what is left here is the live reading and the caveat, printed with
    the submit confirmation. Always returns True.
    """
    try:
        stats = get_node_stats()
    except Exception:
        stats = None

    if stats and getattr(stats, "live_stats", False):
        total_gb = stats.gpu_mem_total_mb / 1024
        used_gb  = stats.gpu_mem_used_mb  / 1024
        free_gb  = total_gb - used_gb
        slice_gb = total_gb / SHARDS_PER_GPU
        info(f"GPU VRAM: [green]{free_gb:.1f} GB free[/]  "
             f"[dim]({used_gb:.1f} GB in use / {total_gb:.0f} GB total)[/]")
        info(f"Your slice's fair share is about {slice_gb:.0f} GB. "
             f"VRAM is shared between concurrent jobs and is not enforced, so "
             f"this is a budget, not a guarantee — going over can OOM someone else.")
    else:
        info("Live GPU stats unavailable. VRAM is shared between concurrent jobs "
             "and is not enforced — treat your fair share as a budget.")
    return True


def _wait_or_keypress(timeout: float) -> bool:
    """Block for up to *timeout* seconds, or until any key is pressed.

    Used as wait_ready's should_stop hook so a long wait never leaves the
    terminal silently unresponsive — a plain time.sleep() loop here once
    swallowed 'q' and Ctrl-C alike; an uncaught KeyboardInterrupt from the
    latter crashed the whole TUI process, which is this user's login shell,
    so it took the SSH session down with it. This makes any single keypress
    an ordinary, ungraceful-free exit from the wait, and ui.py's outer catch
    covers the remaining KeyboardInterrupt case (SSH-level signals, non-tty).
    """
    import select
    import sys
    import time
    try:
        import termios
        import tty
    except ImportError:
        time.sleep(timeout)
        return False
    if not sys.stdin.isatty():
        time.sleep(timeout)
        return False
    old_settings = None
    try:
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            sys.stdin.read(1)
            return True
        return False
    except (termios.error, OSError):
        time.sleep(timeout)
        return False
    finally:
        if old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except termios.error:
                pass


def _post_submit_notebook(job_id: str, folder: str) -> None:
    """Wait for the job's readiness marker, then show the Connect card.

    The card is parsed from the job's own stdout — the authoritative source —
    so it cannot disagree with what the server actually bound. The wait is
    interruptible by any keypress (see _wait_or_keypress) and additionally
    guarded by a bare KeyboardInterrupt catch, so this can never crash the
    session even if the terminal isn't in a state where a keypress is caught.
    """
    import time
    from iitgpu.connect import parse_connect, render_card, wait_ready
    from iitgpu.slurm import queue as _q
    from iitgpu.ui import console as _con

    def _alive() -> bool:
        return any(e.job_id == str(job_id) and e.state in ("PENDING", "RUNNING")
                   for e in _q(all_users=True))

    def _read_out() -> str:
        outs = sorted(Path(folder).glob("slurm-*.out"))
        return outs[-1].read_text() if outs else ""

    info("Starting JupyterLab… (this can take a minute on first launch)")
    info("Press any key to stop waiting — the job keeps running either way.")
    try:
        state = wait_ready(folder, is_alive=_alive, timeout=90,
                           should_stop=lambda: _wait_or_keypress(2.0))
    except KeyboardInterrupt:
        state = "cancelled"

    if state == "cancelled":
        info("Stopped waiting — the job is still starting in the background.")
        info("Check the dashboard (option 3) or press T on the job there for the Connect card.")
        return

    if state == "ready":
        # NFS close-to-open: the readiness marker can be visible slightly
        # before the .out write is flushed. Retry the parse briefly rather
        # than reporting a job that IS ready as "still starting".
        cinfo = parse_connect(_read_out())
        for _ in range(5):
            if cinfo:
                break
            time.sleep(1)
            cinfo = parse_connect(_read_out())
        if cinfo:
            _con.print(render_card(cinfo))
        else:
            warn("JupyterLab is up, but its connection info has not appeared in the job log yet.")
            info("Open the dashboard and press T on the job for the Connect card.")
        return

    if state == "gone":
        out_text = _read_out()
        errs = sorted(Path(folder).glob("slurm-*.err"))
        err_text = errs[-1].read_text() if errs else ""
        err("The job ended before JupyterLab came up.")
        if out_text.strip():
            info("Last output (.out):")
            for line in out_text.splitlines()[-10:]:
                info(f"  {line}")
        if err_text.strip():
            info("Last output (.err):")
            for line in err_text.splitlines()[-15:]:
                info(f"  {line}")
        if not out_text.strip() and not err_text.strip():
            info("No output was produced — the job may have failed before starting; check the dashboard.")
        return

    # state == "timeout"
    warn("Still starting after 90s (large envs can be slow).")
    cinfo = parse_connect(_read_out())
    if cinfo:
        _con.print(render_card(cinfo))
        info("The tunnel may not answer until the dashboard shows RUNNING.")
    else:
        info("Watch it in the dashboard — press T on the job for the Connect card.")


def _run_own_sbatch(cfg, user: str, jdir: str) -> None:
    """Submit a ready-made .sbatch verbatim — the "I already know what I want"
    escape hatch, moved out of the old linear flow unchanged.

    A job folder is still created so the script and its output land where every
    other job's do, the file still passes validate_sbatch (via
    `_tier3_own_script`), and the same job_submit audit pair still runs. What is
    gone is only its old position: it used to interrupt everyone mid-wizard with
    a confirm; now it lives under "Other", where the people who want it look.
    """
    defaults = resource_defaults("custom")
    folder = make_job_folder(jdir, JobSpec(
        job_name="custom", partition=cfg.partition,
        gpu_shards=defaults.gpu_shards, cpus=defaults.cpus, mem_gb=defaults.mem_gb,
        time_limit=defaults.time_limit or "02:00:00", run_command="",
        task_type="custom",
    ))
    text = _tier3_own_script(user, cfg)
    if text is None:
        shutil.rmtree(folder, ignore_errors=True)
        info("Discarded.")
        return

    sbatch_path = str(Path(folder) / "job.sbatch")
    Path(sbatch_path).write_text(text)
    Path(sbatch_path).chmod(0o644)
    kv("Script saved", sbatch_path)

    if not auditclient.log_or_block("job_submit", detail="custom",
                                    meta={"own_sbatch": True}):
        err("Audit logging failed. Refusing to submit (safety policy).")
        shutil.rmtree(folder, ignore_errors=True)
        return

    success, result = submit_job(sbatch_path)
    if success:
        ok(f"Job submitted! ID: {result}")
        auditclient.log("job_submitted_ok", detail="custom", job_id=result)
    else:
        err(f"Submission failed: {result}")
        auditclient.log("job_submit_failed", detail=result)


def _notebook_run_command(ls: LaunchSpec) -> str:
    """Bash that runs a .ipynb top-to-bottom as a batch job.

    papermill streams each cell's stdout/stderr into the job log as it executes
    (`--log-output`), so a long training cell shows progress instead of looking
    hung; `jupyter nbconvert --execute` is the fallback when papermill is not in
    the environment. Either way the executed copy is written into the job folder
    (the script's cwd), so the user's own notebook is never modified in place.
    """
    nb = shlex.quote(ls.script)
    out = shlex.quote(f"{Path(ls.script).stem}.executed.ipynb")
    label = shlex.quote(f"Executing notebook: {Path(ls.script).name}")
    return pip_install_block(ls.requirements, ls.packages) + (
        f"echo {label}\n"
        "if command -v papermill >/dev/null 2>&1; then\n"
        f"    papermill --log-output {nb} {out}\n"
        "else\n"
        f"    jupyter nbconvert --to notebook --execute --output {out} {nb}\n"
        "fi"
    )


def _build_run_command(ls: LaunchSpec) -> str:
    """The single command the batch job runs. `ls.args` is baked in here — it is
    part of the command line, never a JobSpec field.

    The args are re-cleaned even though the hub's editor already cleaned what it
    accepted: a LaunchSpec can also arrive from a template file on disk, which
    nothing in this process typed.
    """
    args = clean_run_command(ls.args) if ls.args else ""
    if ls.script.endswith(".ipynb"):
        if args:
            warn("Extra arguments do not apply to a notebook — ignoring them.")
        return _notebook_run_command(ls)
    runner = "bash" if ls.script.endswith(".sh") else "python3"
    cmd = f"{runner} {shlex.quote(ls.script)}"
    return f"{cmd} {args}".rstrip() if args else cmd


_DEFAULT_ENV_NAME = "data-science"


def _apply_default_env(ls: LaunchSpec, cfg) -> None:
    """Default a fresh launch to the shared prebuilt env when one is installed.

    `default_spec` leaves conda_env empty, which the hub renders as "(not set)"
    and the renderer turns into system python — so a user who accepted every
    default got a JupyterLab session with no torch in it. The filesystem probe
    lives here rather than in launchspec.default_spec so that module stays pure.
    Absent env = no change: the empty default is still a valid launch.
    """
    if ls.intent not in ("notebook", "batch"):
        return                      # a shell renders no sbatch — see review.py
    if ls.conda_env or ls.venv_path or ls.container_image:
        return                      # a template/rerun already said what to use
    env = Path(getattr(cfg, "nfs_root", "/shared")) / "envs" / _DEFAULT_ENV_NAME
    try:
        present = env.is_dir()
    except OSError:
        present = False
    if present:
        ls.env_kind, ls.conda_env = "prebuilt", str(env)


def _preview_sbatch(ls: LaunchSpec, cfg, user: str) -> str:
    """Render the job script this spec would produce, for the hub's Advanced →
    "view generated sbatch". Display only: no folder is created and nothing is
    written, so the real folder name is a placeholder.

    Uses the same to_job_spec + renderer + `_build_run_command` the submit paths
    use, so what the user reads here is what will actually be submitted.
    """
    folder = "<job-folder-assigned-at-submit>"
    job_name = _job_name_for(ls)
    task_type = _task_type_for(ls)
    if ls.intent == "notebook":
        spec = to_job_spec(ls, user=user, partition=cfg.partition,
                           job_name=job_name, task_type=task_type)
        return render_notebook_sbatch(
            spec, folder, port=ls.port,
            gateway_host=cfg.gateway_host, gateway_port=int(cfg.gateway_port),
            requirements=ls.requirements, packages=ls.packages,
        )
    spec = to_job_spec(ls, user=user, partition=cfg.partition,
                       job_name=job_name, task_type=task_type,
                       run_command=_build_run_command(ls))
    return render_sbatch(spec, folder)


def _task_type_for(ls: LaunchSpec) -> str:
    """Internal task_type for the audit trail (not a question the user answers)."""
    if ls.intent == "batch" and ls.script.endswith(".ipynb"):
        return "notebook-script"
    return _TASK_TYPE.get(ls.intent, "custom")


_NAME_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _job_name_for(ls: LaunchSpec) -> str:
    """Name the job after what it runs, so the queue is readable at a glance."""
    if ls.intent == "batch" and ls.script:
        stem = _NAME_SAFE.sub("_", Path(ls.script).stem).strip("_")[:40]
        if stem:
            return stem
    return {"notebook": "notebook", "shell": "interactive"}.get(ls.intent, "job")


def _usable_script(path: str, browse_jail) -> bool:
    """The three things that must hold before we run a path: it is a kind of file
    we know how to run, it is inside the caller's allowed directories, and it is
    actually there.

    Shared by the typed intake and the rerun prefill on purpose — a path lifted
    out of a months-old sbatch on disk has had no more validation than one
    somebody just typed, and rather less recency.
    """
    if not path.endswith(_BATCH_EXTS):
        warn(f"Not a runnable script. Allowed: {', '.join(_BATCH_EXTS)}")
        return False
    if not browse_jail(path):
        warn("That path is outside the directories you are allowed to use.")
        return False
    if not Path(path).is_file():
        warn(f"No such file: {path}")
        return False
    return True


def _pick_batch_script(jdir: str, user: str, browse_jail, start_dir: str) -> str | None:
    """Type a path or pick a recent one — the batch flow's only required question.

    Typed input skips the browser, so validation happens here: this is the
    boundary, not the browser.
    """
    choices = recent_scripts(jdir, user) + ["[browse…]"]
    for _ in range(6):
        raw = questionary.autocomplete(
            "Script or notebook (.py/.ipynb/.sh) — type a path or pick:",
            choices=choices, style=_STYLE,
        ).ask()
        if raw is None:
            return None
        raw = raw.strip()
        if not raw:
            return None
        if raw == "[browse…]":
            picked = _browse_script(start_dir, browse_jail, exts=_BATCH_EXTS)
            if picked is None:
                return None
            raw = picked
        if not _usable_script(raw, browse_jail):
            continue
        return raw
    return None


def run_wizard(prefill: dict | None = None) -> None:  # noqa: C901 (one flow, read top to bottom)
    """Intent → intake → review hub → submit.

    Three screens, not seven prompts: pick what you are doing, say what you are
    running (batch only), then land on the hub — one editable summary that shows
    live GPU availability where the decision is made. Everything the old linear
    flow interrogated up front (environment, size, time, data, model, args,
    arrays, dependencies, mail) is a row on that hub, defaulted and skippable.
    """
    cfg = load_config()
    jdir = jobs_dir(cfg)
    user = getpass.getuser()

    # Role-aware file-browser jail (mirrors files.py). Regular users browse and
    # pick data/scripts from their own shared/users/<user> area (plus read-only
    # shared models/envs); admins get the full NFS jail.
    from iitgpu.config import is_admin
    from iitgpu.validate import in_user_browse_jail
    _admin = is_admin(cfg)
    _browse_jail = (
        in_jail if _admin
        else (lambda p: in_user_browse_jail(p, cfg.nfs_root, user))
    )

    def _user_browse_start() -> str:
        """The user's own folder, created if missing, so the browser always opens
        inside shared/users/<user> instead of falling back to the NFS root."""
        if _admin:
            return cfg.nfs_root
        start = user_dir(cfg, user)
        try:
            Path(start).mkdir(parents=True, exist_ok=True)
        except OSError:
            return start if _browse_jail(start) else cfg.nfs_root
        return start

    header("New Job")

    # ── Intake ───────────────────────────────────────────────────────────────
    ls: LaunchSpec | None = None
    if prefill:
        # Re-run: the previous job's own sbatch is the source of truth for
        # sizing; the hub is where anything about it gets changed.
        ls = from_rerun(prefill, prefill.get("script_path", ""))
        info("Pre-filled from the previous run — change anything below.")
        # The script path came out of a file on disk, not out of this session:
        # the job may have been archived, renamed, or written by hand. Anything
        # that fails the same checks a typed path faces drops through to the
        # normal intake rather than being launched on trust.
        if ls.script and not _usable_script(ls.script, _browse_jail):
            warn(f"The previous script is no longer usable: {ls.script}")
            ls.script = ""

    while ls is None:
        choice = questionary.select(
            "What do you want to do?",
            choices=[label for _, label in _INTENTS]
                    + [questionary.Separator(), _OTHER_CHOICE],
            style=_STYLE,
        ).ask()
        if choice is None:
            return

        if choice != _OTHER_CHOICE:
            intent = next((k for k, label in _INTENTS if label == choice), None)
            if intent is None:
                return
            ls = default_spec(intent)
            _apply_default_env(ls, cfg)
            break

        sub = questionary.select(
            "Other:",
            choices=["Submit my own .sbatch", "Load a template", "back"],
            style=_STYLE,
        ).ask()
        if sub == "Submit my own .sbatch":
            _run_own_sbatch(cfg, user, jdir)
            return
        if sub is None:
            return
        if sub == "back":
            continue                      # back to the intent list, not out
        from iitgpu.templates import pick_template
        tdata = pick_template(cfg)
        if not tdata:
            continue                      # nothing picked — still in the wizard
        ls = from_template(tdata)

    if ls.intent == "notebook":
        ls.port = 8888

    if ls.intent == "batch" and not ls.script:
        picked = _pick_batch_script(jdir, user, _browse_jail, _user_browse_start())
        if not picked:
            return
        ls.script = picked

    # ── Review hub — every remaining field lives here ────────────────────────
    def _hub_browse_script():
        start = _user_browse_start()
        if ls.script and Path(ls.script).parent.is_dir() and _browse_jail(str(Path(ls.script).parent)):
            start = str(Path(ls.script).parent)
        return _browse_script(start, _browse_jail, exts=_BATCH_EXTS)

    def _hub_browse_data():
        return _browse_data_folder(_user_browse_start(), _browse_jail)

    def _hub_deps():
        return _notebook_deps_prompt(
            ls.script, _browse_jail, _user_browse_start(),
            question="Optional — Pre-install packages for this session?",
        )

    def _hub_preview(spec: LaunchSpec) -> str:
        return _preview_sbatch(spec, cfg, user)

    while True:
        outcome = run_hub(ls, cfg, user, browse_script=_hub_browse_script,
                          browse_data=_hub_browse_data, deps_prompt=_hub_deps,
                          preview=None if ls.intent == "shell" else _hub_preview)
        if outcome is None:
            info("Cancelled.")
            return
        if outcome != "template":
            break
        tname = questionary.text("Template name:", default=_job_name_for(ls),
                                 style=_STYLE).ask()
        if tname and tname.strip():
            from iitgpu.templates import save_template
            tspec = to_job_spec(ls, user=user, partition=cfg.partition,
                                job_name=tname.strip(), task_type=_task_type_for(ls),
                                run_command=_build_run_command(ls) if ls.intent == "batch" else "")
            if save_template(cfg, tname.strip(), tspec):
                ok(f"Template '{tname.strip()}' saved.")
                auditclient.log("job_template_saved", detail=tname.strip())

    job_name = _job_name_for(ls)
    task_type = _task_type_for(ls)

    # ── Shell: an allocation, not a job file ─────────────────────────────────
    if ls.intent == "shell":
        spec = to_job_spec(ls, user=user, partition=cfg.partition,
                           job_name="interactive", task_type="interactive")
        cmd = build_interactive_cmd(spec, partition=cfg.partition)
        info("Requesting an interactive GPU allocation — you will land in a shell")
        info("ON the compute node. It ends when you type 'exit' or the time limit hits.")
        info(f"GPU share: {gpu_share_note(spec.gpu_shards)}")
        panel("Interactive command", " ".join(cmd))
        if not questionary.confirm(
            "Start interactive session now?", default=True, style=_STYLE
        ).ask():
            return
        if not auditclient.log_or_block("interactive_start", detail="srun_pty"):
            err("Audit logging failed. Refusing to start (safety policy).")
            return
        import subprocess
        try:
            subprocess.run(cmd)
        except (OSError, KeyboardInterrupt):
            pass
        auditclient.log("interactive_end")
        info("Interactive session ended.")
        return

    # ── Notebook: a JupyterLab session on the node ───────────────────────────
    if ls.intent == "notebook":
        spec = to_job_spec(ls, user=user, partition=cfg.partition,
                           job_name=job_name, task_type=task_type)
        # Auto-populate the SLURM mail directive from users.db — but only when
        # the user has not turned notifications off in the hub's Advanced menu.
        if ls.mail:
            from iitgpu.notify import mta_present
            from iitgpu import daemonclient
            if mta_present():
                _registered_email = daemonclient.email_for(user)
                if _registered_email:
                    spec.mail_user = _registered_email

        folder = make_job_folder(jdir, spec)
        (Path(folder) / ".iit-jupyter").write_text("")  # marks this job as a JupyterLab session
        script_text = render_notebook_sbatch(
            spec, folder, port=ls.port,
            gateway_host=cfg.gateway_host, gateway_port=int(cfg.gateway_port),
            requirements=ls.requirements, packages=ls.packages,
        )
        info(f"GPU share: {gpu_share_note(spec.gpu_shards)}")
        _vram_check()
        panel("Generated notebook sbatch script", script_text)

        if not questionary.confirm(
            "Launch this JupyterLab session?", default=True, style=_STYLE
        ).ask():
            shutil.rmtree(folder, ignore_errors=True)
            info("Discarded.")
            return

        sbatch_path = str(Path(folder) / "job.sbatch")
        Path(sbatch_path).write_text(script_text)
        Path(sbatch_path).chmod(0o644)
        kv("Script saved", sbatch_path)

        if not auditclient.log_or_block("notebook_submit", detail=job_name):
            err("Audit logging failed. Refusing to submit (safety policy).")
            return

        success, result = submit_job(sbatch_path)
        if success:
            ok(f"Notebook job submitted! ID: {result}  ({ls.time_limit} session)")
            auditclient.log("notebook_submitted_ok", detail=job_name, job_id=result)
            auditclient.log(
                "notebook_session_start",
                detail=job_name,
                job_id=result,
                meta={"env": spec.conda_env or spec.container_image or "system",
                      "gpu_shards": spec.gpu_shards},
            )
            _post_submit_notebook(result, folder)
            if questionary.confirm(
                "Watch job output now?", default=False, style=_STYLE
            ).ask():
                try:
                    from iitgpu.dashboard import run_dashboard
                    run_dashboard(job_id=result)
                except ImportError:
                    info("Live dashboard not available.")
        else:
            err(f"Submission failed: {result}")
            auditclient.log("notebook_submit_failed", detail=result)
            # Nothing ran and nothing ever will, so nothing will write here.
            # Leaving the folder behind puts a job in the dashboard's listing
            # that does not exist — the declined paths above already know this.
            shutil.rmtree(folder, ignore_errors=True)
        return

    # ── Batch: a script or notebook, run to completion ───────────────────────
    run_cmd = _build_run_command(ls)
    spec = to_job_spec(ls, user=user, partition=cfg.partition,
                       job_name=job_name, task_type=task_type, run_command=run_cmd)
    if ls.mail:
        from iitgpu.notify import mta_present
        from iitgpu import daemonclient
        if mta_present():
            _registered_email = daemonclient.email_for(user)
            if _registered_email:
                spec.mail_user = _registered_email

    folder = make_job_folder(jdir, spec)
    script_text = render_sbatch(spec, folder)

    _env_display = (spec.container_image or spec.conda_env or spec.venv_path
                    or "none (system python)")
    panel("Job Summary", (
        f"  GPU share  : {gpu_share_note(spec.gpu_shards)}\n"
        f"  Time limit : {spec.time_limit or 'no limit'}\n"
        f"  Script     : {ls.script or '(none)'}\n"
        f"  Environment: {_env_display}\n"
        f"  Data path  : {spec.data_path or 'not set'}\n"
        f"  Model path : {spec.model_path or 'not set'}"
    ))
    panel("Generated sbatch script", script_text)
    _vram_check()

    if not questionary.confirm("Submit this job?", default=True, style=_STYLE).ask():
        shutil.rmtree(folder, ignore_errors=True)
        info("Discarded.")
        return

    sbatch_path = str(Path(folder) / "job.sbatch")
    Path(sbatch_path).write_text(script_text)
    Path(sbatch_path).chmod(0o644)
    kv("Script saved", sbatch_path)

    _submit_meta: dict = {"run_command": spec.run_command, "task_type": spec.task_type}
    if spec.conda_env:
        _submit_meta["conda_env"] = spec.conda_env
    if spec.venv_path:
        _submit_meta["venv_path"] = spec.venv_path
    if spec.container_image:
        _submit_meta["container_image"] = spec.container_image
    if spec.model_path:
        _submit_meta["model_path"] = spec.model_path
    if spec.data_path:
        _submit_meta["data_path"] = spec.data_path
    if spec.array:
        _submit_meta["array"] = spec.array
    if spec.dependency:
        _submit_meta["dependency"] = spec.dependency
    if not auditclient.log_or_block("job_submit", detail=job_name, meta=_submit_meta):
        err("Audit logging failed. Refusing to submit (safety policy).")
        return

    success, result = submit_job(sbatch_path)
    if success:
        ok(f"Job submitted! ID: {result}")
        auditclient.log("job_submitted_ok", detail=job_name, job_id=result)
        if spec.mail_user:
            info(f"SLURM will email [cyan]{spec.mail_user}[/] when the job ends.")
        if questionary.confirm(
            "Watch live output now?", default=True, style=_STYLE
        ).ask():
            try:
                from iitgpu.dashboard import run_dashboard
                run_dashboard(job_id=result)
            except ImportError:
                info("Live dashboard not available. Check job output manually.")
        elif questionary.confirm(
            "Wait here for the result? (silent poll)", default=False, style=_STYLE
        ).ask():
            from iitgpu.notify import poll_until_done
            info("Waiting for the job to finish (Ctrl-C to stop waiting)…")
            try:
                final = poll_until_done(result, interval=10)
                ok(f"Job {result} finished: {final}")
            except KeyboardInterrupt:
                info("Stopped waiting (job keeps running).")
    else:
        err(f"Submission failed: {result}")
        auditclient.log("job_submit_failed", detail=result)
