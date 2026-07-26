# TUI Design Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every menu screen in the IIT GPU Manager TUI (Main Menu, Jobs, Settings, My Workspace, file manager, upload, models, Admin's ~11 screens, Monitor, Setup, Accounting, Notebooks/services, and the already-good New Job wizard/review hub) renders through the same two primitives — a bordered header Panel with a live status line where one is cheap and true, and one canonical `← Back` — instead of today's five different `questionary.Style` objects and four different back-button spellings (`"back"`, `"Back"`, `"[back]"`, `"Back to main menu"`).

**Architecture:** Two new primitives land in `iitgpu/ui.py` (`screen()` for the header Panel, `select_menu()` for the picker+back). Every other TUI module is migrated to call them, screen by screen. No business logic changes — this is chrome only.

**Tech Stack:** Python 3, `questionary` (prompts), `rich` (Panel/Table rendering), `pytest` (existing 770-test suite).

**Repo:** `/home/slurmadmin/IIT-Secure-SLURM-Job-Gateway` on the login node (`ssh slurmadmin@192.168.122.10`). Run all commands there. Deploy is `bash deploy/redeploy-igm.sh` run AS `slurmadmin` (not sudo).

## Global Constraints

- No change to business logic, SLURM/audit calls, or file-jail rules — screen chrome only (per spec Non-goals).
- No change to prompts *within* a flow (e.g. a `questionary.text` asking "Reason:", or an in-flow file-browser's `[cancel]`/`[skip]` sentinel) — only screen-entry header/status/back.
- One spelling of back: `BACK = "← Back"` (arrow glyph is an explicit exception per commit `8702b1e` — "Selection-cursor/arrow glyphs... are left as-is — navigation affordances, not decorative emoji"). Do NOT introduce any other glyph/emoji.
- Every status body must come from a function that already exists — no new live-data plumbing (per spec Goal 4).
- Full `python3 -m pytest -q` must show 0 failures before every commit in this plan (baseline: 770 passed).
- Commit author on this repo is Daham only — no Claude co-author trailer (`git -c user.name=Daham -c user.email=daham.20242053@iit.ac.lk commit ...`), per existing project convention.
- Final version is `1.3.0` in `iitgpu/__init__.py` only (the sole file carrying the version string).

---

## Task 1: `iitgpu/ui.py` shared primitives

**Files:**
- Modify: `iitgpu/ui.py`
- Test: `tests/test_ui.py` (new file)

**Interfaces:**
- Produces: `ui.STYLE` (a `questionary.Style`), `ui.BACK = "← Back"`, `ui.BACK_TO_MAIN = "← Back to main menu"`, `ui.screen(title: str, *, status=None) -> None`, `ui.select_menu(prompt: str, choices: list, *, back: str = BACK) -> str | None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ui.py`:
```python
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from iitgpu import ui


def test_style_is_one_shared_object():
    assert ui.STYLE is not None
    # Same object every import — not re-built per module.
    from iitgpu.ui import STYLE as style_again
    assert ui.STYLE is style_again


def test_back_constants_use_only_the_permitted_arrow_glyph():
    assert ui.BACK == "\u2190 Back"
    assert ui.BACK_TO_MAIN == "\u2190 Back to main menu"


def test_screen_renders_a_panel_with_the_title(monkeypatch):
    buf = Console(file=None, force_terminal=True, width=80)
    captured = []
    monkeypatch.setattr(ui, "console", type("C", (), {
        "print": lambda self, renderable: captured.append(renderable)
    })())
    ui.screen("My Screen")
    assert len(captured) == 1
    assert isinstance(captured[0], Panel)


def test_screen_accepts_a_status_body(monkeypatch):
    captured = []
    monkeypatch.setattr(ui, "console", type("C", (), {
        "print": lambda self, renderable: captured.append(renderable)
    })())
    table = Table()
    ui.screen("Admin", status=table)
    assert captured[0].renderable is table


def test_select_menu_returns_none_on_back(monkeypatch):
    class FakeSelect:
        def __init__(self, *a, **kw):
            self.kw = kw
        def ask(self):
            return ui.BACK
    monkeypatch.setattr(ui.questionary, "select", lambda *a, **kw: FakeSelect(*a, **kw))
    result = ui.select_menu("Pick:", ["a", "b"])
    assert result is None


def test_select_menu_returns_none_on_escape(monkeypatch):
    class FakeSelect:
        def ask(self):
            return None
    monkeypatch.setattr(ui.questionary, "select", lambda *a, **kw: FakeSelect())
    assert ui.select_menu("Pick:", ["a", "b"]) is None


def test_select_menu_returns_the_real_choice(monkeypatch):
    class FakeSelect:
        def ask(self):
            return "a"
    monkeypatch.setattr(ui.questionary, "select", lambda *a, **kw: FakeSelect())
    assert ui.select_menu("Pick:", ["a", "b"]) == "a"


def test_select_menu_appends_separator_and_back_to_choices(monkeypatch):
    seen = {}
    class FakeSelect:
        def ask(self):
            return None
    def fake_select(prompt, choices, style):
        seen["choices"] = choices
        seen["style"] = style
        return FakeSelect()
    monkeypatch.setattr(ui.questionary, "select", fake_select)
    ui.select_menu("Pick:", ["a", "b"])
    assert seen["choices"][-1] == ui.BACK
    assert seen["style"] is ui.STYLE
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/IIT-Secure-SLURM-Job-Gateway && python3 -m pytest tests/test_ui.py -v`
Expected: FAIL — `AttributeError: module 'iitgpu.ui' has no attribute 'STYLE'` (and similar for `BACK`, `screen`, `select_menu`).

- [ ] **Step 3: Implement the primitives**

Replace the full contents of `iitgpu/ui.py` with:
```python
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
BACK = "\u2190 Back"
BACK_TO_MAIN = "\u2190 Back to main menu"


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/IIT-Secure-SLURM-Job-Gateway && python3 -m pytest tests/test_ui.py -v`
Expected: 7 passed.

- [ ] **Step 5: Run the full suite to confirm nothing else broke**

Run: `cd ~/IIT-Secure-SLURM-Job-Gateway && python3 -m pytest -q`
Expected: 777 passed (770 existing + 7 new). `header()`/`ui.console`/`ui.info` etc. are unchanged, so no existing test should break yet.

- [ ] **Step 6: Commit**

```bash
cd ~/IIT-Secure-SLURM-Job-Gateway
git add iitgpu/ui.py tests/test_ui.py
git -c user.name=Daham -c user.email=daham.20242053@iit.ac.lk commit -m "feat(ui): add screen() and select_menu() shared TUI primitives"
```

---

## Task 2: `iitgpu/splash.py` — reusable resource verdict + Main Menu status line

**Files:**
- Modify: `iitgpu/splash.py`
- Test: `tests/test_dashboard.py` (has the existing splash tests — confirm file/section, add one) or `tests/test_splash.py` if that's where they actually live — check with `grep -rn "_resource_seg\|_build_status_line" tests/` before writing.

**Interfaces:**
- Consumes: nothing new (uses `NodeStats` fields already read by `_resource_seg`).
- Produces: `splash.resource_status_line(stats) -> str` (renamed/exposed from the current private `_resource_seg`, same signature and return value, just public and reused elsewhere).

- [ ] **Step 1: Locate the existing tests for `_resource_seg`**

Run: `cd ~/IIT-Secure-SLURM-Job-Gateway && grep -rln "_resource_seg\|_build_status_line" tests/`

- [ ] **Step 2: Write the failing test**

In whichever file that grep finds (append near the existing splash tests):
```python
from iitgpu.splash import resource_status_line


def test_resource_status_line_is_the_public_name_for_the_verdict():
    stats = SimpleNamespace(
        cpu_total=32, cpu_alloc=8, mem_total_mb=61440, mem_alloc_mb=8192,
        shard_total=4, shard_alloc=2, gpu_total=1, gpu_alloc=0,
    )
    line = resource_status_line(stats)
    assert "GPU" in line
    assert "slices free" in line
```
(If `SimpleNamespace` isn't already imported in that test file, add `from types import SimpleNamespace` to its imports.)

- [ ] **Step 3: Run test to verify it fails**

Run: `python3 -m pytest tests/test_dashboard.py -k resource_status_line -v` (adjust path to whatever Step 1 found)
Expected: FAIL — `ImportError: cannot import name 'resource_status_line' from 'iitgpu.splash'`.

- [ ] **Step 4: Rename and re-export**

In `iitgpu/splash.py`, rename `_resource_seg` to `resource_status_line` (the function body is unchanged — only the name), and update its one call site inside `_build_status_line`:
```python
# before
    res_seg = _resource_seg(stats)
# after
    res_seg = resource_status_line(stats)
```
Keep a one-line back-compat alias directly below the renamed function so nothing else that might reference the old private name breaks silently:
```python
_resource_seg = resource_status_line  # back-compat alias, splash.py internal use only
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_dashboard.py -v` (or wherever) then the full suite:
```bash
python3 -m pytest -q
```
Expected: all pass, count +1 over Task 1's total.

- [ ] **Step 6: Commit**

```bash
git add iitgpu/splash.py tests/test_dashboard.py
git -c user.name=Daham -c user.email=daham.20242053@iit.ac.lk commit -m "refactor(splash): expose resource_status_line for reuse outside splash"
```

---

## Task 3: `iitgpu/menu.py` — Main Menu, Jobs, Settings, cluster status

**Files:**
- Modify: `iitgpu/menu.py`
- Test: `tests/test_menu.py` if it exists, else add assertions inline — check first: `grep -rln "run_menu\|_jobs_menu\|_settings_menu" tests/`

**Interfaces:**
- Consumes: `ui.screen`, `ui.select_menu`, `ui.BACK_TO_MAIN`, `splash.resource_status_line`, `slurm.get_node_stats`.

- [ ] **Step 1: Check for existing menu.py tests**

Run: `cd ~/IIT-Secure-SLURM-Job-Gateway && grep -rln "run_menu\|_jobs_menu\|_settings_menu\|_MAIN_ITEMS" tests/`

If a test file exists that patches `questionary.select` and asserts on `"5."`/`"Back to main menu"` literal choices, update those assertions to match the new choice list (same items, now routed through `select_menu`, so the underlying `questionary.select` still receives a `choices=[...]` list — the assertion just needs the trailing `Separator()+BACK_TO_MAIN` appended). Write the updated assertion now (Red step) before touching `menu.py`.

- [ ] **Step 2: Run existing menu tests to verify they fail against old code expectations**

Run: `python3 -m pytest tests/test_menu.py -v` (or wherever found)
Expected: FAIL on the updated assertions (they now expect `BACK_TO_MAIN`, which doesn't exist in the choices yet).
(If no test file exists for menu.py, skip to Step 3 — there's nothing to Red/Green here, this is pure migration; the full-suite run in Step 5 is the safety net.)

- [ ] **Step 3: Migrate `iitgpu/menu.py`**

Replace the whole file:
```python
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


def _main_status() -> str:
    """One-line cluster verdict — the same live availability language the
    launch review hub already shows, at the point the user is about to
    decide what to do."""
    try:
        from iitgpu.slurm import get_node_stats
        from iitgpu.splash import resource_status_line
        return resource_status_line(get_node_stats())
    except Exception:
        return "[dim]cluster stats unavailable[/]"


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

        screen("Main Menu", status=_main_status())
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
    """My queued/running count + free GPU slices — the two numbers someone
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
            parts.append(f"[dim]·[/] {free}/{stats.shard_total} GPU slices free")
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
```

Note: `check_cluster_health()` returns a `bool`/tuple in `setup.py` today per its earlier signature seen in Task 8 — this file already called `console.print(result)` on it before this migration too (pre-existing behavior, not introduced here; leave as-is, it's out of scope to fix that return-type mismatch in this plan).

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_menu.py -v   # if it exists
python3 -m pytest -q
```
Expected: full suite green.

- [ ] **Step 5: Commit**

```bash
git add iitgpu/menu.py tests/test_menu.py 2>/dev/null
git -c user.name=Daham -c user.email=daham.20242053@iit.ac.lk commit -m "refactor(menu): unify Main Menu/Jobs/Settings on screen()/select_menu()"
```

---

## Task 4: `iitgpu/workspace.py`

**Files:**
- Modify: `iitgpu/workspace.py`
- Test: `grep -rln "run_workspace\|Back to main menu" tests/` — likely none dedicated; rely on full-suite run.

**Interfaces:**
- Consumes: `ui.screen`, `ui.select_menu`, `ui.BACK_TO_MAIN`.

- [ ] **Step 1: Check for existing workspace tests referencing the old back string**

Run: `grep -rln "Back to main menu" tests/test_workspace.py 2>/dev/null; grep -rl "run_workspace" tests/`
If found, update the expected choice string to `ui.BACK_TO_MAIN` in that test now (Red step), matching whatever mock pattern it uses.

- [ ] **Step 2: Run to verify it fails (if a test was updated)**

`python3 -m pytest tests/test_workspace.py -v` (skip if no such file — full-suite run at Step 4 is the safety net for a pure migration).

- [ ] **Step 3: Migrate**

In `iitgpu/workspace.py`:
```python
# before
from iitgpu.ui import console, header, info, warn
...
_STYLE = Style([
    ("qmark", "fg:cyan bold"),
    ("question", "bold"),
    ("answer", "fg:magenta bold"),
    ("pointer", "fg:cyan bold"),
    ("highlighted", "fg:cyan bold"),
])
```
```python
# after
from iitgpu.ui import BACK_TO_MAIN, console, info, screen, select_menu, warn
```
(Drop the `_STYLE` block and the now-unused `from questionary import Style` import entirely.)

Then, inside `run_workspace()`:
```python
# before
    while True:
        header("My Workspace")
```
```python
# after
    while True:
        screen("My Workspace")
```
(The existing Disk/Files/Environments/Models Panels printed right after this line stay exactly as they are — they're already the "status body" the design calls for; they just print below the new bordered header Panel instead of below a bare rule.)

And the action picker:
```python
# before
        choice = questionary.select(
            "Action:",
            choices=[
                "Browse my files",
                "Upload data",
                "Download a model",
                "Build / manage environments",
                "Delete a model",
                "Back to main menu",
            ],
            style=_STYLE,
        ).ask()

        if choice is None or choice == "Back to main menu":
            return
```
```python
# after
        choice = select_menu(
            "Action:",
            [
                "Browse my files",
                "Upload data",
                "Download a model",
                "Build / manage environments",
                "Delete a model",
            ],
            back=BACK_TO_MAIN,
        )

        if choice is None:
            return
```

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest -q
```
Expected: full suite green.

- [ ] **Step 5: Commit**

```bash
git add iitgpu/workspace.py tests/test_workspace.py 2>/dev/null
git -c user.name=Daham -c user.email=daham.20242053@iit.ac.lk commit -m "refactor(workspace): unify My Workspace on screen()/select_menu()"
```

---

## Task 5: `iitgpu/admin.py` part 1 — top-level menu, QOS, provisioning, roster

**Files:**
- Modify: `iitgpu/admin.py`
- Test: `tests/test_admin.py`

**Interfaces:**
- Consumes: `ui.screen`, `ui.select_menu`, `ui.BACK`, `ui.STYLE`.

- [ ] **Step 1: Find and update the back-string assertions this task touches**

Run: `grep -n '"Back"\|Back to\|list_qos\|_qos_menu\|_view_users' tests/test_admin.py`
Update any assertion expecting the literal `"Back"` for the QOS menu or the top-level admin choices to expect `ui.BACK` instead (Red step).

- [ ] **Step 2: Run to verify it fails**

`python3 -m pytest tests/test_admin.py -v`
Expected: FAIL on the updated assertions.

- [ ] **Step 3: Migrate `_qos_menu`**

```python
# before
def _qos_menu(style) -> None:
    import questionary
    from rich.table import Table
    from iitgpu.ui import console, header, info, ok, err, warn

    while True:
        header("QOS / Limits")
        rows = list_qos()
        if not rows:
            warn("No QOS data (sacctmgr unavailable)."); return

        t = Table(show_header=True, header_style="bold cyan", show_lines=False)
        t.add_column("QOS", style="magenta")
        t.add_column("Max Wall Time")
        t.add_column("Max GPUs / User")
        t.add_column("Priority")
        for r in rows:
            t.add_row(r["name"], r["max_wall"], str(r["max_gpu"]), r["priority"])
        console.print(t)

        qos_names = [r["name"] for r in rows]
        qname = questionary.select(
            "Select QOS to edit:", choices=qos_names + ["Back"], style=style).ask()
        if qname is None or qname == "Back":
            return

        current = next((r for r in rows if r["name"] == qname), {})
        field = questionary.select(
            "Field to change:",
            choices=["Max Wall Time", "Max GPUs per user", "Priority", "Back"],
            style=style).ask()
        if field is None or field == "Back":
            continue
```
```python
# after
def _qos_menu(style) -> None:
    import questionary
    from rich.table import Table
    from iitgpu.ui import BACK, console, info, ok, err, screen, select_menu, warn

    while True:
        rows = list_qos()
        if not rows:
            screen("QOS / Limits")
            warn("No QOS data (sacctmgr unavailable)."); return

        t = Table(show_header=True, header_style="bold cyan", show_lines=False)
        t.add_column("QOS", style="magenta")
        t.add_column("Max Wall Time")
        t.add_column("Max GPUs / User")
        t.add_column("Priority")
        for r in rows:
            t.add_row(r["name"], r["max_wall"], str(r["max_gpu"]), r["priority"])
        screen("QOS / Limits", status=t)

        qos_names = [r["name"] for r in rows]
        qname = select_menu("Select QOS to edit:", qos_names)
        if qname is None:
            return

        current = next((r for r in rows if r["name"] == qname), {})
        field = select_menu(
            "Field to change:",
            ["Max Wall Time", "Max GPUs per user", "Priority"])
        if field is None:
            continue
```
(The rest of `_qos_menu`'s body — the three `elif field == ...:` branches — is unchanged; only the two `questionary.select(...)` blocks and the header/table placement above move.)

- [ ] **Step 4: Migrate `admin_menu()` (top-level)**

```python
# before
def admin_menu() -> None:
    import questionary
    from questionary import Style, Separator
    from rich.table import Table
    from iitgpu.ui import console, header, info, ok, err, warn

    cfg = load_config()
    if not is_admin(cfg):
        warn("Admin panel is restricted to members of the admin group.")
        return

    style = Style([("qmark", "fg:cyan bold"), ("pointer", "fg:cyan bold")])
    node_default = "iit-MS-7E06"

    while True:
        header("Admin Panel")
        _mail_state = "OFF — disabled" if is_mail_disabled() else "ON"
        choice = questionary.select(
            "Select action:",
            choices=[
                Separator("──  User Management  ──────────────────────────"),
                "  Provision user",
                "  Offboard user",
                "  View users",
                "  Log in as user",
                Separator("──  Jobs & Usage  ─────────────────────────────"),
                "  All-user job history",
                "  Cluster usage (all users)",
                "  Disk usage by user",
                "  Any user's job output",
                Separator("──  Cluster Control  ──────────────────────────"),
                "  Drain node",
                "  Resume node",
                "  QOS / limits",
                "  Maintenance notice",
                Separator("──  Monitoring  ───────────────────────────────"),
                "  Audit log",
                "  Service health",
                "  Mail delivery log",
                f"  Mail service: {_mail_state}",
                Separator("───────────────────────────────────────────────"),
                "  Back",
            ],
            style=style,
        ).ask()

        if choice is None:
            return
        choice = choice.strip()
        if choice == "Back":
            return
```
```python
# after
def admin_menu() -> None:
    from questionary import Separator
    from rich.table import Table
    from iitgpu.ui import STYLE, console, info, ok, err, screen, select_menu, warn

    cfg = load_config()
    if not is_admin(cfg):
        warn("Admin panel is restricted to members of the admin group.")
        return

    style = STYLE
    node_default = "iit-MS-7E06"

    while True:
        _mail_state = "OFF — disabled" if is_mail_disabled() else "ON"
        _maint = get_maintenance()
        _n_users = len(list_gpuusers())
        status = (
            f"[bold]{_n_users}[/] active user(s)   [dim]·[/]   Mail: {_mail_state}"
            + (f"   [dim]·[/]   [yellow]Maintenance ON[/]" if _maint else "")
        )
        screen("Admin Panel", status=status)
        choice = select_menu(
            "Select action:",
            [
                Separator("──  User Management  ──────────────────────────"),
                "  Provision user",
                "  Offboard user",
                "  View users",
                "  Log in as user",
                Separator("──  Jobs & Usage  ─────────────────────────────"),
                "  All-user job history",
                "  Cluster usage (all users)",
                "  Disk usage by user",
                "  Any user's job output",
                Separator("──  Cluster Control  ──────────────────────────"),
                "  Drain node",
                "  Resume node",
                "  QOS / limits",
                "  Maintenance notice",
                Separator("──  Monitoring  ───────────────────────────────"),
                "  Audit log",
                "  Service health",
                "  Mail delivery log",
                f"  Mail service: {_mail_state}",
            ],
        )

        if choice is None:
            return
        choice = choice.strip()
```
(Every subsequent `if choice == "Drain node":` / etc. branch in the rest of `admin_menu()` is unchanged — they compare against the same stripped labels as before.)

- [ ] **Step 5: Migrate `_view_users`**

```python
# before
def _view_users(style) -> None:
    import questionary
    from rich.table import Table
    from iitgpu.ui import console, header, info, warn

    header("User Roster")
```
```python
# after
def _view_users(style) -> None:
    import questionary
    from rich.table import Table
    from iitgpu.ui import console, info, screen, warn

    screen("User Roster")
```
(Body unchanged — the table it builds is printed after `screen()` the same way it printed after `header()`; this screen has no back-button loop to migrate, it's a one-shot view ending in `press_any_key_to_continue`.)

- [ ] **Step 6: Migrate `_provision_menu`'s internal `header()` calls, if any, to `screen()`**

Run: `grep -n "header(" iitgpu/admin.py` and for every occurrence still inside `_provision_menu` (lines ~482-585), replace `header("...")` with `screen("...")` and add `screen` to that function's local `from iitgpu.ui import ...` line (leave `questionary.select` calls inside `_provision_menu` as-is for this task — they're in-flow data-entry prompts, not screen-level pickers with a back button, per the plan's non-goal on in-flow prompts; only its `Style` object gets swapped for `STYLE` where it constructs one locally).

- [ ] **Step 7: Run tests**

```bash
python3 -m pytest tests/test_admin.py -v
python3 -m pytest -q
```
Expected: full suite green.

- [ ] **Step 8: Commit**

```bash
git add iitgpu/admin.py tests/test_admin.py
git -c user.name=Daham -c user.email=daham.20242053@iit.ac.lk commit -m "refactor(admin): unify top-level menu, QOS, and user roster on screen()/select_menu()"
```

---

## Task 6: `iitgpu/admin.py` part 2 — remaining monitoring/control submenus

**Files:**
- Modify: `iitgpu/admin.py`
- Test: `tests/test_admin.py`

**Interfaces:**
- Consumes: same `ui.screen`/`ui.select_menu`/`ui.BACK` as Task 5.

- [ ] **Step 1: Update back-string assertions for the remaining submenus**

Run: `grep -n '"Back"' iitgpu/admin.py` after Task 5 — the remaining hits are in `_view_job_output` (`choices=files + ["Back"]`) and `_view_service_health` (`choices=_UNITS + ["Back"]`). Update `tests/test_admin.py` assertions covering these two functions to expect `ui.BACK` (Red step).

- [ ] **Step 2: Run to verify it fails**

`python3 -m pytest tests/test_admin.py -v`

- [ ] **Step 3: Migrate `_view_job_output`**

```python
# before
def _view_job_output(style) -> None:
    import questionary
    from iitgpu.ui import console, header, info, err

    header("User Job Output")
    target_user = questionary.text("Username:", style=style).ask()
    ...
    fname = questionary.select("Select file:", choices=files + ["Back"],
                               style=style).ask()
    if fname is None or fname == "Back":
        return
```
```python
# after
def _view_job_output(style) -> None:
    import questionary
    from iitgpu.ui import console, info, err, screen, select_menu

    screen("User Job Output")
    target_user = questionary.text("Username:", style=style).ask()
    ...
    fname = select_menu("Select file:", files)
    if fname is None:
        return
```
(Its later `header(f"Job output: ...")` call becomes `screen(f"Job output: ...")` too.)

- [ ] **Step 4: Migrate `_view_service_health`**

```python
# before
def _view_service_health(style) -> None:
    import questionary
    from iitgpu.ui import console, header, info

    _UNITS = ["iit-gpu-audit", "slurmctld", "slurmd", "mariadb", "slurmdbd"]
    header("Service Health")
    unit = questionary.select("Select service:", choices=_UNITS + ["Back"],
                              style=style).ask()
    if unit is None or unit == "Back":
        return
```
```python
# after
def _view_service_health(style) -> None:
    import questionary
    from iitgpu.ui import console, info, screen, select_menu

    _UNITS = ["iit-gpu-audit", "slurmctld", "slurmd", "mariadb", "slurmdbd"]
    screen("Service Health")
    unit = select_menu("Select service:", _UNITS)
    if unit is None:
        return
```

- [ ] **Step 5: Migrate the plain `header()` → `screen()` calls with no back-button change needed**

For `_view_audit_log`, `_view_maillog`, `_view_disk_usage`, and `_mail_service_menu` (each already has exactly one `header("...")` call and no `"Back"`-style picker of its own — they're one-shot views), replace `header(` with `screen(` and update each function's local `from iitgpu.ui import ...` line to import `screen` instead of `header`.

- [ ] **Step 6: Add a header to `_maintenance_menu`, which currently has none**

```python
# before
def _maintenance_menu(style) -> None:
    import os
    import questionary
    from iitgpu.ui import info, ok, err
    current = get_maintenance()
    if current:
        info(f"  [yellow]Active notice:[/] {current.get('reason', '')}")
        action = questionary.select(
            "Maintenance notice:",
            choices=["Update notice", "Clear notice", "Back"],
            style=style,
        ).ask()
        if action == "Clear notice":
```
```python
# after
def _maintenance_menu(style) -> None:
    import os
    import questionary
    from iitgpu.ui import info, ok, err, screen, select_menu
    screen("Maintenance Notice")
    current = get_maintenance()
    if current:
        info(f"  [yellow]Active notice:[/] {current.get('reason', '')}")
        action = select_menu(
            "Maintenance notice:", ["Update notice", "Clear notice"])
        if action == "Clear notice":
```
(The `else:` branch below — prompting for a brand-new reason when there's no active notice — is unchanged.)

- [ ] **Step 7: Migrate `_login_as_menu`'s `"[cancel]"` to `BACK`**

```python
# before
def _login_as_menu(style) -> None:
    ...
    import questionary
    from iitgpu.ui import header, info, ok, err, warn
    header("Log in as user")
    me = getpass.getuser()
    targets = [u for u in list_gpuusers() if u != me]
    if not targets:
        warn("No other users found to log in as.")
        questionary.press_any_key_to_continue("").ask()
        return
    target = questionary.select(
        "Log in as which user?  (you'll get THEIR TUI; quit it to return here)",
        choices=targets + ["[cancel]"], style=style,
    ).ask()
    if not target or target == "[cancel]":
        return
```
```python
# after
def _login_as_menu(style) -> None:
    ...
    import questionary
    from iitgpu.ui import info, ok, err, screen, select_menu, warn
    screen("Log in as user")
    me = getpass.getuser()
    targets = [u for u in list_gpuusers() if u != me]
    if not targets:
        warn("No other users found to log in as.")
        questionary.press_any_key_to_continue("").ask()
        return
    target = select_menu(
        "Log in as which user?  (you'll get THEIR TUI; quit it to return here)",
        targets)
    if not target:
        return
```

- [ ] **Step 8: Run tests**

```bash
python3 -m pytest tests/test_admin.py -v
python3 -m pytest -q
```
Expected: full suite green.

- [ ] **Step 9: Commit**

```bash
git add iitgpu/admin.py tests/test_admin.py
git -c user.name=Daham -c user.email=daham.20242053@iit.ac.lk commit -m "refactor(admin): unify remaining monitoring/control submenus on screen()/select_menu()"
```

---

## Task 7: `iitgpu/monitor.py`

**Files:**
- Modify: `iitgpu/monitor.py`
- Test: `tests/test_monitor_completeness.py`, plus `grep -rln "\\[back\\]" tests/` for any other file exercising these pickers.

**Interfaces:**
- Consumes: `ui.screen`, `ui.select_menu`, `ui.BACK_TO_MAIN`.

- [ ] **Step 1: Update back-string assertions**

Run: `grep -n '\[back\]\|Back to main menu' tests/test_monitor_completeness.py`
Update any found to expect `ui.BACK` (for the five `"[back]"` pickers: `manage_job`'s job list, `manage_job`'s action list, `browse_and_tail_log`'s folder+file pickers, `follow_log`'s folder picker, `rerun_job`'s job picker) or `ui.BACK_TO_MAIN` (for `monitor_menu`'s "Back to main menu").

- [ ] **Step 2: Run to verify it fails**

`python3 -m pytest tests/test_monitor_completeness.py -v`

- [ ] **Step 3: Migrate the module header and each picker**

```python
# before
import questionary
from questionary import Style
from rich.table import Table

from iitgpu import auditclient
from iitgpu.slurm import (cancel, hold, release, requeue, queue,
                          job_detail, job_efficiency, filtered_history)
from iitgpu.ui import console, err, header, info, kv, ok, warn
from iitgpu.validate import in_jail, safe_listdir

_STYLE = Style([
    ("qmark", "fg:cyan bold"),
    ("question", "bold"),
    ("answer", "fg:magenta bold"),
    ("pointer", "fg:cyan bold"),
])
```
```python
# after
import questionary
from rich.table import Table

from iitgpu import auditclient
from iitgpu.slurm import (cancel, hold, release, requeue, queue,
                          job_detail, job_efficiency, filtered_history)
from iitgpu.ui import (BACK_TO_MAIN, STYLE, console, err, info, kv, ok,
                       screen, select_menu, warn)
from iitgpu.validate import in_jail, safe_listdir

_STYLE = STYLE
```
(Keeping the module-level `_STYLE = STYLE` alias means every remaining `style=_STYLE` call in this file — e.g. on `questionary.confirm`, which `select_menu` doesn't wrap — keeps working unchanged.)

Every `header(` call in this file becomes `screen(`:  `show_queue`, `manage_job`, `browse_and_tail_log`, `cluster_status`, `monitor_menu`, `follow_log`, `show_history`, and the inline `header(f"Log: ...")`/`header(f"Job {job_id} detail")`/`header(f"Following {...}")` calls.

`manage_job`:
```python
# before
    choices = [f"{e.job_id}  {e.name}  [{e.state}]" for e in entries] + ["[back]"]
    choice = questionary.select("Select a job:", choices=choices, style=_STYLE).ask()
    if choice is None or choice == "[back]":
        return
    job_id = choice.split()[0]

    action = questionary.select(
        f"Action for job {job_id}:",
        choices=["Cancel", "Hold", "Release", "Requeue", "Details + efficiency", "[back]"],
        style=_STYLE,
    ).ask()
    if action is None or action == "[back]":
        return
```
```python
# after
    choices = [f"{e.job_id}  {e.name}  [{e.state}]" for e in entries]
    choice = select_menu("Select a job:", choices)
    if choice is None:
        return
    job_id = choice.split()[0]

    action = select_menu(
        f"Action for job {job_id}:",
        ["Cancel", "Hold", "Release", "Requeue", "Details + efficiency"])
    if action is None:
        return
```

`browse_and_tail_log`:
```python
# before
    choice = questionary.select(
        "Select job folder:", choices=sorted(folders, reverse=True) + ["[back]"], style=_STYLE
    ).ask()
    if choice is None or choice == "[back]":
        return
    ...
    log_choice = questionary.select("Select log file:", choices=logs + ["[back]"], style=_STYLE).ask()
    if log_choice is None or log_choice == "[back]":
        return
```
```python
# after
    choice = select_menu("Select job folder:", sorted(folders, reverse=True))
    if choice is None:
        return
    ...
    log_choice = select_menu("Select log file:", logs)
    if log_choice is None:
        return
```

`follow_log`:
```python
# before
    choice = questionary.select(
        "Follow which job folder?", choices=sorted(folders, reverse=True) + ["[back]"], style=_STYLE
    ).ask()
    if choice is None or choice == "[back]":
        return
```
```python
# after
    choice = select_menu("Follow which job folder?", sorted(folders, reverse=True))
    if choice is None:
        return
```

`rerun_job`:
```python
# before
    choice = questionary.select(
        "Rerun which job?",
        choices=sorted(folders, reverse=True) + ["[back]"],
        style=_STYLE,
    ).ask()
    if choice is None or choice == "[back]":
        return
```
```python
# after
    choice = select_menu("Rerun which job?", sorted(folders, reverse=True))
    if choice is None:
        return
```

`monitor_menu` (dead code today — zero call sites — migrated anyway for consistency, since it's still a public function `test_monitor_completeness.py` may exercise):
```python
# before
def monitor_menu() -> None:
    while True:
        header("Monitor")
        choice = questionary.select(
            "Monitor options:",
            choices=[
                "View my queue",
                "Cancel a job",
                "View job log",
                "View hardware stats",
                "Back to main menu",
            ],
            style=_STYLE,
        ).ask()
        if choice is None or choice == "Back to main menu":
            return
```
```python
# after
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
```

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest tests/test_monitor_completeness.py -v
python3 -m pytest -q
```
Expected: full suite green.

- [ ] **Step 5: Commit**

```bash
git add iitgpu/monitor.py tests/test_monitor_completeness.py
git -c user.name=Daham -c user.email=daham.20242053@iit.ac.lk commit -m "refactor(monitor): unify job/log/history pickers on screen()/select_menu()"
```

---

## Task 8: `iitgpu/setup.py`

**Files:**
- Modify: `iitgpu/setup.py`
- Test: `tests/test_setup.py`

**Interfaces:**
- Consumes: `ui.screen`, `ui.select_menu`, `ui.BACK_TO_MAIN`.

- [ ] **Step 1: Update back-string assertions**

Run: `grep -n "Back to main menu\|\\[back\\]" tests/test_setup.py`
This is `tests/test_setup.py:192/218/224` (the `run_setup` "Back to main menu" flow) and the `_run_install_prebuilt` `"[back]"` picker. Update both to expect `ui.BACK_TO_MAIN` / `ui.BACK` respectively (Red step).

- [ ] **Step 2: Run to verify it fails**

`python3 -m pytest tests/test_setup.py -v`

- [ ] **Step 3: Migrate the module header block**

```python
# before
import questionary
from questionary import Style

from iitgpu import auditclient
from iitgpu.config import Config, conda_sh, jobs_dir, load_config
from iitgpu import slurm as _slurm
from iitgpu.slurm import submit_job
from iitgpu.ui import err, header, info, kv, ok, warn
from iitgpu.validate import in_jail, safe_listdir

_STYLE = Style([
    ("qmark", "fg:cyan bold"),
    ("question", "bold"),
    ("answer", "fg:magenta bold"),
    ("pointer", "fg:cyan bold"),
])
```
```python
# after
import questionary

from iitgpu import auditclient
from iitgpu.config import Config, conda_sh, jobs_dir, load_config
from iitgpu import slurm as _slurm
from iitgpu.slurm import submit_job
from iitgpu.ui import BACK_TO_MAIN, STYLE, err, info, kv, ok, screen, select_menu, warn
from iitgpu.validate import in_jail, safe_listdir

_STYLE = STYLE
```

- [ ] **Step 4: Replace every screen-entry `header(` with `screen(`**

These are one-shot views with no back-button of their own, so only the call name changes: `_run_health_check`, `_run_env_setup`, `_run_data_upload`, `_run_smoke_test`, `_run_model_download`. (`_browse_file`'s own `questionary.select` with `"[cancel]"` stays untouched — it's an in-flow file browser, same category as `wizard._browse_script`, not a screen-level menu.)

- [ ] **Step 5: Migrate `_run_install_prebuilt`'s back button**

```python
# before
def _run_install_prebuilt(cfg: Config) -> None:
    header("Install Prebuilt Environment")
    ...
    choices = [f"{name}  — {desc}" for name, desc in available.items()] + ["[back]"]
    choice = questionary.select(
        "Which prebuilt environment?", choices=choices, style=_STYLE
    ).ask()
    if choice is None or choice == "[back]":
        return
```
```python
# after
def _run_install_prebuilt(cfg: Config) -> None:
    screen("Install Prebuilt Environment")
    ...
    choices = [f"{name}  — {desc}" for name, desc in available.items()]
    choice = select_menu("Which prebuilt environment?", choices)
    if choice is None:
        return
```

- [ ] **Step 6: Migrate `run_setup()`**

```python
# before
def run_setup() -> None:
    cfg = load_config()
    header("Setup")

    if not _run_health_check(cfg):
        return

    steps = [
        ("Environment (conda/venv)", _run_env_setup),
        ("Install a prebuilt environment", _run_install_prebuilt),
        ("Manage environments & containers", _run_env_manager),
        ("Data upload",              _run_data_upload),
        ("Model download",           _run_model_download),
        ("Smoke test",               _run_smoke_test),
    ]

    by_label = dict(steps)
    while True:
        choice = questionary.select(
            "What would you like to set up?",
            choices=[label for label, _ in steps] + ["Back to main menu"],
            style=_STYLE,
        ).ask()
        if choice is None or choice == "Back to main menu":
            return
        by_label[choice](cfg)
```
```python
# after
def run_setup() -> None:
    cfg = load_config()

    if not _run_health_check(cfg):
        return

    steps = [
        ("Environment (conda/venv)", _run_env_setup),
        ("Install a prebuilt environment", _run_install_prebuilt),
        ("Manage environments & containers", _run_env_manager),
        ("Data upload",              _run_data_upload),
        ("Model download",           _run_model_download),
        ("Smoke test",               _run_smoke_test),
    ]

    by_label = dict(steps)
    while True:
        screen("Setup")
        choice = select_menu(
            "What would you like to set up?",
            [label for label, _ in steps],
            back=BACK_TO_MAIN,
        )
        if choice is None:
            return
        by_label[choice](cfg)
```

- [ ] **Step 7: Run tests**

```bash
python3 -m pytest tests/test_setup.py -v
python3 -m pytest -q
```
Expected: full suite green.

- [ ] **Step 8: Commit**

```bash
git add iitgpu/setup.py tests/test_setup.py
git -c user.name=Daham -c user.email=daham.20242053@iit.ac.lk commit -m "refactor(setup): unify env/prebuilt/smoke-test screens on screen()/select_menu()"
```

---

## Task 9: `iitgpu/accounting.py`

**Files:**
- Modify: `iitgpu/accounting.py`
- Test: `tests/test_accounting.py`

**Interfaces:**
- Consumes: `ui.screen`, `ui.select_menu`.

- [ ] **Step 1: Update back-string assertions**

Run: `grep -n '"Back"' tests/test_accounting.py` — update to `ui.BACK` (Red step).

- [ ] **Step 2: Run to verify it fails**

`python3 -m pytest tests/test_accounting.py -v`

- [ ] **Step 3: Migrate `usage_menu`**

```python
# before
def usage_menu() -> None:
    import questionary
    from questionary import Style
    from rich.table import Table
    from iitgpu.ui import console, header, info

    style = Style([("qmark", "fg:cyan bold"), ("pointer", "fg:cyan bold")])
    while True:
        header("Usage & Accounting")
        choice = questionary.select(
            "Report:",
            choices=["GPU/CPU hours per user (30d)", "Fairshare standing",
                     "Raw sreport (30d)", "Back"],
            style=style,
        ).ask()
        if choice is None or choice == "Back":
            return
```
```python
# after
def usage_menu() -> None:
    from rich.table import Table
    from iitgpu.ui import console, info, screen, select_menu

    while True:
        screen("Usage & Accounting")
        choice = select_menu(
            "Report:",
            ["GPU/CPU hours per user (30d)", "Fairshare standing",
             "Raw sreport (30d)"])
        if choice is None:
            return
```
(The `questionary.press_any_key_to_continue("").ask()` at the end of the loop body is unchanged — it needs no style, and stays as a bare `questionary` import at the top of the function: keep `import questionary` in the function body since `press_any_key_to_continue` is still used.)

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest tests/test_accounting.py -v
python3 -m pytest -q
```
Expected: full suite green.

- [ ] **Step 5: Commit**

```bash
git add iitgpu/accounting.py tests/test_accounting.py
git -c user.name=Daham -c user.email=daham.20242053@iit.ac.lk commit -m "refactor(accounting): unify usage menu on screen()/select_menu()"
```

---

## Task 10: `iitgpu/notebooks.py`

**Files:**
- Modify: `iitgpu/notebooks.py`
- Test: `tests/test_notebooks.py`

**Interfaces:**
- Consumes: `ui.screen`, `ui.select_menu`.

- [ ] **Step 1: Update back-string assertions**

Run: `grep -n '"Back"' tests/test_notebooks.py` — update to `ui.BACK` (Red step).

- [ ] **Step 2: Run to verify it fails**

`python3 -m pytest tests/test_notebooks.py -v`

- [ ] **Step 3: Migrate `services_menu`**

```python
# before
def services_menu() -> None:
    import questionary
    from questionary import Style
    from iitgpu.ui import header, info, ok, err, console

    style = Style([("qmark", "fg:cyan bold"), ("pointer", "fg:cyan bold")])
    while True:
        header("My Running Services")
        svcs = running_services()
        if not svcs:
            info("No active notebooks / TensorBoard / interactive sessions.")
            return
        for s in svcs:
            console.print(f"  [magenta]{s.job_id}[/]  {s.name}  [{s.state}]")
            console.print(f"      [dim]{s.tunnel}[/]")
        choices = [f"Stop {s.job_id} ({s.name})" for s in svcs] + ["Refresh", "Back"]
        choice = questionary.select("Action:", choices=choices, style=style).ask()
        if choice is None or choice == "Back":
            return
        if choice == "Refresh":
            continue
        jid = choice.split()[1]
        if questionary.confirm(f"Stop service job {jid}?", default=False, style=style).ask():
```
```python
# after
def services_menu() -> None:
    import questionary
    from iitgpu.ui import STYLE, info, ok, err, console, screen, select_menu

    style = STYLE
    while True:
        screen("My Running Services")
        svcs = running_services()
        if not svcs:
            info("No active notebooks / TensorBoard / interactive sessions.")
            return
        for s in svcs:
            console.print(f"  [magenta]{s.job_id}[/]  {s.name}  [{s.state}]")
            console.print(f"      [dim]{s.tunnel}[/]")
        choices = [f"Stop {s.job_id} ({s.name})" for s in svcs] + ["Refresh"]
        choice = select_menu("Action:", choices)
        if choice is None:
            return
        if choice == "Refresh":
            continue
        jid = choice.split()[1]
        if questionary.confirm(f"Stop service job {jid}?", default=False, style=style).ask():
```

- [ ] **Step 4: Replace `launch_tensorboard`'s `header(` with `screen(`**

```python
# before
    style = Style([("qmark", "fg:cyan bold"), ("pointer", "fg:cyan bold")])
    cfg = load_config()
    header("Launch TensorBoard")
```
```python
# after
    style = STYLE
    cfg = load_config()
    screen("Launch TensorBoard")
```
And update its local import line from `from iitgpu.ui import header, info, ok, err, kv, panel` to `from iitgpu.ui import STYLE, info, ok, err, kv, panel, screen` (drop the now-unused `from questionary import Style` in that function too).

- [ ] **Step 5: Run tests**

```bash
python3 -m pytest tests/test_notebooks.py -v
python3 -m pytest -q
```
Expected: full suite green.

- [ ] **Step 6: Commit**

```bash
git add iitgpu/notebooks.py tests/test_notebooks.py
git -c user.name=Daham -c user.email=daham.20242053@iit.ac.lk commit -m "refactor(notebooks): unify services/TensorBoard screens on screen()/select_menu()"
```

---

## Task 11: `iitgpu/upload.py`

**Files:**
- Modify: `iitgpu/upload.py`
- Test: `tests/test_upload.py`

**Interfaces:**
- Consumes: `ui.screen`, `ui.select_menu`, `ui.BACK_TO_MAIN`.

- [ ] **Step 1: Update back-string assertions**

`tests/test_upload.py` lines 206-228 and 534 use plain `"back"` for `run_upload`'s destination picker (whose real sentinel today is `"__cancel__"`, not `"back"` — check carefully, these two are different pickers in the same function) and its action picker. Run: `grep -n '"back"\|__cancel__\|__new__' tests/test_upload.py` and confirm which picker each mocked-answer sequence targets before editing. Update the ones targeting the ACTION picker (`"What would you like to do?"`) to expect `ui.BACK`; the destination picker's `"__cancel__"` sentinel becomes `ui.BACK` too once migrated (Step 3 below folds both pickers onto `select_menu`).

- [ ] **Step 2: Run to verify it fails**

`python3 -m pytest tests/test_upload.py -v`

- [ ] **Step 3: Migrate the module header and `run_upload`**

```python
# before
import questionary
from questionary import Style

from iitgpu import auditclient
from iitgpu.config import load_config, make_shared_writable
from iitgpu.ui import console, ok, err, info, header
from iitgpu.validate import in_jail

_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")

_STYLE = Style([
    ("qmark",       "fg:cyan bold"),
    ("question",    "bold"),
    ("answer",      "fg:magenta bold"),
    ("pointer",     "fg:cyan bold"),
    ("highlighted", "fg:cyan bold"),
])
```
```python
# after
import questionary

from iitgpu import auditclient
from iitgpu.config import load_config, make_shared_writable
from iitgpu.ui import BACK_TO_MAIN, STYLE, console, ok, err, info, screen, select_menu
from iitgpu.validate import in_jail

_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")

_STYLE = STYLE
```

Every `header(` in this file (`_show_scp_instructions`, `_unzip_in_folder`, `_browse_folder`, `_download_from_url`) becomes `screen(`.

`run_upload`'s destination picker:
```python
# before
    _folder_choices = (
        [questionary.Choice(f"{n}  ({base / n})", str(base / n)) for n in _existing]
        + [questionary.Choice("[upload here — my data folder]", str(base)),
           questionary.Choice("[create new sub-folder]",        "__new__"),
           questionary.Choice("[cancel]",                        "__cancel__")]
    )
    prompt = f"Select a destination inside your folder ({base}):"

    sel = questionary.select(prompt, choices=_folder_choices, style=_STYLE).ask()

    if sel is None or sel == "__cancel__":
        return
```
```python
# after
    _folder_choices = (
        [questionary.Choice(f"{n}  ({base / n})", str(base / n)) for n in _existing]
        + [questionary.Choice("[upload here — my data folder]", str(base)),
           questionary.Choice("[create new sub-folder]",        "__new__")]
    )
    prompt = f"Select a destination inside your folder ({base}):"

    sel = select_menu(prompt, _folder_choices)

    if sel is None:
        return
```

`run_upload`'s action picker:
```python
# before
    choices = [
        questionary.Choice("Upload from my computer  (zip + SCP instructions)",    "scp"),
        questionary.Choice("Unzip an uploaded .zip  (extract here, with progress)", "unzip"),
        questionary.Choice("Download from a URL  (wget / curl on the server)",      "url"),
        questionary.Choice("Browse folder contents",                                "browse"),
        questionary.Choice("Back to main menu",                                     "back"),
    ]

    while True:
        action = questionary.select(
            "What would you like to do?", choices=choices, style=_STYLE
        ).ask()
        if action is None or action == "back":
            break
```
```python
# after
    choices = [
        questionary.Choice("Upload from my computer  (zip + SCP instructions)",    "scp"),
        questionary.Choice("Unzip an uploaded .zip  (extract here, with progress)", "unzip"),
        questionary.Choice("Download from a URL  (wget / curl on the server)",      "url"),
        questionary.Choice("Browse folder contents",                                "browse"),
    ]

    while True:
        action = select_menu("What would you like to do?", choices, back=BACK_TO_MAIN)
        if action is None:
            break
```

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest tests/test_upload.py -v
python3 -m pytest -q
```
Expected: full suite green.

- [ ] **Step 5: Commit**

```bash
git add iitgpu/upload.py tests/test_upload.py
git -c user.name=Daham -c user.email=daham.20242053@iit.ac.lk commit -m "refactor(upload): unify destination/action pickers on screen()/select_menu()"
```

---

## Task 12: `iitgpu/files.py`

**Files:**
- Modify: `iitgpu/files.py`
- Test: `tests/test_files.py` if present, else full-suite run.

**Interfaces:**
- Consumes: `ui.screen`, `ui.select_menu`.

- [ ] **Step 1: Check for existing back-string assertions**

Run: `grep -rln "\\[ back \\]\|file_manager" tests/`
Update any found expecting `"[ back ]"` to expect `ui.BACK` (Red step).

- [ ] **Step 2: Run to verify it fails (if a test was updated)**

`python3 -m pytest tests/test_files.py -v` (skip if none found).

- [ ] **Step 3: Migrate `file_manager`'s outer loop**

```python
# before
    import getpass
    import questionary
    from questionary import Style
    from iitgpu.config import load_config, user_dir, is_admin
    from iitgpu.validate import in_user_browse_jail, in_user_upload_jail
    from iitgpu.ui import console, header, info, ok, err

    style = Style([("qmark", "fg:cyan bold"), ("pointer", "fg:cyan bold")])
```
```python
# after
    import getpass
    import questionary
    from iitgpu.config import load_config, user_dir, is_admin
    from iitgpu.validate import in_user_browse_jail, in_user_upload_jail
    from iitgpu.ui import STYLE, console, info, ok, err, screen, select_menu

    style = STYLE
```

```python
# before
        header(f"Files — {cur}")
        if total:
            info(f"Disk: {fmt_size(free)} free of {fmt_size(total)}")
        if not writable_cur:
            info("[dim]Read-only area — browsing only[/]")
        entries = list_dir(cur)
        extra_rows = ["[ + new folder ]"] if writable_cur else []
        rows = ["[.. up]"] + [
            (f"[dir] {e.name}" if e.is_dir else f"      {e.name}  ({fmt_size(e.size)})")
            for e in entries
        ] + extra_rows + ["[ refresh ]", "[ back ]"]
        choice = questionary.select(f"{cur}", choices=rows, style=style).ask()
        if choice is None or choice == "[ back ]":
            return
```
```python
# after
        _status = f"Disk: {fmt_size(free)} free of {fmt_size(total)}" if total else None
        screen(f"Files — {cur}", status=_status)
        if not writable_cur:
            info("[dim]Read-only area — browsing only[/]")
        entries = list_dir(cur)
        extra_rows = ["[ + new folder ]"] if writable_cur else []
        rows = ["[.. up]"] + [
            (f"[dir] {e.name}" if e.is_dir else f"      {e.name}  ({fmt_size(e.size)})")
            for e in entries
        ] + extra_rows + ["[ refresh ]"]
        choice = select_menu(f"{cur}", rows)
        if choice is None:
            return
```
(Leave the `dir_choices`/`file_choices` sub-pickers — `["Open", "Rename", "Delete", "Cancel"]` and `["Rename", "Delete", "Show size", "Cancel"]` — untouched: these are one-off action confirmations on a single selected entry, not a screen with its own back button, matching the plan's non-goal on in-flow prompts.)

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest -q
```
Expected: full suite green (no dedicated `test_files.py` back-string tests were found in the repo scan during design — the full-suite run is the safety net for this file).

- [ ] **Step 5: Commit**

```bash
git add iitgpu/files.py
git -c user.name=Daham -c user.email=daham.20242053@iit.ac.lk commit -m "refactor(files): unify file manager browser on screen()/select_menu()"
```

---

## Task 13: `iitgpu/models.py`

**Files:**
- Modify: `iitgpu/models.py`
- Test: `grep -rln "model_menu\|_remove_interactive\|\\[back\\]" tests/`

**Interfaces:**
- Consumes: `ui.screen`, `ui.select_menu`, `ui.BACK_TO_MAIN`.

- [ ] **Step 1: Update back-string assertions**

Run: `grep -rn '"\[back\]"\|Back to main menu' tests/ | grep -i model`
Update found assertions in `_remove_interactive` (`"[back]"` → `ui.BACK`) and `model_menu` (`"Back to main menu"` → `ui.BACK_TO_MAIN`). Do **not** touch anything asserting `"[none / skip]"` — that's `pick_model`'s legitimate "no selection" sentinel, not a back button, and stays exactly as-is.

- [ ] **Step 2: Run to verify it fails**

`python3 -m pytest tests/test_models.py -v` (adjust to whatever file Step 1's grep found).

- [ ] **Step 3: Migrate the module header block**

```python
# before
from iitgpu.ui import err, header, info, kv, ok, warn

_STYLE = Style([
    ("qmark", "fg:cyan bold"),
    ("question", "bold"),
    ...
])
```
```python
# after
from iitgpu.ui import STYLE, err, info, kv, ok, screen, select_menu, warn

_STYLE = STYLE
```
(Drop the now-unused `from questionary import Style` import; keep `import questionary` since `questionary.text`/`questionary.confirm` calls remain throughout the file.)

Replace every screen-entry `header(` with `screen(`: `_list_models`, `_download_hf_interactive`, `_download_url_interactive`, `model_menu`.

- [ ] **Step 4: Migrate `_remove_interactive`**

```python
# before
    choices = [f"{e.name}  ({e.source}, {e.size_mb} MB)" for e in entries] + ["[back]"]
    choice = questionary.select("Select model to remove from registry:", choices=choices, style=_STYLE).ask()
    if choice is None or choice == "[back]":
        return
```
```python
# after
    choices = [f"{e.name}  ({e.source}, {e.size_mb} MB)" for e in entries]
    choice = select_menu("Select model to remove from registry:", choices)
    if choice is None:
        return
```

- [ ] **Step 5: Migrate `model_menu`**

```python
# before
def model_menu(cfg: Config) -> None:
    while True:
        header("Model Library")
        choice = questionary.select(
            "Model options:",
            choices=[
                "List models",
                "Download from HuggingFace Hub",
                "Download from URL",
                "Remove from registry",
                "Back to main menu",
            ],
            style=_STYLE,
        ).ask()
        if choice is None or choice == "Back to main menu":
            return
```
```python
# after
def model_menu(cfg: Config) -> None:
    while True:
        screen("Model Library")
        choice = select_menu(
            "Model options:",
            [
                "List models",
                "Download from HuggingFace Hub",
                "Download from URL",
                "Remove from registry",
            ],
            back=BACK_TO_MAIN,
        )
        if choice is None:
            return
```
(`pick_model`'s `"[none / skip]"` picker is intentionally left untouched — see Task 13 Step 1 note.)

- [ ] **Step 6: Run tests**

```bash
python3 -m pytest -q
```
Expected: full suite green.

- [ ] **Step 7: Commit**

```bash
git add iitgpu/models.py
git -c user.name=Daham -c user.email=daham.20242053@iit.ac.lk commit -m "refactor(models): unify model library menu on screen()/select_menu()"
```

---

## Task 14: `iitgpu/wizard.py` + `iitgpu/review.py` — port the already-good screens to the shared primitives

**Files:**
- Modify: `iitgpu/wizard.py`, `iitgpu/review.py`
- Test: `tests/test_wizard.py`, `tests/test_review.py`

**Interfaces:**
- Consumes: `ui.screen`, `ui.select_menu`, `ui.BACK`, `ui.STYLE`.
- Produces: no change to `run_wizard()`/`run_hub()`'s external signatures — pure internal port.

- [ ] **Step 1: Update back-string assertions**

`tests/test_wizard.py:820` mocks `"back"` for the own-.sbatch/template picker; `tests/test_review.py:428,443` mock `"back"` for the Advanced menu. Update both to `ui.BACK` (Red step).

- [ ] **Step 2: Run to verify it fails**

```bash
python3 -m pytest tests/test_wizard.py tests/test_review.py -v
```

- [ ] **Step 3: Migrate `wizard.py`'s module-level style and header calls**

```python
# before
from iitgpu.ui import err, header, info, kv, ok, panel, warn
...
_STYLE = Style([
    ("qmark", "fg:cyan bold"),
    ("question", "bold"),
    ("answer", "fg:magenta bold"),
    ("pointer", "fg:cyan bold"),
    ("highlighted", "fg:cyan bold"),
    ("selected", "fg:magenta"),
])
```
```python
# after
from iitgpu.ui import BACK, STYLE, err, info, kv, ok, panel, screen, select_menu, warn
...
_STYLE = STYLE
```
(Every existing `style=_STYLE` call site in `wizard.py` keeps working unchanged via this alias.) Replace the one `header("New Job")` call with `screen("New Job")`.

- [ ] **Step 4: Migrate the own-.sbatch/template back button**

```python
# before
            choices=["Submit my own .sbatch", "Load a template", "back"],
        ...
        if sub == "back":
            continue                      # back to the intent list, not out
```
```python
# after
            choices=["Submit my own .sbatch", "Load a template"],
        ...
        if sub is None:
            continue                      # back to the intent list, not out
```
Find this picker's full `questionary.select(...).ask()` call (immediately preceding this block) and wrap it with `select_menu(...)` the same way every other file in this plan does, matching its existing prompt text verbatim.

- [ ] **Step 5: Migrate `review.py`'s Data/model and Advanced back buttons**

```python
# before
    sel = questionary.select("Data / model:", choices=opts + ["back"]).ask()
```
```python
# after
    sel = select_menu("Data / model:", opts)
```
(No `if sel == "back"` check exists after this line today — confirm with `grep -n -A3 'Data / model:' iitgpu/review.py` before editing; if the caller's subsequent `if sel == "data folder (browse)":`/`elif` chain falls through safely on `None` already, no further change is needed there. If it does not, add `if sel is None: return` immediately after.)

```python
# before
        opts += [f"email notifications [{'on' if ls.mail else 'off'}]",
                 "view generated sbatch", "back"]
        sel = questionary.select("Advanced:", choices=opts).ask()
        if sel is None or sel == "back":
            return
```
```python
# after
        opts += [f"email notifications [{'on' if ls.mail else 'off'}]",
                 "view generated sbatch"]
        sel = select_menu("Advanced:", opts)
        if sel is None:
            return
```
Add `from iitgpu.ui import select_menu` to `review.py`'s imports (it currently imports `console, info, warn` from `iitgpu.ui` — extend that line).

- [ ] **Step 6: Port `run_hub`'s own `questionary.select("Select:", ...)` to `select_menu`**

```python
# before
        sel = questionary.select("Select:", choices=choices).ask()
        if sel is None or sel == "Cancel":
            return None
```
```python
# after
        sel = select_menu("Select:", [c for c in choices if c != "Cancel"])
        if sel is None:
            return None
```
("Cancel" already means exactly what `BACK` means here — exit the hub — so it's replaced by `select_menu`'s trailing Back item rather than kept as a duplicate in-list entry. Every other entry in `_HUB_CHOICES` is unchanged.)

- [ ] **Step 7: Run tests**

```bash
python3 -m pytest tests/test_wizard.py tests/test_review.py -v
python3 -m pytest -q
```
Expected: full suite green.

- [ ] **Step 8: Commit**

```bash
git add iitgpu/wizard.py iitgpu/review.py tests/test_wizard.py tests/test_review.py
git -c user.name=Daham -c user.email=daham.20242053@iit.ac.lk commit -m "refactor(wizard,review): port New Job flow onto the same shared primitives"
```

---

## Task 15: Version bump, full verification, deploy

**Files:**
- Modify: `iitgpu/__init__.py`

- [ ] **Step 1: Bump the version**

```python
# before
__version__ = "1.2.x"   # whatever Task 14 left it at — check with:
# grep __version__ iitgpu/__init__.py
```
```python
# after
__version__ = "1.3.0"
```

- [ ] **Step 2: Run the full suite one more time**

```bash
cd ~/IIT-Secure-SLURM-Job-Gateway && python3 -m pytest -q
```
Expected: 0 failures (baseline 770 + the new `test_ui.py` cases from Task 1 + Task 2's one addition).

- [ ] **Step 3: Grep for any leftover inconsistent back-button spelling across the whole app**

```bash
grep -rn '"\[back\]"\|"back"\|"\[ back \]"\|== "Back"\b' iitgpu/ | grep -v "iitgpu/ui.py"
```
Expected: no output. If anything remains, it's a screen this plan's tasks missed — go back and migrate it the same way (Task 1's `select_menu`/`BACK`) before proceeding.

- [ ] **Step 4: Commit the version bump**

```bash
git add iitgpu/__init__.py
git -c user.name=Daham -c user.email=daham.20242053@iit.ac.lk commit -m "chore: bump version to 1.3.0 for the TUI design unification"
```

- [ ] **Step 5: Merge and deploy**

```bash
git log --oneline main..HEAD   # sanity-check the commit list before merging, if this was done on a branch
git checkout main && git merge --ff-only <branch>   # or push directly if committed straight to main
bash deploy/redeploy-igm.sh   # MUST run as slurmadmin directly, not via sudo — see project memory
```

- [ ] **Step 6: Verify live**

```bash
ssh -o BatchMode=yes slurmadmin@192.168.122.10 'iit-gpu-manager --version 2>/dev/null || python3 -c "from iitgpu import __version__; print(__version__)"'
```
Expected: `1.3.0`. Then walk through Main Menu → Jobs → back, My Workspace → back, Settings → back, and (if admin) Admin → QOS/limits → back, confirming every screen shows a bordered header Panel and the same `← Back` at the bottom of its list.

- [ ] **Step 7: Tag the release**

```bash
git tag v1.3.0
git push origin main v1.3.0
```
(Only if the user has confirmed pushing to GitHub is wanted this session — the repo's `origin` PAT has been flagged as compromised/unrotated in project history; confirm before any GitHub push, per standing project caution.)

---

## Self-Review Notes

- **Spec coverage:** Goal 1 (shared header+status Panel) → Task 1 + every migration task. Goal 2 (one back spelling) → Task 1's `BACK`/`BACK_TO_MAIN` + every migration task's before/after. Goal 3 (one Style) → Task 1's `STYLE` + every migration task dropping its local `Style(...)`. Goal 4 (no invented data) → every status body in Tasks 3, 5, 6, 12 sources from a pre-existing function (`get_node_stats`, `queue`, `list_qos`, `list_gpuusers`, `is_mail_disabled`, `get_maintenance`, `disk_usage`). Goal 5 → Task 15. Non-goals (in-flow prompts, `pick_model`'s skip sentinel, `_browse_file`/`_browse_script`'s cancel sentinel, live dashboards) are explicitly called out as untouched in Tasks 8, 12, 13.
- **Every file from the spec's migration list has a task:** `ui.py`(1), `splash.py`(2), `menu.py`(3), `workspace.py`(4), `admin.py`(5,6), `monitor.py`(7), `setup.py`(8), `accounting.py`(9), `notebooks.py`(10), `wizard.py`/`review.py`(14), `__init__.py`(15). Three files surfaced during repo exploration that the original spec's file list under-counted — `upload.py`, `files.py`, `models.py` (all reached from My Workspace and just as much "a TUI menu" as anything else) — got their own tasks (11, 12, 13) so the "full tool" goal is actually met.
- **Type/name consistency check:** `screen(title, *, status=None)` and `select_menu(prompt, choices, *, back=BACK)` signatures are used identically across all 14 migration tasks — no task invents a different parameter name or order.
