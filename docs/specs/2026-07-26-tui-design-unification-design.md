# TUI design unification

**Date:** 2026-07-26
**Status:** Approved for planning

## Motivation

The TUI has grown one file at a time and it shows: `menu.py`, `workspace.py`,
`admin.py`, `monitor.py`, `setup.py`, `accounting.py`, and `notebooks.py` each
define their own `questionary.Style`, their own header treatment (a bare
`console.rule()` via `ui.header()`), and their own "go back" affordance —
`"back"`, `"Back"`, `"[back]"`, `"Back to main menu"`, and indented
`"  Back"` all exist as four different spellings of the same button. The
Launch Review Hub (`iitgpu/review.py`) and the launch-flow intent screen
(`iitgpu/wizard.py`) are the one part of the app that already reads as
designed: a bordered Panel that states the screen's purpose and shows live,
true information (GPU-slice availability) right where the user decides,
rather than a bare list of options.

The user has asked for the rest of the tool to look and feel like that one
part, with a consistent back button everywhere, and confirmed they want the
richer bordered-panel-with-live-status treatment applied broadly, not just a
mechanical style/back-button cleanup.

## Goals

1. One shared look for every screen: a bordered header Panel + (where there's
   something true and cheap to show) a live one-line-or-table status body,
   the same visual language as the review hub.
2. One spelling of "go back", everywhere, with identical semantics: Esc,
   Ctrl-C, and picking Back are all indistinguishable to the caller.
3. One `questionary.Style` object, used by every screen — not five near-copies.
4. No new live-data plumbing: every status panel shows something an existing
   function already computes (`get_node_stats()`, `filtered_history()`,
   `list_qos()`, disk usage helpers already in `workspace.py`/`admin.py`,
   etc.). If a screen has nothing true and cheap to show, it gets the header
   Panel alone — an empty or padded-out status body is worse than none.
5. Ship as v1.3.0 once merged (bump `iitgpu/__init__.py.__version__` only,
   per existing convention — no other file carries the version string).

## Non-goals

- No change to business logic, audit calls, SLURM interaction, or file jail
  rules — this is screen chrome only.
- No change to prompts *within* a flow (e.g. wizard's "Time limit:" text
  input, admin's "Drain node" reason prompt) — only the screen-entry
  header/status/back layer.
- No new keybindings (no Esc-to-go-back beyond what questionary already
  gives via Ctrl-C) — Back stays a selectable list item, matching how every
  existing menu already works and avoiding a prompt_toolkit keybinding layer.
- Non-interactive output (Rich tables printed once, log tailing, live
  dashboards) is untouched.

## Architecture

### `iitgpu/ui.py` gains three things

**`STYLE`** — one `questionary.Style`, promoted from the version already
duplicated (near-identically) in `menu.py`, `wizard.py`, `workspace.py`, and
`monitor.py`:
```python
STYLE = Style([
    ("qmark", "fg:cyan bold"),
    ("question", "bold"),
    ("answer", "fg:magenta bold"),
    ("pointer", "fg:cyan bold"),
    ("highlighted", "fg:cyan bold"),
    ("selected", "fg:magenta"),
])
```
`admin.py`'s narrower inline `Style([("qmark", ...), ("pointer", ...)])` is
replaced with this too — same visual weight everywhere.

**`BACK = "← Back"`** — the one spelling. `menu.py`'s top-level "Back to main
menu" keeps its fuller wording (it's the one place "back" isn't obviously
"back to where I came from") but is defined as `BACK_TO_MAIN = "← Back to main menu"`
alongside it, same arrow glyph, same list position.

**`screen(title, *, status=None) -> None`** — replaces `header()` at every
screen entry point:
```python
def screen(title: str, *, status: RenderableType | str | None = None) -> None:
    body = status if status is not None else ""
    console.print(Panel(body, title=f"[bold]{title}[/]", border_style="cyan"))
```
Called once per loop iteration, same place `header("...")` is called today.
`header()` itself is kept (some non-menu call sites use it for sub-step
banners inside a flow) but no menu screen calls it anymore.

**`select_menu(prompt, choices, *, back=BACK) -> str | None`**:
```python
def select_menu(prompt: str, choices: list, *, back: str = BACK) -> str | None:
    sel = questionary.select(prompt, choices=[*choices, Separator(), back],
                              style=STYLE).ask()
    return None if sel in (None, back) else sel
```
Every call site's `if choice is None or choice == "<whatever back looked like
here>":  return` collapses to `if choice is None: return`. Choices that are
already `Separator()`-grouped (admin.py's grouped sections) keep their
internal separators — `select_menu` only adds the trailing one before Back.

### Status bodies, per screen

Each status body is a thin wrapper around a function that already exists —
nothing new is computed:

| Screen | Status body | Source |
|---|---|---|
| Main Menu | One line: cluster verdict (`OK to submit a GPU job` / `GPU busy, CPU ok` / `Cluster full`) | `splash._build_status_line`'s verdict logic, factored out to a plain-string helper reused by both splash and here |
| Jobs | One line: my running/queued count + free GPU slices | `slurm.get_node_stats()` + `filtered_history()` |
| Settings | One line: last health-check verdict or "not checked this session" | `setup.check_cluster_health()` (only if already run — no forced check on menu entry, that's a real command elsewhere in the menu) |
| My Workspace | Unchanged — already renders Disk/Files/Environments/Models Panels above its action list; those become the "status", the menu just also gets the shared header/back | existing `_disk_usage_summary`, `_recent_jobs`, etc. |
| Admin | One line: active user count + mail service on/off + maintenance on/off | `list_gpuusers()`, `is_mail_disabled()`, `get_maintenance()` |
| Admin → QOS / limits | Small table of QOS name/wall/GPUs/priority | `list_qos()` (already fetched, just rendered before the picker instead of not at all) |
| Admin → View users, Disk usage by user, Audit log, etc. | No separate status body — these screens *are* a data view already (a table is printed); they get the header Panel + `select_menu` back-button treatment, nothing duplicated |

Screens with nothing cheap and true to show (e.g. `_login_as_menu`,
`_maintenance_menu`) get `screen(title)` with no status — consistent chrome,
no invented content.

### Wizard / review hub

`iitgpu/wizard.py`'s intent screen and `iitgpu/review.py`'s hub already use
the Panel-header pattern; they're ported to call `ui.screen()` /
`ui.select_menu()` / `ui.STYLE` / `ui.BACK` instead of their local
equivalents, so there's one implementation, not two that happen to agree
today. The two inline `"back"` list entries in `wizard.py` (own-.sbatch vs
template picker) and `review.py` (Data/model picker, Advanced picker) switch
to `BACK`.

## Migration list (file → what changes)

- **`iitgpu/ui.py`** — add `STYLE`, `BACK`, `BACK_TO_MAIN`, `screen()`, `select_menu()`.
- **`iitgpu/menu.py`** — Main Menu, Jobs submenu, Settings submenu, cluster-status view all move to `screen()`/`select_menu()`. Main Menu's list keeps "New Job" first (already the implicit default cursor position; made explicit isn't needed — `questionary.select` already defaults to the first choice).
- **`iitgpu/workspace.py`** — header/back only; existing Panels/Tables are the status body already.
- **`iitgpu/admin.py`** — `admin_menu()` + all ~9 submenus (`_qos_menu`, `_provision_menu`, `_view_audit_log`, `_view_users`, `_view_maillog`, `_view_job_output`, `_view_service_health`, `_view_disk_usage`, `_maintenance_menu`, `_mail_service_menu`, `_login_as_menu`). Drop the local narrow `Style` and the `"  "`-indentation hack now that a real Panel provides the visual grouping.
- **`iitgpu/monitor.py`** — job list/manage/log-browse/history screens; four different `"[back]"` spellings collapse to one.
- **`iitgpu/setup.py`** — env-setup/prebuilt-install picker screens.
- **`iitgpu/accounting.py`** — usage menu.
- **`iitgpu/notebooks.py`** — services menu.
- **`iitgpu/wizard.py`, `iitgpu/review.py`** — port to shared primitives (see above), no behavior change.
- **`iitgpu/splash.py`** — factor the plain-text verdict out of `_build_status_line` into a reusable function so Main Menu's status line and the splash's live block share one source of truth instead of two copies of the same three-way if/elif.
- **`iitgpu/__init__.py`** — version bump to `1.3.0`, last commit of the branch.

## Testing

- Every existing test that asserts on an exact back-button string (`"back"`,
  `"[back]"`, `"Back"`, `"Back to main menu"`) needs updating to the new
  constant — expected, mechanical, caught immediately by the 770-test pytest
  gate.
- New tests: `ui.select_menu()` returns `None` for both `None` input and the
  back sentinel (2 cases); `ui.screen()` renders a Panel with the given title
  (1 case, Console-capture pattern already used elsewhere in the suite);
  `splash`'s extracted verdict helper returns the three expected strings for
  synthetic stats (already covered indirectly by existing splash tests, just
  confirm the extraction doesn't change output).
- No live-cluster verification needed — this is app-layer TUI code covered
  by the existing pytest suite; the deploy's pytest gate is the acceptance
  bar, same as every other change in this repo.

## Rollout

Standard flow for this repo: build on a branch, pytest gate, whole-diff
review, fix wave if needed, merge to `main`, `redeploy-igm.sh` as
`slurmadmin` (not sudo), verify version + a couple of screens live, tag
`v1.3.0`.
