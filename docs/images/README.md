# Screenshots

All five images `README.md` references are real captures of the actual TUI
running in `--demo` mode, not mockups. None of them touched a real cluster.

| Path | What it shows |
|---|---|
| `splash-and-menu.svg` | Splash screen through the Main Menu. |
| `new-job-review-hub.svg` | The New Job review hub after picking "Open JupyterLab". |
| `live-dashboard.svg` | The live dashboard with one demo job queued. |
| `admin-panel.svg` | The Admin submenu (all 16 items). |
| `install-wizard.svg` | `install-wizard.sh`, mode-select through its first confirm-risky checkpoint (declined -- nothing executed). Static, not an animated recording; see "Improving on this" below if you want a real GIF of a full run. |

## How they were made

No real terminal or GUI was screenshotted:

1. Ran `python3 -m slurmdeck --demo` (or `install-wizard.sh`) inside a real
   PTY (`pexpect.spawn`), scripting the exact prompts/keystrokes needed to
   reach each screen. Isolated `NFS_ROOT`/`HOME` under `/tmp`, and forced
   `SACCT_ENABLED=0` and a fake `GATEWAY_HOST` -- `DEMO_MODE=1` alone does
   **not** fully isolate from the real cluster: `slurm.get_node_stats()`
   still queries live hardware stats and `filtered_history()` still shells
   out to real `sacct` if available, so both were monkeypatched to return
   fixed, obviously-fake numbers before capturing. Worth knowing if you
   reuse this technique against a live cluster -- always check what a
   `--demo` run actually touches before trusting it's isolated.
2. Fed the raw captured bytes through `pyte` (a terminal emulator library)
   to get the correct final screen state -- this matters because live
   panels (the splash screen's status line, the dashboard) redraw in
   place, so a raw byte dump alone shows overlapping frames.
3. Converted `pyte`'s per-cell styled grid into a `rich.text.Text` per row,
   cropped to the rows that matter, printed to a
   `rich.console.Console(record=True)`, and exported with `.save_svg()`.

## Improving on this

- `install-wizard.svg` is a single static frame. A real animated GIF of a
  full run (asciinema + [agg](https://github.com/asciinema/agg), or any
  terminal recorder) would show the confirm-checkpoint flow much better.
- `live-dashboard.svg` shows a job stuck at `PENDING` (demo mode never
  actually starts it) -- capturing right after a state transition, or
  faking `queue()` with a `RUNNING` entry, would look livelier.
- Real usage discovers real gaps: while building these, two site-specific
  values turned out to be hardcoded in the app itself rather than read
  from config (`slurmdeck/slurm.py`'s compute node hostname default, and
  `slurmdeck/splash.py`'s tagline) -- both fixed. `filtered_history()`
  bypassing `DEMO_MODE` (point 1 above) is still open.
