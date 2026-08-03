# Screenshots / GIFs checklist

Images referenced by the main [`README.md`](../../README.md). Drop files at
the exact paths below and the placeholders there start rendering — no other
changes needed.

## Done

| Path | Format | What it shows |
|---|---|---|
| `splash-and-menu.svg` | SVG | Splash screen through the Main Menu — a real capture from a `--demo` run, not a mockup (see "How this one was made" below). |

## Still needed

| Path | Format | What to capture |
|---|---|---|
| `new-job-review-hub.png` | PNG | The New Job wizard's editable review hub (`slurmdeck/review.py`) — job summary with live GPU-slice availability. |
| `live-dashboard.png` | PNG | The live dashboard (`slurmdeck/dashboard.py`) — cluster panel, jobs table, a job's log tail. |
| `admin-panel.png` | PNG | The Admin submenu (`slurmdeck/menu.py`) — the full 16-item admin panel. |
| `install-wizard.gif` | GIF | `install-wizard.sh` running end to end against a real or throwaway machine — mode-select prompt through at least one confirm-risky checkpoint. |

**Capture tips:**
- Terminal recordings: [asciinema](https://asciinema.org/) + [agg](https://github.com/asciinema/agg) (asciinema→GIF) give clean, small GIFs; `ttyrec`/`peek`/`terminalizer` work too.
- Use a terminal profile with a readable font size and a reasonably narrow width (~100 cols) so the images aren't squashed when rendered in GitHub's markdown viewer.
- Redact anything site-specific (real hostnames, IPs, usernames, emails) before capturing, or capture against `DEMO_MODE=1` (see [Demo mode](../../README.md#demo-mode-no-slurm-required)) so nothing needs redacting.

## How `splash-and-menu.svg` was made

No real terminal or GUI was screenshotted — it's a genuine capture of the
actual TUI, not a mockup, produced without touching a real cluster:

1. Ran `python3 -m slurmdeck --demo` inside a real PTY (`pexpect.spawn`,
   `DEMO_MODE=1`, an isolated `NFS_ROOT`/`HOME` under `/tmp`), with
   `slurmdeck.slurm.get_node_stats` monkeypatched to return fixed, clearly
   fake numbers instead of live cluster data.
2. Fed the raw captured bytes through `pyte` (a terminal emulator library)
   to get the correct final screen state — this matters because the splash
   screen's live status panel redraws in place, so a raw byte dump alone
   shows overlapping frames.
3. Converted `pyte`'s per-cell styled grid into a `rich.text.Text` per row,
   printed it to a `rich.console.Console(record=True)`, and called
   `.save_svg()`.

The same technique works for any other screen in this list — swap the
`pexpect` interaction (what's typed, how long to wait) for whatever gets you
to the screen you want, and everything from step 2 onward is unchanged. The
`new-job-review-hub.png`/`live-dashboard.png`/`admin-panel.png` targets
above would need real job data faked similarly (or genuinely submitted, if
capturing on a disposable demo cluster) before the picker widgets would show
anything meaningful.
