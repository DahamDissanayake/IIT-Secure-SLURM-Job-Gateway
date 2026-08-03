# Screenshots / GIFs checklist

Images referenced by the main [`README.md`](../../README.md) but not yet
added. Drop files at the exact paths below and the placeholders there start
rendering — no other changes needed.

| Path | Format | What to capture |
|---|---|---|
| `splash-and-menu.gif` | GIF | A real SSH login landing in the TUI: splash screen (`slurmdeck/splash.py`) through the Main Menu appearing. |
| `new-job-review-hub.png` | PNG | The New Job wizard's editable review hub (`slurmdeck/review.py`) — job summary with live GPU-slice availability. |
| `live-dashboard.png` | PNG | The live dashboard (`slurmdeck/dashboard.py`) — cluster panel, jobs table, a job's log tail. |
| `admin-panel.png` | PNG | The Admin submenu (`slurmdeck/menu.py`) — the full 16-item admin panel. |
| `install-wizard.gif` | GIF | `install-wizard.sh` running end to end against a real or throwaway machine — mode-select prompt through at least one confirm-risky checkpoint. |

**Capture tips:**
- Terminal recordings: [asciinema](https://asciinema.org/) + [agg](https://github.com/asciinema/agg) (asciinema→GIF) give clean, small GIFs; `ttyrec`/`peek`/`terminalizer` work too.
- Use a terminal profile with a readable font size and a reasonably narrow width (~100 cols) so the images aren't squashed when rendered in GitHub's markdown viewer.
- Redact anything site-specific (real hostnames, IPs, usernames, emails) before capturing, or capture against `DEMO_MODE=1` (see [Demo mode](../../README.md#demo-mode-no-slurm-required)) so nothing needs redacting.
