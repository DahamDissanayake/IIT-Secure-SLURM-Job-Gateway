# slurmdeck/connect.py — post-submit notebook connection: readiness marker wait,
# authoritative parse of the job's own stdout, and the Connect card.
#
# The tunnel line and URL are parsed from the job's .out rather than being
# reconstructed, so this can never repeat the advertised-vs-bound port bug:
# whatever the job printed is what the user gets.
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from rich.panel import Panel

_TUNNEL_RE = re.compile(r"^\s*(ssh -p \d+ -N -L \d+:\S+:\d+ \S+@\S+)\s*$", re.M)
_URL_RE = re.compile(r"(http://127\.0\.0\.1:(\d+)/lab\?token=([0-9a-f]+))")


@dataclass(frozen=True)
class ConnectInfo:
    port: int
    token: str
    tunnel: str
    url: str


def parse_connect(out_text: str) -> ConnectInfo | None:
    mt = _TUNNEL_RE.search(out_text or "")
    mu = _URL_RE.search(out_text or "")
    if not (mt and mu):
        return None
    return ConnectInfo(port=int(mu.group(2)), token=mu.group(3),
                       tunnel=mt.group(1), url=mu.group(1))


def render_card(info: ConnectInfo) -> Panel:
    body = (
        "\n  [bold]1.[/] On [bold]YOUR laptop[/], open a terminal and run:\n"
        f"     [bold cyan]{info.tunnel}[/]\n"
        "     [dim](keeps running; an idle terminal is correct)[/]\n\n"
        "  [bold]2.[/] Then open in your browser:\n"
        f"     [bold green]{info.url}[/]\n"
    )
    return Panel(body, title="[bold] Connect to your JupyterLab [/bold]",
                 border_style="green")


def marker_path(folder: str) -> Path:
    return Path(folder) / ".sd-ready"


def wait_ready(folder: str, is_alive: Callable[[], bool],
               timeout: float = 90.0, poll: float = 2.0,
               should_stop: Callable[[], bool] | None = None) -> str:
    """Wait for the job's readiness marker.

    "ready"     — marker appeared
    "gone"      — is_alive() went False first (job failed/cancelled/finished)
    "timeout"   — still alive but no marker within timeout
    "cancelled" — should_stop() returned True (caller asked to bail early)

    *should_stop*, when given, is called once per iteration and stands in for
    the sleep — it must itself block for roughly *poll* seconds and return
    whether the wait should end early. This is how a caller makes the wait
    interruptible (e.g. by a keypress) without this module knowing anything
    about terminals: with should_stop=None the old plain time.sleep(poll)
    behaviour is unchanged, so existing callers and tests are unaffected.
    """
    deadline = time.monotonic() + timeout
    mp = marker_path(folder)
    while True:
        if mp.exists():
            return "ready"
        if not is_alive():
            return "gone"
        if time.monotonic() >= deadline:
            return "timeout"
        if should_stop is not None:
            if should_stop():
                return "cancelled"
        else:
            time.sleep(poll)
