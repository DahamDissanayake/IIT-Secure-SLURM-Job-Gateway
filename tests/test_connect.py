"""Connect card: parse the job's own output — the only authoritative source."""
from pathlib import Path

from rich.console import Console

from iitgpu.connect import ConnectInfo, marker_path, parse_connect, render_card, wait_ready

SAMPLE_OUT = """=================================================
JupyterLab is starting on the GPU node.
Token: bc123979da0269048efef70ff6bfb8fffdbb2ef71827937f
SSH tunnel — open a NEW terminal on YOUR LAPTOP and run:
  ssh -p 2225 -N -L 8930:192.168.122.1:8930 yenuli@10.35.4.100
  (-N = tunnel only, no shell opens — terminal sitting idle is correct)
Then open in browser: http://127.0.0.1:8930/lab?token=bc123979da0269048efef70ff6bfb8fffdbb2ef71827937f
=================================================
"""


def test_parse_connect_extracts_all_fields():
    info = parse_connect(SAMPLE_OUT)
    assert info is not None
    assert info.port == 8930
    assert info.token.startswith("bc1239")
    assert info.tunnel == "ssh -p 2225 -N -L 8930:192.168.122.1:8930 yenuli@10.35.4.100"
    assert info.url == "http://127.0.0.1:8930/lab?token=" + info.token


def test_parse_connect_none_when_not_started_yet():
    assert parse_connect("slurm queued...\n") is None
    assert parse_connect("") is None


def test_render_card_shows_both_steps():
    info = parse_connect(SAMPLE_OUT)
    con = Console(force_terminal=True, width=100)
    with con.capture() as cap:
        con.print(render_card(info))
    out = cap.get()
    assert "ssh -p 2225 -N -L 8930" in out
    assert "http://127.0.0.1:8930/lab?token=" in out
    assert "YOUR laptop" in out


def test_wait_ready_states(tmp_path):
    # ready: marker exists already
    marker_path(str(tmp_path)).touch()
    assert wait_ready(str(tmp_path), is_alive=lambda: True, timeout=1, poll=0.01) == "ready"
    marker_path(str(tmp_path)).unlink()
    # gone: job left RUNNING before marker appeared
    assert wait_ready(str(tmp_path), is_alive=lambda: False, timeout=1, poll=0.01) == "gone"
    # timeout: alive but never ready
    assert wait_ready(str(tmp_path), is_alive=lambda: True, timeout=0.05, poll=0.01) == "timeout"
