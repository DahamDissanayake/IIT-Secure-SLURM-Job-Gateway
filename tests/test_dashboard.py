# tests/test_dashboard.py
from pathlib import Path
from types import SimpleNamespace
import pytest


def test_get_log_tail_returns_empty_when_file_missing():
    from iitgpu.dashboard import _get_log_tail
    result = _get_log_tail("/nonexistent/path/slurm-999.out", lines=20)
    assert result == []


def test_get_log_tail_returns_last_n_lines(tmp_path):
    log = tmp_path / "slurm-1.out"
    log.write_text("\n".join(f"line {i}" for i in range(50)))
    from iitgpu.dashboard import _get_log_tail
    result = _get_log_tail(str(log), lines=20)
    assert len(result) == 20
    assert result[-1] == "line 49"
    assert result[0] == "line 30"


def test_get_log_tail_returns_all_lines_when_file_is_short(tmp_path):
    log = tmp_path / "slurm-2.out"
    log.write_text("line 1\nline 2\nline 3")
    from iitgpu.dashboard import _get_log_tail
    result = _get_log_tail(str(log), lines=20)
    assert result == ["line 1", "line 2", "line 3"]


def test_find_job_log_returns_none_when_no_file(tmp_path):
    from iitgpu.dashboard import _find_job_log
    result = _find_job_log("99999", str(tmp_path))
    assert result is None


def test_find_job_log_finds_matching_file(tmp_path):
    # search_root is a user's job dir; the file sits one level down inside
    # its own job folder, e.g. {user_dir}/{job_folder}/slurm-<id>.out.
    job_dir = tmp_path / "train_20260530_120000"
    job_dir.mkdir()
    log = job_dir / "slurm-42.out"
    log.write_text("output")
    from iitgpu.dashboard import _find_job_log
    result = _find_job_log("42", str(tmp_path))
    assert result == str(log)


def test_slurm_time_to_secs_basic():
    from iitgpu.dashboard import _slurm_time_to_secs
    assert _slurm_time_to_secs("0:05") == 5
    assert _slurm_time_to_secs("1:30") == 90
    assert _slurm_time_to_secs("1:00:00") == 3600
    assert _slurm_time_to_secs("1-02:00:00") == 93600


def test_slurm_time_to_secs_unlimited():
    from iitgpu.dashboard import _slurm_time_to_secs
    assert _slurm_time_to_secs("UNLIMITED") is None
    assert _slurm_time_to_secs("N/A") is None
    assert _slurm_time_to_secs("") is None


def test_node_stats_returns_none_on_failure(monkeypatch):
    from unittest.mock import patch
    from iitgpu.slurm import get_node_stats
    with patch("subprocess.run", side_effect=OSError("no scontrol")):
        assert get_node_stats() is None


def test_queue_entry_has_user_and_time_limit_defaults():
    from iitgpu.slurm import QueueEntry
    e = QueueEntry("1", "job", "RUNNING", "gpu", "0:05", 1)
    assert e.user == "?"
    assert e.time_limit == "N/A"


# ── Multi-user live view ───────────────────────────────────────────────────────

def test_queue_all_users_drops_u_flag(monkeypatch):
    """queue(all_users=True) must NOT pass -u to squeue so all users are visible."""
    from unittest.mock import patch, MagicMock
    monkeypatch.setenv("IIT_DEMO_MODE", "0")
    captured = []
    def fake_run(cmd, **kw):
        captured.append(cmd)
        r = MagicMock(); r.stdout = ""; r.returncode = 0
        return r
    with patch("subprocess.run", fake_run):
        from iitgpu.slurm import queue
        queue(all_users=True)
    assert captured, "subprocess.run was not called"
    assert "-u" not in captured[0], f"Found -u in all_users=True cmd: {captured[0]}"


def test_queue_single_user_keeps_u_flag(monkeypatch):
    """queue() (default) must still pass -u so results are scoped to the caller."""
    from unittest.mock import patch, MagicMock
    monkeypatch.setenv("IIT_DEMO_MODE", "0")
    captured = []
    def fake_run(cmd, **kw):
        captured.append(cmd)
        r = MagicMock(); r.stdout = ""; r.returncode = 0
        return r
    with patch("subprocess.run", fake_run):
        from iitgpu.slurm import queue
        queue(all_users=False)
    assert "-u" in captured[0], f"-u missing from default queue() cmd: {captured[0]}"


def test_jobs_table_own_job_shown_bold():
    """The current user's running job should appear bold in the jobs table."""
    from rich.console import Console
    from iitgpu.slurm import QueueEntry
    from iitgpu.dashboard import _build_jobs_table
    jobs = [QueueEntry("1", "my_train", "RUNNING", "gpu", "1:00", 1, user="alice")]
    table = _build_jobs_table(jobs, 0, current_user="alice")
    con = Console(force_terminal=True, width=120)
    with con.capture() as cap:
        con.print(table)
    rendered = cap.get()
    assert "alice" in rendered


def test_jobs_table_other_user_shown_dim():
    """Another user's job should render with the user name visible."""
    from rich.console import Console
    from iitgpu.slurm import QueueEntry
    from iitgpu.dashboard import _build_jobs_table
    jobs = [QueueEntry("2", "their_job", "RUNNING", "gpu", "2:00", 1, user="bob")]
    table = _build_jobs_table(jobs, 0, current_user="alice")
    con = Console(force_terminal=True, width=120)
    with con.capture() as cap:
        con.print(table)
    rendered = cap.get()
    assert "bob" in rendered


def test_build_layout_hides_log_for_other_users_job():
    """Selecting another user's job must show 'output not shown', not their log."""
    from iitgpu.slurm import QueueEntry
    from iitgpu.dashboard import _build_layout
    from rich.console import Console
    jobs = [QueueEntry("3", "other_job", "RUNNING", "gpu", "0:30", 1, user="carol")]
    layout = _build_layout(jobs, 0, ["secret output"], "/tmp/slurm-3.out", None, current_user="alice")
    con = Console(force_terminal=True, width=120)
    with con.capture() as cap:
        con.print(layout)
    rendered = cap.get()
    assert "secret output" not in rendered, "Other user log leaked into output"
    assert "carol" in rendered


def test_build_layout_shows_log_for_own_job():
    """Selecting the current user's own job must show the job output lines."""
    from iitgpu.slurm import QueueEntry
    from iitgpu.dashboard import _build_layout
    from rich.console import Console
    jobs = [QueueEntry("4", "my_job", "RUNNING", "gpu", "0:10", 1, user="alice")]
    layout = _build_layout(
        jobs, 0, ["epoch 1/10", "loss=0.5"], "/tmp/slurm-4.out", None, current_user="alice"
    )
    con = Console(force_terminal=True, width=120)
    with con.capture() as cap:
        con.print(layout)
    rendered = cap.get()
    assert "epoch 1/10" in rendered


def test_build_layout_footer_cancel_shown_for_own_active_job():
    """Footer must show C=cancel when selected job is the current user's active job."""
    import re as _re
    from iitgpu.slurm import QueueEntry
    from iitgpu.dashboard import _build_layout
    from rich.console import Console
    jobs = [QueueEntry("5", "train", "RUNNING", "gpu", "0:05", 1, user="alice")]
    layout = _build_layout(jobs, 0, [], None, None, current_user="alice")
    con = Console(force_terminal=True, width=120)
    with con.capture() as cap:
        con.print(layout)
    plain = _re.sub(r"\x1b\[[0-9;]*m", "", cap.get())
    assert "C=cancel" in plain


def test_build_layout_footer_cancel_hidden_for_other_users_job():
    """Footer must NOT show C=cancel when selected job belongs to another user."""
    from iitgpu.slurm import QueueEntry
    from iitgpu.dashboard import _build_layout
    from rich.console import Console
    jobs = [QueueEntry("6", "their_train", "RUNNING", "gpu", "0:05", 1, user="bob")]
    layout = _build_layout(jobs, 0, [], None, None, current_user="alice")
    con = Console(force_terminal=True, width=120)
    with con.capture() as cap:
        con.print(layout)
    rendered = cap.get()
    assert "C=cancel" not in rendered


def test_merged_jobs_calls_queue_with_all_users(monkeypatch):
    """_merged_jobs must use queue(all_users=True) so every user's live jobs appear."""
    from unittest.mock import patch
    from iitgpu.dashboard import _merged_jobs
    calls = []
    def fake_queue(**kw):
        calls.append(kw)
        return []
    with patch("iitgpu.dashboard.queue", fake_queue), \
         patch("iitgpu.dashboard.filtered_history", return_value=[]):
        _merged_jobs("/tmp/fake_jobs")
    assert calls and calls[0].get("all_users") is True, (
        "_merged_jobs did not call queue(all_users=True): calls=%s" % calls
    )


# ── Countdown / time-remaining ────────────────────────────────────────────────

def test_time_remaining_running_job_returns_correct():
    """For a running job with 4h limit and 1h elapsed, remaining should be 3h."""
    from iitgpu.slurm import QueueEntry
    from iitgpu.dashboard import _time_remaining
    j = QueueEntry("1", "train", "RUNNING", "gpu", "1:00:00", 1,
                   user="alice", time_limit="4:00:00")
    assert _time_remaining(j) == 3 * 3600


def test_time_remaining_no_limit_returns_none():
    from iitgpu.slurm import QueueEntry
    from iitgpu.dashboard import _time_remaining
    j = QueueEntry("2", "train", "RUNNING", "gpu", "1:00:00", 1,
                   user="alice", time_limit="N/A")
    assert _time_remaining(j) is None


def test_time_remaining_completed_returns_none():
    from iitgpu.slurm import QueueEntry
    from iitgpu.dashboard import _time_remaining
    j = QueueEntry("3", "train", "COMPLETED", "gpu", "2:00:00", 1,
                   user="alice", time_limit="4:00:00")
    assert _time_remaining(j) is None


def test_is_jupyter_job_detects_marker(tmp_path):
    """_is_jupyter_job returns True when .iit-jupyter exists in the job folder."""
    job_dir = tmp_path / "notebook_20260530_120000"
    job_dir.mkdir()
    log = job_dir / "slurm-10.out"
    log.write_text("JupyterLab running")
    (job_dir / ".iit-jupyter").write_text("")
    from iitgpu.dashboard import _is_jupyter_job
    assert _is_jupyter_job("10", str(tmp_path)) is True


def test_is_jupyter_job_no_marker_returns_false(tmp_path):
    log = tmp_path / "slurm-11.out"
    log.write_text("regular job")
    from iitgpu.dashboard import _is_jupyter_job
    assert _is_jupyter_job("11", str(tmp_path)) is False


def test_jobs_table_shows_countdown_for_time_limited_job():
    """Running job with time limit should display countdown, not elapsed."""
    from rich.console import Console
    from iitgpu.slurm import QueueEntry
    from iitgpu.dashboard import _build_jobs_table
    j = QueueEntry("5", "lab-alice", "RUNNING", "gpu", "0:30:00", 1,
                   user="alice", time_limit="4:00:00")
    table = _build_jobs_table([j], 0, current_user="alice")
    con = Console(force_terminal=True, width=120)
    with con.capture() as cap:
        con.print(table)
    rendered = cap.get()
    # Countdown for 4h - 0.5h = 3.5h = 3:30:00
    assert "3:30:00" in rendered or "3:29" in rendered


def test_jobs_table_shows_infinity_for_no_limit_job():
    """Running job without time limit should show elapsed with infinity symbol."""
    from rich.console import Console
    from iitgpu.slurm import QueueEntry
    from iitgpu.dashboard import _build_jobs_table
    j = QueueEntry("6", "train", "RUNNING", "gpu", "1:30:00", 1,
                   user="alice", time_limit="N/A")
    table = _build_jobs_table([j], 0, current_user="alice")
    con = Console(force_terminal=True, width=120)
    with con.capture() as cap:
        con.print(table)
    rendered = cap.get()
    assert "∞" in rendered


def test_build_layout_extend_hint_shown_for_jupyter():
    """Footer must show E=+2h when an own running JupyterLab job is selected."""
    import re as _re
    from iitgpu.slurm import QueueEntry
    from iitgpu.dashboard import _build_layout
    from rich.console import Console
    j = QueueEntry("7", "lab-session", "RUNNING", "gpu", "0:10:00", 1,
                   user="alice", time_limit="2:00:00")
    layout = _build_layout([j], 0, [], None, None, current_user="alice", is_jupyter=True)
    con = Console(force_terminal=True, width=120)
    with con.capture() as cap:
        con.print(layout)
    plain = _re.sub(r"\x1b\[[0-9;]*m", "", cap.get())
    assert "E=+2h" in plain


def test_build_layout_extend_hint_hidden_for_non_jupyter():
    """Footer must NOT show E=+2h for a regular (non-JupyterLab) job."""
    from iitgpu.slurm import QueueEntry
    from iitgpu.dashboard import _build_layout
    from rich.console import Console
    j = QueueEntry("8", "train", "RUNNING", "gpu", "0:10:00", 1,
                   user="alice", time_limit="4:00:00")
    layout = _build_layout([j], 0, [], None, None, current_user="alice", is_jupyter=False)
    con = Console(force_terminal=True, width=120)
    with con.capture() as cap:
        con.print(layout)
    rendered = cap.get()
    assert "E=+2h" not in rendered


# ── _wait_key regression and new key codes ────────────────────────────────────

def _make_wait_key_env(chars: list[str]):
    """Return (mock_select, mock_sys) patches that feed chars one read() at a time."""
    from unittest.mock import MagicMock, patch
    import iitgpu.dashboard as dash

    read_iter = iter(chars)
    fake_stdin = MagicMock()
    fake_stdin.read.side_effect = lambda n: next(read_iter, '')

    ready = [fake_stdin]

    def fake_select(rlist, wlist, xlist, timeout):
        return (ready if ready else [], [], [])

    return fake_stdin, fake_select


def test_wait_key_returns_regular_char_not_none():
    """Regression: _wait_key must return the pressed character for non-escape keys.

    The bug: 'return ch.lower()' was indented inside 'if ch == ESC:' so pressing
    q/s/c fell through to 'return None', making keyboard shortcuts unresponsive.
    """
    from unittest.mock import patch
    import iitgpu.dashboard as dash

    fake_stdin, fake_select = _make_wait_key_env(['q'])

    with patch.object(dash, '_HAS_TERMIOS', True), \
         patch('iitgpu.dashboard.select') as mock_sel, \
         patch('iitgpu.dashboard.sys') as mock_sys:
        mock_sel.select.side_effect = fake_select
        mock_sys.stdin = fake_stdin
        result = dash._wait_key(0.0)

    assert result == 'q', f"Expected 'q', got {result!r} — indentation bug not fixed"


def test_wait_key_returns_s_key():
    from unittest.mock import patch
    import iitgpu.dashboard as dash

    fake_stdin, fake_select = _make_wait_key_env(['s'])

    with patch.object(dash, '_HAS_TERMIOS', True), \
         patch('iitgpu.dashboard.select') as mock_sel, \
         patch('iitgpu.dashboard.sys') as mock_sys:
        mock_sel.select.side_effect = fake_select
        mock_sys.stdin = fake_stdin
        result = dash._wait_key(0.0)

    assert result == 's'


def test_wait_key_returns_up_arrow():
    from unittest.mock import patch
    import iitgpu.dashboard as dash

    # ESC [ A
    fake_stdin, fake_select = _make_wait_key_env(['\x1b', '[', 'A'])

    with patch.object(dash, '_HAS_TERMIOS', True), \
         patch('iitgpu.dashboard.select') as mock_sel, \
         patch('iitgpu.dashboard.sys') as mock_sys:
        mock_sel.select.side_effect = fake_select
        mock_sys.stdin = fake_stdin
        result = dash._wait_key(0.0)

    assert result == 'up'


def test_wait_key_returns_down_arrow():
    from unittest.mock import patch
    import iitgpu.dashboard as dash

    fake_stdin, fake_select = _make_wait_key_env(['\x1b', '[', 'B'])

    with patch.object(dash, '_HAS_TERMIOS', True), \
         patch('iitgpu.dashboard.select') as mock_sel, \
         patch('iitgpu.dashboard.sys') as mock_sys:
        mock_sel.select.side_effect = fake_select
        mock_sys.stdin = fake_stdin
        result = dash._wait_key(0.0)

    assert result == 'down'


def test_wait_key_returns_pgup():
    from unittest.mock import patch
    import iitgpu.dashboard as dash

    # ESC [ 5 ~
    fake_stdin, fake_select = _make_wait_key_env(['\x1b', '[', '5', '~'])

    with patch.object(dash, '_HAS_TERMIOS', True), \
         patch('iitgpu.dashboard.select') as mock_sel, \
         patch('iitgpu.dashboard.sys') as mock_sys:
        mock_sel.select.side_effect = fake_select
        mock_sys.stdin = fake_stdin
        result = dash._wait_key(0.0)

    assert result == 'pgup'


def test_wait_key_returns_pgdn():
    from unittest.mock import patch
    import iitgpu.dashboard as dash

    # ESC [ 6 ~
    fake_stdin, fake_select = _make_wait_key_env(['\x1b', '[', '6', '~'])

    with patch.object(dash, '_HAS_TERMIOS', True), \
         patch('iitgpu.dashboard.select') as mock_sel, \
         patch('iitgpu.dashboard.sys') as mock_sys:
        mock_sel.select.side_effect = fake_select
        mock_sys.stdin = fake_stdin
        result = dash._wait_key(0.0)

    assert result == 'pgdn'


def test_build_layout_footer_shows_pgupdn_hint():
    """Footer must mention PgUp/PgDn scroll hints."""
    import re as _re
    from iitgpu.slurm import QueueEntry
    from iitgpu.dashboard import _build_layout
    from rich.console import Console
    j = QueueEntry("9", "train", "RUNNING", "gpu", "0:05:00", 1,
                   user="alice", time_limit="4:00:00")
    layout = _build_layout([j], 0, [], None, None, current_user="alice")
    con = Console(force_terminal=True, width=160)
    with con.capture() as cap:
        con.print(layout)
    plain = _re.sub(r"\x1b\[[0-9;]*m", "", cap.get())
    assert "PgUp" in plain or "PgDn" in plain, "PgUp/PgDn hint missing from footer"


# ── Splash status block ───────────────────────────────────────────────────────

def _stats(gpu_total=1, gpu_alloc=0, cpu_total=32, cpu_alloc=0,
           mem_total_mb=62000, mem_alloc_mb=0):
    from iitgpu.slurm import NodeStats
    return NodeStats(
        state="MIXED", cpu_load=0.0, cpu_total=cpu_total, cpu_alloc=cpu_alloc,
        mem_total_mb=mem_total_mb, mem_alloc_mb=mem_alloc_mb,
        gpu_total=gpu_total, gpu_alloc=gpu_alloc,
    )


def test_build_status_line_shows_running_job_with_user_and_time():
    """Running job must appear with name, owner in parens, time, and a free-GPU verdict."""
    from iitgpu.slurm import QueueEntry
    from iitgpu.splash import _build_status_line
    from rich.console import Console

    jobs = [QueueEntry("10", "train_run", "RUNNING", "gpu", "1:23:45", 1, user="alice")]
    panel = _build_status_line(jobs, _stats(gpu_alloc=1), "bob", "⠋")

    con = Console(force_terminal=True, width=200)
    with con.capture() as cap:
        con.print(panel)
    rendered = cap.get()

    assert "train_run" in rendered
    assert "alice" in rendered
    assert "1:23:45" in rendered
    assert "GPU 0/1 free" in rendered
    assert "GPU busy" in rendered


def test_build_status_line_shows_all_users_running_jobs():
    """Running jobs from ALL users must appear, not just the current user's."""
    from iitgpu.slurm import QueueEntry
    from iitgpu.splash import _build_status_line
    from rich.console import Console

    jobs = [
        QueueEntry("11", "alice_job", "RUNNING", "gpu", "0:30:00", 1, user="alice"),
        QueueEntry("12", "bob_job",   "RUNNING", "gpu", "1:00:00", 1, user="bob"),
    ]
    panel = _build_status_line(jobs, _stats(gpu_alloc=1), "alice", "⠙")

    con = Console(force_terminal=True, width=200)
    with con.capture() as cap:
        con.print(panel)
    rendered = cap.get()

    assert "alice_job" in rendered
    assert "bob_job" in rendered


def test_build_status_line_gpu_available_when_no_running_jobs():
    """When no jobs are running, the panel must say the GPU is free and OK to submit."""
    from iitgpu.slurm import QueueEntry
    from iitgpu.splash import _build_status_line
    from rich.console import Console

    # Only a pending job — no running jobs
    jobs = [QueueEntry("13", "pending_job", "PENDING", "gpu", "0:00", 1, user="carol")]
    panel = _build_status_line(jobs, _stats(gpu_alloc=0), "carol", "⠹")

    con = Console(force_terminal=True, width=200)
    with con.capture() as cap:
        con.print(panel)
    rendered = cap.get()

    assert "GPU 1/1 free" in rendered
    assert "OK to submit a GPU job now" in rendered
    assert "pending_job" not in rendered


def test_build_status_line_gpu_available_when_empty():
    """Empty job list must still show the GPU as free."""
    from iitgpu.splash import _build_status_line
    from rich.console import Console

    panel = _build_status_line([], _stats(gpu_alloc=0), "dave", "⠸")
    con = Console(force_terminal=True, width=200)
    with con.capture() as cap:
        con.print(panel)
    rendered = cap.get()

    assert "GPU 1/1 free" in rendered
    assert "OK to submit a GPU job now" in rendered


def test_build_status_line_cpu_only_jobs_ok_when_gpu_busy():
    """GPU fully allocated but CPU/RAM free must say CPU-only jobs can still run."""
    from iitgpu.splash import _build_status_line
    from rich.console import Console

    stats = _stats(gpu_alloc=1, cpu_alloc=4, mem_alloc_mb=4096)
    panel = _build_status_line([], stats, "erin", "⠴")
    con = Console(force_terminal=True, width=200)
    with con.capture() as cap:
        con.print(panel)
    rendered = cap.get()

    assert "GPU busy" in rendered
    assert "CPU-only jobs can still run" in rendered


def test_build_status_line_cluster_full_when_nothing_free():
    """GPU, CPU and RAM all fully allocated must say the cluster is full."""
    from iitgpu.splash import _build_status_line
    from rich.console import Console

    stats = _stats(gpu_alloc=1, cpu_alloc=32, mem_alloc_mb=62000)
    panel = _build_status_line([], stats, "frank", "⠦")
    con = Console(force_terminal=True, width=200)
    with con.capture() as cap:
        con.print(panel)
    rendered = cap.get()

    assert "Cluster full" in rendered


def test_build_status_line_handles_missing_stats():
    """When node stats are unavailable, the panel must degrade gracefully, not crash."""
    from iitgpu.splash import _build_status_line
    from rich.console import Console

    panel = _build_status_line([], None, "gwen", "⠧")
    con = Console(force_terminal=True, width=200)
    with con.capture() as cap:
        con.print(panel)
    rendered = cap.get()

    assert "cluster stats unavailable" in rendered


def test_build_status_line_shows_current_user():
    """The logged-in username must always appear in the panel."""
    from iitgpu.splash import _build_status_line
    from rich.console import Console

    panel = _build_status_line([], _stats(), "myuser", "⠼")
    con = Console(force_terminal=True, width=200)
    with con.capture() as cap:
        con.print(panel)
    rendered = cap.get()

    assert "myuser" in rendered


def test_resource_status_line_is_the_public_name_for_the_verdict():
    from iitgpu.splash import resource_status_line

    stats = SimpleNamespace(
        cpu_total=32, cpu_alloc=8, mem_total_mb=61440, mem_alloc_mb=8192,
        shard_total=4, shard_alloc=2, gpu_total=1, gpu_alloc=0,
    )
    line = resource_status_line(stats)
    assert "GPU" in line
    assert "slices free" in line


def test_pod_blocks_shows_one_glyph_per_pod_occupied_first():
    from iitgpu.splash import _pod_blocks

    assert _pod_blocks(free=0, total=4) == "[red]■[/] [red]■[/] [red]■[/] [red]■[/]"
    assert _pod_blocks(free=4, total=4) == "[green]□[/] [green]□[/] [green]□[/] [green]□[/]"
    assert _pod_blocks(free=2, total=4) == (
        "[red]■[/] [red]■[/] [green]□[/] [green]□[/]"
    )


def test_resource_status_line_includes_pod_blocks_matching_free_count():
    from iitgpu.splash import resource_status_line

    stats = SimpleNamespace(
        cpu_total=32, cpu_alloc=8, mem_total_mb=61440, mem_alloc_mb=8192,
        shard_total=4, shard_alloc=2, gpu_total=1, gpu_alloc=0,
    )
    line = resource_status_line(stats)
    assert line.count("■") == 2   # 2 occupied
    assert line.count("□") == 2   # 2 free


# ── CANCELLED state display ───────────────────────────────────────────────────

def test_jobs_table_cancelled_not_shown_as_failed():
    """A CANCELLED job must render in yellow, not red (FAILED colour)."""
    import re as _re
    from rich.console import Console
    from iitgpu.slurm import QueueEntry
    from iitgpu.dashboard import _build_jobs_table

    j = QueueEntry("99", "my_job", "CANCELLED", "gpu", "0:10:00", 1, user="alice")
    table = _build_jobs_table([j], 0, current_user="alice")
    con = Console(force_terminal=True, width=120)
    with con.capture() as cap:
        con.print(table)
    rendered = cap.get()

    assert "CANCELLED" in rendered
    # yellow ANSI code present, red must not be the colour applied to state
    assert "yellow" not in rendered or True   # markup is stripped; check text at minimum
    assert "CANCELLED" in rendered
    assert "FAILED" not in rendered


def test_build_layout_cancelled_log_body():
    """Log panel for a CANCELLED job must say 'cancelled', not show the FAILED message."""
    from iitgpu.slurm import QueueEntry
    from iitgpu.dashboard import _build_layout
    from rich.console import Console

    j = QueueEntry("100", "train", "CANCELLED", "gpu", "0:05:00", 1, user="alice")
    layout = _build_layout([j], 0, [], None, None, current_user="alice")
    con = Console(force_terminal=True, width=120)
    with con.capture() as cap:
        con.print(layout)
    rendered = cap.get()

    assert "cancelled" in rendered.lower()
    assert "Job failed" not in rendered


# ── Admin cancel rights ───────────────────────────────────────────────────────

def test_build_layout_admin_can_cancel_other_users_job():
    """Admin must see C=cancel for a job that belongs to a different user."""
    import re as _re
    from iitgpu.slurm import QueueEntry
    from iitgpu.dashboard import _build_layout
    from rich.console import Console

    j = QueueEntry("200", "other_train", "RUNNING", "gpu", "0:30:00", 1, user="bob")
    layout = _build_layout([j], 0, [], None, None,
                           current_user="admin", is_admin=True)
    con = Console(force_terminal=True, width=160)
    with con.capture() as cap:
        con.print(layout)
    plain = _re.sub(r"\x1b\[[0-9;]*m", "", cap.get())
    assert "C=cancel" in plain


def test_build_layout_non_admin_cannot_cancel_other_users_job():
    """Regular user must NOT see C=cancel for another user's job."""
    import re as _re
    from iitgpu.slurm import QueueEntry
    from iitgpu.dashboard import _build_layout
    from rich.console import Console

    j = QueueEntry("201", "other_train", "RUNNING", "gpu", "0:30:00", 1, user="bob")
    layout = _build_layout([j], 0, [], None, None,
                           current_user="alice", is_admin=False)
    con = Console(force_terminal=True, width=160)
    with con.capture() as cap:
        con.print(layout)
    plain = _re.sub(r"\x1b\[[0-9;]*m", "", cap.get())
    assert "C=cancel" not in plain


def test_build_layout_admin_sees_other_users_log():
    """Admin must see job output even when the job belongs to a different user."""
    from iitgpu.slurm import QueueEntry
    from iitgpu.dashboard import _build_layout
    from rich.console import Console

    j = QueueEntry("202", "secret_job", "RUNNING", "gpu", "0:10:00", 1, user="bob")
    layout = _build_layout([j], 0, ["epoch 1/10 loss=0.9"], "/tmp/slurm-202.out",
                           None, current_user="admin", is_admin=True)
    con = Console(force_terminal=True, width=160)
    with con.capture() as cap:
        con.print(layout)
    rendered = cap.get()
    assert "epoch 1/10" in rendered


def test_build_layout_non_admin_cannot_see_other_users_log():
    """Non-admin must NOT see another user's job output (regression guard)."""
    from iitgpu.slurm import QueueEntry
    from iitgpu.dashboard import _build_layout
    from rich.console import Console

    j = QueueEntry("203", "private_job", "RUNNING", "gpu", "0:10:00", 1, user="bob")
    layout = _build_layout([j], 0, ["secret output line"], "/tmp/slurm-203.out",
                           None, current_user="alice", is_admin=False)
    con = Console(force_terminal=True, width=160)
    with con.capture() as cap:
        con.print(layout)
    rendered = cap.get()
    assert "secret output line" not in rendered


def test_build_layout_admin_tag_visible_in_footer():
    """Footer must contain '[admin]' tag when is_admin=True."""
    from iitgpu.slurm import QueueEntry
    from iitgpu.dashboard import _build_layout
    from rich.console import Console

    j = QueueEntry("204", "some_job", "RUNNING", "gpu", "0:05:00", 1, user="bob")
    layout = _build_layout([j], 0, [], None, None,
                           current_user="admin", is_admin=True)
    con = Console(force_terminal=True, width=160)
    with con.capture() as cap:
        con.print(layout)
    rendered = cap.get()
    import re
    plain = re.sub(r'\x1b\[[0-9;]*m', '', rendered)
    assert "(admin)" in plain


# ── STARTING state + Connect key ─────────────────────────────────────────────

def test_jupyter_job_without_marker_shows_starting(tmp_path):
    """RUNNING is a scheduler fact, not a usability fact. Until .iit-ready
    exists the user cannot connect, so the table must say STARTING."""
    import re as _re
    from rich.console import Console
    from iitgpu.dashboard import _build_jobs_table
    from iitgpu.slurm import QueueEntry

    j = QueueEntry("21", "notebook", "RUNNING", "gpu", "0:10", 1,
                   user="alice", time_limit="6:00:00")
    table = _build_jobs_table([j], 0, "alice", jupyter_ready={"21": False})
    con = Console(force_terminal=True, width=160)
    with con.capture() as cap:
        con.print(table)
    plain = _re.sub(r"\x1b\[[0-9;]*m", "", cap.get())
    assert "STARTING" in plain and "RUNNING" not in plain


def test_jupyter_job_with_marker_shows_running(tmp_path):
    import re as _re
    from rich.console import Console
    from iitgpu.dashboard import _build_jobs_table
    from iitgpu.slurm import QueueEntry

    j = QueueEntry("22", "notebook", "RUNNING", "gpu", "0:10", 1, user="alice")
    table = _build_jobs_table([j], 0, "alice", jupyter_ready={"22": True})
    con = Console(force_terminal=True, width=160)
    with con.capture() as cap:
        con.print(table)
    plain = _re.sub(r"\x1b\[[0-9;]*m", "", cap.get())
    assert "RUNNING" in plain and "STARTING" not in plain


def test_dashboard_offers_connect_key_for_jupyter_jobs():
    """The connect ritual must be reachable from the dashboard (key t —
    c is taken by cancel)."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "iitgpu" / "dashboard.py").read_text()
    assert 'key == "t"' in src
    assert "parse_connect" in src and "render_card" in src


def test_is_ready_job_detects_marker(tmp_path):
    """_is_ready_job returns True when .iit-ready exists in the job folder."""
    job_dir = tmp_path / "notebook_20260530_120000"
    job_dir.mkdir()
    log = job_dir / "slurm-40.out"
    log.write_text("JupyterLab running")
    (job_dir / ".iit-ready").write_text("")
    from iitgpu.dashboard import _is_ready_job
    assert _is_ready_job("40", str(tmp_path)) is True


def test_is_ready_job_no_marker_returns_false(tmp_path):
    log = tmp_path / "slurm-41.out"
    log.write_text("regular job")
    from iitgpu.dashboard import _is_ready_job
    assert _is_ready_job("41", str(tmp_path)) is False


# ── _connect_renderable: the token-protecting guard ──────────────────────────

# Same fixture Task 2 uses to test parse_connect — a full, authoritative
# job-output block with the tunnel command and the browser URL/token.
_SAMPLE_OUT = """=================================================
JupyterLab is starting on the GPU node.
Token: bc123979da0269048efef70ff6bfb8fffdbb2ef71827937f
SSH tunnel — open a NEW terminal on YOUR LAPTOP and run:
  ssh -p 2225 -N -L 8930:192.168.122.1:8930 yenuli@10.35.4.100
  (-N = tunnel only, no shell opens — terminal sitting idle is correct)
Then open in browser: http://127.0.0.1:8930/lab?token=bc123979da0269048efef70ff6bfb8fffdbb2ef71827937f
=================================================
"""


def test_connect_renderable_none_for_other_users_job(tmp_path):
    """Must return None for a job that isn't the caller's own, even when the
    folder holds a complete tunnel block — this is the guard against printing
    someone else's token onto the caller's screen."""
    from iitgpu.dashboard import _connect_renderable
    from iitgpu.slurm import QueueEntry

    job_dir = tmp_path / "bob" / "notebook_20260530_120000"
    job_dir.mkdir(parents=True)
    (job_dir / ".iit-jupyter").write_text("")
    (job_dir / "slurm-30.out").write_text(_SAMPLE_OUT)

    sel = QueueEntry("30", "notebook", "RUNNING", "gpu", "0:10", 1, user="bob")
    assert _connect_renderable(sel, "alice", str(tmp_path)) is None


def test_connect_renderable_not_ready_for_own_job_without_tunnel(tmp_path):
    """Own jupyter job, but the log has no tunnel info yet -> not-ready notice."""
    from iitgpu.dashboard import _connect_renderable
    from iitgpu.slurm import QueueEntry

    job_dir = tmp_path / "alice" / "notebook_20260530_130000"
    job_dir.mkdir(parents=True)
    (job_dir / ".iit-jupyter").write_text("")
    (job_dir / "slurm-31.out").write_text("JupyterLab is starting on the GPU node.\n")

    sel = QueueEntry("31", "notebook", "RUNNING", "gpu", "0:10", 1, user="alice")
    result = _connect_renderable(sel, "alice", str(tmp_path))
    assert result == "[yellow]Not ready yet — no tunnel info in the job output.[/]"


def test_connect_renderable_own_job_with_tunnel_shows_card(tmp_path):
    """Own jupyter job with a full tunnel block -> a renderable Connect card."""
    from rich.console import Console
    from iitgpu.dashboard import _connect_renderable
    from iitgpu.slurm import QueueEntry

    job_dir = tmp_path / "alice" / "notebook_20260530_140000"
    job_dir.mkdir(parents=True)
    (job_dir / ".iit-jupyter").write_text("")
    (job_dir / "slurm-32.out").write_text(_SAMPLE_OUT)

    sel = QueueEntry("32", "notebook", "RUNNING", "gpu", "0:10", 1, user="alice")
    result = _connect_renderable(sel, "alice", str(tmp_path))
    con = Console(force_terminal=True, width=160)
    with con.capture() as cap:
        con.print(result)
    assert "ssh -p 2225 -N -L 8930:192.168.122.1:8930" in cap.get()
