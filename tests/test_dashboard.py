# tests/test_dashboard.py
from pathlib import Path
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
    log = tmp_path / "slurm-42.out"
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
         patch("iitgpu.dashboard.recent_jobs", return_value=[]):
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
    log = tmp_path / "slurm-10.out"
    log.write_text("JupyterLab running")
    (tmp_path / ".iit-jupyter").write_text("")
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

def test_build_status_line_shows_running_job_with_user_and_time():
    """Running job must appear with name, owner in brackets, and time used."""
    from iitgpu.slurm import QueueEntry
    from iitgpu.splash import _build_status_line
    from rich.console import Console

    jobs = [QueueEntry("10", "train_run", "RUNNING", "gpu", "1:23:45", 1, user="alice")]
    panel = _build_status_line(jobs, "bob", "⠋")

    con = Console(force_terminal=True, width=200)
    with con.capture() as cap:
        con.print(panel)
    rendered = cap.get()

    assert "train_run" in rendered
    assert "alice" in rendered
    assert "1:23:45" in rendered


def test_build_status_line_shows_all_users_running_jobs():
    """Running jobs from ALL users must appear, not just the current user's."""
    from iitgpu.slurm import QueueEntry
    from iitgpu.splash import _build_status_line
    from rich.console import Console

    jobs = [
        QueueEntry("11", "alice_job", "RUNNING", "gpu", "0:30:00", 1, user="alice"),
        QueueEntry("12", "bob_job",   "RUNNING", "gpu", "1:00:00", 1, user="bob"),
    ]
    panel = _build_status_line(jobs, "alice", "⠙")

    con = Console(force_terminal=True, width=200)
    with con.capture() as cap:
        con.print(panel)
    rendered = cap.get()

    assert "alice_job" in rendered
    assert "bob_job" in rendered


def test_build_status_line_gpu_available_when_no_running_jobs():
    """When no jobs are running, the panel must say GPU is available."""
    from iitgpu.slurm import QueueEntry
    from iitgpu.splash import _build_status_line
    from rich.console import Console

    # Only a pending job — no running jobs
    jobs = [QueueEntry("13", "pending_job", "PENDING", "gpu", "0:00", 1, user="carol")]
    panel = _build_status_line(jobs, "carol", "⠹")

    con = Console(force_terminal=True, width=200)
    with con.capture() as cap:
        con.print(panel)
    rendered = cap.get()

    assert "GPU is available" in rendered
    assert "pending_job" not in rendered


def test_build_status_line_gpu_available_when_empty():
    """Empty job list must show GPU is available."""
    from iitgpu.splash import _build_status_line
    from rich.console import Console

    panel = _build_status_line([], "dave", "⠸")
    con = Console(force_terminal=True, width=200)
    with con.capture() as cap:
        con.print(panel)
    rendered = cap.get()

    assert "GPU is available" in rendered


def test_build_status_line_shows_current_user():
    """The logged-in username must always appear in the panel."""
    from iitgpu.splash import _build_status_line
    from rich.console import Console

    panel = _build_status_line([], "myuser", "⠼")
    con = Console(force_terminal=True, width=200)
    with con.capture() as cap:
        con.print(panel)
    rendered = cap.get()

    assert "myuser" in rendered
