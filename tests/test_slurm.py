# tests/test_slurm.py
"""Tests for slurm.py — sacct_history, job_history, recent_jobs fallback."""
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest


# ── sacct_history ─────────────────────────────────────────────────────────────

def _make_sacct_output(*rows):
    """Build parsable2 sacct output. Each row: (jobid, name, user, state, elapsed)."""
    lines = []
    for r in rows:
        jid, name, user, state, elapsed = r
        lines.append(f"{jid}|{name}|{user}|{state}|{elapsed}|2026-05-30T10:00:00|2026-05-30T11:00:00|gres/gpu=1")
    return "\n".join(lines)


def test_sacct_history_parses_completed_jobs():
    from slurmdeck.slurm import sacct_history, QueueEntry

    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = _make_sacct_output(
        ("100", "train", "daham", "COMPLETED", "01:00:00"),
        ("101", "infer", "daham", "FAILED", "00:10:00"),
    )
    with patch("subprocess.run", return_value=mock):
        rows = sacct_history(limit=10)
    assert len(rows) == 2
    assert rows[0].job_id == "101"   # newest-first (reversed)
    assert rows[1].job_id == "100"
    assert rows[0].state == "FAILED"
    assert rows[1].state == "COMPLETED"


def test_sacct_history_skips_step_lines():
    from slurmdeck.slurm import sacct_history

    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = (
        "200|train|daham|COMPLETED|02:00:00|2026-05-30T10:00:00|2026-05-30T12:00:00|gres/gpu=1\n"
        "200.batch|batch|daham|COMPLETED|02:00:00|2026-05-30T10:00:00|2026-05-30T12:00:00|\n"
    )
    with patch("subprocess.run", return_value=mock):
        rows = sacct_history()
    assert len(rows) == 1
    assert rows[0].job_id == "200"


def test_sacct_history_returns_empty_on_failure():
    from slurmdeck.slurm import sacct_history

    mock = MagicMock()
    mock.returncode = 1
    mock.stdout = ""
    with patch("subprocess.run", return_value=mock):
        assert sacct_history() == []


def test_sacct_history_returns_empty_on_oserror():
    from slurmdeck.slurm import sacct_history

    with patch("subprocess.run", side_effect=OSError("no sacct")):
        assert sacct_history() == []


def test_sacct_history_respects_limit():
    from slurmdeck.slurm import sacct_history

    rows_data = [
        (str(i), f"job{i}", "daham", "COMPLETED", "00:01:00")
        for i in range(50)
    ]
    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = _make_sacct_output(*rows_data)
    with patch("subprocess.run", return_value=mock):
        result = sacct_history(limit=5)
    assert len(result) == 5


def test_sacct_history_strips_state_suffix():
    """State 'CANCELLED by 1234' should be stripped to 'CANCELLED'."""
    from slurmdeck.slurm import sacct_history

    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = "300|job|daham|CANCELLED by 1234|00:05:00|2026-05-30T10:00:00|2026-05-30T10:05:00|\n"
    with patch("subprocess.run", return_value=mock):
        rows = sacct_history()
    assert rows[0].state == "CANCELLED"


# ── job_history ───────────────────────────────────────────────────────────────

def test_job_history_uses_sacct_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("SACCT_ENABLED", "1")
    monkeypatch.setenv("NFS_ROOT", str(tmp_path))

    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = _make_sacct_output(("400", "sacct_job", "daham", "COMPLETED", "00:30:00"))

    with patch("subprocess.run", return_value=mock):
        from slurmdeck.slurm import job_history
        rows = job_history(str(tmp_path))
    assert any(r.job_id == "400" for r in rows)


def test_job_history_falls_back_to_file_scan_when_sacct_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("SACCT_ENABLED", "0")
    monkeypatch.setenv("NFS_ROOT", str(tmp_path))

    # Create a fake job output file. job_history is always called with
    # jobs_dir() (NFS_ROOT/jobs), not NFS_ROOT itself — match that contract.
    jobs_root = tmp_path / "jobs"
    job_dir = jobs_root / "daham" / "train_20260530_120000"
    job_dir.mkdir(parents=True)
    (job_dir / "slurm-500.out").write_text("output\n")

    from slurmdeck.slurm import job_history
    rows = job_history(str(jobs_root))
    assert any(r.job_id == "500" for r in rows)


def test_job_history_falls_back_when_sacct_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("SACCT_ENABLED", "1")
    monkeypatch.setenv("NFS_ROOT", str(tmp_path))

    # sacct returns nothing
    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = ""

    jobs_root = tmp_path / "jobs"
    job_dir = jobs_root / "daham" / "train_20260530_130000"
    job_dir.mkdir(parents=True)
    (job_dir / "slurm-600.out").write_text("output\n")

    with patch("subprocess.run", return_value=mock):
        from slurmdeck.slurm import job_history
        rows = job_history(str(jobs_root))
    assert any(r.job_id == "600" for r in rows)


# ── recent_jobs state detection (file-scan fallback, no sacct) ────────────────
# recent_jobs() has no exit code to go on, so a non-empty .err used to be
# treated as an automatic FAILED. That mislabels interactive-service jobs
# (JupyterLab, TensorBoard) which always log routine INFO/WARNING lines to
# stderr — a cleanly scancel'd or finished notebook job would show FAILED even
# though it never crashed. These lock in the fix: prefer SLURM's own epilogue
# line for the real state, and don't blame routine Jupyter logging as a failure.

def test_recent_jobs_uses_slurm_epilogue_state(tmp_path):
    from slurmdeck.slurm import recent_jobs
    job_dir = tmp_path / "jobs" / "amasha" / "notebook_1"
    job_dir.mkdir(parents=True)
    (job_dir / "slurm-269.out").write_text("stdout\n")
    (job_dir / "slurm-269.err").write_text(
        "[I 2026-07-16 ServerApp] some routine info\n"
        "[2026-07-16T12:47:03.327] error: *** JOB 269 ON iit-MS-7E06 CANCELLED "
        "AT 2026-07-16T12:47:03 DUE to SIGNAL Terminated ***\n"
    )
    rows = recent_jobs(str(tmp_path / "jobs"), limit=5)
    assert rows[0].state == "CANCELLED"


def test_recent_jobs_does_not_fail_notebook_for_routine_stderr(tmp_path):
    from slurmdeck.slurm import recent_jobs
    job_dir = tmp_path / "jobs" / "amasha" / "notebook_2"
    job_dir.mkdir(parents=True)
    (job_dir / ".sd-jupyter").write_text("")
    (job_dir / "slurm-300.out").write_text("stdout\n")
    (job_dir / "slurm-300.err").write_text(
        "[I 2026-07-16 ServerApp] JupyterLab extension loaded from ...\n"
        "[W 2026-07-16 LabApp] Could not determine jupyterlab build status without nodejs\n"
    )
    rows = recent_jobs(str(tmp_path / "jobs"), limit=5)
    assert rows[0].state == "COMPLETED"


def test_recent_jobs_still_flags_traceback_as_failed(tmp_path):
    from slurmdeck.slurm import recent_jobs
    job_dir = tmp_path / "jobs" / "daham" / "train_1"
    job_dir.mkdir(parents=True)
    (job_dir / ".sd-jupyter").write_text("")  # even a jupyter job, a real crash still counts
    (job_dir / "slurm-400.out").write_text("training...\n")
    (job_dir / "slurm-400.err").write_text(
        "Traceback (most recent call last):\n  File \"x.py\", line 1\nValueError: boom\n"
    )
    rows = recent_jobs(str(tmp_path / "jobs"), limit=5)
    assert rows[0].state == "FAILED"


def test_recent_jobs_flags_failed_for_plain_script_stderr(tmp_path):
    """Non-notebook jobs keep the old behavior: any stderr content is FAILED."""
    from slurmdeck.slurm import recent_jobs
    job_dir = tmp_path / "jobs" / "daham" / "train_2"
    job_dir.mkdir(parents=True)
    (job_dir / "slurm-401.out").write_text("training...\n")
    (job_dir / "slurm-401.err").write_text("some warning printed by the script\n")
    rows = recent_jobs(str(tmp_path / "jobs"), limit=5)
    assert rows[0].state == "FAILED"


def test_recent_jobs_completed_when_err_empty():
    from slurmdeck.slurm import recent_jobs
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path
        job_dir = Path(td) / "jobs" / "daham" / "train_3"
        job_dir.mkdir(parents=True)
        (job_dir / "slurm-402.out").write_text("training...\n")
        (job_dir / "slurm-402.err").write_text("")
        rows = recent_jobs(str(Path(td) / "jobs"), limit=5)
        assert rows[0].state == "COMPLETED"


# ── config.SACCT_ENABLED ──────────────────────────────────────────────────────

def test_config_sacct_enabled_explicit_true(monkeypatch):
    monkeypatch.setenv("SACCT_ENABLED", "1")
    import importlib
    import slurmdeck.config as cfg_mod
    importlib.reload(cfg_mod)
    cfg = cfg_mod.load_config()
    assert cfg.sacct_enabled is True


def test_config_sacct_enabled_explicit_false(monkeypatch):
    monkeypatch.setenv("SACCT_ENABLED", "0")
    import importlib
    import slurmdeck.config as cfg_mod
    importlib.reload(cfg_mod)
    cfg = cfg_mod.load_config()
    assert cfg.sacct_enabled is False


def test_config_sacct_auto_detects_via_which(monkeypatch):
    monkeypatch.setenv("SACCT_ENABLED", "auto")
    with patch("shutil.which", return_value="/usr/bin/sacct"):
        import importlib
        import slurmdeck.config as cfg_mod
        importlib.reload(cfg_mod)
        cfg = cfg_mod.load_config()
    assert cfg.sacct_enabled is True


def test_config_sacct_auto_returns_false_when_sacct_missing(monkeypatch):
    monkeypatch.setenv("SACCT_ENABLED", "auto")
    with patch("shutil.which", return_value=None):
        import importlib
        import slurmdeck.config as cfg_mod
        importlib.reload(cfg_mod)
        cfg = cfg_mod.load_config()
    assert cfg.sacct_enabled is False


# ── Regression: sacct CLI must use -S window and NOT --state (drops completed) ──

def test_sacct_history_uses_start_window_not_state_filter():
    """sacct_history must pass -S (start window) and must NOT pass --state=,
    because sacct's --state filter silently drops already-completed jobs."""
    from slurmdeck.slurm import sacct_history
    captured = {}

    def fake_run(cmd, *a, **k):
        captured["cmd"] = cmd
        class R:
            returncode = 0
            stdout = ""
        return R()

    with patch("subprocess.run", side_effect=fake_run):
        sacct_history()

    cmd = captured["cmd"]
    assert "-S" in cmd, "sacct_history must pass an explicit -S start window"
    joined = " ".join(cmd)
    assert "--state=" not in joined, (
        "sacct_history must NOT use --state= (it drops completed jobs); "
        "filter terminal states in Python instead"
    )


def test_sacct_history_filters_running_and_pending_in_python():
    """Rows in RUNNING/PENDING must be excluded (they belong to queue())."""
    from slurmdeck.slurm import sacct_history
    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = (
        "10|done|daham|COMPLETED|00:05:00|s|e|gres/gpu=1\n"
        "11|run|daham|RUNNING|00:01:00|s|e|gres/gpu=1\n"
        "12|wait|daham|PENDING|00:00:00|s|e|\n"
        "13|oom|daham|OUT_OF_MEMORY|00:02:00|s|e|gres/gpu=1\n"
    )
    with patch("subprocess.run", return_value=mock):
        rows = sacct_history()
    states = {r.state for r in rows}
    ids = {r.job_id for r in rows}
    assert "RUNNING" not in states and "PENDING" not in states
    assert ids == {"10", "13"}, f"expected terminal jobs only, got {ids}"


def test_sacct_history_accepts_days_param():
    from slurmdeck.slurm import sacct_history
    captured = {}

    def fake_run(cmd, *a, **k):
        captured["cmd"] = cmd
        class R:
            returncode = 0
            stdout = ""
        return R()

    with patch("subprocess.run", side_effect=fake_run):
        sacct_history(days=7)
    assert "now-7days" in " ".join(captured["cmd"])


# ── extend_job_time ────────────────────────────────────────────────────────────

def test_extend_job_time_calls_scontrol_update(monkeypatch):
    """extend_job_time must call scontrol update with the correct TimeLimit arg."""
    from unittest.mock import patch, MagicMock
    monkeypatch.setenv("DEMO_MODE", "0")
    captured = []
    def fake_run(cmd, **kw):
        captured.append(cmd)
        r = MagicMock(); r.returncode = 0; r.stderr = ""
        return r
    with patch("subprocess.run", fake_run):
        from slurmdeck.slurm import extend_job_time
        ok, msg = extend_job_time("42", extra_hours=2)
    assert ok
    assert any("scontrol" in str(c) for c in captured)
    full_cmd = " ".join(captured[0])
    assert "TimeLimit=+02:00:00" in full_cmd


def test_extend_job_time_returns_false_on_failure(monkeypatch):
    from unittest.mock import patch, MagicMock
    monkeypatch.setenv("DEMO_MODE", "0")
    def fake_run(cmd, **kw):
        r = MagicMock(); r.returncode = 1; r.stderr = "Access denied"
        return r
    with patch("subprocess.run", fake_run):
        from slurmdeck.slurm import extend_job_time
        ok, msg = extend_job_time("99")
    assert not ok
    assert "Access denied" in msg


def test_extend_job_time_demo_mode(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "1")
    import importlib, slurmdeck.slurm as _sm; importlib.reload(_sm)
    from slurmdeck.slurm import extend_job_time
    ok, msg = extend_job_time("7", 3)
    assert ok
    assert "demo" in msg
