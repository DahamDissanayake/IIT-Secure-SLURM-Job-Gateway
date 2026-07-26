# tests/test_manage_job_admin.py
"""'Manage a job' menu must let admins act on any user's job, matching the
live dashboard's existing admin-cancel behaviour. Previously it always
called queue(user=getpass.getuser()) with no admin check anywhere, so an
admin could not even SEE another user's job here, let alone cancel it."""
from unittest.mock import MagicMock
import pytest

from iitgpu import monitor
from iitgpu.slurm import QueueEntry


def _entries():
    return [
        QueueEntry(job_id="10", name="notebook", state="RUNNING",
                   partition="gpu", time_used="0:10", nodes=1, user="dahamadmin"),
        QueueEntry(job_id="11", name="train", state="RUNNING",
                   partition="gpu", time_used="0:05", nodes=1, user="yenuli"),
    ]


def test_manage_job_admin_queries_all_users(monkeypatch):
    monkeypatch.setattr(monitor, "is_admin", lambda cfg=None: True)
    monkeypatch.setattr(monitor.getpass, "getuser", lambda: "dahamadmin")
    calls = {}

    def _mock_queue(**kw):
        calls["kw"] = kw
        return _entries()
    monkeypatch.setattr(monitor, "queue", _mock_queue)
    monkeypatch.setattr(monitor, "select_menu", lambda *a, **k: None)  # bail after listing
    monitor.manage_job()
    assert calls["kw"] == {"all_users": True}


def test_manage_job_non_admin_queries_own_jobs_only(monkeypatch):
    monkeypatch.setattr(monitor, "is_admin", lambda cfg=None: False)
    monkeypatch.setattr(monitor.getpass, "getuser", lambda: "yenuli")
    calls = {}

    def _mock_queue(**kw):
        calls["kw"] = kw
        return _entries()
    monkeypatch.setattr(monitor, "queue", _mock_queue)
    monkeypatch.setattr(monitor, "select_menu", lambda *a, **k: None)
    monitor.manage_job()
    assert calls["kw"] == {"user": "yenuli"}


def test_manage_job_admin_can_cancel_others_job(monkeypatch):
    monkeypatch.setattr(monitor, "is_admin", lambda cfg=None: True)
    monkeypatch.setattr(monitor.getpass, "getuser", lambda: "dahamadmin")
    monkeypatch.setattr(monitor, "queue", lambda **kw: _entries())
    picks = iter(["11  train  [RUNNING]  (yenuli)", "Cancel"])
    monkeypatch.setattr(monitor, "select_menu", lambda *a, **k: next(picks))
    monkeypatch.setattr("questionary.confirm",
                         lambda *a, **k: MagicMock(ask=lambda: True))
    monkeypatch.setattr(monitor.auditclient, "log", lambda *a, **k: None)
    cancelled = {}
    monkeypatch.setattr(monitor, "cancel",
                         lambda jid: (cancelled.setdefault("id", jid), (True, "cancelled"))[1])
    monitor.manage_job()
    assert cancelled["id"] == "11"


def test_manage_job_non_admin_blocked_from_others_job(monkeypatch):
    """Defense in depth: even if a non-admin's entries list ever contained
    someone else's job, the action must still be refused before mutating."""
    monkeypatch.setattr(monitor, "is_admin", lambda cfg=None: False)
    monkeypatch.setattr(monitor.getpass, "getuser", lambda: "yenuli")
    monkeypatch.setattr(monitor, "queue", lambda **kw: _entries())
    monkeypatch.setattr(monitor, "select_menu",
                         lambda *a, **k: "10  notebook  [RUNNING]  (dahamadmin)")
    cancelled = {"n": 0}
    monkeypatch.setattr(monitor, "cancel",
                         lambda jid: cancelled.__setitem__("n", cancelled["n"] + 1))
    monitor.manage_job()
    assert cancelled["n"] == 0
