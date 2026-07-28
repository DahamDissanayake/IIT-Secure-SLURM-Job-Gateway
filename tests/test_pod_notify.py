# tests/test_pod_notify.py
"""Pod-availability notify subscriptions: subscribe/unsubscribe are pure
file-backed CRUD; check_and_notify is the one-time-fire logic."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _stats(shard_total=4, shard_alloc=4):
    return SimpleNamespace(shard_total=shard_total, shard_alloc=shard_alloc)


@pytest.fixture(autouse=True)
def _isolated_nfs_root(tmp_path, monkeypatch):
    monkeypatch.setenv("NFS_ROOT", str(tmp_path))
    import importlib
    import iitgpu.pod_notify as pn
    importlib.reload(pn)
    yield pn


def test_subscribe_then_get_subscription(_isolated_nfs_root):
    pn = _isolated_nfs_root
    pn.subscribe("alice", "alice@iit.lk", 2)
    sub = pn.get_subscription("alice")
    assert sub["username"] == "alice"
    assert sub["email"] == "alice@iit.lk"
    assert sub["wanted_pods"] == 2
    assert "subscribed_at" in sub


def test_get_subscription_none_when_not_subscribed(_isolated_nfs_root):
    assert _isolated_nfs_root.get_subscription("bob") is None


def test_subscribe_again_replaces_previous_threshold(_isolated_nfs_root):
    pn = _isolated_nfs_root
    pn.subscribe("alice", "alice@iit.lk", 1)
    pn.subscribe("alice", "alice@iit.lk", 3)
    with pn._open_locked("r+") as f:
        subs = pn._read(f)
    matching = [s for s in subs if s["username"] == "alice"]
    assert len(matching) == 1
    assert matching[0]["wanted_pods"] == 3


def test_unsubscribe_removes_and_reports_true(_isolated_nfs_root):
    pn = _isolated_nfs_root
    pn.subscribe("alice", "alice@iit.lk", 2)
    assert pn.unsubscribe("alice") is True
    assert pn.get_subscription("alice") is None


def test_unsubscribe_false_when_nothing_to_remove(_isolated_nfs_root):
    assert _isolated_nfs_root.unsubscribe("nobody") is False


def test_subscriptions_are_independent_per_user(_isolated_nfs_root):
    pn = _isolated_nfs_root
    pn.subscribe("alice", "alice@iit.lk", 1)
    pn.subscribe("bob", "bob@iit.lk", 2)
    assert pn.get_subscription("alice")["wanted_pods"] == 1
    assert pn.get_subscription("bob")["wanted_pods"] == 2
    pn.unsubscribe("alice")
    assert pn.get_subscription("alice") is None
    assert pn.get_subscription("bob")["wanted_pods"] == 2


def test_check_and_notify_no_op_when_no_subscriptions(_isolated_nfs_root):
    pn = _isolated_nfs_root
    with patch("iitgpu.mailer.send_pod_available_notice") as send:
        n = pn.check_and_notify(stats=_stats(shard_alloc=0))
    assert n == 0
    send.assert_not_called()


def test_check_and_notify_no_op_when_threshold_not_met(_isolated_nfs_root):
    pn = _isolated_nfs_root
    pn.subscribe("alice", "alice@iit.lk", 3)
    with patch("iitgpu.mailer.send_pod_available_notice") as send:
        # only 1 free, alice wants 3
        n = pn.check_and_notify(stats=_stats(shard_total=4, shard_alloc=3))
    assert n == 0
    send.assert_not_called()
    assert pn.get_subscription("alice") is not None  # still subscribed


def test_check_and_notify_fires_and_removes_when_threshold_met(_isolated_nfs_root):
    pn = _isolated_nfs_root
    pn.subscribe("alice", "alice@iit.lk", 2)
    with patch("iitgpu.mailer.send_pod_available_notice") as send:
        n = pn.check_and_notify(stats=_stats(shard_total=4, shard_alloc=1))  # 3 free
    assert n == 1
    send.assert_called_once_with("alice", "alice@iit.lk", 3, 4, 2)
    # one-time: subscription is gone after firing
    assert pn.get_subscription("alice") is None


def test_check_and_notify_only_fires_subscriptions_whose_threshold_is_met(_isolated_nfs_root):
    pn = _isolated_nfs_root
    pn.subscribe("alice", "alice@iit.lk", 1)   # met (2 free)
    pn.subscribe("bob", "bob@iit.lk", 3)       # not met (2 free)
    with patch("iitgpu.mailer.send_pod_available_notice") as send:
        n = pn.check_and_notify(stats=_stats(shard_total=4, shard_alloc=2))
    assert n == 1
    send.assert_called_once_with("alice", "alice@iit.lk", 2, 4, 1)
    assert pn.get_subscription("alice") is None
    assert pn.get_subscription("bob") is not None


def test_check_and_notify_noop_on_unsharded_site(_isolated_nfs_root):
    pn = _isolated_nfs_root
    pn.subscribe("alice", "alice@iit.lk", 1)
    with patch("iitgpu.mailer.send_pod_available_notice") as send:
        n = pn.check_and_notify(stats=_stats(shard_total=0, shard_alloc=0))
    assert n == 0
    send.assert_not_called()
    assert pn.get_subscription("alice") is not None


def test_check_and_notify_noop_when_stats_unavailable(_isolated_nfs_root):
    pn = _isolated_nfs_root
    pn.subscribe("alice", "alice@iit.lk", 1)
    with patch("iitgpu.slurm.get_node_stats", return_value=None):
        n = pn.check_and_notify()
    assert n == 0
