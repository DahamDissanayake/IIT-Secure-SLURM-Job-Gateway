# iitgpu/pod_notify.py
"""One-time "notify me when pods are free" subscriptions.

A user subscribes while the GPU is fully occupied ("email me once at least
K pods are free"); a periodic checker (deploy/iit-gpu-pod-notify, run by a
systemd timer) polls live cluster state and, the first time a subscription's
threshold is met, emails the user and removes the subscription. One
notification per subscription, never a recurring alert — the user has to
subscribe again if they want to be notified a second time.

Subscriptions live in a single JSON file on NFS (same convention as the
admin ".mail-disabled" flag file) so both the interactive TUI and the
standalone checker script see the same state. All reads/writes hold an
exclusive flock across the whole read-modify-write to stay safe against
concurrent subscribers and the checker running at the same moment.
"""
from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime, timezone


def _sub_file() -> str:
    try:
        from iitgpu.config import load_config
        return f"{load_config().nfs_root}/.pod-notify-subscriptions.json"
    except Exception:
        return f"{os.environ.get('NFS_ROOT', '/shared')}/.pod-notify-subscriptions.json"


def _open_locked(mode: str):
    """Open the subscriptions file for an exclusive read-modify-write,
    creating it with an empty list on first use."""
    path = _sub_file()
    if not os.path.exists(path):
        try:
            with open(path, "x") as f:
                f.write("[]")
        except FileExistsError:
            pass
    f = open(path, mode)
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    return f


def _read(f) -> list[dict]:
    f.seek(0)
    try:
        data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, ValueError):
        return []


def _write(f, subs: list[dict]) -> None:
    f.seek(0)
    f.truncate()
    json.dump(subs, f)
    f.flush()
    os.fsync(f.fileno())


def subscribe(username: str, email: str, wanted_pods: int) -> None:
    """Create or replace this user's subscription. Subscribing again
    overwrites the previous threshold -- one active subscription per user."""
    with _open_locked("r+") as f:
        subs = [s for s in _read(f) if s.get("username") != username]
        subs.append({
            "username": username,
            "email": email,
            "wanted_pods": int(wanted_pods),
            "subscribed_at": datetime.now(timezone.utc).isoformat(),
        })
        _write(f, subs)


def unsubscribe(username: str) -> bool:
    """Remove this user's subscription if one exists. Returns whether one was removed."""
    with _open_locked("r+") as f:
        subs = _read(f)
        remaining = [s for s in subs if s.get("username") != username]
        removed = len(remaining) != len(subs)
        if removed:
            _write(f, remaining)
        return removed


def get_subscription(username: str) -> dict | None:
    with _open_locked("r+") as f:
        subs = _read(f)
    return next((s for s in subs if s.get("username") == username), None)


def check_and_notify(stats=None) -> int:
    """Fire (and remove) every subscription whose threshold is now met.

    Returns how many notifications were sent. Pass `stats` directly in
    tests; the real checker script (deploy/iit-gpu-pod-notify) leaves it
    unset to fetch live cluster state. Sites without GPU sharing configured
    (stats.shard_total == 0) have no pod concept, so nothing ever fires
    there -- this is a no-op, not an error.
    """
    from iitgpu import mailer

    if stats is None:
        from iitgpu.slurm import get_node_stats
        stats = get_node_stats()
    if stats is None or not stats.shard_total:
        return 0

    total = stats.shard_total
    free = max(0, total - stats.shard_alloc)

    with _open_locked("r+") as f:
        candidates = [s for s in _read(f) if free >= s.get("wanted_pods", 1)]
    if not candidates:
        return 0

    # Send outside the lock (a slow/blocked mail send must not hold up
    # subscribe/unsubscribe for the whole 2-minute cycle), then only remove
    # the subscriptions that actually got delivered -- a failed send must
    # not silently consume the user's one-time notification. Failures stay
    # subscribed and are retried next cycle.
    sent_usernames: set[str] = set()
    for s in candidates:
        ok, _msg = mailer.send_pod_available_notice(
            s["username"], s["email"], free, total, s.get("wanted_pods", 1))
        if ok:
            sent_usernames.add(s["username"])

    if sent_usernames:
        with _open_locked("r+") as f:
            remaining = [s for s in _read(f) if s.get("username") not in sent_usernames]
            _write(f, remaining)

    return len(sent_usernames)
