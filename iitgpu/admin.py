# iitgpu/admin.py
"""Admin panel — gated to the admin group (config.is_admin()).

Permission gatekeepers applied to every privileged subprocess call:
  • stdin=subprocess.DEVNULL — questionary/prompt_toolkit leaves the PTY in
    raw mode after each prompt; inheriting that as sudo stdin causes
    "A terminal is required to authenticate" even when NOPASSWD rules match.
    Explicit DEVNULL ensures sudo never tries to read from the terminal.
  • sudo -n (non-interactive) — fails immediately with a clear error if a
    NOPASSWD rule is ever missing, instead of hanging for input.
  • Full absolute paths — avoids PATH-resolution ambiguity in sudo matching.
"""
from __future__ import annotations

import getpass
import re
import secrets
import string
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

from iitgpu import auditclient
from iitgpu.config import load_config, is_admin
from iitgpu import daemonclient
from iitgpu.slurm import get_node_stats

# Sri Lanka Standard Time = UTC+5:30
def _cluster_tz():
    try:
        from iitgpu.config import cluster_tz
        return cluster_tz()
    except Exception:
        return timezone(timedelta(hours=5, minutes=30))

_LK = _cluster_tz()

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
# POSIX-safe usernames only: lowercase start, then lowercase/digits/_/-, max 32.
# Blocks path traversal (../) and shell/sudo argument injection in provisioning.
_USERNAME_RE = re.compile(r"^[a-z_][a-z0-9_-]{0,31}$")


def valid_username(username: str) -> bool:
    return bool(_USERNAME_RE.match(username or ""))


def _gen_password() -> str:
    """Generate a random 8-character password with lowercase, uppercase, digits, and symbols."""
    symbols = "!@#$%&*"
    pool = string.ascii_letters + string.digits + symbols
    parts = [
        secrets.choice(string.ascii_lowercase),
        secrets.choice(string.ascii_uppercase),
        secrets.choice(string.digits),
        secrets.choice(symbols),
        *[secrets.choice(pool) for _ in range(4)],
    ]
    for i in range(len(parts) - 1, 0, -1):
        j = secrets.randbelow(i + 1)
        parts[i], parts[j] = parts[j], parts[i]
    return "".join(parts)


def _fmt_ts(ts_str: str) -> str:
    """Convert ISO-8601 UTC timestamp to GMT+5:30 display string."""
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.astimezone(_LK).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, AttributeError):
        return ts_str[:19]


def _run(cmd: list[str], timeout: int = 15,
         stdin_data: str | None = None) -> tuple[int, str, str]:
    """Run a subprocess with stdin always closed (DEVNULL) unless stdin_data is given."""
    try:
        kw: dict = {"capture_output": True, "text": True, "timeout": timeout}
        if stdin_data is not None:
            kw["input"] = stdin_data
        else:
            kw["stdin"] = subprocess.DEVNULL
        r = subprocess.run(cmd, **kw)
        return r.returncode, r.stdout, r.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, "", str(exc)


# ── Node control ─────────────────────────────────────────────────────────────────

def get_jobs_on_node(node: str) -> list[dict]:
    rc, out, _ = _run(["squeue", "--noheader",
                        "--format=%i|%u|%j|%T", f"--nodelist={node}"])
    if rc != 0 or not out.strip():
        return []
    jobs = []
    for line in out.strip().splitlines():
        parts = line.split("|")
        if len(parts) == 4:
            jobs.append({"id": parts[0].strip(), "user": parts[1].strip(),
                         "name": parts[2].strip(), "state": parts[3].strip()})
    return jobs


def pod_occupancy(node: str, total_pods: int) -> list[dict | None]:
    """Which running job holds each of the node's pod cells, oldest job
    first. Purely a rendering convention: SLURM only tracks a shard COUNT
    per job, not a specific pod identity, so cell assignment is recomputed
    fresh on every call, never persisted."""
    rc, out, _ = _run(["squeue", "--noheader", "--states=RUNNING",
                        "--format=%i|%u|%j|%S|%b", f"--nodelist={node}"])
    cells: list[dict | None] = [None] * total_pods
    if rc != 0 or not out.strip():
        return cells
    jobs = []
    for line in out.strip().splitlines():
        parts = line.split("|")
        if len(parts) != 5:
            continue
        jid, user, name, start, gres = (p.strip() for p in parts)
        shards = 0
        for item in gres.split(","):
            item = item.strip()
            if item.startswith("gres/shard:") or item.startswith("shard:"):
                try:
                    shards = int(item.split(":")[-1].split("(")[0])
                except ValueError:
                    shards = 0
        if shards > 0:
            jobs.append({"id": jid, "user": user, "name": name,
                        "start": start, "shards": shards})
    jobs.sort(key=lambda j: j["start"])
    idx = 0
    for j in jobs:
        for _ in range(j["shards"]):
            if idx >= total_pods:
                break
            cells[idx] = {"id": j["id"], "user": j["user"], "name": j["name"]}
            idx += 1
    return cells


def cancel_jobs_on_node(node: str) -> tuple[int, list[str]]:
    jobs = get_jobs_on_node(node)
    cancelled = []
    for j in jobs:
        rc, _, _ = _run(["sudo", "-n", "scancel", j["id"]])
        if rc == 0:
            cancelled.append(j["id"])
            auditclient.log("admin_job_cancel", detail=f"force-drain:{node}",
                            job_id=j["id"])
    return len(cancelled), cancelled


def drain_node(node: str, reason: str,
               cancel_running: bool = False) -> tuple[bool, str]:
    if not node or not reason:
        return False, "node and reason are required"
    cancelled_ids: list[str] = []
    if cancel_running:
        _, cancelled_ids = cancel_jobs_on_node(node)
    rc, _, err = _run(["sudo", "-n", "scontrol", "update",
                        f"nodename={node}", "state=drain", f"reason={reason}"])
    auditclient.log("admin_node_drain", detail=f"{node}:{reason}")
    if rc != 0:
        return False, err.strip() or "drain failed"
    if cancelled_ids:
        return True, f"draining — cancelled {len(cancelled_ids)} job(s): {', '.join(cancelled_ids)}"
    return True, "draining (running jobs will finish before node reaches DRAINED)"


def resume_node(node: str) -> tuple[bool, str]:
    rc, _, err = _run(["sudo", "-n", "scontrol", "update",
                        f"nodename={node}", "state=resume"])
    auditclient.log("admin_node_resume", detail=node)
    return (rc == 0), ("resumed" if rc == 0 else (err.strip() or "resume failed"))


# ── Users ─────────────────────────────────────────────────────────────────────────

def list_gpuusers() -> list[str]:
    cfg = load_config()
    import grp
    import pwd
    try:
        g = grp.getgrnam(cfg.gpuusers_group)
        members = set(g.gr_mem)
        for u in pwd.getpwall():
            if u.pw_gid == g.gr_gid:
                members.add(u.pw_name)
        return sorted(members)
    except KeyError:
        return []


def set_user_password(username: str, password: str) -> tuple[bool, str]:
    rc, _, err = _run(["sudo", "-n", "chpasswd"],
                       stdin_data=f"{username}:{password}\n")
    return (rc == 0), (err.strip() or "")


def provision_user(username: str, admin: bool = False,
                   password: str = "",
                   role: str = "",
                   email: str = "",
                   full_name: str = "",
                   notes: str = "") -> tuple[bool, str]:
    """Create user on both nodes + SLURM association, then write users.db row."""
    if not valid_username(username):
        return False, (f"invalid username {username!r} — must match "
                       f"[a-z_][a-z0-9_-]{{0,31}} (no slashes, spaces, or dots)")
    cmd = ["sudo", "-n", "/usr/local/bin/iit-gpu-adduser", username]
    if admin or role == "admin":
        cmd.append("--admin")
    elif role == "shell":
        cmd.append("--shell-user")
    rc, out, err = _run(cmd, timeout=120)
    if rc != 0:
        return False, err.strip() or "provision failed"
    msg = out.strip()
    if email:
        effective_role = "admin" if (admin or role == "admin") else role or "tool"
        ok_db, db_msg = daemonclient.create_user(
            username, email, effective_role, full_name, notes,
            must_change_pw=bool(password))
        if ok_db:
            msg += "\n  OK  user DB record created"
        else:
            msg += f"\n  WARN  user DB record failed: {db_msg}"
    auditclient.log("admin_provision_user",
                    detail=username,
                    meta={"role": role or ("admin" if admin else "tool"),
                          "email": email})
    ok_pw = False
    if password:
        ok_pw, perr = set_user_password(username, password)
        msg += "\n  OK  password set" if ok_pw else f"\n  WARN  password not set: {perr or 'chpasswd failed'}"
        if ok_pw:
            auditclient.log("password_change_required", detail=username)
    if email and ok_pw:
        from iitgpu import mailer as _mailer
        mail_ok, mail_msg = _mailer.send_welcome(username, email, full_name, password)
        if mail_ok:
            msg += "\n  OK  welcome email sent (with initial password)"
            auditclient.log("welcome_sent", detail=username, meta={"email": email})
        else:
            msg += f"\n  WARN  welcome email failed: {mail_msg} — hand credentials in person"
            auditclient.log("mail_failed", detail=f"welcome:{username}",
                            meta={"error": mail_msg})

    # Admins are NOT notified about user creation or any other user update.
    # User-facing mail (welcome/login/offboard) goes only to the user it concerns;
    # admins only ever receive mail about their OWN account, to their own address.

    return True, msg


def offboard_user(username: str, purge: bool = False) -> tuple[bool, str]:
    if not valid_username(username):
        return False, f"invalid username {username!r} — refusing to offboard"
    user_record = daemonclient.get_user(username)
    cmd = ["sudo", "-n", "/usr/local/bin/iit-gpu-deluser", username]
    if purge:
        cmd.append("--purge-data")
    rc, out, err = _run(cmd, timeout=120)
    if rc == 0:
        daemonclient.offboard_user(username)
        auditclient.log("admin_offboard_user", detail=username)
        if user_record and user_record.get("email"):
            from iitgpu import mailer as _mailer
            mail_ok, mail_msg = _mailer.send_offboard(
                username, user_record["email"], user_record.get("full_name", ""))
            if not mail_ok:
                auditclient.log("mail_failed", detail=f"offboard:{username}",
                                meta={"error": mail_msg})
    return (rc == 0), (out.strip() if rc == 0 else (err.strip() or "offboard failed"))


# ── Audit log ─────────────────────────────────────────────────────────────────────

def read_audit(limit: int = 40, action_filter: str = "",
               user_filter: str = "",
               date_from: str = "", date_to: str = "") -> list[dict]:
    """Read recent audit events via daemon (SQLite), newest first."""
    return daemonclient.query_audit(
        user=user_filter, action=action_filter,
        date_from=date_from, date_to=date_to, limit=limit)


# ── Maintenance notice ────────────────────────────────────────────────────────────

def _maintenance_path() -> str:
    cfg = load_config()
    return f"{cfg.nfs_root}/.maintenance.json"


def get_maintenance() -> dict | None:
    import json
    try:
        data = json.loads(open(_maintenance_path()).read())
        if data.get("active"):
            return data
    except (OSError, ValueError):
        pass
    return None


def set_maintenance(reason: str, set_by: str) -> tuple[bool, str]:
    import json
    import os
    data = {
        "active": True,
        "reason": reason,
        "set_by": set_by,
        "since": datetime.now(timezone.utc).isoformat(),
    }
    try:
        p = _maintenance_path()
        with open(p, "w") as f:
            json.dump(data, f)
        os.chmod(p, 0o666)
        auditclient.log("admin_maintenance_set", detail=reason)
        return True, f"Maintenance notice active: {reason}"
    except OSError as exc:
        return False, str(exc)


def clear_maintenance() -> tuple[bool, str]:
    import os
    try:
        os.remove(_maintenance_path())
    except FileNotFoundError:
        pass
    except OSError as exc:
        return False, str(exc)
    auditclient.log("admin_maintenance_clear", detail="")
    return True, "Maintenance notice cleared."


# ── Mail service kill-switch ────────────────────────────────────────────────────
# A single flag file on the share (presence = OFF) is checked at every send
# point: the transactional client (mailer._daemon_mail), the daemon's authoritative
# mail.send handler, AND the SLURM-job MailProg (iit-gpu-mailer). Toggling it here
# stops/restores ALL outbound email immediately.

def is_mail_disabled() -> bool:
    from iitgpu.mailer import mail_disabled
    return mail_disabled()


def set_mail_disabled(set_by: str) -> tuple[bool, str]:
    import json
    import os
    from iitgpu.mailer import _mail_flag_path
    try:
        p = _mail_flag_path()
        with open(p, "w") as f:
            json.dump({"disabled": True, "set_by": set_by,
                       "since": datetime.now(timezone.utc).isoformat()}, f)
        os.chmod(p, 0o666)  # readable by the daemon (root) + slurm MailProg
        auditclient.log("admin_mail_disabled", detail=set_by)
        return True, "Mail service DISABLED — no emails will be sent."
    except OSError as exc:
        return False, str(exc)


def enable_mail(set_by: str = "") -> tuple[bool, str]:
    import os
    from iitgpu.mailer import _mail_flag_path
    try:
        os.remove(_mail_flag_path())
    except FileNotFoundError:
        pass
    except OSError as exc:
        return False, str(exc)
    auditclient.log("admin_mail_enabled", detail=set_by)
    return True, "Mail service ENABLED — emails will be sent again."


# ── QOS / partitions ──────────────────────────────────────────────────────────────

def list_qos() -> list[dict]:
    rc, out, _ = _run(["sacctmgr", "-n", "--parsable2", "show", "qos",
                        "format=Name,MaxWall,MaxTRESPerUser,Priority"])
    rows: list[dict] = []
    for line in out.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 4 or not parts[0].strip():
            continue
        name = parts[0].strip()
        max_wall = parts[1].strip() or "unlimited"
        tres = parts[2].strip()
        max_gpu = "unlimited"
        max_pods = "unlimited"
        for item in tres.split(","):
            if item.startswith("gres/gpu="):
                max_gpu = item.split("=", 2)[-1]
            elif item.startswith("gres/shard="):
                max_pods = item.split("=", 2)[-1]
        rows.append({
            "name": name,
            "max_wall": max_wall,
            "max_gpu": max_gpu,
            "max_pods": max_pods,
            "priority": parts[3].strip() or "0",
        })
    return rows


def set_qos_maxwall(qos_name: str, max_wall: str) -> tuple[bool, str]:
    rc, out, err = _run(
        ["sudo", "-n", "sacctmgr", "-i", "modify", "qos", qos_name,
         "set", f"MaxWall={max_wall}"], timeout=20)
    auditclient.log("admin_qos_modify", detail=f"{qos_name}:MaxWall={max_wall!r}")
    return (rc == 0), (out.strip() or "updated") if rc == 0 else (err.strip() or "failed")


def set_qos_merge_tres(qos_name: str, **tres_updates) -> tuple[bool, str]:
    """Modify one or more components of MaxTRESPerUser without clobbering the
    others. A value of None clears that component; a positive int sets it.

    Fixes a latent bug: the old set_qos_maxgpu() overwrote the WHOLE
    MaxTRESPerUser string, so setting the GPU cap silently dropped any
    existing shard (pod) cap, and vice versa."""
    rc, out, err = _run(["sacctmgr", "-n", "--parsable2", "show", "qos",
                        qos_name, "format=MaxTRESPerUser"])
    if rc != 0:
        return False, (err.strip() or "could not read current MaxTRESPerUser")
    parts: dict[str, str] = {}
    first_line = out.strip().splitlines()[0] if out.strip() else ""
    for item in first_line.split(","):
        if "=" in item:
            k, _, v = item.partition("=")
            parts[k.strip()] = v.strip()
    for key, val in tres_updates.items():
        if val is None:
            parts.pop(key, None)
        else:
            parts[key] = str(val)
    tres = ",".join(f"{k}={v}" for k, v in parts.items())
    rc, out, err = _run(
        ["sudo", "-n", "sacctmgr", "-i", "modify", "qos", qos_name,
         "set", f"MaxTRESPerUser={tres}"], timeout=20)
    auditclient.log("admin_qos_modify", detail=f"{qos_name}:MaxTRESPerUser={tres!r}")
    return (rc == 0), (out.strip() or "updated") if rc == 0 else (err.strip() or "failed")


def set_qos_maxgpu(qos_name: str, max_gpu: int | None) -> tuple[bool, str]:
    return set_qos_merge_tres(qos_name, **{"gres/gpu": max_gpu})


def set_qos_max_pods_per_user(qos_name: str, max_pods: int | None) -> tuple[bool, str]:
    """Caps how many pods (GPU shards) one user can hold across their
    concurrently running jobs on this QOS -- enforced by SLURM itself
    (sacctmgr/slurmctld), not custom application logic."""
    return set_qos_merge_tres(qos_name, **{"gres/shard": max_pods})


def set_qos_priority(qos_name: str, priority: int) -> tuple[bool, str]:
    rc, out, err = _run(
        ["sudo", "-n", "sacctmgr", "-i", "modify", "qos", qos_name,
         "set", f"Priority={priority}"], timeout=20)
    auditclient.log("admin_qos_modify", detail=f"{qos_name}:Priority={priority}")
    return (rc == 0), (out.strip() or "updated") if rc == 0 else (err.strip() or "failed")


# ── Pods sub-menu ─────────────────────────────────────────────────────────────────

def resize_pod_count(new_n: int, node: str = "iit-MS-7E06") -> tuple[bool, str]:
    """Admin action: change the cluster's pod count. Refuses if any job is
    running/queued anywhere, or if new_n would floor CPU/mem per pod to
    zero. Otherwise shells out to resize-pods.sh, which does the actual
    cross-node config rewrite + restart + verify."""
    rc, out, _ = _run(["squeue", "--noheader"])
    if rc != 0:
        # Fail CLOSED: a squeue we could not read is NOT proof of an empty
        # queue, and resizing under live jobs is the one thing this gate exists
        # to prevent.
        return False, "Cannot read the job queue (squeue failed) -- refusing to resize"
    if out.strip():
        n_jobs = len(out.strip().splitlines())
        return False, f"{n_jobs} job(s) still active cluster-wide -- refusing to resize"

    stats = get_node_stats(node)
    if stats is None:
        return False, "Cannot read live node stats -- refusing to resize blind"

    from iitgpu.pods import fits_new_pod_count
    fits, msg = fits_new_pod_count(new_n, stats)
    if not fits:
        return False, msg

    # -u slurmadmin is REQUIRED and must stay in lockstep with the grant in
    # deploy/sudoers-gateway-admin:
    #     %gpuadmins ALL=(slurmadmin) NOPASSWD: /opt/iit-gpu/deploy/resize-pods.sh *
    # Without -u, sudo targets root, which that line does not authorise (and
    # the script's own `id -un` guard would refuse anyway). Same shape as the
    # "log in as user" action's `sudo -H -u <target>` below.
    rc, out, err = _run(
        ["sudo", "-n", "-u", "slurmadmin",
         "/opt/iit-gpu/deploy/resize-pods.sh", str(new_n)], timeout=120)
    auditclient.log("admin_pod_resize", detail=f"new_n={new_n} rc={rc}")
    if rc != 0:
        return False, (err.strip() or out.strip() or "resize failed")
    return True, (out.strip() or f"resize applied: pod count is now {new_n}")


def _pods_menu(style, node: str = "iit-MS-7E06") -> None:
    import questionary
    from rich.table import Table
    from iitgpu.pods import fits_new_pod_count, pod_count, pod_count_known, pod_resources
    from iitgpu.ui import console, ok, err, screen

    stats = get_node_stats(node)
    known = pod_count_known(stats)
    n = pod_count(stats)
    size = pod_resources(stats)
    cells = pod_occupancy(node, n)

    t = Table(show_header=True, header_style="bold cyan")
    t.add_column("Pod")
    t.add_column("Status")
    t.add_column("Job")
    seen: set[str] = set()
    for i, cell in enumerate(cells, start=1):
        if cell is None:
            t.add_row(str(i), "[dim]free[/]", "")
        else:
            label = "" if cell["id"] in seen else f"{cell['id']} {cell['user']} ({cell['name']})"
            seen.add(cell["id"])
            t.add_row(str(i), "[green]in use[/]", label)
    # pod_count()/pod_resources() floor to 1/PodSize(1,1) when live stats are
    # unreadable -- a safe number for arithmetic, but presenting it to an admin
    # as fact would be a lie. Gate on pod_count_known() like every other
    # pods.py consumer (validate.py, launchspec.py, review.py, wizard.py).
    if known:
        status = (f"{n} pod(s) configured — {size.cpus} CPU / "
                  f"{size.mem_gb} GB RAM each")
    else:
        status = "pod count unknown — cluster stats unavailable"
    screen("Pods", status=status)
    console.print(t)

    if questionary.confirm("Resize pod count?", default=False, style=style).ask():
        current = str(n) if known else "unknown"
        val = questionary.text(f"New pod count (currently {current}):", style=style).ask()
        try:
            new_n = int((val or "").strip())
        except ValueError:
            err("Enter a whole number."); return

        # Spec: the confirm dialog shows the REAL derived per-pod CPU/mem for
        # the candidate N before anything is committed. fits_new_pod_count()
        # already returns exactly that sentence; resize_pod_count() re-runs the
        # same check atomically at execution time (cheap, and the queue/stats
        # can change between the preview and the commit).
        if stats is None:
            err("Cannot read live node stats -- refusing to resize blind."); return
        fits, preview = fits_new_pod_count(new_n, stats)
        if not fits:
            err(preview); return
        if not questionary.confirm(
                f"{preview} Apply this resize?", default=False, style=style).ask():
            return

        good, msg = resize_pod_count(new_n, node=node)
        (ok if good else err)(msg)


# ── QOS sub-menu ──────────────────────────────────────────────────────────────────

def _qos_menu(style) -> None:
    import questionary
    from rich.table import Table
    from iitgpu.ui import console, info, ok, err, screen, select_menu, warn

    while True:
        rows = list_qos()
        if not rows:
            screen("QOS / Limits")
            warn("No QOS data (sacctmgr unavailable)."); return

        t = Table(show_header=True, header_style="bold cyan", show_lines=False)
        t.add_column("QOS", style="magenta")
        t.add_column("Max Wall Time")
        t.add_column("Max GPUs / User")
        t.add_column("Max Pods / User")
        t.add_column("Priority")
        for r in rows:
            t.add_row(r["name"], r["max_wall"], str(r["max_gpu"]),
                     str(r["max_pods"]), r["priority"])
        screen("QOS / Limits", status=t)

        qos_names = [r["name"] for r in rows]
        qname = select_menu("Select QOS to edit:", qos_names)
        if qname is None:
            return

        current = next((r for r in rows if r["name"] == qname), {})
        field = select_menu(
            "Field to change:",
            ["Max Wall Time", "Max GPUs per user", "Max Pods per user", "Priority"])
        if field is None:
            continue

        if field == "Max Wall Time":
            info(f"  Current: [magenta]{current.get('max_wall', '?')}[/]")
            info("  Format: HH:MM:SS or D-HH:MM:SS  |  leave blank = unlimited")
            val = questionary.text("New MaxWall:", style=style).ask()
            if val is None:
                continue
            if questionary.confirm(
                    f"Set [magenta]{qname}[/] MaxWall to "
                    f"[magenta]{val.strip() or 'unlimited'}[/]?",
                    default=True, style=style).ask():
                good, msg = set_qos_maxwall(qname, val.strip())
                (ok if good else err)(msg)

        elif field == "Max GPUs per user":
            info(f"  Current: [magenta]{current.get('max_gpu', '?')}[/]")
            val = questionary.text(
                "New max GPUs (positive integer; blank = unlimited):",
                style=style).ask()
            if val is None:
                continue
            val = val.strip()
            gpu_val: int | None = None
            if val:
                try:
                    gpu_val = int(val)
                    if gpu_val <= 0:
                        raise ValueError
                except ValueError:
                    err("Enter a positive integer or leave blank."); continue
            if questionary.confirm(
                    f"Set [magenta]{qname}[/] Max GPUs to "
                    f"[magenta]{gpu_val if gpu_val is not None else 'unlimited'}[/]?",
                    default=True, style=style).ask():
                good, msg = set_qos_maxgpu(qname, gpu_val)
                (ok if good else err)(msg)

        elif field == "Max Pods per user":
            info(f"  Current: [magenta]{current.get('max_pods', '?')}[/]")
            val = questionary.text(
                "New max pods per user (positive integer; blank = unlimited):",
                style=style).ask()
            if val is None:
                continue
            val = val.strip()
            pods_val: int | None = None
            if val:
                try:
                    pods_val = int(val)
                    if pods_val <= 0:
                        raise ValueError
                except ValueError:
                    err("Enter a positive integer or leave blank."); continue
            if questionary.confirm(
                    f"Set [magenta]{qname}[/] Max Pods per user to "
                    f"[magenta]{pods_val if pods_val is not None else 'unlimited'}[/]?",
                    default=True, style=style).ask():
                good, msg = set_qos_max_pods_per_user(qname, pods_val)
                (ok if good else err)(msg)

        elif field == "Priority":
            info(f"  Current: [magenta]{current.get('priority', '0')}[/]")
            val = questionary.text("New priority (integer):", style=style).ask()
            if val is None:
                continue
            try:
                prio = int(val.strip())
            except ValueError:
                err("Enter an integer."); continue
            if questionary.confirm(
                    f"Set [magenta]{qname}[/] Priority to [magenta]{prio}[/]?",
                    default=True, style=style).ask():
                good, msg = set_qos_priority(qname, prio)
                (ok if good else err)(msg)


# ── Provision-user sub-flow ───────────────────────────────────────────────────────

def _provision_menu(style) -> None:
    import questionary
    from iitgpu.ui import ok, err, info, warn

    u = questionary.text("New username:", style=style).ask()
    if not u or not u.strip():
        return
    u = u.strip()
    if not valid_username(u):
        err(f"Invalid username {u!r} — must match [a-z_][a-z0-9_-]{{0,31}} "
            f"(no slashes, spaces, or dots).")
        return

    role = questionary.select(
        "User type:",
        choices=[
            questionary.Choice("tool   — forced-TUI, audited (default)", "tool"),
            questionary.Choice("admin  — forced-TUI + admin panel",      "admin"),
            questionary.Choice("shell  — real bash shell, NOT audited",  "shell"),
        ],
        style=style,
    ).ask()
    if role is None:
        return

    if role == "shell":
        warn("[yellow bold]Shell user warning:[/]")
        warn("[yellow]This grants a real shell on the login node. Their activity is[/]")
        warn("[yellow]NOT audited by the tool. Use only for edge cases the tool[/]")
        warn("[yellow]cannot handle. They are still SLURM-capped via their association.[/]")
        if not questionary.confirm(
                "Understood — proceed with shell user creation?",
                default=False, style=style).ask():
            return

    full_name = questionary.text("Full name:", style=style).ask() or ""
    email = questionary.text("Email address:", style=style).ask() or ""
    if email and not _EMAIL_RE.match(email.strip()):
        err("Invalid email address — user DB record will be skipped.")
        email = ""
    email = email.strip()
    notes = questionary.text("Notes (optional):", style=style).ask() or ""
    _pw_mode = questionary.select(
        "Initial password:",
        choices=["Set it myself", "Generate a random password"],
        style=style,
    ).ask()
    if _pw_mode is None:
        return

    if _pw_mode == "Set it myself":
        pw = questionary.password(
            "Password — user will be required to change on first login:", style=style).ask() or ""
        if pw:
            pw2 = questionary.password("Confirm password:", style=style).ask() or ""
            if pw != pw2:
                err("Passwords do not match — user not created.")
                return
    else:
        pw = _gen_password()
        ok(f"Generated password:  [bold cyan]{pw}[/]  ← note this down before continuing")

    # ── Workspace exists check (re-provision of offboarded user) ──────────────
    from iitgpu.config import load_config, user_dir
    _cfg = load_config()
    _ws = Path(user_dir(_cfg, u))
    if _ws.exists():
        warn(f"[yellow]Workspace already exists: {_ws}[/]")
        _ws_choice = questionary.select(
            "A workspace from a previous tenant exists. What should happen?",
            choices=[
                "Reuse existing workspace (new user inherits old files)",
                "Wipe and recreate workspace (delete all old files)",
                "Cancel — do not provision",
            ],
            style=style,
        ).ask()
        if _ws_choice is None or _ws_choice.startswith("Cancel"):
            info("Provisioning cancelled.")
            return
        auditclient.log(
            "workspace_decision",
            detail=u,
            meta={"decision": "reuse" if "Reuse" in _ws_choice else "wipe"},
        )
        if "Wipe" in _ws_choice:
            import shutil
            try:
                shutil.rmtree(str(_ws))
                info(f"[dim]Workspace wiped: {_ws}[/]")
            except OSError as exc:
                err(f"Could not wipe workspace: {exc}")
                return

    good, msg = provision_user(
        u, admin=(role == "admin"), role=role, password=pw,
        email=email, full_name=full_name.strip(), notes=notes.strip())
    (ok if good else err)(msg)
    if good and not pw:
        info(f"[dim]Set a password: sudo passwd {u}[/]")


# ── Admin log viewers ─────────────────────────────────────────────────────────────

def _view_audit_log(style) -> None:
    import questionary
    from iitgpu.ui import console, screen, info

    screen("Audit Log")
    uf  = questionary.text("Filter by user (blank=all):", style=style).ask() or ""
    af  = questionary.text("Filter by action (blank=all):", style=style).ask() or ""
    df  = questionary.text("Date from (YYYY-MM-DD, blank=any):", style=style).ask() or ""
    dt  = questionary.text("Date to   (YYYY-MM-DD, blank=any):", style=style).ask() or ""
    lim = questionary.text("Limit (default 40):", style=style).ask() or "40"
    try:
        limit = int(lim)
    except ValueError:
        limit = 40
    date_from = (df.strip() + "T00:00:00+00:00") if df.strip() else ""
    date_to   = (dt.strip() + "T23:59:59+00:00") if dt.strip() else ""
    events = read_audit(limit=limit, action_filter=af.strip(),
                        user_filter=uf.strip(),
                        date_from=date_from, date_to=date_to)
    if not events:
        info("No matching events.")
        return
    for ev in events:
        meta_str = ""
        if ev.get("meta"):
            meta_str = f"  [dim]{ev['meta']}[/]"
        console.print(
            f"  [dim]{_fmt_ts(ev.get('ts', ''))}[/]  "
            f"[magenta]{ev.get('user', '?')}[/]  "
            f"{ev.get('action', '?')}  "
            f"[dim]{ev.get('detail', '')}[/]"
            f"{meta_str}"
        )


def _view_users(style) -> None:
    import questionary
    from rich.table import Table
    from iitgpu.ui import console, info, screen, warn

    screen("User Roster")
    data = daemonclient.view_roster()
    users = data.get("users", [])
    drift = data.get("drift", {})
    db_only = set(drift.get("db_only", []))
    os_only = set(drift.get("os_only", []))

    if not users and not os_only:
        info("No users in database (daemon may be unavailable)."); return

    t = Table(show_header=True, header_style="bold cyan", show_lines=False)
    t.add_column("Username",   style="magenta")
    t.add_column("Full name")
    t.add_column("Email")
    t.add_column("Role")
    t.add_column("Status")
    t.add_column("Created at")
    t.add_column("Created by")
    t.add_column("Flags",      style="yellow")
    # Active users first; offboarded/inactive sink to the bottom (stable within group).
    ordered = sorted(users, key=lambda u: u.get("status", "") != "active")
    for u in ordered:
        flags = []
        if u["username"] in db_only:
            flags.append("DB-only")
        status = u.get("status", "")
        status_cell = f"[green]{status}[/]" if status == "active" else status
        t.add_row(
            u["username"], u.get("full_name", ""), u.get("email", ""),
            u["role"], status_cell, _fmt_ts(u.get("created_at", "")),
            u.get("created_by", ""), ", ".join(flags))
    console.print(t)

    if os_only:
        warn(f"[yellow]OS-only (in group but no DB row):[/] {', '.join(sorted(os_only))}")
    if db_only:
        warn(f"[yellow]DB-only (DB row but no OS group):[/] {', '.join(sorted(db_only))}")

    questionary.press_any_key_to_continue("").ask()


def _view_maillog(style) -> None:
    import questionary
    from iitgpu.ui import console, screen, info

    screen("Mail Delivery Log  (/var/log/msmtp.log)")
    lines = daemonclient.tail_maillog(lines=60)
    if not lines:
        info("Log empty or unavailable (check daemon + /var/log/msmtp.log).")
    else:
        for line in lines:
            console.print(f"  [dim]{line}[/]")
    questionary.press_any_key_to_continue("").ask()


def _view_job_output(style) -> None:
    import questionary
    from iitgpu.ui import console, info, err, screen, select_menu

    screen("User Job Output")
    target_user = questionary.text("Username:", style=style).ask()
    if not target_user or not target_user.strip():
        return
    target_user = target_user.strip()
    cfg = load_config()
    jobs_base = Path(cfg.nfs_root) / cfg.jobs_subdir / target_user
    if not jobs_base.exists():
        info(f"No job directory for {target_user}")
        return
    # Collect .out/.err files
    files = sorted(
        f.name for jd in jobs_base.iterdir() if jd.is_dir()
        for f in jd.iterdir()
        if f.suffix in (".out", ".err")
    ) if jobs_base.exists() else []
    if not files:
        info(f"No .out/.err files for {target_user}")
        return
    fname = select_menu("Select file:", files)
    if fname is None:
        return
    # Find the full relative path
    rel = None
    for jd in jobs_base.iterdir():
        if jd.is_dir() and (jd / fname).exists():
            rel = f"{jd.name}/{fname}"
            break
    if rel is None:
        err("File not found.")
        return
    good, content = daemonclient.read_job_log(target_user, rel)
    if not good:
        err(f"Cannot read: {content}")
        return
    screen(f"Job output: {target_user}/{rel}")
    console.print(content[:8000])   # first 8k chars
    questionary.press_any_key_to_continue("").ask()


def _view_service_health(style) -> None:
    import questionary
    from iitgpu.ui import console, info, screen, select_menu

    _UNITS = ["iit-gpu-audit", "slurmctld", "slurmd", "mariadb", "slurmdbd"]
    screen("Service Health")
    unit = select_menu("Select service:", _UNITS)
    if unit is None:
        return
    good, data = daemonclient.service_status(unit)
    if not good:
        from iitgpu.ui import err
        err(f"Cannot get status: {data.get('error', '?')}"); return
    active = data.get("active", "unknown")
    color = "green" if active == "active" else "red"
    console.print(f"\n  [{color}]● {unit}[/]  status: [{color}]{active}[/]\n")
    journal = data.get("journal", "")
    if journal:
        info("[dim]Recent journal entries:[/]")
        for line in journal.splitlines()[-20:]:
            console.print(f"  [dim]{line}[/]")
    questionary.press_any_key_to_continue("").ask()


def disk_usage_by_user(jobs_root: str) -> list[dict]:
    """Return per-user disk usage under jobs_root, sorted by bytes descending.

    Each entry: {"user": str, "bytes": int, "human": str}
    """
    import shutil as _shutil

    def _fmt(n: int) -> str:
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if n < 1024:
                return f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} PB"

    root = Path(jobs_root)
    if not root.is_dir():
        return []
    rows = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        total = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
        rows.append({"user": entry.name, "bytes": total, "human": _fmt(total)})
    rows.sort(key=lambda r: r["bytes"], reverse=True)
    return rows


def _view_disk_usage(style) -> None:
    import questionary
    from rich.table import Table
    from iitgpu.ui import console, screen, info

    screen("Disk Usage by User  (/shared/jobs)")
    cfg   = load_config()
    root  = Path(cfg.nfs_root) / cfg.jobs_subdir
    rows  = disk_usage_by_user(str(root))
    if not rows:
        info("No job directories found.")
        questionary.press_any_key_to_continue("").ask()
        return
    t = Table(show_header=True, header_style="bold cyan")
    t.add_column("User", style="magenta")
    t.add_column("Used", justify="right")
    for r in rows:
        t.add_row(r["user"], r["human"])
    console.print(t)
    questionary.press_any_key_to_continue("").ask()


# ── Main admin menu ───────────────────────────────────────────────────────────────

def _maintenance_menu(style) -> None:
    import os
    import questionary
    from iitgpu.ui import info, ok, err, screen, select_menu
    screen("Maintenance Notice")
    current = get_maintenance()
    if current:
        info(f"  [yellow]Active notice:[/] {current.get('reason', '')}")
        action = select_menu(
            "Maintenance notice:", ["Update notice", "Clear notice"])
        if action == "Clear notice":
            if questionary.confirm("Clear the maintenance notice?",
                                   default=True, style=style).ask():
                good, msg = clear_maintenance()
                (ok if good else err)(msg)
        elif action == "Update notice":
            reason = questionary.text(
                "New maintenance reason:", style=style).ask()
            if reason and reason.strip():
                good, msg = set_maintenance(
                    reason.strip(), set_by=os.environ.get("USER", "admin"))
                (ok if good else err)(msg)
    else:
        reason = questionary.text(
            "Maintenance reason (shown to all users on login):",
            style=style).ask()
        if reason and reason.strip():
            good, msg = set_maintenance(
                reason.strip(), set_by=os.environ.get("USER", "admin"))
            (ok if good else err)(msg)


def _mail_service_menu(style) -> None:
    import questionary
    from iitgpu.ui import screen, info, ok, err, warn
    screen("Mail Service")
    disabled = is_mail_disabled()
    if disabled:
        info("  [yellow]Current state: DISABLED[/] — no emails are being sent.")
        if questionary.confirm(
            "Re-ENABLE the mail service?", default=False, style=style
        ).ask():
            good, msg = enable_mail(getpass.getuser())
            (ok if good else err)(msg)
    else:
        info("  [green]Current state: ENABLED[/] — emails are being sent.")
        warn("Disabling stops ALL outbound email: welcome / login / offboard "
             "notices AND SLURM job (BEGIN/END/FAIL) mail.")
        if questionary.confirm(
            "DISABLE the mail service now?", default=False, style=style
        ).ask():
            good, msg = set_mail_disabled(getpass.getuser())
            (ok if good else err)(msg)
    questionary.press_any_key_to_continue("").ask()


def _login_as_menu(style) -> None:
    """Drop the admin into another user's TUI via `sudo -iu <user>` so they can
    inspect that account exactly as the user sees it. Requires the gpuadmins
    sudoers Runas rule (deploy/sudoers-gateway-admin). Actions inside the spawned
    session are audited as the TARGET user (kernel SO_PEERCRED), so the switch
    itself is logged here for accountability."""
    import questionary
    from iitgpu.ui import info, ok, err, screen, select_menu, warn
    screen("Log in as user")
    me = getpass.getuser()
    targets = [u for u in list_gpuusers() if u != me]
    if not targets:
        warn("No other users found to log in as.")
        questionary.press_any_key_to_continue("").ask()
        return
    target = select_menu(
        "Log in as which user?  (you'll get THEIR TUI; quit it to return here)",
        targets)
    if not target:
        return
    if not valid_username(target):
        err(f"invalid username {target!r}")
        return
    info(f"  Switching to [bold]{target}[/] — you'll see exactly what they see.")
    info("  Quit their TUI (main menu → Quit, or Ctrl-D) to come back here.")
    auditclient.log("admin_login_as", detail=target)
    # `sudo -H -u <user> <launcher>`: runs the TUI with the target's real UID and
    # HOME. We deliberately do NOT use `sudo -i` — login-shell mode makes sudo
    # match `bash -c <cmd>`, which a tight "launcher-only" sudoers rule can't
    # authorize without also whitelisting a full shell. -H gives the right HOME;
    # USER/LOGNAME are set to the target by sudo; identity is enforced by the
    # kernel (SO_PEERCRED) regardless.
    try:
        subprocess.run(
            ["sudo", "-H", "-u", target, "/usr/local/bin/iit-gpu-manager"])
    except (OSError, KeyboardInterrupt) as exc:
        err(f"Could not start a session as {target}: {exc}")
    auditclient.log("admin_login_as_end", detail=target)
    info(f"  Returned from {target}'s session.")
    questionary.press_any_key_to_continue("").ask()


def admin_menu() -> None:
    import questionary
    from questionary import Separator
    from rich.table import Table
    from iitgpu.ui import STYLE, console, info, ok, err, screen, select_menu, warn

    cfg = load_config()
    if not is_admin(cfg):
        warn("Admin panel is restricted to members of the admin group.")
        return

    style = STYLE
    node_default = "iit-MS-7E06"

    while True:
        _mail_state = "OFF — disabled" if is_mail_disabled() else "ON"
        _maint = get_maintenance()
        _n_users = len(list_gpuusers())
        status = (
            f"[bold]{_n_users}[/] active user(s)   [dim]·[/]   Mail: {_mail_state}"
            + ("   [dim]·[/]   [yellow]Maintenance ON[/]" if _maint else "")
        )
        screen("Admin Panel", status=status)
        choice = select_menu(
            "Select action:",
            [
                Separator("──  User Management  ──────────────────────────"),
                "  Provision user",
                "  Offboard user",
                "  View users",
                "  Log in as user",
                Separator("──  Jobs & Usage  ─────────────────────────────"),
                "  All-user job history",
                "  Cluster usage (all users)",
                "  Disk usage by user",
                "  Any user's job output",
                Separator("──  Cluster Control  ──────────────────────────"),
                "  Drain node",
                "  Resume node",
                "  QOS / limits",
                "  Pods (GPU slots)",
                "  Maintenance notice",
                Separator("──  Monitoring  ───────────────────────────────"),
                "  Audit log",
                "  Service health",
                "  Mail delivery log",
                f"  Mail service: {_mail_state}",
            ],
        )

        if choice is None:
            return
        choice = choice.strip()

        if choice == "Drain node":
            node = questionary.text("Node:", default=node_default, style=style).ask()
            node = (node or node_default).strip()
            reason = questionary.text("Reason:", style=style).ask()
            if not reason or not reason.strip():
                err("A drain reason is required.")
                questionary.press_any_key_to_continue("").ask()
                continue
            running = get_jobs_on_node(node)
            cancel_running = False
            if running:
                info(f"  [yellow]{len(running)} job(s) currently on {node}:[/]")
                for j in running:
                    info(f"    job {j['id']}  user={j['user']}  "
                         f"name={j['name']}  [{j['state']}]")
                cancel_running = questionary.confirm(
                    "Cancel these jobs now? (force drain)",
                    default=False, style=style).ask()
            else:
                info(f"  [dim]No jobs running on {node}.[/]")
            good, msg = drain_node(node, reason.strip(), cancel_running=cancel_running)
            (ok if good else err)(msg)

        elif choice == "Resume node":
            node = questionary.text("Node:", default=node_default, style=style).ask()
            good, msg = resume_node(node or node_default)
            (ok if good else err)(msg)

        elif choice == "Provision user":
            _provision_menu(style)

        elif choice == "Offboard user":
            u = questionary.text("Username to remove:", style=style).ask()
            if u and questionary.confirm(
                    f"Offboard {u}?", default=False, style=style).ask():
                purge = questionary.confirm(
                    "Purge their /shared data?", default=False, style=style).ask()
                good, msg = offboard_user(u.strip(), purge=purge)
                (ok if good else err)(msg)

        elif choice == "View users":
            _view_users(style)
            continue

        elif choice == "Audit log":
            _view_audit_log(style)

        elif choice == "All-user job history":
            from iitgpu.config import jobs_dir
            from iitgpu.slurm import filtered_history
            cfg = load_config()
            t = Table(show_header=True, header_style="bold cyan")
            for c in ("Job ID", "User", "Name", "State", "Elapsed", "Partition"):
                t.add_column(c)
            rows = filtered_history(jobs_dir(cfg), all_users=True, days=30)
            if not rows:
                info("No job history found.")
            for entry in rows:
                t.add_row(entry.job_id, entry.user, entry.name, entry.state,
                          entry.time_used, entry.partition)
            console.print(t)
            questionary.press_any_key_to_continue("").ask()

        elif choice == "Cluster usage (all users)":
            from iitgpu.accounting import usage_by_user
            t = Table(show_header=True, header_style="bold cyan")
            for c in ("User", "GPU-h", "CPU-h", "Jobs"):
                t.add_column(c)
            for r in usage_by_user(days=30):
                t.add_row(r.user, f"{r.gpu_hours:.1f}",
                          f"{r.cpu_hours:.1f}", str(r.job_count))
            console.print(t)
            questionary.press_any_key_to_continue("").ask()

        elif choice == "Disk usage by user":
            _view_disk_usage(style)
            continue

        elif choice == "Any user's job output":
            _view_job_output(style)
            continue

        elif choice == "Mail delivery log":
            _view_maillog(style)
            continue

        elif choice == "Log in as user":
            _login_as_menu(style)
            continue

        elif choice.startswith("Mail service:"):
            _mail_service_menu(style)
            continue

        elif choice == "Service health":
            _view_service_health(style)
            continue

        elif choice == "QOS / limits":
            _qos_menu(style)

        elif choice == "Pods (GPU slots)":
            _pods_menu(style)
            questionary.press_any_key_to_continue("").ask()

        elif choice == "Maintenance notice":
            _maintenance_menu(style)

        else:
            questionary.press_any_key_to_continue("").ask()
            continue

        questionary.press_any_key_to_continue("").ask()
