# tests/test_admin.py
"""Phase 7: admin panel — gating, node control, users, audit, QOS."""
from unittest.mock import patch, MagicMock, call
import json
import pytest
from iitgpu import admin


def _proc(rc=0, out="", err=""):
    m = MagicMock(); m.returncode = rc; m.stdout = out; m.stderr = err
    return m


# ── Gate ──────────────────────────────────────────────────────────────────────

def test_admin_menu_blocked_for_non_admin(capsys):
    with patch("iitgpu.admin.is_admin", return_value=False):
        admin.admin_menu()


# ── Timestamp formatting ──────────────────────────────────────────────────────

def test_fmt_ts_converts_utc_to_lk():
    # 2026-06-01T00:00:00+00:00 UTC  →  2026-06-01 05:30:00 GMT+5:30
    result = admin._fmt_ts("2026-06-01T00:00:00+00:00")
    assert result == "2026-06-01 05:30:00"


def test_fmt_ts_handles_z_suffix():
    result = admin._fmt_ts("2026-06-01T00:00:00Z")
    assert result == "2026-06-01 05:30:00"


def test_fmt_ts_handles_bad_input():
    result = admin._fmt_ts("not-a-timestamp")
    assert result == "not-a-timestamp"  # shorter than 19 chars, returned as-is


def test_fmt_ts_handles_empty():
    result = admin._fmt_ts("")
    assert result == ""


# ── Node control ──────────────────────────────────────────────────────────────

def test_drain_node_uses_sudo_n():
    with patch("subprocess.run", return_value=_proc()) as r:
        ok, msg = admin.drain_node("node1", "maintenance")
    # drain_node calls squeue (get_jobs_on_node) then scontrol
    scontrol_call = next(c for c in r.call_args_list
                         if "scontrol" in c[0][0])
    cmd = scontrol_call[0][0]
    assert cmd[:3] == ["sudo", "-n", "scontrol"]
    assert "nodename=node1" in cmd
    assert "state=drain" in cmd
    assert "reason=maintenance" in cmd
    assert ok


def test_drain_node_requires_reason():
    ok, _ = admin.drain_node("node1", "")
    assert not ok


def test_drain_node_force_cancels_jobs():
    squeue_out = "42|public|train|RUNNING\n"
    responses = [_proc(out=squeue_out), _proc(), _proc()]  # squeue, scancel, scontrol
    with patch("subprocess.run", side_effect=responses) as r:
        ok, msg = admin.drain_node("node1", "maint", cancel_running=True)
    assert ok
    assert "42" in msg
    scancel_call = r.call_args_list[1][0][0]
    assert "scancel" in scancel_call
    assert "42" in scancel_call


def test_get_jobs_on_node_parses_squeue():
    out = "42|public|train|RUNNING\n99|daham|test|RUNNING\n"
    with patch("subprocess.run", return_value=_proc(out=out)):
        jobs = admin.get_jobs_on_node("iit-MS-7E06")
    assert len(jobs) == 2
    assert jobs[0]["id"] == "42" and jobs[0]["user"] == "public"
    assert jobs[1]["id"] == "99"


def test_get_jobs_on_node_empty():
    with patch("subprocess.run", return_value=_proc(out="")):
        jobs = admin.get_jobs_on_node("iit-MS-7E06")
    assert jobs == []


def test_pod_occupancy_assigns_cells_by_start_time():
    out = ("42|public|train|2026-07-27T08:00:00|gres/shard:2\n"
           "99|daham|nb|2026-07-27T09:00:00|gres/shard:1\n")
    with patch("subprocess.run", return_value=_proc(out=out)):
        cells = admin.pod_occupancy("iit-MS-7E06", total_pods=4)
    assert len(cells) == 4
    assert cells[0]["id"] == "42" and cells[1]["id"] == "42"
    assert cells[2]["id"] == "99"
    assert cells[3] is None


def test_pod_occupancy_all_free_when_no_jobs():
    with patch("subprocess.run", return_value=_proc(out="")):
        cells = admin.pod_occupancy("iit-MS-7E06", total_pods=4)
    assert cells == [None, None, None, None]


def test_pod_occupancy_ignores_jobs_with_no_gpu():
    out = "7|public|cpujob|2026-07-27T08:00:00|\n"
    with patch("subprocess.run", return_value=_proc(out=out)):
        cells = admin.pod_occupancy("iit-MS-7E06", total_pods=4)
    assert cells == [None, None, None, None]


def test_pod_occupancy_caps_at_total_pods_if_over_allocated():
    """Defensive: never index past the grid even if SLURM briefly reports
    more shards allocated than the grid has cells for (e.g. mid-resize)."""
    out = "1|a|j|2026-07-27T08:00:00|gres/shard:9\n"
    with patch("subprocess.run", return_value=_proc(out=out)):
        cells = admin.pod_occupancy("iit-MS-7E06", total_pods=4)
    assert len(cells) == 4
    assert all(c is not None and c["id"] == "1" for c in cells)


# ── Pod count resize ────────────────────────────────────────────────────────

def test_resize_pod_count_refuses_when_jobs_running():
    out = "42|public|train|RUNNING\n"
    with patch("subprocess.run", return_value=_proc(out=out)):
        ok, msg = admin.resize_pod_count(5)
    assert not ok
    assert "running" in msg.lower() or "active" in msg.lower()


def test_resize_pod_count_refuses_degenerate_n(monkeypatch):
    from iitgpu.slurm import NodeStats
    stats = NodeStats(state="MIXED", cpu_load=0.0, cpu_total=32, cpu_alloc=0,
                      mem_total_mb=62000, mem_alloc_mb=0, gpu_total=1, gpu_alloc=0,
                      shard_total=4, shard_alloc=0)
    with patch("subprocess.run", return_value=_proc(out="")), \
         patch("iitgpu.admin.get_node_stats", return_value=stats):
        ok, msg = admin.resize_pod_count(40)
    assert not ok
    assert "0 CPU" in msg


def test_resize_pod_count_runs_the_script_when_clear(monkeypatch):
    from iitgpu.slurm import NodeStats
    stats = NodeStats(state="MIXED", cpu_load=0.0, cpu_total=32, cpu_alloc=0,
                      mem_total_mb=62000, mem_alloc_mb=0, gpu_total=1, gpu_alloc=0,
                      shard_total=4, shard_alloc=0)

    seen = []

    def fake_run(cmd, timeout=15, stdin_data=None):
        if cmd[0] == "squeue":
            return 0, "", ""
        seen.append(cmd)
        assert cmd[-1] == "5"
        assert "resize-pods.sh" in cmd[-2]
        return 0, "resize applied: pod count is now 5", ""

    with patch("iitgpu.admin._run", side_effect=fake_run), \
         patch("iitgpu.admin.get_node_stats", return_value=stats):
        ok, msg = admin.resize_pod_count(5)
    assert ok
    assert "5" in msg
    # The runas MUST be slurmadmin: sudo with no -u targets root, which the
    # sudoers grant does not authorise (and the script's own id -un guard
    # would refuse anyway).
    assert seen[0][:4] == ["sudo", "-n", "-u", "slurmadmin"]


def test_resize_sudo_runas_matches_the_sudoers_grant():
    """Couples the sudo invocation to deploy/sudoers-gateway-admin so the two
    cannot silently drift apart again: the ALL=(<runas>) in the grant for
    resize-pods.sh must be the same account admin.py passes to `sudo -u`."""
    import re
    from pathlib import Path
    from iitgpu.slurm import NodeStats
    sudoers = (Path(__file__).resolve().parents[1]
               / "deploy" / "sudoers-gateway-admin").read_text()
    grant = [ln for ln in sudoers.splitlines() if "resize-pods.sh" in ln]
    assert len(grant) == 1, grant
    m = re.search(r"ALL=\(([^)]+)\)", grant[0])
    assert m, grant[0]
    runas = m.group(1)
    script_path = grant[0].split("NOPASSWD:")[1].strip().rstrip(" *")

    stats = NodeStats(state="MIXED", cpu_load=0.0, cpu_total=32, cpu_alloc=0,
                      mem_total_mb=62000, mem_alloc_mb=0, gpu_total=1, gpu_alloc=0,
                      shard_total=4, shard_alloc=0)
    seen = []

    def fake_run(cmd, timeout=15, stdin_data=None):
        if cmd[0] == "squeue":
            return 0, "", ""
        seen.append(cmd)
        return 0, "ok", ""

    with patch("iitgpu.admin._run", side_effect=fake_run), \
         patch("iitgpu.admin.get_node_stats", return_value=stats):
        admin.resize_pod_count(5)

    assert seen[0] == ["sudo", "-n", "-u", runas, script_path, "5"]


def test_resize_pod_count_refuses_when_squeue_fails():
    """Fail CLOSED: a squeue we could not run is not evidence of an empty
    queue, and resizing under live jobs is what this gate exists to stop."""
    with patch("subprocess.run", return_value=_proc(rc=1, err="squeue: error")):
        ok, msg = admin.resize_pod_count(5)
    assert not ok
    assert "queue" in msg.lower()


def test_resize_pod_count_refuses_when_stats_unreadable():
    """No live node stats means no way to check the candidate N would leave a
    usable per-pod size -- refuse rather than resize blind."""
    with patch("subprocess.run", return_value=_proc(out="")), \
         patch("iitgpu.admin.get_node_stats", return_value=None):
        ok, msg = admin.resize_pod_count(5)
    assert not ok
    assert "blind" in msg.lower() or "stats" in msg.lower()


# ── Pods screen (_pods_menu) ────────────────────────────────────────────────

def _stats(shard_total=4):
    from iitgpu.slurm import NodeStats
    return NodeStats(state="MIXED", cpu_load=0.0, cpu_total=32, cpu_alloc=0,
                     mem_total_mb=62000, mem_alloc_mb=0, gpu_total=1, gpu_alloc=0,
                     shard_total=shard_total, shard_alloc=0)


def _answer(value):
    a = MagicMock(); a.ask.return_value = value
    return a


def test_pods_menu_reports_unknown_when_stats_unavailable(capsys):
    """pod_count()/pod_resources() floor to 1 / 1 CPU / 1 GB when live stats
    are unreadable. Presenting that to an admin as fact is a lie -- every other
    pods.py consumer gates on pod_count_known() and so must this one."""
    with patch("iitgpu.admin.get_node_stats", return_value=None), \
         patch("iitgpu.admin.pod_occupancy", return_value=[None]), \
         patch("questionary.confirm", return_value=_answer(False)):
        admin._pods_menu(None)
    out = capsys.readouterr().out
    assert "unknown" in out.lower()
    assert "1 pod(s) configured" not in out


def test_pods_menu_confirms_with_the_derived_sizing_before_resizing():
    """Spec: the confirm dialog shows the REAL derived per-pod CPU/mem for the
    candidate N before anything is committed."""
    prompts = []

    def fake_confirm(msg, **kw):
        prompts.append(msg)
        return _answer(True)

    with patch("iitgpu.admin.get_node_stats", return_value=_stats()), \
         patch("iitgpu.admin.pod_occupancy", return_value=[None] * 4), \
         patch("questionary.confirm", side_effect=fake_confirm), \
         patch("questionary.text", return_value=_answer("5")), \
         patch("iitgpu.admin.resize_pod_count",
               return_value=(True, "done")) as rz:
        admin._pods_menu(None, node="iit-MS-7E06")

    # 5 pods out of 32 CPUs / (62000MB//1024 - 2 headroom) = 6 CPU / 11 GB
    assert any("6 CPU" in p and "11 GB" in p for p in prompts), prompts
    rz.assert_called_once()
    assert rz.call_args.args[0] == 5
    # M2: the screen's own node must be forwarded, not silently defaulted
    assert rz.call_args.kwargs.get("node") == "iit-MS-7E06"


def test_pods_menu_declining_the_sizing_confirm_does_not_resize():
    answers = iter([True, False])  # "Resize?" yes, "Apply this resize?" no

    with patch("iitgpu.admin.get_node_stats", return_value=_stats()), \
         patch("iitgpu.admin.pod_occupancy", return_value=[None] * 4), \
         patch("questionary.confirm",
               side_effect=lambda *a, **k: _answer(next(answers))), \
         patch("questionary.text", return_value=_answer("5")), \
         patch("iitgpu.admin.resize_pod_count") as rz:
        admin._pods_menu(None)
    rz.assert_not_called()


def test_pods_menu_rejects_a_degenerate_n_before_confirming(capsys):
    with patch("iitgpu.admin.get_node_stats", return_value=_stats()), \
         patch("iitgpu.admin.pod_occupancy", return_value=[None] * 4), \
         patch("questionary.confirm", return_value=_answer(True)), \
         patch("questionary.text", return_value=_answer("40")), \
         patch("iitgpu.admin.resize_pod_count") as rz:
        admin._pods_menu(None)
    rz.assert_not_called()
    assert "0 CPU" in capsys.readouterr().out


def test_resume_node_uses_sudo_n():
    with patch("subprocess.run", return_value=_proc()) as r:
        ok, _ = admin.resume_node("node1")
    cmd = r.call_args[0][0]
    assert cmd[:3] == ["sudo", "-n", "scontrol"]
    assert "state=resume" in cmd
    assert ok


# ── Users ─────────────────────────────────────────────────────────────────────

def test_provision_user_uses_full_path_and_sudo_n():
    with patch("subprocess.run", return_value=_proc(out="done")) as r:
        ok, _ = admin.provision_user("alice", admin=True)
    cmd = r.call_args_list[0][0][0]
    assert cmd[0] == "sudo"
    assert cmd[1] == "-n"
    assert cmd[2] == "/usr/local/bin/iit-gpu-adduser"
    assert "alice" in cmd
    assert "--admin" in cmd
    assert ok


def test_provision_user_sets_password_via_chpasswd():
    with patch("subprocess.run", return_value=_proc(out="done")) as r, \
         patch("iitgpu.admin.daemonclient.create_user", return_value=(True, "ok")), \
         patch("iitgpu.admin.auditclient.log"):
        ok, msg = admin.provision_user("alice", password="s3cr3t", email="a@b.com")
    assert ok
    cmds = [c[0][0] for c in r.call_args_list]
    assert any("iit-gpu-adduser" in " ".join(c) for c in cmds)
    chpasswd_call = next(c for c in r.call_args_list if "chpasswd" in c[0][0])
    assert "alice:s3cr3t\n" in (chpasswd_call[1].get("input") or "")


def test_provision_user_welcome_sent_with_password():
    """send_welcome must receive the initial password so it can be emailed to the user."""
    with patch("subprocess.run", return_value=_proc(out="done")), \
         patch("iitgpu.admin.daemonclient.create_user", return_value=(True, "ok")), \
         patch("iitgpu.admin.auditclient.log"), \
         patch("iitgpu.mailer.send_welcome", return_value=(True, "sent")) as mock_welcome:
        admin.provision_user("alice", password="s3cr3t",
                             email="alice@iit.lk", full_name="Alice")
        import time; time.sleep(0.05)
    assert mock_welcome.called
    args, kwargs = mock_welcome.call_args
    assert "s3cr3t" in str(args) or "s3cr3t" in str(kwargs), \
        "send_welcome must receive the initial password"


def test_provision_user_must_change_pw_flag_set_when_password_given():
    """create_user must be called with must_change_pw=True when a password is set."""
    with patch("subprocess.run", return_value=_proc(out="done")), \
         patch("iitgpu.admin.daemonclient.create_user", return_value=(True, "ok")) as mock_cu, \
         patch("iitgpu.admin.auditclient.log"):
        admin.provision_user("alice", password="s3cr3t",
                             email="alice@iit.lk", role="tool")
    assert mock_cu.called
    _, kwargs = mock_cu.call_args
    assert kwargs.get("must_change_pw") is True


def test_provision_user_must_change_pw_false_when_no_password():
    """must_change_pw must be False when no password is provided at provision time."""
    with patch("subprocess.run", return_value=_proc(out="done")), \
         patch("iitgpu.admin.daemonclient.create_user", return_value=(True, "ok")) as mock_cu, \
         patch("iitgpu.admin.auditclient.log"):
        admin.provision_user("alice", password="", email="alice@iit.lk", role="tool")
    assert mock_cu.called
    _, kwargs = mock_cu.call_args
    assert kwargs.get("must_change_pw") is False


def test_provision_user_skips_password_on_adduser_failure():
    with patch("subprocess.run", return_value=_proc(rc=1, err="adduser failed")) as r:
        ok, msg = admin.provision_user("alice", password="s3cr3t")
    assert not ok
    assert r.call_count == 1  # chpasswd never called


def test_set_user_password_pipes_to_chpasswd():
    with patch("subprocess.run", return_value=_proc()) as r:
        ok, _ = admin.set_user_password("bob", "pass123")
    assert ok
    cmd = r.call_args[0][0]
    assert cmd == ["sudo", "-n", "chpasswd"]
    assert r.call_args[1].get("input") == "bob:pass123\n"


def test_offboard_user_uses_full_path_and_sudo_n():
    with patch("subprocess.run", return_value=_proc(out="done")) as r:
        ok, _ = admin.offboard_user("bob", purge=True)
    cmd = r.call_args[0][0]
    assert cmd[0] == "sudo"
    assert cmd[1] == "-n"
    assert cmd[2] == "/usr/local/bin/iit-gpu-deluser"
    assert "--purge-data" in cmd
    assert ok


def test_run_always_uses_devnull_stdin():
    """_run passes stdin=DEVNULL unless stdin_data is given."""
    import subprocess as sp
    with patch("subprocess.run", return_value=_proc()) as r:
        admin._run(["echo", "hi"])
    kwargs = r.call_args[1]
    assert kwargs["stdin"] == sp.DEVNULL


def test_run_uses_pipe_when_stdin_data_given():
    import subprocess as sp
    with patch("subprocess.run", return_value=_proc()) as r:
        admin._run(["cat"], stdin_data="hello\n")
    kwargs = r.call_args[1]
    assert kwargs["input"] == "hello\n"
    assert "stdin" not in kwargs


# ── Audit log ─────────────────────────────────────────────────────────────────

def test_read_audit_filters_by_action():
    evs_data = [{"ts": "2026-05-31T10:00:00+00:00", "user": "alice",
                 "action": "job_submit"}]
    with patch("iitgpu.admin.daemonclient.query_audit", return_value=evs_data):
        evs = admin.read_audit(action_filter="job_submit")
    assert len(evs) == 1 and evs[0]["user"] == "alice"


def test_read_audit_filters_by_user():
    evs_data = [{"ts": "2026-05-31T10:01:00+00:00", "user": "bob",
                 "action": "job_cancel"}]
    with patch("iitgpu.admin.daemonclient.query_audit", return_value=evs_data):
        evs = admin.read_audit(user_filter="bob")
    assert len(evs) == 1 and evs[0]["action"] == "job_cancel"


# ── QOS ───────────────────────────────────────────────────────────────────────

_QOS_OUTPUT = "normal|08:00:00|gres/gpu=10,gres/shard=4|0\nlong|7-00:00:00||0\n"


def test_list_qos_parses_sacctmgr_output():
    with patch("subprocess.run", return_value=_proc(out=_QOS_OUTPUT)):
        rows = admin.list_qos()
    assert len(rows) == 2
    normal = rows[0]
    assert normal["name"] == "normal"
    assert normal["max_wall"] == "08:00:00"
    assert normal["max_gpu"] == "10"
    assert normal["max_pods"] == "4"
    assert normal["priority"] == "0"
    long_qos = rows[1]
    assert long_qos["max_wall"] == "7-00:00:00"
    assert long_qos["max_gpu"] == "unlimited"
    assert long_qos["max_pods"] == "unlimited"


def test_set_qos_maxwall_calls_sacctmgr():
    with patch("subprocess.run", return_value=_proc(out="Modified")) as r:
        ok, _ = admin.set_qos_maxwall("normal", "12:00:00")
    cmd = r.call_args[0][0]
    assert cmd[:3] == ["sudo", "-n", "sacctmgr"]
    assert "modify" in cmd and "qos" in cmd and "normal" in cmd
    assert "MaxWall=12:00:00" in cmd
    assert ok


def test_set_qos_maxwall_empty_clears_limit():
    with patch("subprocess.run", return_value=_proc(out="Modified")) as r:
        ok, _ = admin.set_qos_maxwall("normal", "")
    cmd = r.call_args[0][0]
    assert "MaxWall=" in cmd
    assert ok


def test_set_qos_maxgpu_sets_tres():
    with patch("subprocess.run", return_value=_proc(out="Modified")) as r:
        ok, _ = admin.set_qos_maxgpu("normal", 2)
    write_cmd = r.call_args[0][0]
    assert write_cmd[:3] == ["sudo", "-n", "sacctmgr"]
    assert "MaxTRESPerUser=gres/gpu=2" in write_cmd
    assert ok


def test_set_qos_maxgpu_none_clears_limit():
    with patch("subprocess.run", return_value=_proc(out="Modified")) as r:
        ok, _ = admin.set_qos_maxgpu("long", None)
    write_cmd = r.call_args[0][0]
    assert "MaxTRESPerUser=" in write_cmd
    assert ok


def test_set_qos_maxgpu_preserves_existing_shard_cap():
    """Regression test for the clobber bug: setting the GPU cap must not wipe
    an existing shard (pod) cap already present in MaxTRESPerUser."""
    read_proc = _proc(out="gres/gpu=10,gres/shard=4\n")
    write_proc = _proc(out="Modified")
    with patch("subprocess.run", side_effect=[read_proc, write_proc]) as r:
        ok, _ = admin.set_qos_maxgpu("normal", 2)
    assert ok
    write_cmd = r.call_args_list[-1][0][0]
    tres_arg = next(a for a in write_cmd if a.startswith("MaxTRESPerUser="))
    assert "gres/gpu=2" in tres_arg
    assert "gres/shard=4" in tres_arg


def test_set_qos_max_pods_per_user_preserves_existing_gpu_cap():
    read_proc = _proc(out="gres/gpu=10,gres/shard=4\n")
    write_proc = _proc(out="Modified")
    with patch("subprocess.run", side_effect=[read_proc, write_proc]) as r:
        ok, _ = admin.set_qos_max_pods_per_user("normal", 2)
    assert ok
    write_cmd = r.call_args_list[-1][0][0]
    tres_arg = next(a for a in write_cmd if a.startswith("MaxTRESPerUser="))
    assert "gres/shard=2" in tres_arg
    assert "gres/gpu=10" in tres_arg


def test_set_qos_max_pods_per_user_none_clears_only_shard():
    read_proc = _proc(out="gres/gpu=10,gres/shard=4\n")
    write_proc = _proc(out="Modified")
    with patch("subprocess.run", side_effect=[read_proc, write_proc]) as r:
        ok, _ = admin.set_qos_max_pods_per_user("normal", None)
    assert ok
    write_cmd = r.call_args_list[-1][0][0]
    tres_arg = next(a for a in write_cmd if a.startswith("MaxTRESPerUser="))
    assert "gres/shard" not in tres_arg
    assert "gres/gpu=10" in tres_arg


def test_set_qos_priority():
    with patch("subprocess.run", return_value=_proc(out="Modified")) as r:
        ok, _ = admin.set_qos_priority("normal", 100)
    cmd = r.call_args[0][0]
    assert "Priority=100" in cmd
    assert ok


# ── All-user job history ──────────────────────────────────────────────────────

def test_filtered_history_accepts_all_users_flag():
    """filtered_history must accept (search_root, all_users=True) without TypeError."""
    from iitgpu.slurm import filtered_history, QueueEntry
    fake = [QueueEntry("10", "alice", "COMPLETED", "gpu", "1:00", 1)]
    with patch("iitgpu.slurm._sacct_history_user", return_value=fake):
        rows = filtered_history("/shared/jobs", all_users=True, days=30)
    assert any(r.job_id == "10" for r in rows)


# ── list_gpuusers ─────────────────────────────────────────────────────────────

def test_list_gpuusers_returns_sorted():
    fake_grp = MagicMock(gr_mem=["bob", "alice"], gr_gid=1500)
    with patch("grp.getgrnam", return_value=fake_grp), \
         patch("pwd.getpwall", return_value=[]):
        users = admin.list_gpuusers()
    assert users == ["alice", "bob"]


# ── Disk usage ────────────────────────────────────────────────────────────────

def test_disk_usage_by_user_sums_per_user(tmp_path):
    alice = tmp_path / "alice" / "job1"
    alice.mkdir(parents=True)
    (alice / "out.log").write_bytes(b"x" * 1024)
    (alice / "err.log").write_bytes(b"y" * 512)

    bob = tmp_path / "bob" / "job1"
    bob.mkdir(parents=True)
    (bob / "out.log").write_bytes(b"z" * 2048)

    rows = admin.disk_usage_by_user(str(tmp_path))
    by_user = {r["user"]: r for r in rows}

    assert by_user["alice"]["bytes"] == 1536
    assert by_user["bob"]["bytes"] == 2048


def test_disk_usage_by_user_sorted_descending(tmp_path):
    for user, size in [("alice", 100), ("charlie", 5000), ("bob", 300)]:
        d = tmp_path / user / "j"
        d.mkdir(parents=True)
        (d / "f").write_bytes(b"x" * size)

    rows = admin.disk_usage_by_user(str(tmp_path))
    assert rows[0]["user"] == "charlie"
    assert rows[-1]["user"] == "alice"


def test_disk_usage_by_user_empty_dir(tmp_path):
    assert admin.disk_usage_by_user(str(tmp_path)) == []


def test_disk_usage_by_user_nonexistent_root(tmp_path):
    assert admin.disk_usage_by_user(str(tmp_path / "no_such_dir")) == []


def test_disk_usage_human_readable_units(tmp_path):
    d = tmp_path / "alice" / "j"
    d.mkdir(parents=True)
    (d / "f").write_bytes(b"x" * (2 * 1024 * 1024))  # 2 MB

    rows = admin.disk_usage_by_user(str(tmp_path))
    assert rows[0]["human"] == "2.0 MB"

# ── M4: username validation ───────────────────────────────────────────────────

def test_valid_username_accepts_normal():
    assert admin.valid_username("alice")
    assert admin.valid_username("bob_2")
    assert admin.valid_username("user-name")
    assert admin.valid_username("_svc")


def test_valid_username_rejects_dangerous():
    assert not admin.valid_username("../etc")
    assert not admin.valid_username("a/b")
    assert not admin.valid_username("has space")
    assert not admin.valid_username("Has.Dot")
    assert not admin.valid_username("UPPER")
    assert not admin.valid_username("")
    assert not admin.valid_username("9startsdigit")
    assert not admin.valid_username("x" * 40)


def test_provision_user_rejects_bad_username():
    with patch("subprocess.run") as r:
        ok, msg = admin.provision_user("../../etc/passwd", password="x", email="a@b.com")
    assert not ok
    assert "invalid username" in msg.lower()
    r.assert_not_called()   # never reaches sudo adduser


def test_offboard_user_rejects_bad_username():
    with patch("subprocess.run") as r:
        ok, msg = admin.offboard_user("../../etc")
    assert not ok
    r.assert_not_called()


# ── Mail service kill-switch ────────────────────────────────────────────────────

def test_mail_kill_switch_toggle(tmp_path, monkeypatch):
    from iitgpu import mailer
    flag = tmp_path / ".mail-disabled"
    monkeypatch.setattr(mailer, "_mail_flag_path", lambda: str(flag))
    monkeypatch.setattr("iitgpu.admin.auditclient.log", lambda *a, **k: None)

    assert admin.is_mail_disabled() is False
    good, _ = admin.set_mail_disabled("tester")
    assert good and flag.exists() and admin.is_mail_disabled() is True
    good, _ = admin.enable_mail("tester")
    assert good and not flag.exists() and admin.is_mail_disabled() is False


def test_enable_mail_is_idempotent_when_already_on(tmp_path, monkeypatch):
    from iitgpu import mailer
    flag = tmp_path / ".mail-disabled"
    monkeypatch.setattr(mailer, "_mail_flag_path", lambda: str(flag))
    monkeypatch.setattr("iitgpu.admin.auditclient.log", lambda *a, **k: None)
    good, _ = admin.enable_mail("tester")   # nothing to remove
    assert good and not flag.exists()


# ── Log in as user ─────────────────────────────────────────────────────────────

def test_login_as_runs_sudo_iu_for_selected_user(monkeypatch):
    calls = {}
    monkeypatch.setattr(admin, "list_gpuusers", lambda: ["sanuth", "alice", "dahamadmin"])
    monkeypatch.setattr(admin.getpass, "getuser", lambda: "dahamadmin")
    monkeypatch.setattr("questionary.select",
                        lambda *a, **k: MagicMock(ask=lambda: "sanuth"))
    monkeypatch.setattr("questionary.press_any_key_to_continue",
                        lambda *a, **k: MagicMock(ask=lambda: None))
    monkeypatch.setattr(admin.auditclient, "log", lambda *a, **k: None)
    monkeypatch.setattr(admin.subprocess, "run",
                        lambda cmd, *a, **k: calls.setdefault("cmd", cmd))

    admin._login_as_menu(style=None)
    assert calls["cmd"] == ["sudo", "-H", "-u", "sanuth", "/usr/local/bin/iit-gpu-manager"]


def test_login_as_cancel_does_not_launch(monkeypatch):
    """Picking Back in the real select_menu yields target=None; the menu
    must return without ever launching sudo."""
    ran = {"n": 0}
    monkeypatch.setattr(admin, "list_gpuusers", lambda: ["sanuth"])
    monkeypatch.setattr(admin.getpass, "getuser", lambda: "dahamadmin")
    monkeypatch.setattr("iitgpu.ui.select_menu", lambda *a, **k: None)
    monkeypatch.setattr(admin.subprocess, "run",
                        lambda *a, **k: ran.__setitem__("n", ran["n"] + 1))
    admin._login_as_menu(style=None)
    assert ran["n"] == 0


def test_login_as_invalid_username_does_not_launch(monkeypatch):
    """Defense in depth: even if select_menu ever returned something that
    fails valid_username, _login_as_menu must bail out before launching
    sudo rather than passing it through."""
    ran = {"n": 0}
    monkeypatch.setattr(admin, "list_gpuusers", lambda: ["sanuth"])
    monkeypatch.setattr(admin.getpass, "getuser", lambda: "dahamadmin")
    monkeypatch.setattr("iitgpu.ui.select_menu", lambda *a, **k: "[cancel]")
    monkeypatch.setattr(admin.subprocess, "run",
                        lambda *a, **k: ran.__setitem__("n", ran["n"] + 1))
    admin._login_as_menu(style=None)
    assert ran["n"] == 0


def test_gen_password_length_and_charset():
    import string
    from iitgpu.admin import _gen_password
    pw = _gen_password()
    assert len(pw) == 8
    assert any(c in string.ascii_lowercase for c in pw)
    assert any(c in string.ascii_uppercase for c in pw)
    assert any(c in string.digits for c in pw)
    assert any(c in "!@#$%&*" for c in pw)


def test_gen_password_unique():
    from iitgpu.admin import _gen_password
    passwords = {_gen_password() for _ in range(50)}
    assert len(passwords) > 1
