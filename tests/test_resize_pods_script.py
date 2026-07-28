"""deploy/resize-pods.sh in --dry-run mode: parses/validates without
touching real system files or restarting anything. Full cross-node
execution is verified live (see Plan B Task 5), not in this unit test."""
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "deploy" / "resize-pods.sh"


def _run(args, env):
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True,
                          text=True, env=env, timeout=10)


def _fake_conf_dir(tmp_path):
    d = tmp_path / "slurm"
    d.mkdir()
    (d / "gres.conf").write_text(
        "Name=gpu File=/dev/nvidia0\nName=shard Count=4 File=/dev/nvidia0\n")
    (d / "slurm.conf").write_text(
        "SelectType=select/cons_tres\n"
        "NodeName=iit-MS-7E06 NodeAddr=192.168.122.1 CPUs=32 RealMemory=62000 "
        "Gres=gpu:1,shard:4 State=UNKNOWN\n"
        "PartitionName=gpu Nodes=iit-MS-7E06 Default=YES MaxTime=1-00:00:00 State=UP\n")
    return d


def test_dry_run_rewrites_gres_conf_count_in_place(tmp_path, monkeypatch):
    conf = _fake_conf_dir(tmp_path)
    env = {**dict(**{}), "SLURM_CONF_DIR": str(conf), "PATH": "/usr/bin:/bin"}
    result = _run(["5", "--dry-run"], env)
    assert result.returncode == 0, result.stderr
    assert "Name=shard Count=5" in (conf / "gres.conf").read_text()
    assert "Gres=gpu:1,shard:5" in (conf / "slurm.conf").read_text()


def test_dry_run_backs_up_originals(tmp_path):
    conf = _fake_conf_dir(tmp_path)
    env = {"SLURM_CONF_DIR": str(conf), "PATH": "/usr/bin:/bin"}
    _run(["5", "--dry-run"], env)
    backups = list(conf.glob("gres.conf.bak.*"))
    assert len(backups) == 1


def test_rejects_non_positive_pod_count(tmp_path):
    conf = _fake_conf_dir(tmp_path)
    env = {"SLURM_CONF_DIR": str(conf), "PATH": "/usr/bin:/bin"}
    result = _run(["0", "--dry-run"], env)
    assert result.returncode != 0
    assert "positive" in (result.stderr + result.stdout).lower()


def test_missing_conf_dir_fails_loudly(tmp_path):
    env = {"SLURM_CONF_DIR": str(tmp_path / "nope"), "PATH": "/usr/bin:/bin"}
    result = _run(["5", "--dry-run"], env)
    assert result.returncode != 0


def test_lock_file_default_is_writable_by_slurmadmin():
    """The script runs as slurmadmin, which cannot create files in /var/run
    (root:root 0755) -- a default lockfile there aborts the whole resize under
    `set -e` before it does anything (final-review C3)."""
    src = SCRIPT.read_text()
    line = [ln for ln in src.splitlines() if ln.startswith("LOCK_FILE=")]
    assert len(line) == 1, line
    assert "/var/run" not in line[0] and "/run/" not in line[0]
    assert "/tmp/" in line[0]


# ── Full (non-dry-run) cross-node paths, driven through stub ssh/scp/sudo ────
#
# The dry-run tests above stop before any GPU-host work, which is exactly why
# the rollback bugs found in final review were invisible. These tests put stub
# `ssh`/`scp`/`sudo`/`squeue`/`id` binaries first on PATH and run the REAL
# non-dry-run path end to end, so the sync/verify/rollback control flow is
# actually exercised.

_STUBS = {
    "id": '#!/bin/bash\necho slurmadmin\n',
    "squeue": '#!/bin/bash\nexit 0\n',
    "sudo": '#!/bin/bash\necho "sudo $*" >> "$RESIZE_TEST_LOG"\nexit 0\n',
    "scp": (
        '#!/bin/bash\n'
        'echo "scp $*" >> "$RESIZE_TEST_LOG"\n'
        'if [ -n "${FAIL_SCP_MATCH:-}" ] && [[ "$*" == *"$FAIL_SCP_MATCH"* ]]; then\n'
        '  echo "scp: no such remote directory" >&2; exit 1\n'
        'fi\n'
        'exit 0\n'
    ),
    "ssh": (
        '#!/bin/bash\n'
        'echo "ssh $*" >> "$RESIZE_TEST_LOG"\n'
        'if [ -n "${FAIL_SSH_MATCH:-}" ] && [[ "$*" == *"$FAIL_SSH_MATCH"* ]]; then\n'
        '  echo "ssh: remote command failed" >&2; exit 255\n'
        'fi\n'
        'if [[ "$*" == *"slurmd -G"* ]]; then\n'
        '  echo "Gres Name=shard Type=(null) Count=${REPORTED_SHARDS:-0} shard:${REPORTED_SHARDS:-0}"\n'
        'fi\n'
        'exit 0\n'
    ),
}


def _stub_bin(tmp_path):
    d = tmp_path / "fakebin"
    d.mkdir()
    for name, body in _STUBS.items():
        p = d / name
        p.write_text(body)
        p.chmod(0o755)
    return d


def _live_env(tmp_path, conf, **extra):
    log = tmp_path / "calls.log"
    log.write_text("")
    env = {
        "PATH": f"{_stub_bin(tmp_path)}:/usr/bin:/bin",
        "SLURM_CONF_DIR": str(conf),
        "RESIZE_LOCK_FILE": str(tmp_path / "resize.lock"),
        "GPU_HOST_SSH": "stub@gpuhost",
        "NODE_NAME": "iit-MS-7E06",
        "RESIZE_TEST_LOG": str(log),
        "REPORTED_SHARDS": "5",
    }
    env.update(extra)
    return env, log


def test_live_path_creates_the_remote_staging_dir_before_scp(tmp_path):
    conf = _fake_conf_dir(tmp_path)
    env, log = _live_env(tmp_path, conf)
    result = _run(["5"], env)
    assert result.returncode == 0, result.stderr
    assert "resize applied" in result.stdout
    calls = log.read_text()
    mkdir_i = calls.index("mkdir -p '/tmp/slurm-deck-resize-")
    scp_i = calls.index("scp ")
    assert mkdir_i < scp_i, calls  # dir must exist before scp targets it


def test_verify_mismatch_rolls_back_the_gpu_host_too(tmp_path):
    """The rollback must create ITS OWN remote staging dir -- before the fix it
    scp'd into a directory nothing ever created, so the GPU host silently kept
    the new config while the script claimed a clean rollback (C2)."""
    conf = _fake_conf_dir(tmp_path)
    env, log = _live_env(tmp_path, conf, REPORTED_SHARDS="4")  # wrong count
    result = _run(["5"], env)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "rollback performed" in result.stderr
    calls = log.read_text()
    assert "mkdir -p '/tmp/slurm-deck-resize-rollback-" in calls
    assert "slurm-deck-resize-rollback-" in calls.split("mkdir -p '/tmp/slurm-deck-resize-rollback-")[1]
    # local config really went back to the pre-resize count
    assert "Name=shard Count=4" in (conf / "gres.conf").read_text()
    assert "Gres=gpu:1,shard:4" in (conf / "slurm.conf").read_text()


def test_gpu_sync_failure_routes_into_the_same_rollback(tmp_path):
    """A hard ssh/scp failure in the MAIN path used to abort via `set -e`
    before the rollback/resume block ever ran, leaving the login node rewritten
    and the GPU host untouched (I3)."""
    conf = _fake_conf_dir(tmp_path)
    env, log = _live_env(tmp_path, conf, FAIL_SCP_MATCH="/tmp/slurm-deck-resize-2")
    result = _run(["5"], env)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "GPU-host sync/restart failed" in result.stderr
    assert "rollback performed" in result.stderr
    assert "resize applied" not in result.stdout
    assert "Name=shard Count=4" in (conf / "gres.conf").read_text()
    calls = log.read_text()
    assert "/tmp/slurm-deck-resize-rollback-" in calls
    assert "state=resume" in calls.lower()


def test_failed_rollback_is_reported_not_silently_swallowed(tmp_path):
    """If the rollback's own push to the GPU host fails, the script must NOT
    claim a clean rollback -- that is the half-applied state the whole
    mechanism exists to surface."""
    conf = _fake_conf_dir(tmp_path)
    env, log = _live_env(tmp_path, conf, REPORTED_SHARDS="4",
                         FAIL_SCP_MATCH="slurm-deck-resize-rollback-")
    result = _run(["5"], env)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "CRITICAL" in result.stderr
    assert "rollback performed" not in result.stderr
