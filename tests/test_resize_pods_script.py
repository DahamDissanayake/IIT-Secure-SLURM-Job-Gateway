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
