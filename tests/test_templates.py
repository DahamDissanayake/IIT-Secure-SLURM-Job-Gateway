# tests/test_templates.py
import os
import pytest


def _node(shard_total):
    from iitgpu.slurm import NodeStats
    return NodeStats(state="MIXED", cpu_load=0.0, cpu_total=32, cpu_alloc=0,
                     mem_total_mb=62000, mem_alloc_mb=0,
                     gpu_total=1, gpu_alloc=0,
                     shard_total=shard_total, shard_alloc=0)


def test_builtin_presets_gpu_count_matches_cluster(tmp_path, monkeypatch):
    """The presets fit today's cluster: nothing asks for more pods than exist.

    Note this checks the COMPARISON, not the presets — see the companion test
    below and the NOTE above `_BUILTIN_PRESETS`: the preset sizing is still a
    hardcoded 4 pods / 16 CPU / 60 GB pending Plan B, so on a cluster split into
    fewer pods this assertion is *meant* to fail. That is the point: it is a
    live check against `pod_count()`, not a restatement of the same literal.
    """
    monkeypatch.setenv("NFS_ROOT", str(tmp_path))
    from iitgpu.pods import pod_count
    from iitgpu.templates import _BUILTIN_PRESETS
    n = pod_count(_node(4))
    for name, preset in _BUILTIN_PRESETS.items():
        shards = preset["gpu_shards"]
        assert shards <= n, (
            f"Preset '{name}' requests {shards} GPU slices "
            f"but the cluster has {n}"
        )


def test_builtin_preset_gpu_check_is_live_derived_not_vacuous(tmp_path, monkeypatch):
    """Guards the test above from becoming a tautology.

    It derives n from a shard_total=4 fixture and asserts presets <= 4 — the
    same literal 4 the presets themselves hardcode, so it would still "pass" if
    the comparison were wired to a constant. Running the identical comparison
    against a 2-pod node must REJECT the 4-pod presets; if it doesn't, the check
    isn't reading the live pod count at all.
    """
    monkeypatch.setenv("NFS_ROOT", str(tmp_path))
    from iitgpu.pods import pod_count
    from iitgpu.templates import _BUILTIN_PRESETS

    def _too_big(stats):
        n = pod_count(stats)
        return [name for name, p in _BUILTIN_PRESETS.items() if p["gpu_shards"] > n]

    assert _too_big(_node(4)) == []
    assert pod_count(_node(2)) == 2, "pod_count must follow the fixture, not a constant"
    assert _too_big(_node(2)), (
        "a 2-pod cluster must reject the 4-pod presets — the comparison above "
        "is not reading the live pod count")


def test_builtin_presets_cpus_within_cluster_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("NFS_ROOT", str(tmp_path))
    from iitgpu.templates import _BUILTIN_PRESETS
    for name, preset in _BUILTIN_PRESETS.items():
        assert preset["cpus"] <= 16, f"Preset '{name}' requests {preset['cpus']} CPUs but cluster has 16"


def test_builtin_presets_mem_within_cluster_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("NFS_ROOT", str(tmp_path))
    from iitgpu.templates import _BUILTIN_PRESETS
    for name, preset in _BUILTIN_PRESETS.items():
        assert preset["mem_gb"] <= 60, f"Preset '{name}' requests {preset['mem_gb']}GB but cluster has ~60GB"


def test_builtin_preset_partition_is_gpu(tmp_path, monkeypatch):
    monkeypatch.setenv("NFS_ROOT", str(tmp_path))
    from iitgpu.templates import _BUILTIN_PRESETS
    for name, preset in _BUILTIN_PRESETS.items():
        assert preset["partition"] == "gpu", f"Preset '{name}' uses partition '{preset['partition']}' not 'gpu'"


# ── Saving a template actually works ─────────────────────────────────────────
# Regression: save_template() called make_shared_writable() without importing
# it. Every save raised NameError — and because the function only catches
# OSError, it escaped as a crash rather than the "Could not save template"
# path. Nothing exercised save_template end-to-end, so the whole feature was
# broken in the deployed tool while the suite stayed green. These tests run the
# real thing against a temp NFS root.

def _tmp_cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("NFS_ROOT", str(tmp_path))
    monkeypatch.setenv("IIT_SITE_ENV", "/nonexistent")
    from iitgpu.config import load_config
    return load_config()


def test_save_template_writes_a_file_and_returns_true(tmp_path, monkeypatch):
    from iitgpu.jobs import JobSpec
    from iitgpu.templates import _template_path, save_template

    cfg = _tmp_cfg(tmp_path, monkeypatch)
    monkeypatch.setattr("iitgpu.auditclient.log", lambda *a, **kw: None)
    spec = JobSpec(job_name="nightly", partition="gpu", gpu_shards=1, cpus=8,
                   mem_gb=14, time_limit="04:00:00",
                   run_command="python3 /shared/users/alice/train.py --lr 3e-4",
                   task_type="custom", conda_env="/shared/envs/pytorch")

    assert save_template(cfg, "nightly run", spec) is True
    assert _template_path(cfg, "nightly run").is_file()


def test_saved_template_round_trips_through_load(tmp_path, monkeypatch):
    """What pick_template hands the wizard must be the spec that went in."""
    import json
    from iitgpu.jobs import JobSpec
    from iitgpu.templates import _template_path, load_template, save_template

    cfg = _tmp_cfg(tmp_path, monkeypatch)
    monkeypatch.setattr("iitgpu.auditclient.log", lambda *a, **kw: None)
    spec = JobSpec(job_name="sweep", partition="gpu", gpu_shards=4, cpus=16,
                   mem_gb=60, time_limit="08:00:00",
                   run_command="python3 /shared/users/alice/sweep.py",
                   task_type="custom", data_path="/shared/users/alice/data",
                   array="0-9")
    assert save_template(cfg, "sweep", spec) is True

    # Valid JSON on disk (pick_template -> load_template parses it).
    raw = json.loads(_template_path(cfg, "sweep").read_text())
    assert raw["run_command"] == spec.run_command

    data = load_template(cfg, "sweep")
    assert data is not None
    for field in ("gpu_shards", "cpus", "mem_gb", "time_limit",
                  "run_command", "task_type", "data_path", "array"):
        assert data[field] == getattr(spec, field), field


def test_save_template_reports_failure_instead_of_crashing(tmp_path, monkeypatch):
    """An unwritable templates dir is an expected condition on a shared FS: it
    must come back as False, not an exception in the user's face."""
    from iitgpu.jobs import JobSpec
    from iitgpu.templates import save_template

    cfg = _tmp_cfg(tmp_path, monkeypatch)
    monkeypatch.setattr("iitgpu.auditclient.log", lambda *a, **kw: None)
    monkeypatch.setattr("pathlib.Path.write_text",
                        lambda *a, **kw: (_ for _ in ()).throw(OSError("read-only")))
    spec = JobSpec(job_name="j", partition="gpu", gpu_shards=1, cpus=4, mem_gb=8,
                   time_limit="01:00:00", run_command="python3 x.py")

    assert save_template(cfg, "doomed", spec) is False
