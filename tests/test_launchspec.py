"""Launch flow pure logic: sizes, availability wording, spec derivation."""
from pathlib import Path

import pytest

from iitgpu.launchspec import (
    LaunchSpec, apply_pods, availability_line, default_spec,
    from_rerun, from_template, pod_availability, pod_label,
    recent_scripts, to_job_spec,
)
from iitgpu.slurm import NodeStats

NODE_CPUS = 32
NODE_MEM_GB = 60


def _stats(free: int, shard_total: int = 4) -> NodeStats:
    return NodeStats(state="MIXED", cpu_load=0.0, cpu_total=32, cpu_alloc=0,
                     mem_total_mb=62000, mem_alloc_mb=0,
                     gpu_total=1, gpu_alloc=0,
                     shard_total=shard_total,
                     shard_alloc=shard_total - free)


def test_apply_pods_sizes_from_live_stats():
    ls = LaunchSpec(intent="batch")
    apply_pods(ls, 2, _stats(free=4))
    assert (ls.gpu_shards, ls.cpus, ls.mem_gb) == (2, 16, 28)


def test_apply_pods_clamps_to_pod_count():
    ls = LaunchSpec(intent="batch")
    apply_pods(ls, 9, _stats(free=4))
    assert ls.gpu_shards == 4


def test_apply_pods_keeps_sane_defaults_when_the_pod_count_is_unknown():
    """C1: with no stats, resources_for() answers pods.pod_resources(None)'s
    degenerate 1 CPU / 1 GB. Writing that into the LaunchSpec sized every job
    the wizard launched without a reachable cluster at 1/1. The sizing fields
    must be left alone instead; only gpu_shards is recorded."""
    ls = LaunchSpec(intent="batch")           # dataclass defaults: 8 CPU / 14 GB
    apply_pods(ls, 1, None)
    assert (ls.cpus, ls.mem_gb) == (8, 14)
    assert ls.gpu_shards == 1

    # Same rule for a node that answers but reports no shards at all.
    ls2 = LaunchSpec(intent="batch", cpus=16, mem_gb=60)
    apply_pods(ls2, 2, _stats(free=0, shard_total=0))
    assert (ls2.cpus, ls2.mem_gb) == (16, 60)


def test_default_spec_without_stats_is_pod_sized_not_degenerate():
    """The exact C1 reproduction: wizard.py's main intent path called
    default_spec(intent) with no stats and got a 1 CPU / 1 GB notebook."""
    ls = default_spec("notebook")
    assert (ls.cpus, ls.mem_gb) != (1, 1)
    assert (ls.gpu_shards, ls.cpus, ls.mem_gb, ls.time_limit) == (1, 8, 14, "06:00:00")


def test_default_spec_with_live_stats_still_uses_the_live_numbers():
    """The fallback must not shadow a real reading when one is available."""
    ls = default_spec("batch", _stats(free=4, shard_total=8))
    assert (ls.cpus, ls.mem_gb) == (4, 7)     # 32//8 CPU, (60-2)//8 GB


def test_a_full_cards_worth_of_one_pod_jobs_fits_the_node():
    stats = _stats(free=4)
    ls = LaunchSpec(intent="notebook")
    apply_pods(ls, 1, stats)
    n = stats.shard_total // ls.gpu_shards
    assert ls.cpus * n <= NODE_CPUS and ls.mem_gb * n <= NODE_MEM_GB


def test_default_specs_per_intent():
    stats = _stats(free=4)
    assert default_spec("notebook", stats).time_limit == "06:00:00"
    assert default_spec("batch", stats).time_limit == "04:00:00"
    sh = default_spec("shell", stats)
    assert (sh.gpu_shards, sh.time_limit) == (1, "02:00:00")


def test_apply_pods_then_change_pods_updates_sizing():
    ls = default_spec("batch", _stats(free=4))
    apply_pods(ls, 4, _stats(free=4))
    assert (ls.gpu_shards, ls.cpus, ls.mem_gb) == (4, 32, 56)
    apply_pods(ls, 1, _stats(free=4))
    assert (ls.gpu_shards, ls.cpus, ls.mem_gb) == (1, 8, 14)


def test_pod_label_shows_fraction_and_resources():
    ls = LaunchSpec(intent="batch")
    apply_pods(ls, 2, _stats(free=4))
    label = pod_label(ls, _stats(free=4))
    assert "2 pod" in label and "2/4 GPU" in label and "16 CPU" in label and "28 GB" in label


def test_pod_label_says_whole_gpu_at_full_pod_count():
    ls = LaunchSpec(intent="batch")
    apply_pods(ls, 4, _stats(free=4))
    assert "whole GPU" in pod_label(ls, _stats(free=4))


def test_pod_label_says_share_unknown_without_a_live_pod_count():
    """I1: pod_count() floors at 1, so a spec with 1 pod and no stats used to
    render "1 pod - whole GPU". Unknown must read as unknown."""
    ls = LaunchSpec(intent="batch")
    for stats in (None, _stats(free=0, shard_total=0)):
        label = pod_label(ls, stats)
        assert "GPU share unknown" in label
        assert "whole GPU" not in label
        assert "8 CPU" in label and "14 GB" in label


def test_pod_availability_reports_starts_now_or_queues():
    stats = _stats(free=2)
    assert "starts now" in pod_availability(2, stats)
    assert "will queue" in pod_availability(3, stats)


def test_availability_line_unknown_without_stats():
    assert availability_line(None) == "GPU availability unknown"


def test_availability_line_reports_free_pods():
    assert availability_line(_stats(free=3)) == "GPU now: 3/4 pods free"


def test_to_job_spec_maps_fields():
    ls = default_spec("batch")
    ls.conda_env = "/shared/envs/data-science"; ls.data_path = "/shared/data/cifar10"
    ls.args = "--epochs 5"; ls.array = "0-3"; ls.dependency = "afterok:12"
    js = to_job_spec(ls, user="u", partition="gpu", job_name="train",
                     task_type="custom", run_command="python3 x.py")
    assert js.gpu_shards == ls.gpu_shards and js.cpus == ls.cpus and js.mem_gb == ls.mem_gb
    assert js.conda_env == ls.conda_env and js.data_path == ls.data_path
    assert js.array == "0-3" and js.dependency == "afterok:12"
    assert js.time_limit == "04:00:00" and js.user == "u"


def test_recent_scripts_excludes_scripts_that_no_longer_exist(tmp_path):
    """A recent entry pointing at a deleted file would 404 the user at intake."""
    jobs = tmp_path / "jobs" / "u" / "a_1"
    jobs.mkdir(parents=True)
    (jobs / "job.sbatch").write_text("#!/bin/bash\npython3 /data/proj/deleted.py\n")
    assert recent_scripts(str(tmp_path / "jobs"), "u", limit=5) == []


def test_recent_scripts_returns_existing_newest_first(tmp_path):
    proj = tmp_path / "proj"; proj.mkdir()
    s1 = proj / "one.py"; s2 = proj / "two.ipynb"
    s1.write_text("x"); s2.write_text("y")
    jobs = tmp_path / "jobs" / "u"
    import os, time
    for i, script in enumerate((s1, s2)):
        d = jobs / f"run_{i}"; d.mkdir(parents=True)
        (d / "job.sbatch").write_text(f"python3 {script}\n")
        t = time.time() + i
        os.utime(d, (t, t))
    got = recent_scripts(str(tmp_path / "jobs"), "u", limit=5)
    assert got == [str(s2), str(s1)]


def test_from_template_maps_task_type_and_nearest_size():
    ls = from_template({"task_type": "notebook", "gpu_shards": 1, "cpus": 8,
                        "mem_gb": 14, "time_limit": "08:00:00",
                        "conda_env": "/shared/envs/data-science"})
    assert ls.intent == "notebook"
    assert ls.time_limit == "08:00:00" and ls.conda_env.endswith("data-science")
    ls2 = from_template({"task_type": "train", "gpu_shards": 4, "cpus": 16, "mem_gb": 60})
    assert ls2.intent == "batch" and ls2.gpu_shards == 4
    ls3 = from_template({"task_type": "custom", "gpu_shards": 1, "cpus": 6, "mem_gb": 20})
    assert ls3.cpus == 6


def test_from_rerun_uses_parsed_resources():
    ls = from_rerun({"gpu_shards": 1, "cpus": 4, "mem_gb": 8,
                     "time_limit": "01:00:00", "array": "0-9"}, script="/p/s.py")
    assert ls.intent == "batch" and ls.script == "/p/s.py"
    assert ls.cpus == 4 and ls.array == "0-9"


def test_from_rerun_carries_environment_data_and_args():
    """Re-run has to mean re-run. Carrying the sizing but dropping the env, the
    data path and the arguments produces a job that looks like the original in
    the queue and does something else entirely."""
    ls = from_rerun({"gpu_shards": 4, "cpus": 16, "mem_gb": 60,
                     "conda_env": "/shared/envs/pytorch-cifar",
                     "data_path": "/shared/users/alice/data",
                     "extra_args": "--epochs 10 --lr 3e-4"},
                    script="/shared/users/alice/train.py")
    assert ls.conda_env == "/shared/envs/pytorch-cifar"
    assert ls.env_kind == "conda"
    assert ls.data_path == "/shared/users/alice/data"
    assert ls.args == "--epochs 10 --lr 3e-4"


def test_from_rerun_carries_a_container_image():
    ls = from_rerun({"container_image": "/shared/images/llm.sif"}, script="/p/s.py")
    assert ls.container_image == "/shared/images/llm.sif"
    assert ls.env_kind == "container"
    assert ls.conda_env == ""


def test_from_template_maps_an_interactive_template_to_a_shell():
    """A saved shell allocation must load as a shell. Loaded as batch it becomes
    a job with no script — the hub would refuse to launch it, and the user would
    have no idea why."""
    ls = from_template({"task_type": "interactive", "gpu_shards": 1,
                        "cpus": 8, "mem_gb": 14, "time_limit": "02:00:00"})
    assert ls.intent == "shell"
    assert ls.time_limit == "02:00:00"
