"""Launch flow pure logic: sizes, availability wording, spec derivation."""
from pathlib import Path

import pytest

from iitgpu.jobs import SHARDS_PER_GPU
from iitgpu.launchspec import (
    SIZES, LaunchSpec, apply_size, availability_line, default_spec,
    from_rerun, from_template, recent_scripts, size_availability,
    size_label, size_name_for, to_job_spec,
)
from iitgpu.slurm import NodeStats

NODE_CPUS = 32
NODE_MEM_GB = 60


def _stats(free: int) -> NodeStats:
    return NodeStats(state="MIXED", cpu_load=0.0, cpu_total=32, cpu_alloc=0,
                     mem_total_mb=62000, mem_alloc_mb=0,
                     gpu_total=1, gpu_alloc=0,
                     shard_total=SHARDS_PER_GPU,
                     shard_alloc=SHARDS_PER_GPU - free)


def test_sizes_table_matches_spec():
    assert SIZES["small"].gpu_shards == 1 and SIZES["small"].cpus == 4 and SIZES["small"].mem_gb == 8
    assert SIZES["standard"].cpus == 8 and SIZES["standard"].mem_gb == 14
    assert SIZES["whole"].gpu_shards == SHARDS_PER_GPU and SIZES["whole"].mem_gb == 60


def test_a_full_cards_worth_of_standard_jobs_fits_the_node():
    s = SIZES["standard"]
    n = SHARDS_PER_GPU // s.gpu_shards
    assert s.cpus * n <= NODE_CPUS and s.mem_gb * n <= NODE_MEM_GB


def test_default_specs_per_intent():
    assert default_spec("notebook").time_limit == "06:00:00"
    assert default_spec("batch").time_limit == "04:00:00"
    sh = default_spec("shell")
    assert (sh.gpu_shards, sh.cpus, sh.mem_gb, sh.time_limit) == (1, 4, 8, "02:00:00")


def test_apply_size_and_roundtrip_name():
    ls = default_spec("batch")
    apply_size(ls, "whole")
    assert (ls.gpu_shards, ls.cpus, ls.mem_gb) == (SHARDS_PER_GPU, 16, 60)
    assert size_name_for(ls) == "whole"
    ls.cpus = 6
    assert size_name_for(ls) is None
    assert size_label(ls).startswith("Custom")


def test_availability_wording():
    assert availability_line(_stats(3)) == "GPU now: 3/4 slices free"
    assert availability_line(None) == "GPU availability unknown"
    assert size_availability(1, _stats(3)) == "— starts now (3 slices free)"
    assert size_availability(4, _stats(3)) == "— will queue (needs 4 free, 3 free)"
    assert size_availability(1, None) == "— availability unknown"


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
    assert ls.intent == "notebook" and size_name_for(ls) == "standard"
    assert ls.time_limit == "08:00:00" and ls.conda_env.endswith("data-science")
    ls2 = from_template({"task_type": "train", "gpu_shards": 4, "cpus": 16, "mem_gb": 60})
    assert ls2.intent == "batch" and size_name_for(ls2) == "whole"
    ls3 = from_template({"task_type": "custom", "gpu_shards": 1, "cpus": 6, "mem_gb": 20})
    assert size_name_for(ls3) is None and ls3.cpus == 6


def test_from_rerun_uses_parsed_resources():
    ls = from_rerun({"gpu_shards": 1, "cpus": 4, "mem_gb": 8,
                     "time_limit": "01:00:00", "array": "0-9"}, script="/p/s.py")
    assert ls.intent == "batch" and ls.script == "/p/s.py"
    assert ls.cpus == 4 and ls.array == "0-9"
