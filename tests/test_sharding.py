"""GPU sharding: several jobs share the one physical GPU.

Before sharding a job asked for --gres=gpu:1 and took the whole card, so a
second GPU job (e.g. a colleague's JupyterLab) could not start at all. Jobs now
request slices (--gres=shard:N) instead.
"""
import pytest

from iitgpu.jobs import (
    SHARDS_PER_GPU, JobSpec, TASK_DEFAULTS, gpu_share_note, gres_directive,
    render_sbatch, render_notebook_sbatch, resource_defaults,
)


def _spec(**kw):
    base = dict(
        job_name="j", partition="gpu", gpu_shards=1, cpus=4, mem_gb=8,
        time_limit="01:00:00", run_command="python x.py",
    )
    base.update(kw)
    return JobSpec(**base)


# ── GRES directive ────────────────────────────────────────────────────────────

def test_gres_directive_requests_shards_not_whole_gpu():
    assert gres_directive(1) == "shard:1"
    assert gres_directive(SHARDS_PER_GPU) == f"shard:{SHARDS_PER_GPU}"


def test_gres_directive_empty_when_no_gpu_wanted():
    assert gres_directive(0) == ""


# ── Rendered scripts ──────────────────────────────────────────────────────────

def test_rendered_sbatch_never_requests_a_whole_gpu(tmp_path):
    """--gres=gpu:N takes the device *and* all its slices — the exact blocking
    that sharding removes. No renderer may emit it."""
    script = render_sbatch(_spec(gpu_shards=1), str(tmp_path))
    assert "--gres=shard:1" in script
    assert "--gres=gpu:" not in script


def test_two_notebooks_each_take_one_slice(tmp_path):
    """Two JupyterLab jobs must fit on the card at once."""
    nb = resource_defaults("notebook")
    assert nb.gpu_shards == 1
    assert nb.gpu_shards * 2 <= SHARDS_PER_GPU, "two notebooks must fit together"


def test_notebook_sbatch_requests_a_slice(tmp_path):
    script = render_notebook_sbatch(
        _spec(gpu_shards=1, task_type="notebook"), str(tmp_path), port=8888)
    assert "--gres=shard:1" in script
    assert "--gres=gpu:" not in script


def test_cpu_only_job_omits_gres_entirely(tmp_path):
    """An empty --gres= would be rejected by sbatch, so the line must be absent."""
    script = render_sbatch(_spec(gpu_shards=0), str(tmp_path))
    assert "--gres" not in script


# ── Task defaults ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("task", ["notebook", "interactive", "test", "inference"])
def test_light_tasks_leave_room_for_others(task):
    assert resource_defaults(task).gpu_shards < SHARDS_PER_GPU


@pytest.mark.parametrize("task", ["train", "finetune"])
def test_training_still_gets_the_whole_card(task):
    assert resource_defaults(task).gpu_shards == SHARDS_PER_GPU


def test_no_task_default_over_subscribes_the_gpu():
    for name, d in TASK_DEFAULTS.items():
        assert 0 <= d.gpu_shards <= SHARDS_PER_GPU, name


# ── User-facing wording ───────────────────────────────────────────────────────

def test_gpu_share_note_explains_what_is_left():
    assert "no GPU" in gpu_share_note(0)
    assert "whole GPU" in gpu_share_note(SHARDS_PER_GPU)
    partial = gpu_share_note(1)
    assert f"1/{SHARDS_PER_GPU}" in partial and "left for others" in partial


# ── Validation ────────────────────────────────────────────────────────────────

def test_shard_request_is_not_measured_against_whole_gpu_limit(monkeypatch):
    """A site with one physical GPU may set MAX_GPUS=1; shard:4 is still legal."""
    monkeypatch.setenv("MAX_GPUS", "1")
    monkeypatch.setenv("MAX_GPU_SHARDS", "4")
    import importlib, iitgpu.validate as v
    importlib.reload(v)
    try:
        errors = v.validate_sbatch("#SBATCH --gres=shard:4\n", "alice")
        assert not any("GPU" in e for e in errors), errors
    finally:
        importlib.reload(v)


def test_shard_request_beyond_capacity_is_rejected(monkeypatch):
    monkeypatch.setenv("MAX_GPU_SHARDS", "4")
    import importlib, iitgpu.validate as v
    importlib.reload(v)
    try:
        errors = v.validate_sbatch("#SBATCH --gres=shard:9\n", "alice")
        assert any("slices" in e for e in errors), errors
    finally:
        importlib.reload(v)


# ── Slice sizing ──────────────────────────────────────────────────────────────
#
# Splitting the GPU is not enough on its own: if a one-slice job still asks for
# a third of the node's RAM, memory becomes the new bottleneck and the second
# notebook queues on "Resources" with the GPU almost idle. This is the bug that
# survived the first sharding attempt, so it is pinned here.

NODE_CPUS = 32
NODE_MEM_GB = 60      # RealMemory=62000 MB


@pytest.mark.parametrize("task", ["notebook", "interactive", "inference"])
def test_a_full_cards_worth_of_slice_jobs_fits_on_the_node(task):
    d = resource_defaults(task)
    concurrent = SHARDS_PER_GPU // d.gpu_shards
    assert d.cpus * concurrent <= NODE_CPUS, (
        f"{concurrent}x {task} needs {d.cpus * concurrent} CPUs, node has {NODE_CPUS}")
    assert d.mem_gb * concurrent <= NODE_MEM_GB, (
        f"{concurrent}x {task} needs {d.mem_gb * concurrent} GB RAM, "
        f"node has {NODE_MEM_GB}")


def test_two_notebooks_fit_side_by_side():
    """The originally reported failure: a second JupyterLab could not start."""
    d = resource_defaults("notebook")
    assert d.gpu_shards * 2 <= SHARDS_PER_GPU
    assert d.cpus * 2 <= NODE_CPUS
    assert d.mem_gb * 2 <= NODE_MEM_GB


def test_whole_card_tasks_still_fit_on_the_node():
    for task in ("train", "finetune", "custom"):
        d = resource_defaults(task)
        assert d.cpus <= NODE_CPUS and d.mem_gb <= NODE_MEM_GB, task
