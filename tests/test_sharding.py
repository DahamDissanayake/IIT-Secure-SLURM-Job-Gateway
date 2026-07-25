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


def _nb_script(tmp_path):
    from iitgpu.jobs import JobSpec, render_notebook_sbatch
    spec = JobSpec(job_name="notebook", partition="gpu", gpu_shards=1, cpus=8,
                   mem_gb=14, time_limit="08:00:00", run_command="",
                   task_type="notebook", conda_env="/shared/envs/data-science")
    return render_notebook_sbatch(spec, str(tmp_path), port=8888,
                                  gateway_host="gw.edu", gateway_port=2225)


def test_notebook_is_rooted_at_the_users_own_folder(tmp_path):
    """Serving /shared made the per-user jail meaningless in the notebook."""
    script = _nb_script(tmp_path)
    assert "--ServerApp.root_dir=$IIT_USER_ROOT" in script
    assert "--notebook-dir=/shared" not in script


def test_notebook_exposes_shared_assets_by_symlink(tmp_path):
    """Users still need the datasets they train on."""
    script = _nb_script(tmp_path)
    for asset in ("models", "envs", "data", "datasets"):
        assert f"ln -sfn /shared/{asset}" in script, f"{asset} must be reachable"


def test_symlink_creation_is_idempotent(tmp_path):
    """The job reruns on every launch; -sfn must not fail on an existing link."""
    script = _nb_script(tmp_path)
    assert "ln -s " not in script.replace("ln -sfn ", "")


def test_symlink_creation_guards_existing_non_symlink(tmp_path):
    """users/hassan2/envs and users/public/data are REAL directories, not
    symlinks. ln -sfn does not clobber a real file/dir at the destination --
    it links *inside* it instead (envs/envs), silently hiding the shared
    asset and leaving a stray link behind on every relaunch. The script must
    check before linking (skip + warn instead of a bare ln -sfn) rather than
    linking unconditionally."""
    script = _nb_script(tmp_path)
    # Every asset link must be preceded by a symlink-or-absent guard, not a
    # bare unconditional ln -sfn.
    assert '[ -L "$_iit_target" ] || [ ! -e "$_iit_target" ]' in script
    assert "WARNING" in script and "already exists and is not a symlink" in script
    # And the guard must actually wrap the ln -sfn call, not just appear
    # somewhere else in the script.
    for asset in ("models", "envs", "data", "datasets"):
        guarded = (
            f'if [ -L "$_iit_target" ] || [ ! -e "$_iit_target" ]; then\n'
            f'    ln -sfn /shared/{asset} "$_iit_target"'
        )
        assert guarded in script, f"{asset} link must be guarded, not unconditional"


def test_notebook_docstring_does_not_claim_loopback_only(tmp_path):
    """It binds the routable NodeAddr; a false security comment is a trap."""
    from iitgpu.jobs import render_notebook_sbatch
    assert "127.0.0.1 only" not in (render_notebook_sbatch.__doc__ or "")
