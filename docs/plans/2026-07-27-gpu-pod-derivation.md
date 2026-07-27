# GPU Pod Derivation (Plan A of 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded `SHARDS_PER_GPU=4` / `_SLICE_CPUS=8` / `_SLICE_MEM_GB=14` constants with pod sizing derived live from `NodeStats` (itself read from `scontrol show node`), so nothing in the app hardcodes a 4-way split — everything divides by whatever the cluster currently reports.

**Architecture:** One new pure module, `iitgpu/pods.py`, is the single place pod-count and per-pod CPU/RAM math happens. `iitgpu/jobs.py`, `iitgpu/launchspec.py`, `iitgpu/review.py`, and `iitgpu/validate.py` all call into it instead of doing their own division or referencing a constant. This plan ships against **today's cluster** (a fixed 4-shard `gres.conf`) — it changes nothing about how many pods exist, only how that number is discovered and used. Admin-driven resizing of the pod count is Plan B (`2026-07-27-gpu-pod-admin-resize.md`), built on top of this.

**Tech Stack:** Python 3.14, pytest, existing `iitgpu` package conventions (dataclasses, `rich`/`questionary` for TUI, `unittest.mock.patch` for tests).

## Global Constraints

- Repo: `/home/slurmadmin/IIT-Secure-SLURM-Job-Gateway` on the login node (`ssh slurmadmin@192.168.122.10`), branch `main`. All edits happen there directly (no local clone on the GPU host).
- Full pytest gate (`python3 -m pytest -q`) must pass before every commit, run on the login node.
- Deploy only via `bash deploy/redeploy-igm.sh` run **as `slurmadmin`, not sudo** (sudo breaks the self-reexec guard and skips post-gate steps).
- Commit author: Daham only (`git -c user.name="Daham Dissanayake" -c user.email="dahamdissanayake05@gmail.com" commit ...`), no co-author line. Push to `origin main` without asking, as long as it's a clean fast-forward.
- Live cluster's current real values (used throughout this plan's expected numbers): `CPUTot=32`, `RealMemory=62000` (MB), `Gres=gpu:1,shard:4` → `pod_count()==4`, per-pod `cpus=8`, per-pod `mem_gb=14` (see Task 1's headroom derivation — this must land on 14 to match the existing, already-correct `_SLICE_MEM_GB` value).
- VRAM figures are always an **estimate**, never an enforced ceiling — every piece of UI text touching VRAM must keep saying "not enforced" / "shared" per existing wording precedent.

---

### Task 1: `iitgpu/pods.py` — pure pod-sizing module

**Files:**
- Create: `iitgpu/pods.py`
- Test: `tests/test_pods.py`

**Interfaces:**
- Consumes: `iitgpu.slurm.NodeStats` (fields used: `cpu_total: int`, `mem_total_mb: int`, `shard_total: int`, `gpu_mem_total_mb: int`, `live_stats: bool` — all already exist on `NodeStats`).
- Produces (used by Tasks 2-4, 6):
  - `PodSize` — frozen dataclass, fields `cpus: int`, `mem_gb: int`.
  - `pod_count(stats: NodeStats | None) -> int`
  - `pod_resources(stats: NodeStats | None) -> PodSize`
  - `resources_for(k: int, stats: NodeStats | None) -> tuple[int, int, int]` — returns `(cpus, mem_gb, gpu_shards)`, `gpu_shards` always equal to the clamped `k`.
  - `estimated_vram_gb(k: int, stats: NodeStats | None) -> float | None` — `None` when live GPU stats aren't available.
  - `fits_new_pod_count(new_n: int, stats: NodeStats) -> tuple[bool, str]` — pre-flight sanity check for Plan B's resize action; included here since it's pure math with no SLURM I/O.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pods.py`:

```python
"""Pure pod-sizing math: cpu/mem/VRAM per pod, derived live from NodeStats,
never from a hardcoded constant."""
from iitgpu.pods import (PodSize, estimated_vram_gb, fits_new_pod_count,
                          pod_count, pod_resources, resources_for)
from iitgpu.slurm import NodeStats


def _stats(shard_total=4, shard_alloc=0, cpu_total=32, mem_total_mb=62000,
           gpu_mem_total_mb=0, live_stats=False):
    return NodeStats(state="MIXED", cpu_load=0.0, cpu_total=cpu_total, cpu_alloc=0,
                     mem_total_mb=mem_total_mb, mem_alloc_mb=0,
                     gpu_total=1, gpu_alloc=0,
                     shard_total=shard_total, shard_alloc=shard_alloc,
                     gpu_mem_total_mb=gpu_mem_total_mb, live_stats=live_stats)


def test_pod_count_reads_live_shard_total():
    assert pod_count(_stats(shard_total=4)) == 4
    assert pod_count(_stats(shard_total=5)) == 5


def test_pod_count_falls_back_to_one_when_stats_unavailable():
    assert pod_count(None) == 1
    assert pod_count(_stats(shard_total=0)) == 1


def test_pod_resources_matches_the_existing_hand_picked_slice_size():
    """This is the load-bearing assertion: the new live-derived math must land
    on exactly the same 8 CPU / 14 GB the old hardcoded _SLICE_CPUS/_SLICE_MEM_GB
    constants used, on today's real cluster numbers (32 CPU, 62000 MB, 4 pods)."""
    size = pod_resources(_stats(shard_total=4, cpu_total=32, mem_total_mb=62000))
    assert size == PodSize(cpus=8, mem_gb=14)


def test_pod_resources_uneven_division_floors():
    size = pod_resources(_stats(shard_total=5, cpu_total=32, mem_total_mb=62000))
    assert size.cpus == 6     # 32 // 5
    assert size.mem_gb == 11  # (60 - 2 headroom) // 5


def test_pod_resources_falls_back_to_minimal_size_with_no_stats():
    assert pod_resources(None) == PodSize(cpus=1, mem_gb=1)


def test_resources_for_scales_linearly_with_k():
    stats = _stats(shard_total=4, cpu_total=32, mem_total_mb=62000)
    assert resources_for(1, stats) == (8, 14, 1)
    assert resources_for(2, stats) == (16, 28, 2)
    assert resources_for(4, stats) == (32, 56, 4)


def test_resources_for_caps_k_at_pod_count():
    stats = _stats(shard_total=4, cpu_total=32, mem_total_mb=62000)
    assert resources_for(9, stats) == resources_for(4, stats)


def test_resources_for_floors_k_at_one():
    stats = _stats(shard_total=4, cpu_total=32, mem_total_mb=62000)
    assert resources_for(0, stats) == resources_for(1, stats)


def test_estimated_vram_gb_scales_with_k_and_pod_count():
    stats = _stats(shard_total=4, gpu_mem_total_mb=32768, live_stats=True)
    assert estimated_vram_gb(1, stats) == 8.0
    assert estimated_vram_gb(2, stats) == 16.0
    assert estimated_vram_gb(4, stats) == 32.0


def test_estimated_vram_gb_none_when_live_stats_unavailable():
    stats = _stats(shard_total=4, gpu_mem_total_mb=0, live_stats=False)
    assert estimated_vram_gb(1, stats) is None
    assert estimated_vram_gb(1, None) is None


def test_fits_new_pod_count_rejects_zero_cpu_per_pod():
    stats = _stats(cpu_total=32, mem_total_mb=62000)
    ok, msg = fits_new_pod_count(40, stats)
    assert not ok and "0 CPU" in msg


def test_fits_new_pod_count_rejects_zero_mem_per_pod():
    stats = _stats(cpu_total=32, mem_total_mb=62000)
    ok, msg = fits_new_pod_count(30, stats)
    assert not ok and ("0 GB" in msg or "0 CPU" in msg)


def test_fits_new_pod_count_accepts_reasonable_n():
    stats = _stats(cpu_total=32, mem_total_mb=62000)
    ok, msg = fits_new_pod_count(5, stats)
    assert ok
    assert "6 CPU" in msg and "11 GB" in msg


def test_fits_new_pod_count_rejects_below_one():
    stats = _stats(cpu_total=32, mem_total_mb=62000)
    ok, _ = fits_new_pod_count(0, stats)
    assert not ok
```

- [ ] **Step 2: Run tests to verify they fail**

Run (on the login node): `cd ~/IIT-Secure-SLURM-Job-Gateway && python3 -m pytest tests/test_pods.py -v`
Expected: every test fails with `ModuleNotFoundError: No module named 'iitgpu.pods'`.

- [ ] **Step 3: Implement `iitgpu/pods.py`**

```python
# iitgpu/pods.py — pure pod-sizing math, no I/O. Pod count and per-pod CPU/RAM
# are always derived from a live NodeStats snapshot (itself read from
# `scontrol show node` by iitgpu.slurm.get_node_stats), never from a stored
# constant. This is the ONE place that math happens; jobs.py, launchspec.py,
# review.py and validate.py all call in here instead of doing their own
# division.
from __future__ import annotations
from dataclasses import dataclass

from iitgpu.slurm import NodeStats

# Reserved for the OS/system services, never handed to a pod. Matches the
# margin the previous hand-picked _SLICE_MEM_GB=14 constant already baked in
# (62000 MB / 4 pods would be 15 GB/pod with zero headroom; the old constant
# was deliberately 1 GB under that "so four really fit").
_MEM_HEADROOM_GB = 2


@dataclass(frozen=True)
class PodSize:
    cpus: int
    mem_gb: int


def pod_count(stats: NodeStats | None) -> int:
    """How many pods the node is currently split into, read straight from live
    SLURM state (gres/shard total) -- never a separately-stored config value."""
    if stats is None or stats.shard_total <= 0:
        return 1
    return stats.shard_total


def pod_resources(stats: NodeStats | None) -> PodSize:
    """CPU/RAM for a single pod, floor-divided from the node's real totals."""
    if stats is None:
        return PodSize(cpus=1, mem_gb=1)
    n = pod_count(stats)
    cpus = max(1, stats.cpu_total // n)
    usable_mem_gb = max(0, (stats.mem_total_mb // 1024) - _MEM_HEADROOM_GB)
    mem_gb = max(1, usable_mem_gb // n)
    return PodSize(cpus=cpus, mem_gb=mem_gb)


def resources_for(k: int, stats: NodeStats | None) -> tuple[int, int, int]:
    """(cpus, mem_gb, gpu_shards) for a job requesting k pods. k is clamped to
    [1, pod_count] -- gpu_shards in the return value IS the clamped k, since a
    pod and a GPU shard are the same unit by construction."""
    n = pod_count(stats)
    k = max(1, min(k, n))
    size = pod_resources(stats)
    return size.cpus * k, size.mem_gb * k, k


def estimated_vram_gb(k: int, stats: NodeStats | None) -> float | None:
    """Fair-share VRAM estimate for k pods -- an ESTIMATE, never enforced (GPU
    shards are a scheduling split only; nothing partitions VRAM). Returns None
    when live GPU stats aren't available, so callers show a plain caveat with
    no number instead of a wrong one."""
    if not stats or not getattr(stats, "live_stats", False) or not stats.gpu_mem_total_mb:
        return None
    n = pod_count(stats)
    k = max(1, min(k, n))
    total_gb = stats.gpu_mem_total_mb / 1024
    return total_gb * k / n


def fits_new_pod_count(new_n: int, stats: NodeStats) -> tuple[bool, str]:
    """Sanity-check a candidate pod-count resize BEFORE any SLURM config is
    touched (used by Plan B's admin resize flow). Rejects an N that would
    floor CPU or mem per pod to zero."""
    if new_n < 1:
        return False, "Pod count must be at least 1."
    cpus = stats.cpu_total // new_n
    usable_mem_gb = max(0, (stats.mem_total_mb // 1024) - _MEM_HEADROOM_GB)
    mem_gb = usable_mem_gb // new_n
    if cpus < 1:
        return False, (f"{new_n} pods would give each pod 0 CPUs "
                        f"({stats.cpu_total} total) -- pick a smaller N.")
    if mem_gb < 1:
        return False, (f"{new_n} pods would give each pod 0 GB RAM "
                        f"({usable_mem_gb} GB usable) -- pick a smaller N.")
    return True, f"{new_n} pods -> {cpus} CPU / {mem_gb} GB RAM each."
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_pods.py -v`
Expected: all PASS. If `test_pod_resources_matches_the_existing_hand_picked_slice_size` fails, double-check the headroom constant — it must reproduce `cpus=8, mem_gb=14` on `cpu_total=32, mem_total_mb=62000, shard_total=4`.

- [ ] **Step 5: Commit**

```bash
git add iitgpu/pods.py tests/test_pods.py
git -c user.name="Daham Dissanayake" -c user.email="dahamdissanayake05@gmail.com" commit -m "feat(pods): add live pod-sizing derivation module

Pure math only -- pod count and per-pod CPU/RAM/VRAM-estimate, always
derived from a NodeStats snapshot instead of a hardcoded constant."
```

---

### Task 2: `iitgpu/jobs.py` — replace hardcoded constants with pod-derived defaults

**Files:**
- Modify: `iitgpu/jobs.py:21-81` (the `SHARDS_PER_GPU`/`_SLICE_CPUS`/`_SLICE_MEM_GB`/`TASK_DEFAULTS`/`resource_defaults`/`gres_directive`/`gpu_share_note` block)
- Modify: `tests/test_jobs.py:100-131` (existing `resource_defaults` tests, expectations change for train/finetune/custom)
- Modify: `tests/test_notebook.py:1-20` (`TASK_DEFAULTS` membership test → `TASK_POD_DEFAULTS`)
- Modify: `tests/test_sharding.py` (every `SHARDS_PER_GPU`/`TASK_DEFAULTS` reference)

**Interfaces:**
- Consumes: `iitgpu.pods.pod_count`, `resources_for` (Task 1).
- Produces (used by Tasks 3-5): `resource_defaults(task_type: str, stats: NodeStats | None = None) -> TaskDefaults` (same name/return type as before, new optional `stats` param; fetches live stats internally when not given). `gres_directive(gpu_shards: int) -> str` (unchanged). `gpu_share_note(gpu_shards: int, total_shards: int) -> str` (was `(gpu_shards)` only — **now requires `total_shards` explicitly, no default, no internal I/O** — every call site must be updated; see Task 5). `TASK_POD_DEFAULTS: dict[str, int | str]` (new name, replaces the old `TASK_DEFAULTS` dict of literal values — maps task type to either an int pod count or the sentinel `"all"`).

- [ ] **Step 1: Update the failing/changing tests first**

In `tests/test_jobs.py`, replace the block starting `from iitgpu.jobs import TaskDefaults, resource_defaults, TASK_DEFAULTS` through `test_resource_defaults_unknown_falls_back_to_custom` with:

```python
from iitgpu.jobs import TaskDefaults, resource_defaults, TASK_POD_DEFAULTS
from iitgpu.slurm import NodeStats


def _stats():
    return NodeStats(state="MIXED", cpu_load=0.0, cpu_total=32, cpu_alloc=0,
                     mem_total_mb=62000, mem_alloc_mb=0, gpu_total=1, gpu_alloc=0,
                     shard_total=4, shard_alloc=0)


def test_resource_defaults_train_takes_the_whole_node():
    """Unified pod model: training defaults to ALL pods, which on today's live
    cluster means the whole node -- not the old arbitrary half-the-CPUs value."""
    d = resource_defaults("train", _stats())
    assert d.gpu_shards == 4
    assert d.cpus == 32
    assert d.mem_gb == 56
    assert d.time_limit == ""


def test_resource_defaults_inference():
    d = resource_defaults("inference", _stats())
    assert d.cpus == 8
    assert d.mem_gb == 14
    assert d.gpu_shards == 1
    assert d.time_limit == "04:00:00"


def test_resource_defaults_test_is_a_fixed_smoke_allocation():
    """'test' is a deliberately fixed tiny smoke-test job, not pod-derived."""
    d = resource_defaults("test", _stats())
    assert d.cpus == 4
    assert d.mem_gb == 8
    assert d.time_limit == "00:30:00"


def test_resource_defaults_unknown_falls_back_to_custom():
    d = resource_defaults("nonexistent_task", _stats())
    assert d == resource_defaults("custom", _stats())


def test_resource_defaults_without_stats_degrades_gracefully(monkeypatch):
    """No live stats available (e.g. scontrol unreachable) -- must not crash,
    falls back to a single-pod-sized allocation."""
    import iitgpu.jobs as jobs
    monkeypatch.setattr(jobs, "_live_stats", lambda: None)
    d = resource_defaults("notebook")
    assert d.gpu_shards == 1 and d.cpus >= 1 and d.mem_gb >= 1


def test_task_pod_defaults_covers_every_task_type():
    for name in ("train", "finetune", "custom", "inference", "notebook", "interactive"):
        assert name in TASK_POD_DEFAULTS
```

In `tests/test_notebook.py`, replace:

```python
def test_notebook_in_task_defaults():
    from iitgpu.jobs import TASK_DEFAULTS
    assert "notebook" in TASK_DEFAULTS
```

with:

```python
def test_notebook_in_task_pod_defaults():
    from iitgpu.jobs import TASK_POD_DEFAULTS
    assert "notebook" in TASK_POD_DEFAULTS
```

And update `test_notebook_defaults_correct` to pass stats explicitly:

```python
def test_notebook_defaults_correct():
    from iitgpu.jobs import resource_defaults
    from iitgpu.slurm import NodeStats
    stats = NodeStats(state="MIXED", cpu_load=0.0, cpu_total=32, cpu_alloc=0,
                      mem_total_mb=62000, mem_alloc_mb=0, gpu_total=1, gpu_alloc=0,
                      shard_total=4, shard_alloc=0)
    d = resource_defaults("notebook", stats)
    assert d.gpu_shards == 1
    assert d.cpus == 8
    assert d.mem_gb == 14
    assert d.time_limit == "08:00:00"
```

In `tests/test_sharding.py`, replace the whole file's imports and every `SHARDS_PER_GPU`/`TASK_DEFAULTS` reference:

```python
"""GPU sharding: several jobs share the one physical GPU.

Before sharding a job asked for --gres=gpu:1 and took the whole card, so a
second GPU job (e.g. a colleague's JupyterLab) could not start at all. Jobs now
request slices (--gres=shard:N) instead. Pod count (how many slices the card
is split into) is read live from NodeStats, not a hardcoded constant -- these
tests pin it at today's real cluster value (4) via an explicit stats fixture.
"""
import pytest

from iitgpu.jobs import (
    JobSpec, TASK_POD_DEFAULTS, gpu_share_note, gres_directive,
    render_sbatch, render_notebook_sbatch, resource_defaults,
)
from iitgpu.pods import pod_count
from iitgpu.slurm import NodeStats

NODE_CPUS = 32
NODE_MEM_GB = 60      # RealMemory=62000 MB


def _stats(shard_total=4):
    return NodeStats(state="MIXED", cpu_load=0.0, cpu_total=NODE_CPUS, cpu_alloc=0,
                     mem_total_mb=62000, mem_alloc_mb=0, gpu_total=1, gpu_alloc=0,
                     shard_total=shard_total, shard_alloc=0)


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
    assert gres_directive(4) == "shard:4"


def test_gres_directive_empty_when_no_gpu_wanted():
    assert gres_directive(0) == ""


# ── Rendered scripts ──────────────────────────────────────────────────────────

def test_rendered_sbatch_never_requests_a_whole_gpu(tmp_path):
    script = render_sbatch(_spec(gpu_shards=1), str(tmp_path))
    assert "--gres=shard:1" in script
    assert "--gres=gpu:" not in script


def test_two_notebooks_each_take_one_slice(tmp_path):
    stats = _stats()
    nb = resource_defaults("notebook", stats)
    assert nb.gpu_shards == 1
    assert nb.gpu_shards * 2 <= pod_count(stats), "two notebooks must fit together"


def test_notebook_sbatch_requests_a_slice(tmp_path):
    script = render_notebook_sbatch(
        _spec(gpu_shards=1, task_type="notebook"), str(tmp_path), port=8888)
    assert "--gres=shard:1" in script
    assert "--gres=gpu:" not in script


def test_cpu_only_job_omits_gres_entirely(tmp_path):
    script = render_sbatch(_spec(gpu_shards=0), str(tmp_path))
    assert "--gres" not in script


# ── Task defaults ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("task", ["notebook", "interactive", "test", "inference"])
def test_light_tasks_leave_room_for_others(task):
    stats = _stats()
    assert resource_defaults(task, stats).gpu_shards < pod_count(stats)


@pytest.mark.parametrize("task", ["train", "finetune"])
def test_training_still_gets_the_whole_card(task):
    stats = _stats()
    assert resource_defaults(task, stats).gpu_shards == pod_count(stats)


def test_no_task_default_over_subscribes_the_gpu():
    stats = _stats()
    for name in TASK_POD_DEFAULTS:
        d = resource_defaults(name, stats)
        assert 0 <= d.gpu_shards <= pod_count(stats), name


# ── User-facing wording ───────────────────────────────────────────────────────

def test_gpu_share_note_explains_what_is_left():
    assert "no GPU" in gpu_share_note(0, 4)
    assert "whole GPU" in gpu_share_note(4, 4)
    partial = gpu_share_note(1, 4)
    assert "1/4" in partial and "left for others" in partial


@pytest.mark.parametrize("task", ["notebook", "interactive", "inference"])
def test_a_full_cards_worth_of_slice_jobs_fits_on_the_node(task):
    stats = _stats()
    d = resource_defaults(task, stats)
    concurrent = pod_count(stats) // d.gpu_shards
    assert d.cpus * concurrent <= NODE_CPUS, (
        f"{concurrent}x {task} needs {d.cpus * concurrent} CPUs, node has {NODE_CPUS}")
    assert d.mem_gb * concurrent <= NODE_MEM_GB, (
        f"{concurrent}x {task} needs {d.mem_gb * concurrent} GB RAM, "
        f"node has {NODE_MEM_GB}")


def test_two_notebooks_fit_side_by_side():
    stats = _stats()
    d = resource_defaults("notebook", stats)
    assert d.gpu_shards * 2 <= pod_count(stats)
    assert d.cpus * 2 <= NODE_CPUS
    assert d.mem_gb * 2 <= NODE_MEM_GB


def test_whole_card_tasks_still_fit_on_the_node():
    stats = _stats()
    for task in ("train", "finetune", "custom"):
        d = resource_defaults(task, stats)
        assert d.cpus <= NODE_CPUS and d.mem_gb <= NODE_MEM_GB, task
```

(If the rest of `test_sharding.py` beyond line ~160 contains additional tests not shown above, leave them as-is unless they reference `SHARDS_PER_GPU`/`TASK_DEFAULTS` directly — grep the file for both names after this edit and fix any remaining hits the same way: replace `SHARDS_PER_GPU` with `pod_count(stats)` against an explicit `_stats()` fixture, replace `TASK_DEFAULTS` with `TASK_POD_DEFAULTS`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_jobs.py tests/test_notebook.py tests/test_sharding.py -v`
Expected: FAIL — `ImportError: cannot import name 'TASK_POD_DEFAULTS'` and assertion mismatches against the still-old `jobs.py`.

- [ ] **Step 3: Implement the `iitgpu/jobs.py` change**

Replace lines 21-81 of `iitgpu/jobs.py` (from the `# The cluster's single GPU...` comment through the end of `gpu_share_note`) with:

```python
@dataclass(frozen=True)
class TaskDefaults:
    gpu_shards: int   # GPU slices (pods) to request
    cpus: int
    mem_gb: int
    time_limit: str  # "" means no time limit (SLURM INFINITE)


_ALL_PODS = "all"

# How many pods each task type requests by default. "all" means "every pod
# currently on the node" -- on today's cluster (4 pods) that's the whole
# node; if an admin resizes pod count later (Plan B), this keeps working with
# no change here, because it's resolved against live NodeStats at call time,
# never a stored number.
TASK_POD_DEFAULTS: dict[str, int | str] = {
    "train":       _ALL_PODS,
    "finetune":    _ALL_PODS,
    "custom":      _ALL_PODS,
    "inference":   1,
    "notebook":    1,
    "interactive": 1,
}

_DEFAULT_TIME_LIMIT: dict[str, str] = {
    "train": "", "finetune": "", "custom": "",
    "inference": "04:00:00", "notebook": "08:00:00", "interactive": "02:00:00",
}

# "test" is a fixed tiny smoke-test allocation, deliberately NOT tied to pod
# count -- it's a connectivity check, not real work, and always the same size
# regardless of how the cluster is currently split.
_TEST_DEFAULTS = TaskDefaults(gpu_shards=1, cpus=4, mem_gb=8, time_limit="00:30:00")


def _live_stats():
    try:
        from iitgpu.slurm import get_node_stats
        return get_node_stats()
    except Exception:
        return None


def resource_defaults(task_type: str, stats=None) -> TaskDefaults:
    """Default sizing for a task type, derived live from the cluster's current
    pod split. Falls back to a live scontrol read when *stats* isn't given
    (most call sites don't already have a NodeStats handy)."""
    if task_type == "test":
        return _TEST_DEFAULTS
    if stats is None:
        stats = _live_stats()
    from iitgpu.pods import pod_count, resources_for
    n = pod_count(stats)
    want = TASK_POD_DEFAULTS.get(task_type, _ALL_PODS)
    k = n if want == _ALL_PODS else min(int(want), n)
    cpus, mem_gb, gpu_shards = resources_for(k, stats)
    return TaskDefaults(gpu_shards=gpu_shards, cpus=cpus, mem_gb=mem_gb,
                        time_limit=_DEFAULT_TIME_LIMIT.get(task_type, ""))


def gres_directive(gpu_shards: int) -> str:
    """SLURM --gres value for a shard request, or "" when the job needs no GPU."""
    return f"shard:{gpu_shards}" if gpu_shards > 0 else ""


def gpu_share_note(gpu_shards: int, total_shards: int) -> str:
    """Plain-language description of how much of the GPU a request reserves.

    Pure -- total_shards must be passed in by the caller (e.g.
    `pods.pod_count(get_node_stats())`), never fetched internally, so this
    stays a fast, testable, no-I/O function."""
    if gpu_shards <= 0:
        return "no GPU (CPU-only)"
    if gpu_shards >= total_shards:
        return "the whole GPU (no one else can use it)"
    return (f"{gpu_shards}/{total_shards} of the GPU "
            f"({total_shards - gpu_shards}/{total_shards} left for others)")
```

Also delete the old module-level `SHARDS_PER_GPU = 4` line and its explanatory comment (a few lines above where `@dataclass(frozen=True) class TaskDefaults` used to start) — nothing in this file should reference a hardcoded shard count any more.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_jobs.py tests/test_notebook.py tests/test_sharding.py -v`
Expected: all PASS. Also run `python3 -m pytest tests/test_pods.py -v` again to confirm no regression.

- [ ] **Step 5: Commit**

```bash
git add iitgpu/jobs.py tests/test_jobs.py tests/test_notebook.py tests/test_sharding.py
git -c user.name="Daham Dissanayake" -c user.email="dahamdissanayake05@gmail.com" commit -m "refactor(jobs): derive task-type sizing from live pod count

TASK_DEFAULTS -> TASK_POD_DEFAULTS (pod counts, not literal cpu/mem
numbers). resource_defaults() and gpu_share_note() no longer read the
SHARDS_PER_GPU constant -- both take/derive the live pod count instead.
Training now defaults to all pods (the whole node today) instead of an
arbitrary half-the-CPUs allocation."
```

---

### Task 3: `iitgpu/launchspec.py` — replace named size presets with pod selection

**Files:**
- Modify: `iitgpu/launchspec.py` (remove `Size`/`SIZES`/`apply_size`/`size_name_for`/`_frac`/`size_label`/`size_availability`; add `apply_pods`/`pod_label`/`pod_availability`)
- Modify: `tests/test_launchspec.py`

**Interfaces:**
- Consumes: `iitgpu.pods.pod_count`, `resources_for`, `estimated_vram_gb` (Task 1).
- Produces (used by Task 4): `apply_pods(ls: LaunchSpec, k: int, stats) -> None` (sets `ls.gpu_shards, ls.cpus, ls.mem_gb` — no new field; a pod IS a GPU shard, so `ls.gpu_shards` already means "pods requested"). `default_spec(intent: str, stats=None) -> LaunchSpec` (same name/signature shape as before, new optional `stats`). `pod_label(ls, stats) -> str`. `pod_availability(k: int, stats) -> str` (replaces `size_availability`). `availability_line(stats) -> str` (unchanged — it never referenced `SHARDS_PER_GPU`).

- [ ] **Step 1: Update the failing tests first**

In `tests/test_launchspec.py`, replace the import block and the `SIZES`/`apply_size`-based tests:

```python
from iitgpu.jobs import TaskDefaults  # unused-import check: keep only if still needed elsewhere in the file
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


def test_pod_availability_reports_starts_now_or_queues():
    stats = _stats(free=2)
    assert "starts now" in pod_availability(2, stats)
    assert "will queue" in pod_availability(3, stats)


def test_availability_line_unknown_without_stats():
    assert availability_line(None) == "GPU availability unknown"


def test_availability_line_reports_free_slices():
    assert availability_line(_stats(free=3)) == "GPU now: 3/4 slices free"
```

(Keep the rest of the existing file's `from_template`/`from_rerun`/`recent_scripts`/`to_job_spec` tests unchanged below this point — they don't reference `SIZES`/`SHARDS_PER_GPU` and remain valid as-is.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_launchspec.py -v`
Expected: FAIL — `ImportError: cannot import name 'apply_pods'`.

- [ ] **Step 3: Implement the `iitgpu/launchspec.py` change**

Replace the block from `@dataclass(frozen=True) class Size:` through the end of `size_availability` (roughly lines 12-112 of the original file) with:

```python
from iitgpu.pods import estimated_vram_gb, pod_count, resources_for


@dataclass
class LaunchSpec:
    intent: str
    script: str = ""
    env_kind: str = "prebuilt"      # prebuilt | conda | venv | container | none
    conda_env: str = ""
    venv_path: str = ""
    container_image: str = ""
    gpu_shards: int = 1     # this IS the pod count -- 1 shard == 1 pod
    cpus: int = 8
    mem_gb: int = 14
    time_limit: str = "04:00:00"
    data_path: str = ""
    model_path: str = ""
    args: str = ""
    array: str = ""
    dependency: str = ""
    mail: bool = True
    requirements: str = ""
    packages: str = ""
    port: int = 8888


_INTENT_DEFAULT_PODS = {"notebook": 1, "batch": 1, "shell": 1}
_INTENT_DEFAULT_TIME = {"notebook": "06:00:00", "shell": "02:00:00"}


def apply_pods(ls: LaunchSpec, k: int, stats) -> None:
    """Set ls to request k pods, sizing cpus/mem_gb to match live node totals."""
    ls.cpus, ls.mem_gb, ls.gpu_shards = resources_for(k, stats)


def default_spec(intent: str, stats=None) -> LaunchSpec:
    ls = LaunchSpec(intent=intent)
    apply_pods(ls, _INTENT_DEFAULT_PODS.get(intent, 1), stats)
    ls.time_limit = _INTENT_DEFAULT_TIME.get(intent, "04:00:00")
    return ls


def pod_label(ls: LaunchSpec, stats) -> str:
    n = pod_count(stats)
    frac = "whole GPU" if ls.gpu_shards >= n else f"{ls.gpu_shards}/{n} GPU"
    plural = "" if ls.gpu_shards == 1 else "s"
    return f"{ls.gpu_shards} pod{plural} — {frac} · {ls.cpus} CPU · {ls.mem_gb} GB"


def _slices_free(stats) -> int | None:
    if stats and getattr(stats, "shard_total", 0):
        return max(0, stats.shard_total - stats.shard_alloc)
    return None


def availability_line(stats) -> str:
    free = _slices_free(stats)
    if free is None:
        return "GPU availability unknown"
    return f"GPU now: {free}/{stats.shard_total} slices free"


def pod_availability(k: int, stats) -> str:
    free = _slices_free(stats)
    if free is None:
        return "— availability unknown"
    if k <= free:
        return f"— starts now ({free} free)"
    return f"— will queue (needs {k}, {free} free)"
```

Remove the old `Size`/`SIZES`/`_INTENT_DEFAULTS`/`apply_size`/`size_name_for`/`_frac`/`size_label`/`size_availability` definitions entirely, and remove the now-unused `from iitgpu.jobs import SHARDS_PER_GPU, JobSpec` line at the top of the file — replace with `from iitgpu.jobs import JobSpec` (still needed by `to_job_spec`).

`to_job_spec`, `from_template`, `from_rerun`, `recent_scripts` are unchanged (they already read/write `gpu_shards`/`cpus`/`mem_gb` directly, which still exist with the same names and meaning).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_launchspec.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add iitgpu/launchspec.py tests/test_launchspec.py
git -c user.name="Daham Dissanayake" -c user.email="dahamdissanayake05@gmail.com" commit -m "refactor(launchspec): pod selection replaces named size presets

SIZES (Small/Standard/Whole GPU) removed. apply_pods()/pod_label()/
pod_availability() replace apply_size()/size_label()/size_availability(),
sized live from pods.resources_for() instead of a fixed table. gpu_shards
now IS the pod count -- no new field, no dual bookkeeping."
```

---

### Task 4: `iitgpu/review.py` — pod stepper in the review hub

**Files:**
- Modify: `iitgpu/review.py:1-125` (imports, `_vram_note`, `render_hub`, `_edit_size` → `_edit_pods`)
- Modify: `iitgpu/review.py:313-360` (the `"Change size"` choice in `_HUB_CHOICES` and its dispatch in `run_hub`)
- Modify: `tests/test_review.py`

**Interfaces:**
- Consumes: `iitgpu.launchspec.{pod_label, pod_availability, apply_pods}`, `iitgpu.pods.{pod_count, resources_for, estimated_vram_gb}` (Tasks 1, 3).
- Produces: `render_hub(ls, stats) -> Panel` (unchanged signature), `_edit_pods(ls, stats) -> None` (was `_edit_size`).

- [ ] **Step 1: Update the failing tests first**

In `tests/test_review.py`, replace the import line `from iitgpu.jobs import SHARDS_PER_GPU` with nothing (no longer needed) and update `_stats()`'s hardcoded `shard_total=SHARDS_PER_GPU` to `shard_total=4`:

```python
from iitgpu.launchspec import default_spec
from iitgpu.review import render_hub, run_hub
from iitgpu.slurm import NodeStats
from iitgpu.ui import BACK


def _stats(free=3):
    return NodeStats(state="MIXED", cpu_load=0.0, cpu_total=32, cpu_alloc=0,
                     mem_total_mb=62000, mem_alloc_mb=0, gpu_total=1,
                     gpu_alloc=0, shard_total=4,
                     shard_alloc=4 - free)


def test_hub_shows_every_field_and_availability():
    ls = default_spec("batch", _stats(3))
    ls.script = "/shared/users/u/train.py"
    ls.conda_env = "/shared/envs/data-science"
    from rich.console import Console
    import re
    con = Console(force_terminal=True, width=120)
    with con.capture() as cap:
        con.print(render_hub(ls, _stats(3)))
    out = re.sub(r"\x1b\[[0-9;]*m", "", cap.get())
    assert "train.py" in out
    assert "data-science" in out
    assert "1 pod" in out and "8 CPU" in out
    assert "3/4 slices free" in out
    assert "4h" in out or "04:00:00" in out
```

(Keep the rest of `test_review.py` as-is except: every other `_stats()` call and every `SHARDS_PER_GPU` reference in the file gets the same treatment — replace with the literal `4`, since pod count is now a live-derived number pinned by the fixture, not an imported constant. Any test asserting old size-preset wording like `"Standard"` should instead assert the new `"N pod(s)"` wording from `pod_label`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_review.py -v`
Expected: FAIL — old wording assertions (`"Standard"`) no longer match, `SHARDS_PER_GPU` import error if left in.

- [ ] **Step 3: Implement the `iitgpu/review.py` change**

Replace the top imports:

```python
from iitgpu.jobs import gpu_share_note
from iitgpu.launchspec import (LaunchSpec, apply_pods, availability_line,
                               pod_availability, pod_label)
from iitgpu.pods import estimated_vram_gb, pod_count
from iitgpu.slurm import get_node_stats
from iitgpu.ui import console, info, select_menu, warn
```

Replace `_vram_note`:

```python
def _vram_note(ls: LaunchSpec, stats) -> str:
    """The VRAM caveat, with the actual per-pod share when the node reports it."""
    base = "GPU memory is shared between jobs and not enforced"
    share_gb = estimated_vram_gb(ls.gpu_shards, stats)
    if share_gb is None:
        return f"{base}."
    total_gb = stats.gpu_mem_total_mb / 1024
    return (f"{base} — your fair share is about "
            f"{share_gb:.0f} GB of {total_gb:.0f}.")
```

In `render_hub`, change the "Size" row and the `gpu_share_note` call:

```python
    rows += [
        ("Environment", escape(_env_display(ls))),
        ("Pods", f"{pod_label(ls, stats)}   [dim]{pod_availability(ls.gpu_shards, stats)}[/]"),
        ("Time limit", _fmt_time(ls.time_limit)),
        ("Data / model", escape(ls.data_path or ls.model_path or "(none)")),
    ]
```

and further down:

```python
    share = gpu_share_note(ls.gpu_shards, pod_count(stats))
```

Update `_NOOP_FIELDS`/`_CHOICE_FOR_FIELD` keys from `"Size"` to `"Pods"` (they're currently unused for the size row per the existing `_NOOP_FIELDS` table, since every intent shows sizing — no change needed to the dict *values*, only make sure no stale `"Size"` string lingers if present).

Replace `_edit_size` with:

```python
def _edit_pods(ls: LaunchSpec, stats) -> None:
    n = pod_count(stats)
    choices, mapping = [], {}
    for k in range(1, n + 1):
        probe = LaunchSpec(intent=ls.intent)
        apply_pods(probe, k, stats)
        vram = estimated_vram_gb(k, stats)
        vram_txt = f", ~{vram:.0f}GB VRAM" if vram is not None else ""
        label = (f"{pod_label(probe, stats)}{vram_txt}  "
                 f"{pod_availability(k, stats)}")
        choices.append(label); mapping[label] = k
    sel = select_menu("Pods:", choices)
    if sel:
        apply_pods(ls, mapping[sel], stats)
```

In `_HUB_CHOICES`, change `"Change size"` to `"Change pods"`, and in `run_hub`'s dispatch (search for the `elif sel == "Change size":` branch near line 353) change it to:

```python
        elif sel == "Change pods":
            _edit_pods(ls, stats)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_review.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add iitgpu/review.py tests/test_review.py
git -c user.name="Daham Dissanayake" -c user.email="dahamdissanayake05@gmail.com" commit -m "feat(review): pod stepper replaces the size preset menu in the hub

Change size -> Change pods: one row per available pod, each showing the
resulting CPU/mem/VRAM-estimate live, instead of three fixed named sizes."
```

---

### Task 5: `iitgpu/wizard.py` — remove the last `SHARDS_PER_GPU` references

**Files:**
- Modify: `iitgpu/wizard.py:15-16` (import line)
- Modify: `iitgpu/wizard.py:270-286` (`_vram_check`)
- Modify: `iitgpu/wizard.py:~825, ~864, ~928` (three `gpu_share_note(spec.gpu_shards)` call sites)
- Modify: `iitgpu/wizard.py:~422` (`_run_own_sbatch`'s `resource_defaults("custom")` call — verify only, likely no change needed)
- Modify: `tests/test_wizard.py` (`_make_stats` fixture, `test_gpu_share_note_describes_a_partial_card`)

**Interfaces:**
- Consumes: `iitgpu.pods.pod_count`, `iitgpu.jobs.gpu_share_note` (now requires `total_shards`, Task 2).

- [ ] **Step 1: Update the failing tests first**

In `tests/test_wizard.py`, the `_make_stats` fixture (around line 454) currently omits `shard_total`, defaulting it to `0`. Fix it to reflect the real live cluster:

```python
def _make_stats(gpu_mem_used_mb: int = 12288, gpu_mem_total_mb: int = 32768):
    from iitgpu.slurm import NodeStats
    return NodeStats(
        state="ALLOCATED", cpu_load=1.0, cpu_total=32, cpu_alloc=16,
        mem_total_mb=131072, mem_alloc_mb=65536, gpu_total=1, gpu_alloc=1,
        shard_total=4, shard_alloc=1,
        gpu_util=50, gpu_mem_used_mb=gpu_mem_used_mb,
        gpu_mem_total_mb=gpu_mem_total_mb,
        gpu_temp=60, gpu_power_w=200.0, cpu_util=40, cpu_load5=1.0,
        mem_used_mb=40000, live_stats=True,
    )
```

Update `test_gpu_share_note_describes_a_partial_card`:

```python
def test_gpu_share_note_describes_a_partial_card():
    """A one-slice job must say so — resource sizing changed and nothing said."""
    from iitgpu.jobs import gpu_share_note
    note = gpu_share_note(1, 4)
    assert "1/4" in note
    assert "left for others" in note
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_wizard.py -v -k "vram_check or gpu_share_note"`
Expected: FAIL — `gpu_share_note()` still takes one positional arg in the current source (TypeError on the 2-arg call), and `_vram_check` still imports/uses the module constant, so `"about 8 GB"` may or may not already hold depending on order of task execution (Task 2 already removed `SHARDS_PER_GPU` from jobs.py, so `wizard.py`'s `from iitgpu.jobs import SHARDS_PER_GPU, ...` import will now raise `ImportError` — confirms this task is needed).

- [ ] **Step 3: Implement the `iitgpu/wizard.py` change**

Change the import line (around line 15-16) from:

```python
from iitgpu.jobs import (SHARDS_PER_GPU, JobSpec, build_interactive_cmd,
                         gpu_share_note, make_job_folder, pip_install_block,
```

to:

```python
from iitgpu.jobs import (JobSpec, build_interactive_cmd,
                         gpu_share_note, make_job_folder, pip_install_block,
```

(keep whatever else was on that `import (...)` continuation unchanged).

In `_vram_check` (around line 270-286), replace:

```python
        slice_gb = total_gb / SHARDS_PER_GPU
```

with:

```python
        from iitgpu.pods import pod_count
        slice_gb = total_gb / pod_count(stats)
```

Add a small local helper right above `_vram_check` (or anywhere module-level before its first use) so the three `gpu_share_note(spec.gpu_shards)` call sites keep working with one argument, fetching the live total themselves:

```python
def _current_share_note(gpu_shards: int) -> str:
    """gpu_share_note() is pure and needs the live pod total passed in --
    this is the one place in the wizard flow that fetches it, so the three
    call sites below don't each need their own get_node_stats() call."""
    from iitgpu.pods import pod_count
    return gpu_share_note(gpu_shards, pod_count(get_node_stats()))
```

Then replace each of the three occurrences of `gpu_share_note(spec.gpu_shards)` (around lines 825, 864, 928) with `_current_share_note(spec.gpu_shards)`.

Leave `_run_own_sbatch`'s `resource_defaults("custom")` call (around line 422) unchanged — `resource_defaults`'s new optional `stats` parameter defaults to a live fetch internally (Task 2), so this call site already gets correct live-derived sizing with no code change here.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_wizard.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add iitgpu/wizard.py tests/test_wizard.py
git -c user.name="Daham Dissanayake" -c user.email="dahamdissanayake05@gmail.com" commit -m "refactor(wizard): drop the last SHARDS_PER_GPU references

_vram_check and the three gpu_share_note() call sites now pass the live
pod count explicitly instead of reading a hardcoded constant."
```

---

### Task 6: `iitgpu/validate.py` — live ceiling instead of a fixed env var

**Files:**
- Modify: `iitgpu/validate.py:7-10` (add a live-derived ceiling function)
- Modify: `iitgpu/validate.py:~304` (the `--gres` shard check)
- Modify: `tests/test_validate.py` (add a new test; existing `MAX_GPUS` test at line ~287 is unaffected since it tests `--gres=gpu:` not `--gres=shard:`)

**Interfaces:**
- Consumes: `iitgpu.pods.pod_count` (Task 1).
- Produces: `_max_gpu_shards() -> int` (internal helper, not part of the public API other modules import).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_validate.py`:

```python
def test_validate_sbatch_rejects_excess_shards_against_live_pod_count(tmp_path):
    """--gres=shard:N is checked against the LIVE pod count, not a fixed env
    var -- on today's 4-pod cluster, requesting 5 must be rejected."""
    os.environ.update({"NFS_ROOT": str(tmp_path)})
    _user_dir(tmp_path)
    import importlib, iitgpu.validate as v; importlib.reload(v)
    from unittest.mock import patch
    from iitgpu.slurm import NodeStats
    stats = NodeStats(state="MIXED", cpu_load=0.0, cpu_total=32, cpu_alloc=0,
                      mem_total_mb=62000, mem_alloc_mb=0, gpu_total=1, gpu_alloc=0,
                      shard_total=4, shard_alloc=0)
    with patch("iitgpu.daemonclient.email_for", return_value="alice@iit.lk"), \
         patch("iitgpu.slurm.get_node_stats", return_value=stats):
        errors = v.validate_sbatch("#SBATCH --gres=shard:5\n", "alice")
    assert any("slice" in e.lower() or "GPU" in e for e in errors)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_validate.py -k excess_shards -v`
Expected: FAIL or pass-for-the-wrong-reason — the current code checks against the fixed `MAX_GPU_SHARDS=4` env var, which happens to also reject 5, so this test might accidentally pass already. To confirm the change actually matters, also check with a stats fixture where `shard_total=8` (a hypothetically-resized cluster) — a request of `shard:5` should then be ALLOWED, which the old fixed-env-var code would incorrectly still reject. Add:

```python
def test_validate_sbatch_ceiling_tracks_a_resized_pod_count(tmp_path):
    """If the live cluster reports 8 pods, a request for 5 must be allowed --
    proves the check isn't still pinned to the old fixed default of 4."""
    os.environ.update({"NFS_ROOT": str(tmp_path)})
    _user_dir(tmp_path)
    import importlib, iitgpu.validate as v; importlib.reload(v)
    from unittest.mock import patch
    from iitgpu.slurm import NodeStats
    stats = NodeStats(state="MIXED", cpu_load=0.0, cpu_total=64, cpu_alloc=0,
                      mem_total_mb=124000, mem_alloc_mb=0, gpu_total=1, gpu_alloc=0,
                      shard_total=8, shard_alloc=0)
    with patch("iitgpu.daemonclient.email_for", return_value="alice@iit.lk"), \
         patch("iitgpu.slurm.get_node_stats", return_value=stats):
        errors = v.validate_sbatch("#SBATCH --gres=shard:5\n", "alice")
    assert errors == []
```

Run both: expected the second one FAILS against the current code (rejects 5 because `MAX_GPU_SHARDS` env default is still 4), confirming the fix is needed.

- [ ] **Step 3: Implement the `iitgpu/validate.py` change**

Keep the existing `MAX_GPU_SHARDS` env var as a defensive fallback, but check live pod count first. Change:

```python
MAX_GPU_SHARDS = int(os.environ.get("MAX_GPU_SHARDS", "4"))
```

to add, right after it:

```python
MAX_GPU_SHARDS = int(os.environ.get("MAX_GPU_SHARDS", "4"))


def _max_gpu_shards() -> int:
    """Live ceiling: never more shards than the cluster currently has pods.
    Falls back to the MAX_GPU_SHARDS env var if scontrol is unreachable."""
    try:
        from iitgpu.pods import pod_count
        from iitgpu.slurm import get_node_stats
        stats = get_node_stats()
        if stats is not None:
            return pod_count(stats)
    except Exception:
        pass
    return MAX_GPU_SHARDS
```

Then change the check site (around the `if key == "gres" and val.lower().lstrip("-").startswith("shard"):` block) from:

```python
                if key == "gres" and val.lower().lstrip("-").startswith("shard"):
                    if n > MAX_GPU_SHARDS:
                        errors.append(
                            f"--gres requests {n} GPU slices; "
                            f"the cluster has {MAX_GPU_SHARDS}")
```

to:

```python
                if key == "gres" and val.lower().lstrip("-").startswith("shard"):
                    ceiling = _max_gpu_shards()
                    if n > ceiling:
                        errors.append(
                            f"--gres requests {n} GPU slices; "
                            f"the cluster has {ceiling}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_validate.py -v`
Expected: all PASS, including both new tests.

- [ ] **Step 5: Commit**

```bash
git add iitgpu/validate.py tests/test_validate.py
git -c user.name="Daham Dissanayake" -c user.email="dahamdissanayake05@gmail.com" commit -m "fix(validate): shard-request ceiling tracks live pod count

Was pinned to a fixed MAX_GPU_SHARDS env var (default 4) regardless of
the cluster's actual current pod split. Falls back to that env var only
when live stats are unreachable."
```

---

### Task 7: Full regression, deploy, and live verification

**Files:** none (verification-only task)

- [ ] **Step 1: Run the full pytest gate**

On the login node: `cd ~/IIT-Secure-SLURM-Job-Gateway && python3 -m pytest -q`
Expected: all tests pass (baseline was 786 before this plan; expect the count to grow by roughly the number of new tests added across Tasks 1-6, and no failures/errors).

- [ ] **Step 2: Deploy**

`bash deploy/redeploy-igm.sh` (run as `slurmadmin`, **not** via sudo).
Expected: gate passes, deploy completes, no manual post-gate steps needed (per this project's established working invocation).

- [ ] **Step 3: Live-verify nothing regressed on today's real 4-pod cluster**

As a real provisioned user (not `root-daham`/`slurmadmin`), submit two jobs each requesting a partial pod share and confirm they run concurrently with the expected `AllocTRES`, e.g.:

```bash
ssh slurmadmin@192.168.122.10
sudo -u dahamadmin sbatch --wrap="sleep 120" --gres=shard:2 --cpus-per-task=16 --mem=28G --partition=gpu -J pod-verify-a
sudo -u yenuli   sbatch --wrap="sleep 120" --gres=shard:2 --cpus-per-task=16 --mem=28G --partition=gpu -J pod-verify-b
squeue -o "%.8i %.9P %.20j %.8u %.2t %.10M %b"
```

Expected: both `ST=R` simultaneously (2+2=4 shards, exactly the pod count — should NOT queue), `scontrol show node` reports `AllocTRES` summing correctly. Cancel both afterward: `sudo scancel <id1> <id2>`.

- [ ] **Step 4: Confirm no stray references remain**

`grep -rn "SHARDS_PER_GPU\|_SLICE_CPUS\|_SLICE_MEM_GB" iitgpu/ tests/` on the login node.
Expected: no output (clean removal). If anything remains, it's a missed call site from Tasks 2-6 — fix it and re-run the full gate before considering this plan done.

---

**Explicit follow-up:** admin-configurable pod count (resizing N, the occupancy grid, the per-user pod cap, and `deploy/resize-pods.sh`) is Plan B — `docs/superpowers/plans/2026-07-27-gpu-pod-admin-resize.md` — built on top of this plan's `pods.py`.
