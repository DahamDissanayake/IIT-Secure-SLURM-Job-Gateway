# iitgpu/launchspec.py — pure launch-flow logic: no prompts, no I/O beyond
# the recents scan. The review hub (review.py) renders what this produces.
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from iitgpu.jobs import JobSpec
from iitgpu.pods import pod_count, pod_count_known, resources_for


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
    """Set ls to request k pods, sizing cpus/mem_gb to match live node totals.

    When the live pod count is UNKNOWN (no stats, or SLURM reporting zero
    shards) cpus/mem_gb are left exactly as they are — the LaunchSpec dataclass
    defaults, or whatever a template/rerun already put there. pods.pod_resources
    answers a degenerate 1 CPU / 1 GB in that case, and writing that in would
    silently shrink every job the wizard launches without a reachable cluster
    to 1/1 (the C1 regression). gpu_shards still records what was asked for,
    floored at 1, since a pod request of "at least one" is safe to keep.
    """
    if not pod_count_known(stats):
        ls.gpu_shards = max(1, k)
        return
    ls.cpus, ls.mem_gb, ls.gpu_shards = resources_for(k, stats)


def default_spec(intent: str, stats=None) -> LaunchSpec:
    """A fresh LaunchSpec for *intent*, sized from live node stats when given.

    *stats* is optional and this module stays I/O-free by design: callers that
    can reach SLURM (wizard.py) pass `get_node_stats()` so sizing is live;
    callers that cannot get the LaunchSpec dataclass defaults (8 CPU / 14 GB),
    which is a reasonable pod-sized job, not the degenerate 1/1 fallback.
    """
    ls = LaunchSpec(intent=intent)
    apply_pods(ls, _INTENT_DEFAULT_PODS.get(intent, 1), stats)
    ls.time_limit = _INTENT_DEFAULT_TIME.get(intent, "04:00:00")
    return ls


def pod_label(ls: LaunchSpec, stats) -> str:
    plural = "" if ls.gpu_shards == 1 else "s"
    if not pod_count_known(stats):
        # We do not know how many pods the node is split into, so we cannot
        # claim any fraction of it — least of all "whole GPU", which is what
        # pod_count()'s floor of 1 would otherwise make this say.
        frac = "GPU share unknown"
    else:
        n = pod_count(stats)
        frac = "whole GPU" if ls.gpu_shards >= n else f"{ls.gpu_shards}/{n} GPU"
    return f"{ls.gpu_shards} pod{plural} — {frac} · {ls.cpus} CPU · {ls.mem_gb} GB"


def _slices_free(stats) -> int | None:
    if pod_count_known(stats):
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


def to_job_spec(ls: LaunchSpec, *, user: str, partition: str, job_name: str,
                task_type: str, run_command: str = "") -> JobSpec:
    return JobSpec(
        job_name=job_name, partition=partition,
        gpu_shards=ls.gpu_shards, cpus=ls.cpus, mem_gb=ls.mem_gb,
        time_limit=ls.time_limit, run_command=run_command, user=user,
        model_path=ls.model_path, conda_env=ls.conda_env, venv_path=ls.venv_path,
        task_type=task_type, container_image=ls.container_image,
        array=ls.array, dependency=ls.dependency, data_path=ls.data_path,
    )


_SCRIPT_RE = re.compile(r"(/[^\s\"']+\.(?:py|ipynb|sh))\b")


def recent_scripts(jobs_dir: str, user: str, limit: int = 5) -> list[str]:
    """Last distinct scripts referenced by the user's past jobs, newest first.

    Scans <jobs_dir>/<user>/*/job.sbatch for absolute paths ending .py/.ipynb/.sh
    and keeps only those that still exist on disk.
    """
    base = Path(jobs_dir) / user
    if not base.is_dir():
        return []
    seen: list[str] = []
    folders = sorted((d for d in base.iterdir() if d.is_dir()),
                     key=lambda d: d.stat().st_mtime, reverse=True)
    for d in folders:
        sb = d / "job.sbatch"
        if not sb.is_file():
            continue
        try:
            text = sb.read_text()
        except OSError:
            continue
        for m in _SCRIPT_RE.findall(text):
            if m not in seen and Path(m).is_file():
                seen.append(m)
                if len(seen) >= limit:
                    return seen
    return seen


# A template stores the JobSpec's task_type, which is the internal label, not
# the intent the user picked. "interactive" is a shell allocation — loading one
# as a batch job would silently turn a saved shell into a job that runs nothing.
_TEMPLATE_INTENT = {"notebook": "notebook", "interactive": "shell"}


def from_template(tdata: dict, stats=None) -> LaunchSpec:
    """A stored template normally carries its own gpu_shards/cpus/mem_gb; when a
    field is missing the base default_spec sizing shows through, so *stats* is
    threaded here too — live pod sizing when the caller has it, the LaunchSpec
    dataclass defaults (8 CPU / 14 GB) when it does not. Never 1 CPU / 1 GB."""
    intent = _TEMPLATE_INTENT.get(tdata.get("task_type", ""), "batch")
    ls = default_spec(intent, stats)
    for f_ in ("conda_env", "venv_path", "container_image", "data_path",
               "model_path", "array", "dependency"):
        if tdata.get(f_):
            setattr(ls, f_, tdata[f_])
    for f_ in ("gpu_shards", "cpus", "mem_gb"):
        if tdata.get(f_) is not None:
            setattr(ls, f_, int(tdata[f_]))
    if tdata.get("time_limit"):
        ls.time_limit = tdata["time_limit"]
    if tdata.get("extra_args"):
        ls.args = tdata["extra_args"]
    if tdata.get("run_command") and not ls.args:
        ls.args = ""
    return ls


def from_rerun(parsed: dict, script: str, stats=None) -> LaunchSpec:
    """Rebuild a launch from a previous job's parsed sbatch.

    "Re-run" has to mean re-run: carrying the sizing but dropping the
    environment, the data path and the arguments produces a job that looks like
    the original in the queue and does something else entirely.

    *stats* only backs the fields the parsed sbatch did not supply (same rule as
    from_template) — anything the old job actually recorded still wins.
    """
    ls = default_spec("batch", stats)
    ls.script = script
    for f_ in ("gpu_shards", "cpus", "mem_gb"):
        if parsed.get(f_) is not None:
            setattr(ls, f_, int(parsed[f_]))
    for f_ in ("time_limit", "array", "dependency",
               "conda_env", "venv_path", "container_image", "data_path"):
        if parsed.get(f_):
            setattr(ls, f_, parsed[f_])
    if parsed.get("extra_args"):
        ls.args = parsed["extra_args"]
    if ls.container_image:
        ls.env_kind = "container"
    elif ls.conda_env:
        ls.env_kind = "conda"
    elif ls.venv_path:
        ls.env_kind = "venv"
    return ls
