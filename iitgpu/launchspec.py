# iitgpu/launchspec.py — pure launch-flow logic: no prompts, no I/O beyond
# the recents scan. The review hub (review.py) renders what this produces.
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from iitgpu.jobs import SHARDS_PER_GPU, JobSpec


@dataclass(frozen=True)
class Size:
    name: str
    gpu_shards: int
    cpus: int
    mem_gb: int
    default_time: str


SIZES: dict[str, Size] = {
    "small":    Size("Small", 1, 4, 8, "02:00:00"),
    "standard": Size("Standard", 1, 8, 14, "04:00:00"),
    "whole":    Size("Whole GPU", SHARDS_PER_GPU, 16, 60, "08:00:00"),
}

_INTENT_DEFAULTS = {  # (size key, time override or None)
    "notebook": ("standard", "06:00:00"),
    "batch":    ("standard", None),
    "shell":    ("small", None),
}


@dataclass
class LaunchSpec:
    intent: str
    script: str = ""
    env_kind: str = "prebuilt"      # prebuilt | conda | venv | container | none
    conda_env: str = ""
    venv_path: str = ""
    container_image: str = ""
    gpu_shards: int = 1
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


def apply_size(ls: LaunchSpec, name: str) -> None:
    s = SIZES[name]
    ls.gpu_shards, ls.cpus, ls.mem_gb = s.gpu_shards, s.cpus, s.mem_gb
    ls.time_limit = s.default_time


def default_spec(intent: str) -> LaunchSpec:
    size_key, time_override = _INTENT_DEFAULTS.get(intent, ("standard", None))
    ls = LaunchSpec(intent=intent)
    apply_size(ls, size_key)
    if time_override:
        ls.time_limit = time_override
    return ls


def size_name_for(ls: LaunchSpec) -> str | None:
    for key, s in SIZES.items():
        if (ls.gpu_shards, ls.cpus, ls.mem_gb) == (s.gpu_shards, s.cpus, s.mem_gb):
            return key
    return None


def _frac(shards: int) -> str:
    if shards >= SHARDS_PER_GPU:
        return f"{SHARDS_PER_GPU}/{SHARDS_PER_GPU} GPU"
    if shards == 1 and SHARDS_PER_GPU == 4:
        return "¼ GPU"
    return f"{shards}/{SHARDS_PER_GPU} GPU"


def size_label(ls: LaunchSpec) -> str:
    key = size_name_for(ls)
    name = SIZES[key].name if key else "Custom"
    return f"{name} — {_frac(ls.gpu_shards)} · {ls.cpus} CPU · {ls.mem_gb} GB"


def _slices_free(stats) -> int | None:
    if stats and getattr(stats, "shard_total", 0):
        return max(0, stats.shard_total - stats.shard_alloc)
    return None


def availability_line(stats) -> str:
    free = _slices_free(stats)
    if free is None:
        return "GPU availability unknown"
    return f"GPU now: {free}/{stats.shard_total} slices free"


def size_availability(gpu_shards: int, stats) -> str:
    free = _slices_free(stats)
    if free is None:
        return "— availability unknown"
    if gpu_shards <= free:
        return f"— starts now ({free} slices free)"
    return f"— will queue (needs {gpu_shards} free, {free} free)"


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


def from_template(tdata: dict) -> LaunchSpec:
    intent = "notebook" if tdata.get("task_type") == "notebook" else "batch"
    ls = default_spec(intent)
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


def from_rerun(parsed: dict, script: str) -> LaunchSpec:
    ls = default_spec("batch")
    ls.script = script
    for f_ in ("gpu_shards", "cpus", "mem_gb"):
        if parsed.get(f_) is not None:
            setattr(ls, f_, int(parsed[f_]))
    for f_ in ("time_limit", "array", "dependency"):
        if parsed.get(f_):
            setattr(ls, f_, parsed[f_])
    return ls
