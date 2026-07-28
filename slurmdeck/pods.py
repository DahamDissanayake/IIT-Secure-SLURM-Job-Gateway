# slurmdeck/pods.py — pure pod-sizing math, no I/O. Pod count and per-pod CPU/RAM
# are always derived from a live NodeStats snapshot (itself read from
# `scontrol show node` by slurmdeck.slurm.get_node_stats), never from a stored
# constant. This is the ONE place that math happens; jobs.py, launchspec.py,
# review.py and validate.py all call in here instead of doing their own
# division.
from __future__ import annotations
from dataclasses import dataclass

from slurmdeck.slurm import NodeStats

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


def pod_count_known(stats: NodeStats | None) -> bool:
    """True only when live SLURM state actually told us how the node is split.

    pod_count() answers 1 for BOTH "the node really has one pod" and "we could
    not read the node at all" -- a safe floor for arithmetic, but indistinguishable
    to a caller. Anything that would otherwise assert a fraction to the user
    ("the whole GPU"), resize a job, or impose a ceiling must check this first
    and degrade to "unknown" instead of asserting something it cannot know.
    """
    return stats is not None and getattr(stats, "shard_total", 0) > 0


def pod_resources(stats: NodeStats | None) -> PodSize:
    """CPU/RAM for a single pod, floor-divided from the node's real totals.

    With no stats this returns a deliberately minimal 1 CPU / 1 GB: the pure
    math layer has nothing to divide, and under-promising beats inventing a
    size. It is NOT a sensible default to hand a real job -- callers that size
    a LaunchSpec/JobSpec must gate on pod_count_known() and keep their own
    defaults rather than overwrite them with this (see launchspec.apply_pods).
    """
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
