"""Pure pod-sizing math: cpu/mem/VRAM per pod, derived live from NodeStats,
never from a hardcoded constant."""
from slurmdeck.pods import (PodSize, estimated_vram_gb, fits_new_pod_count,
                          pod_count, pod_count_known, pod_resources,
                          resources_for)
from slurmdeck.slurm import NodeStats


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


def test_pod_count_known_separates_a_real_one_pod_node_from_no_reading():
    """pod_count() answers 1 for both "one pod" and "no idea". Callers that
    would otherwise assert a fraction, resize a job or set a ceiling need to
    tell those apart -- conflating them is what made an unreachable cluster
    report "the whole GPU" and shrink jobs to 1 CPU / 1 GB."""
    assert pod_count_known(_stats(shard_total=1)) is True
    assert pod_count_known(_stats(shard_total=4)) is True
    assert pod_count_known(None) is False
    assert pod_count_known(_stats(shard_total=0)) is False


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
    # Use a fixture where memory is the scarcer resource, so the mem_gb < 1
    # branch (not the CPU branch) is actually triggered. With cpu_total=64 and
    # usable_mem_gb=8 (10240 MB / 1024 - 2), N=9 gives cpus=7 (passes) but
    # mem_gb=0 (fails mem check).
    stats = _stats(cpu_total=64, mem_total_mb=10240)
    ok, msg = fits_new_pod_count(9, stats)
    assert not ok and "0 GB" in msg


def test_fits_new_pod_count_accepts_reasonable_n():
    stats = _stats(cpu_total=32, mem_total_mb=62000)
    ok, msg = fits_new_pod_count(5, stats)
    assert ok
    assert "6 CPU" in msg and "11 GB" in msg


def test_fits_new_pod_count_rejects_below_one():
    stats = _stats(cpu_total=32, mem_total_mb=62000)
    ok, _ = fits_new_pod_count(0, stats)
    assert not ok
