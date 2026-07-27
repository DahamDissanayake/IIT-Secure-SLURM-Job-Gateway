# GPU Pod Allocation — Admin-Configurable Shared GPU Slots

## Problem statement

Investigating a live "two jobs running slow" report traced the root cause to
the fixed GPU-sharing slice sizing introduced earlier (`SHARDS_PER_GPU=4`,
`_SLICE_CPUS=8`, `_SLICE_MEM_GB=14` in `iitgpu/jobs.py`): both running jobs'
14GB memory cgroups were pinned at their cap and swap-thrashing (confirmed via
cgroup `memory.stat` pgmajfault counts and live `iostat`/`vmstat` showing
~1.2GB/s sustained NVMe reads at ~85-90% iowait, while GPU compute utilization
sat at 1%). The fixed quarter-node slice was undersized for the actual
workloads, and — separately — is wasteful when only one user is on the
cluster, since they're still capped to a quarter of the node even with
nothing else running.

The fix direction agreed on: replace the fixed 4-way split with an
admin-configurable number of "pods" (N). Resources split evenly into N pods.
A user can select multiple pods for one job to get a proportional share, up
to what's actually free and up to a per-user cap. Resizing N is only allowed
when no jobs are running.

## Goals

- Admin can set pod count N from the admin panel; the whole node's CPU/RAM
  and the GPU's shard scheduling split evenly into N.
- A job can request 1..N pods (subject to what's free and a per-user cap),
  getting a proportional share of CPU/RAM and GPU scheduling weight.
- A lone job on an otherwise-idle cluster is not artificially capped below
  what pod selection allows it to request (existing whole-GPU task types
  already default to requesting all pods).
- Admin dashboard shows pod occupancy: which running job holds how many pods.
- Resizing N is blocked while any job is running, cluster-wide.
- No new persistent config value duplicates what SLURM already knows — pod
  count and per-pod sizing are always derived live from the cluster's actual
  reported state, never stored separately in app config.

## Non-goals (explicitly out of scope for this spec)

- **Enforced VRAM partitioning.** GPU shards remain a *scheduling* count, not
  a hardware/driver-level VRAM partition — that gap already exists today and
  is unrelated to pod count being configurable. Closing it requires NVIDIA
  MPS (`CUDA_MPS_PINNED_DEVICE_MEM_LIMIT`), which was discussed separately and
  is left as explicit future work. Any VRAM figure this feature displays is a
  labeled **estimate**, never an enforced ceiling.
- **CPU-only jobs.** They already request cpu/mem directly via `cons_tres`
  with no observed bottleneck; they are not pulled into the pod model.
- Runtime resize (changing N while jobs are running) — deliberately
  unsupported per the resolved decision below.

## Resolved decisions (from earlier discussion)

| Question | Decision |
|---|---|
| "New user only sees 1 pod free" | Live availability only — no special first-timer quota, just whatever's actually unclaimed. |
| Per-user cap on concurrent pods | Yes — enforced via QOS `MaxTRESPerUser=gres/shard=<cap>`, not app-side logic. |
| Uneven division (e.g. N=5) | Floor `total/N` per pod; remainder sits unused. |
| Resize cost (slurmd restart, brief downtime) | Acceptable, since it only runs when the queue is empty. |
| Do training jobs (`train`/`finetune`/`custom`) also go through pod selection? | Yes — unify all task types under "select k of N pods"; training just defaults to selecting all pods. |
| Do CPU-only jobs join the pod model? | No — left untouched. |
| Do named size presets (Small/Standard/Whole GPU) survive? | No — replaced entirely by the pod stepper. |
| Where does pod count N live? | Nowhere as separate config — always derived live from `scontrol show node` / `NodeStats.shard_total`, which itself comes from `gres.conf`/`slurm.conf`. SLURM's actual reported state is the only source of truth. |

## Architecture

Three layers:

1. **Cluster layer (SLURM config)** — `gres.conf` (`Name=shard Count=N`) and
   `slurm.conf`'s node `Gres=gpu:1,shard:N` line, identical on both the login
   node and the GPU host. Unchanged mechanism from today, just admin-driven
   instead of a one-time manual edit.
2. **Derivation layer (pure Python, no I/O)** — new `iitgpu/pods.py`. Given a
   `NodeStats` snapshot (already fetched by the existing `get_node_stats()`),
   it is the single place that computes pod count, per-pod CPU/RAM, and an
   estimated VRAM share for a requested pod count `k`. Nothing else hardcodes
   this math.
3. **Surface layer** — three consumers of the derivation layer: the
   submission wizard/review hub (pod stepper + live tally), the admin panel
   (new Pods screen: resize action, per-user cap, occupancy grid), and the
   dashboard/splash status line (already generic, needs one hardcoded
   `SHARDS_PER_GPU == 4` special-case string removed).

The per-user pod cap is enforced by SLURM itself (QOS `MaxTRESPerUser`), not
custom application code, so it holds even against a raw `sbatch` bypassing
the TUI entirely.

## Components

**New: `iitgpu/pods.py`** (pure, no I/O)
- `pod_count(stats: NodeStats) -> int` — reads `stats.shard_total`.
- `pod_resources(stats) -> PodSize` — `cpu = floor(cpu_total / pod_count)`,
  `mem_gb = floor(mem_total_gb / pod_count)`.
- `resources_for(k, stats) -> (cpu, mem_gb, gpu_shards)` and
  `estimated_vram_gb(k, stats)` (explicitly labeled as an estimate; reuses the
  math `review.py` already has in `_vram_note`, parameterized by live
  `pod_count` instead of the constant `SHARDS_PER_GPU`).
- Replaces `jobs.py`'s `SHARDS_PER_GPU`/`_SLICE_CPUS`/`_SLICE_MEM_GB`
  constants and the hardcoded fraction math in `gpu_share_note()`.

**`iitgpu/jobs.py`** — `TASK_DEFAULTS` becomes "default pod count per task
type" (train/finetune/custom default to all pods; notebook/interactive/
inference default to 1), resolved through `pods.py` instead of literal
`cpus=`/`mem_gb=` numbers.

**`iitgpu/launchspec.py`** — `SIZES` dict removed. `LaunchSpec` gains
`pods_requested: int` as the primary sizing field. `apply_size()`/
`size_label()`/`size_availability()` become pod-stepper equivalents
(`set_pods(ls, k)`, `pod_label(ls)`). `_frac()`'s hardcoded
`SHARDS_PER_GPU == 4` → "¼ GPU" special case is removed in favor of a
generic "1/N GPU" phrasing.

**`iitgpu/review.py`** — the hub's size row becomes a live stepper: current
pod selection plus resulting CPU/mem/VRAM-estimate, re-rendering on each
+/-. Smallest change of the surface layer since the estimate math already
exists here.

**`iitgpu/validate.py`** — `MAX_GPU_SHARDS` env var is replaced by "can't
exceed live `pod_count`"; the per-user ceiling comes from the QOS cap, with
validate.py giving a friendly pre-check message before SLURM would reject it.

**`iitgpu/admin.py`** — new "Pods" screen:
- Occupancy grid: which job holds how many pods. Pod-index assignment is a
  **UI-side rendering convention only** (first-come-first-served over
  currently running jobs, recomputed fresh every draw, never persisted) —
  SLURM itself only tracks a shard *count* per job, not a specific pod
  identity.
- Resize action: gated on cluster-wide `squeue` being empty, confirm dialog
  showing the new derived per-pod sizing before committing, then runs
  `deploy/resize-pods.sh`.
- `set_qos_maxgpu()` is replaced by a generalized
  `set_qos_merge_tres(qos_name, **tres_updates)` that reads the current
  `MaxTRESPerUser`, merges in only the changed component, and writes the full
  string back — fixing today's latent bug where setting one TRES component
  silently drops any other (e.g. would currently wipe an existing shard cap
  when setting a GPU cap).

**New: `deploy/resize-pods.sh`** — mirrors this project's existing cross-node
admin-script pattern: backs up `gres.conf`/`slurm.conf` on both nodes
(timestamped, same convention as prior live changes), rewrites
`Count=N`/`Gres=gpu:1,shard:N` identically on both, restarts `slurmctld`
(login node) then `slurmd` (GPU host) in that order, runs `scontrol update
State=RESUME` unconditionally (a GRES change reliably drains the node
otherwise), and verifies via `slurmd -G` + `scontrol show node`.

**`iitgpu/splash.py`/`dashboard.py`** — status line is already generic
(`GPU {free}/{shard_total} slices free`); only the `_frac()` special case in
`launchspec.py` needs fixing.

## Data flow

**1. Admin resize (N → N')**
1. Admin opens Pods screen, sees live `pod_count`, occupancy grid, resize
   action.
2. Enters N'. `admin.py` checks `squeue` cluster-wide is empty; refuses
   otherwise.
3. Confirm dialog shows the real derived per-pod CPU/mem for N' (computed
   against the same live `NodeStats` totals) before committing.
4. `admin.py` invokes `resize-pods.sh` (`sudo -n`, same pattern as other
   privileged admin actions): backup → rewrite both nodes → restart in order
   → resume → verify.
5. Every subsequent `get_node_stats()` call anywhere in the app automatically
   reflects N' — nothing else stores N independently, so nothing else needs
   to be told about the change.
6. Existing caveat still applies: an already-open long-running TUI session
   keeps whatever it had cached at launch; a resize doesn't retroactively
   update a session mid-flow.

**2. Job submission (user selects k pods)**
1. Review hub reads live `pod_count` + free pods
   (`shard_total - shard_alloc`) via `get_node_stats()`.
2. Stepper defaults `k` per task type (overridable).
3. Each +/- recomputes cpu/mem/gpu_shards/VRAM-estimate via `pods.py` — pure,
   no round trip.
4. `k` is capped by two independent, separately-enforced ceilings: pods
   actually free right now (else the job queues, existing behavior
   unchanged), and the user's QOS `MaxTRESPerUser=gres/shard=<cap>` (enforced
   by SLURM itself — if the wizard's own pre-check and SLURM ever disagree,
   SLURM's rejection is shown verbatim, never a different app-invented
   number).
5. `to_job_spec()` builds `JobSpec` from `pods_requested` instead of a named
   `SIZES` key.

**3. Admin occupancy grid**
1. `admin.py` reads all running jobs on the node (existing
   `get_jobs_on_node()` pattern) and each job's `gres/shard=k` allocation.
2. Assigns each job to the next `k` unassigned pod cells by job start time —
   a rendering convention recomputed fresh every draw, nothing persisted.
3. Unassigned cells render as free.

## Error handling

**Resize preconditions and races**
- Empty-queue check happens twice: once for the confirm dialog, again
  atomically immediately before `resize-pods.sh` executes — if a job slipped
  in between, the script aborts before touching any config and reports which
  job blocked it.
- Sanity floor: before confirming, reject/strongly-warn on an N that would
  floor per-pod CPU or mem to a degenerate value (e.g. N=40 → 0 CPU/pod),
  using the same `pods.py` math ahead of time.

**Resize execution failure**
- Configs backed up (timestamped) on both nodes before either is rewritten.
- Login-node `slurmctld` restart failure: script stops before touching the
  GPU host; admin sees the raw restart error.
- GPU-host `slurmd` restart failure after the login node already succeeded
  (the dangerous half-applied state — matches the "stale DRAIN after a GRES
  change" gotcha from the original sharding rollout): script detects the
  mismatch via `slurmd -G` vs. intended count and auto-rolls back **both**
  nodes to the backed-up configs, restarting again — admin is told explicitly
  "rollback performed" vs. "resize applied," never a bare "done."
- `scontrol update NodeName=... State=RESUME` always runs after a successful
  rewrite, since a GRES change reliably drains the node.
- Concurrent resize attempts: `resize-pods.sh` takes a lockfile so a second
  invocation fails fast with "resize already in progress" instead of
  interleaving two rewrites.

**Job submission time**
- `get_node_stats()` can already return `None`; wizard/review hub falls back
  to the same degraded "cluster stats unavailable" mode the dashboard already
  uses, rather than showing wrong numbers.
- More pods requested than currently free: unchanged existing behavior — job
  queues on `Reason=Resources`.
- More pods requested than the user's QOS cap: SLURM rejects at `sbatch`
  time; the wizard surfaces SLURM's actual rejection text.

**QOS merge fix**
- `set_qos_merge_tres()` reads the current `MaxTRESPerUser` first; if that
  read fails, it refuses to write rather than guessing or defaulting —
  never silently drops an existing cap the way today's `set_qos_maxgpu()`
  can.

## Testing

- `iitgpu/pods.py`: pure unit tests against fixture `NodeStats` values —
  division/floor rounding, the degenerate-N sanity check, VRAM-estimate math.
- `set_qos_merge_tres()`: regression test proving it preserves an existing
  TRES component when setting a different one (the exact bug being fixed).
- Occupancy-grid assignment: pure function over a fake job list, deterministic,
  no live cluster needed.
- `resize-pods.sh` itself isn't unit-testable against real SLURM in CI —
  follows this project's established precedent (full pytest gate on the dev
  clone, then one real live resize plus two real concurrent jobs verified
  end-to-end on the actual cluster before considering it done, same as the
  original GPU-sharing and cons_tres rollouts).

## Rollout

Build on the login node's dev clone (`~/IIT-Secure-SLURM-Job-Gateway`), full
pytest gate must pass, deploy via `redeploy-igm.sh` as `slurmadmin` (not
sudo), then live-verify: resize N once with the queue empty, submit two real
jobs as real provisioned users each taking a partial pod share, confirm both
run concurrently with the expected `AllocTRES`, and confirm the admin
occupancy grid renders correctly against them.

## Explicit follow-ups (not in this spec)

- NVIDIA MPS integration for real (enforced) VRAM/compute isolation per pod.
- Any policy beyond "floor division, live-derived N" for pod sizing.
