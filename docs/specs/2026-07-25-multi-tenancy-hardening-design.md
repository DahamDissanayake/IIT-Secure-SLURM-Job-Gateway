# IIT GPU Manager — Multi-Tenancy Hardening

Status: approved, ready for implementation planning
Date: 2026-07-25
Baseline: `main` @ v1.0.4 (1efef71), 648 tests passing

## Goal

Splitting the GPU four ways (v1.0.2–v1.0.4) turned the cluster from effectively
single-user into genuinely multi-tenant. Several latent single-user assumptions
became live problems the moment more than one person could hold the card at
once. This spec closes the three that expose data, plus three cheap clarity
fixes that stop the interface lying about what a user is getting.

Every finding below was reproduced on the live cluster. None is inferred from
reading code.

## Scope

In scope:

- **P0-1** Legacy home directories are world-readable and writable.
- **P0-2** JupyterLab serves all of `/shared`, bypassing the per-user jail.
- **P0-3** No audit record that an interactive session was started.
- **P2-1** `gpu_share_note()` is dead code; users are never told they get a quarter of the GPU.
- **P2-2** The VRAM check predates sharing and now misleads.
- **P2-3** A docstring claims a security posture the code does not implement.

Out of scope, deferred by decision:

- **P1-1** The dashboard's `E=+2h` extend can never succeed. SLURM restricts
  `TimeLimit` increases to operators and `_gateway_prefix()` is empty in this
  deployment; the notebook default of 8h also equals the QOS `MaxWall`.
- **P3-1 / P3-2** Five-step connect ritual; `RUNNING` does not mean ready.

Non-goals:

- Hard per-job VRAM isolation. Shards schedule, they do not isolate. That would
  need NVIDIA MPS, and MIG is unavailable on a GeForce card.
- Rotating the compromised GitHub PAT in the `origin` remote. Real, tracked
  separately, unrelated to this work.

## Verified findings

### P0-1 — Four legacy home directories are world-accessible

Newer accounts provision correctly at `2700`/`2770`. Four older ones were never
remediated:

```
777 dahamadmin   777 hassan   777 public   775 daham
2700 amasha  2700 hansika  2770 yenuli  2770 sankeetha  …
```

Confirmed cross-user read and write:

```
$ sudo -u yenuli head /shared/users/dahamadmin/train_resnet.py
#!/usr/bin/env python3
"""ResNet-50 on ImageNet-100 subset (Imagenette)."""

$ sudo -u yenuli touch /shared/users/dahamadmin/.probe
WRITE SUCCEEDED          # probe removed after testing
```

The shared asset directories are also world-writable, so the "read-only shared
assets" this design symlinks into notebooks are not read-only today:

```
777 public:gpuusers  /shared/data
777 public:gpuusers  /shared/envs
777 public:gpuusers  /shared/models
777 public:gpuusers  /shared/templates
```

### P0-2 — The notebook bypasses the per-user jail

The TUI file manager confines a user to their own folder plus read-only shared
assets. The notebook launcher roots JupyterLab at the entire share:

```
TUI jail   ['/shared/users/yenuli', '/shared/models', '/shared/envs']
notebook   --notebook-dir=/shared

GET /api/contents → templates, images, scripts, data, models,
                    jobs, miniforge3, datasets, users, tmp, envs
```

The jail is enforced in one surface and ignored in the other.

### P0-3 — No record that an interactive session ran

Notebook jobs are audited at submission like any job, but nothing marks them as
interactive execution environments. Two constraints shaped the design:

- Jupyter Server 2.19 ships event schemas for `contents_service`,
  `gateway_client` and `kernel_actions` only. **There is no terminal event
  schema**, so logging terminal creation means subclassing internal handlers.
- The audit socket lives at `/run/iit-gpu/audit.sock` on the **login node**
  only. `/run` is per-host tmpfs; the GPU host has no `/run/iit-gpu` at all, so
  a job **cannot** reach the daemon.

Disabling the terminal API was considered and rejected: a notebook cell can run
`os.system`, `subprocess`, or `!cmd` regardless, so the toggle buys appearance
rather than containment. Containment comes from P0-1 and P0-2.

### P2 — Three clarity defects

- `gpu_share_note()` is defined in `jobs.py` with **zero call sites**. Resource
  sizing changed materially and nothing in the interface says so.
- The pre-submit VRAM check compares the user's estimate against *total* free
  VRAM. With up to four tenants that number is not theirs to spend, and shards
  do not cap VRAM, so two jobs can still OOM each other while it reports plenty.
- `jobs.py:332` claims JupyterLab "Binds JupyterLab to 127.0.0.1 only (not
  exposed to network)". It has bound the routable NodeAddr since the tunnel fix.

## Design

### P0-1 — Permissions

Set `2700` on `/shared/users/{dahamadmin,hassan,daham}`, matching the
already-correct accounts.

`public` is **exempt and stays `777`**. It is a real login account (uid 1003, in
`gpuusers`) that appears to serve as an informal shared drop space; locking it
would remove a working habit without providing a replacement. This exemption is
deliberate and must be encoded, or the drift check below will fight it.

Shared asset directories move to `2775` — owner and `gpuusers` keep write so
people can still contribute datasets and models collaboratively; only `other`
loses write:

```
/shared/data       777 → 2775
/shared/envs       777 → 2775
/shared/models     777 → 2775
/shared/templates  777 → 2775
/shared/datasets   775 → 2775   (already not world-writable; normalised for
                                 setgid group inheritance)
```

Two guards, because filesystem state is not unit-testable:

- `iit-gpu-adduser.sh` asserts a newly created home is `2700` and fails loudly
  otherwise.
- `redeploy-igm.sh` gains a drift check that fails if any `/shared/users/*`
  other than `public` is group- or world-accessible.

### P0-2 — Notebook jail

`render_notebook_sbatch` switches from `--notebook-dir=/shared` to
`--ServerApp.root_dir=/shared/users/$USER`. Both forms were confirmed working
on Jupyter Server 2.19.

Shared assets reach the user through symlinks created by the job script itself,
immediately before launch — idempotent, self-healing for all existing accounts,
and requiring no backfill migration:

```bash
mkdir -p /shared/users/$USER
ln -sfn /shared/models   /shared/users/$USER/models
ln -sfn /shared/envs     /shared/users/$USER/envs
ln -sfn /shared/data     /shared/users/$USER/data
ln -sfn /shared/datasets /shared/users/$USER/datasets
```

Verified live that Jupyter Server follows symlinks pointing outside `root_dir`:

```
GET /api/contents/models → FOLLOWED — mistralai--Mistral-7B-v0.3, microsoft--phi-2, …
```

`validate.user_browse_roots` widens to include `data` and `datasets` so the file
manager and the notebook enforce one identical boundary. This is deliberately
*wider* than today's TUI jail, which blocks the shared datasets people train on.

Relative paths inside notebooks are unaffected: Jupyter gives a kernel its cwd
from the notebook file's own directory, not from `root_dir`.

Resulting view for a regular user:

```
/shared/users/yenuli          ← JupyterLab root
  my-work/          rw
  models   ->       /shared/models      (group-writable, world read-only)
  envs     ->       /shared/envs
  data     ->       /shared/data
  datasets ->       /shared/datasets

hidden: users/*  jobs/*  tmp  miniforge3  scripts  images
```

### P0-3 — Session audit

The wizard's notebook path emits, at submit time on the login node where the
socket is reachable:

```python
auditclient.log("notebook_session_start", job_id=<id>,
                meta={"env": <conda_env>, "gpu_shards": <n>})
```

This records the accountable act — who launched an interactive execution
environment, with what — and does not claim per-command capture or a
server-ready timestamp, neither of which is obtainable from the login node.

### P2 — Clarity

- Wire `gpu_share_note()` into the wizard's `panel("Job Summary", …)` and the
  notebook confirm step, so the user reads "1/4 of the GPU (3/4 left for
  others)" before submitting.
- Reframe the VRAM check around a slice budget (~8 GB of 32 GB) and state
  plainly that VRAM is shared between concurrent jobs and not enforced, so the
  figure is a courtesy budget rather than a guarantee.
- Correct the `jobs.py:332` docstring to describe the real posture: bound to the
  node's SLURM NodeAddr, reachable from the gateway network, gated by a per-job
  random token.

## Testing

Unit tests, in the existing `tests/` layout:

- The generated notebook script roots at `/shared/users/$USER` and never emits
  `--notebook-dir=/shared`.
- The symlink block is present, uses `ln -sfn` (idempotent), and covers all four
  assets.
- `user_browse_roots` includes `data` and `datasets`.
- A `notebook_session_start` audit event is emitted on notebook submit.
- The job summary contains the GPU share wording.
- The VRAM prompt mentions the per-slice budget and that VRAM is unenforced.

Deploy-time checks, because pytest cannot see real filesystem state:

- Home-directory mode drift, `public` exempted.

Live verification, re-running the exact probes that found the defects:

- `sudo -u <userA>` read and write attempts against `<userB>`'s home now fail.
- `GET /api/contents` as a non-admin lists only the user's folder plus the four
  symlinks.
- A notebook still launches, reaches `HTTP 200` with its token, and can read a
  dataset through the `data` symlink.

## Rollout

Ordered so the only user-visible change ships last, and each step is
independently revertible.

1. **Permissions** — homes and shared asset dirs, plus both guards. Reverting is
   a `chmod` back.
2. **Clarity** — share note, VRAM wording, docstring. Display-only, no
   behavioural risk.
3. **Notebook jail** — `root_dir` switch, symlinks, widened TUI jail. Shipped
   last because it changes what users see; reverting restores `--notebook-dir`.

Risks:

- Locking the three homes may break an undiscovered workflow that relied on
  cross-user access. Recent activity is low (7, 2 and 0 items touched in 30
  days), and `public` remains available as a shared space.
- Widening the TUI jail to `data`/`datasets` grants file-manager access users do
  not have today. This is intentional and matches what the notebook will show.
