# IIT GPU Manager — Multi-Tenancy Hardening

Status: approved, ready for implementation planning
Date: 2026-07-25
Baseline: `main` @ v1.0.4 (1efef71), 648 tests passing

## Goal

Splitting the GPU four ways (v1.0.2–v1.0.4) turned the cluster from effectively
single-user into genuinely multi-tenant. Several latent single-user assumptions
became live problems the moment more than one person could hold the card at
once. This spec establishes one access model across the whole share, closes the
three findings that expose data, and fixes three clarity defects that make the
interface misdescribe what a user is getting.

Every finding below was reproduced on the live cluster. None is inferred from
reading code.

## The access model

One rule, applied everywhere:

- **Everything in `/shared` except per-user areas is shared.** `data`,
  `datasets`, `models`, `envs`, `templates`, `scripts`, `images` are readable by
  every `gpuusers` member and writable by the group, so people can contribute
  datasets and models collaboratively. Only `other` loses write.
- **Per-user areas belong to their owner and to admins, nobody else.**
  `/shared/users/<name>` and `/shared/jobs/<name>` are accessible to `<name>`
  and to `gpuadmins`. No other user can read or write them.

Job folders are included deliberately: they hold the user's scripts and job
output, which is the same private content as their home. Locking `users/` while
leaving `jobs/` group-readable would be half a fix — the same data would remain
readable by another path.

## Scope

In scope:

- **P0-1** Per-user areas are readable and writable by other users.
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

## Environment constraints

These were verified and they dictate how the work must be carried out.

**`root_squash` is set on the NFS export.** From `/etc/exports`:

```
/mnt/nvme_storage/shared 192.168.122.0/24(rw,sync,no_subtree_check,root_squash)
```

Root on the login node is squashed and cannot modify `/shared`; `sudo mkdir`
there fails with `Permission denied`. **All ownership and mode changes must run
on the GPU host**, which is the NFS server. `/shared` there is a symlink to
`/mnt/nvme_storage/shared`, so operating on either path works locally as root.

Consequence: `redeploy-igm.sh` runs on the login node and therefore **cannot
repair** permissions. It can only detect drift and report it.

`iit-gpu-adduser.sh` already accounts for this — it ssh's to the GPU host to
create the user area (lines 124–127). That path is correct and only its mode and
group need changing.

**`gpuadmins` membership differs between nodes.** This blocks the whole model:

```
GPU host    gpuadmins:x:1501:daham
login node  gpuadmins:x:1501:daham,slurmadmin,dahamadmin,indrajith
```

Notebooks and jobs run on the GPU host, so `dahamadmin` — the admin tool account
— is not an admin where it matters. Proven directly against a probe directory
owned `yenuli:gpuadmins` mode `2750`:

```
GPU host, before:  yenuli CAN · daham CAN · dahamadmin denied · hassan denied
GPU host, after adding dahamadmin to gpuadmins:  dahamadmin CAN access
```

(The probe and the membership change were both reverted after testing.)

Syncing `gpuadmins` across both nodes is therefore a **prerequisite**, not a
side task. `gpuusers` is already matched at GID 1500; `gpuadmins` exists at GID
1501 on both but with different members.

## Verified findings

### P0-1 — Per-user areas are world-accessible

Newer accounts provision correctly. Four older ones predate that fix:

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

Job folders leak the same class of content to every `gpuusers` member:

```
2770 public:gpuusers  /shared/jobs
$ sudo -u yenuli ls /shared/jobs/dahamadmin
notebook_20260605_115559  notebook_20260605_123333  …
```

The shared asset directories are world-writable, so the "read-only shared
assets" this design symlinks into notebooks are not read-only today:

```
777 public:gpuusers  /shared/{data,envs,models,templates}
775                  /shared/datasets
```

### P0-2 — The notebook bypasses the per-user jail

```
TUI jail   ['/shared/users/yenuli', '/shared/models', '/shared/envs']
notebook   --notebook-dir=/shared

GET /api/contents → templates, images, scripts, data, models,
                    jobs, miniforge3, datasets, users, tmp, envs
```

The jail is enforced in one surface and ignored in the other.

### P0-3 — No record that an interactive session ran

Two constraints shaped this design:

- Jupyter Server 2.19 ships event schemas for `contents_service`,
  `gateway_client` and `kernel_actions` only. **There is no terminal event
  schema**, so logging terminal creation means subclassing internal handlers.
- The audit socket is `/run/iit-gpu/audit.sock` on the **login node** only.
  `/run` is per-host tmpfs and the GPU host has no `/run/iit-gpu`, so a job
  **cannot** reach the daemon.

Disabling the terminal API was considered and rejected: a notebook cell can run
`os.system`, `subprocess` or `!cmd` regardless, so the toggle buys appearance
rather than containment. Containment comes from P0-1 and P0-2.

### P2 — Three clarity defects

- `gpu_share_note()` is defined in `jobs.py` with **zero call sites**.
- The pre-submit VRAM check compares the estimate against *total* free VRAM.
  With four tenants that is not the user's to spend, and shards do not cap VRAM.
- `jobs.py:332` claims JupyterLab "Binds JupyterLab to 127.0.0.1 only (not
  exposed to network)". It has bound the routable NodeAddr since the tunnel fix.

## Design

### P0-1 — Permissions

All commands run **on the GPU host** as root.

Prerequisite — sync admin group membership on the GPU host:

```
usermod -aG gpuadmins slurmadmin
usermod -aG gpuadmins dahamadmin
usermod -aG gpuadmins indrajith
```

Per-user areas, for every user including `public`:

```
chown <user>:gpuadmins /shared/users/<user>  /shared/jobs/<user>
chmod 2770             /shared/users/<user>  /shared/jobs/<user>
```

`2770` gives the owner and `gpuadmins` full access and `other` nothing. Admins
get write rather than read-only because they already have "Log in as user"
(`sudo -H -u <user> /usr/local/bin/iit-gpu-manager`), so a read-only mode would
be a restriction they can bypass through a supported feature — the mode should
state what is actually true. The setgid bit makes new files inside inherit
`gpuadmins`, keeping admin access working for anything created later.

`public` is **not** exempt. It is a normal account under this model; anyone
logging in as `public` still has full access to its area.

Shared asset directories become group-writable, world-read-only:

```
/shared/data       777 → 2775
/shared/envs       777 → 2775
/shared/models     777 → 2775
/shared/templates  777 → 2775
/shared/datasets   775 → 2775   (already not world-writable; normalised for
                                 setgid group inheritance)
```

Parent directories `/shared/users` and `/shared/jobs` stay traversable so users
can reach their own area. Folder *names* therefore remain listable. That is
accepted: the names are usernames, already known from the roster, and removing
read from the parent would break each user's path to their own folder.

Provisioning and creation paths change to match:

- `iit-gpu-adduser.sh` (lines 124–127) currently does `chown $NEW_UID:$NEW_UID`
  and `chmod 0700`. It becomes `chown $NEW_UID:gpuadmins` and `chmod 2770`, and
  asserts the result.
- `jobs.make_job_folder` currently chmods `0770` and chowns the group to
  `gpuusers`. It becomes group `gpuadmins` with mode `2770`. Note: the
  `gpuusers` group was originally chosen so a shared submit account could read
  the script under `gateway_shared_user` mode. That mode is **off** here
  (`shared_user_mode=False`, `_gateway_prefix()` returns `[]`), so the change is
  safe — but if shared-user submission is ever enabled, this needs revisiting.

Drift detection lives in `redeploy-igm.sh`, which can only report because of
`root_squash`: it fails the deploy if any `/shared/users/*` or `/shared/jobs/*`
is accessible to `other`, printing the exact remediation command to run on the
GPU host.

### P0-2 — Notebook jail

`render_notebook_sbatch` switches from `--notebook-dir=/shared` to
`--ServerApp.root_dir=/shared/users/$USER`. Both forms were confirmed working on
Jupyter Server 2.19.

Shared assets reach the user through symlinks created by the job script itself,
immediately before launch — idempotent, self-healing for existing accounts, and
requiring no backfill migration:

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
environment, with what — without claiming per-command capture or a server-ready
timestamp, neither of which is obtainable from the login node.

### P2 — Clarity

- Wire `gpu_share_note()` into the wizard's `panel("Job Summary", …)` and the
  notebook confirm step, so the user reads "1/4 of the GPU (3/4 left for
  others)" before submitting.
- Reframe the VRAM check around a slice budget (~8 GB of 32 GB) and state that
  VRAM is shared between concurrent jobs and not enforced, so the figure is a
  courtesy budget rather than a guarantee.
- Correct the `jobs.py:332` docstring: bound to the node's SLURM NodeAddr,
  reachable from the gateway network, gated by a per-job random token.

## Testing

Unit tests, in the existing `tests/` layout:

- The generated notebook script roots at `/shared/users/$USER` and never emits
  `--notebook-dir=/shared`.
- The symlink block is present, uses `ln -sfn`, and covers all four assets.
- `user_browse_roots` includes `data` and `datasets`.
- `make_job_folder` produces mode `2770` with group `gpuadmins`.
- A `notebook_session_start` audit event is emitted on notebook submit.
- The job summary contains the GPU share wording.
- The VRAM prompt mentions the per-slice budget and that VRAM is unenforced.

Deploy-time check, because pytest cannot see real filesystem state:

- `redeploy-igm.sh` fails if any `/shared/users/*` or `/shared/jobs/*` grants
  access to `other`, and prints the GPU-host command to fix it.

Live verification, re-running the exact probes that found the defects:

- `sudo -u yenuli` read and write against `dahamadmin`'s home and job folder now
  both fail.
- `sudo -u daham` (gpuadmins) and `sudo -u dahamadmin` (after the group sync)
  can both still access them, **on the GPU host**.
- `GET /api/contents` as a non-admin lists only the user's folder plus the four
  symlinks.
- A notebook launches, reaches `HTTP 200` with its token, and reads a dataset
  through the `data` symlink.

## Rollout

Ordered so each step is independently revertible and the only user-visible
change ships last.

1. **Group sync** — add `slurmadmin`, `dahamadmin`, `indrajith` to `gpuadmins`
   on the GPU host. Prerequisite for everything else; reverting is `gpasswd -d`.
2. **Permissions** — per-user areas to `2770 <user>:gpuadmins`, shared assets to
   `2775`, on the GPU host. Provisioning and `make_job_folder` updated, drift
   check added. Reverting is a `chmod` back.
3. **Clarity** — share note, VRAM wording, docstring. Display-only.
4. **Notebook jail** — `root_dir` switch, symlinks, widened TUI jail. Last
   because it changes what users see; reverting restores `--notebook-dir`.

Risks:

- Locking previously-open areas may break an undiscovered workflow that relied
  on cross-user access. Recent activity in the exposed homes is low (7, 2 and 0
  items touched in 30 days).
- `public` loses its informal shared-drop-space role. If that turns out to be
  needed, the replacement is an explicit `/shared/scratch` (group `gpuusers`,
  mode `2770`, sticky) rather than a user home doubling as one.
- Changing `make_job_folder`'s group to `gpuadmins` is safe only while
  `gateway_shared_user` is off. Enabling shared-user submission later requires
  revisiting it.
