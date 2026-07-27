# IIT-GPU-Manager

Secure terminal gateway for SLURM GPU job submission at IIT. Users SSH in and land directly in a forced-command TUI — they can never drop to a real shell (unless explicitly provisioned as a shell user, see below). Every action is logged before it is executed, GPU capacity is split into shares so more than one person can run at once, and every user's files are jailed to their own private area on the shared filesystem.

**Current deployed cluster:**

| | |
|---|---|
| Login / gateway node | `192.168.122.10` — where users SSH in, where SLURM's `slurmctld` runs, where this repo lives |
| GPU / compute node | `192.168.122.1` (`iit-MS-7E06`) — where jobs actually run, and the NFS server for `/shared` |
| GPU | 1× RTX 5090, split into **4 shares** (`gres/shard`) so up to 4 job slices can run at once |
| CPU / RAM | 32 CPUs / 62 GB RAM total on the compute node |
| SLURM | 25.11.2, `cons_tres` scheduling (jobs get exactly the CPU/mem/GPU-share they ask for, not the whole node) |
| Shared storage | NFSv4 export mounted at `/shared` on both nodes |
| Public SSH access | `ssh -p <GATEWAY_PORT> <username>@<GATEWAY_HOST>` (site-specific, see [Configuration Reference](#configuration-reference)) |

---

## Table of Contents

1. [What this is](#what-this-is)
2. [Architecture](#architecture)
3. [Linux users, groups & access model](#linux-users-groups--access-model)
4. [Using the tool — full menu reference](#using-the-tool--full-menu-reference)
   - [Main Menu](#main-menu)
   - [1. New Job](#1-new-job)
   - [2. My Workspace](#2-my-workspace)
   - [3. Jobs](#3-jobs)
   - [4. Settings](#4-settings)
   - [5. Admin](#5-admin-admins-only)
5. [Linux machine setup (before installation)](#linux-machine-setup-before-installation)
6. [Installation](#installation)
7. [Installing via an AI coding agent (igm-installer skill)](#installing-via-an-ai-coding-agent-igm-installer-skill)
8. [Adding and removing users](#adding-and-removing-users)
9. [Configuration reference](#configuration-reference)
10. [Audit logging](#audit-logging)
11. [Mail / notifications](#mail--notifications)
12. [GPU sharing model](#gpu-sharing-model)
13. [Demo mode (no SLURM required)](#demo-mode-no-slurm-required)
14. [Running the test suite](#running-the-test-suite)
15. [Project layout](#project-layout)
16. [Security model / bypass-test checklist](#security-model--bypass-test-checklist)
17. [Day-2 operations (maintainers)](#day-2-operations-maintainers)

---

## What this is

Users with no Linux or SLURM knowledge SSH in, build a conda/venv/container environment, upload a dataset, and launch a JupyterLab session, a batch script, or an interactive shell on the GPU — entirely through menus. The tool then:

- Renders and submits the `sbatch`/`srun` script for them
- Watches the job come up and hands back a ready-to-use SSH tunnel + browser link (for JupyterLab/TensorBoard)
- Shows a live dashboard of everyone's jobs, streams logs, and lets you cancel/extend your own session
- Logs every action (login, job submit/cancel, file access, admin action) to an audit trail nobody but admins can read
- Keeps every regular user confined to their own folder on the shared filesystem — they cannot browse, read, or write anyone else's files, even though everyone shares one NFS export

Admins (a separate Linux group, see below) get everything above plus a full admin panel: provisioning/offboarding users, draining/resuming the compute node, cluster-wide usage reports, and cancelling *any* user's job.

---

## Architecture

```
                          ┌─────────────────────────────┐
   SSH (users)  ────────► │   Login node 192.168.122.10 │
   port 2225              │   sshd  ForceCommand         │
                          │   ├─ iit-gpu-gateway (scp/   │
                          │   │   rsync/sftp passthrough,│
                          │   │   or launch the TUI)     │
                          │   ├─ iit-gpu-manager (TUI,    │
                          │   │   PYTHONPATH=/opt/iit-gpu)│
                          │   ├─ slurmctld                │
                          │   └─ iit-gpu-audit.service     │
                          │      (SQLite + JSONL log)      │
                          └───────────────┬─────────────┘
                                          │ NFS (rw, /shared)
                                          │ SSH (adduser/deluser, sbatch scripts run here)
                          ┌───────────────▼─────────────┐
                          │  GPU / compute node          │
                          │  192.168.122.1 (iit-MS-7E06)  │
                          │  ├─ slurmd                    │
                          │  ├─ NFS server for /shared     │
                          │  ├─ iit-gpu-stats-writer        │
                          │  │   (writes GPU/CPU/RAM json)  │
                          │  └─ RTX 5090 (gres/shard:4)      │
                          └───────────────────────────────┘
```

**Request flow for a typical action** (e.g. launching JupyterLab):

1. User SSHes to the login node. `sshd`'s `Match Group gpuusers` block fires `ForceCommand /usr/local/bin/iit-gpu-gateway` — the user never gets a raw shell.
2. `iit-gpu-gateway` recognises this is an interactive session (not an `scp`/`rsync`/`sftp` transfer) and `exec`s `/usr/local/bin/iit-gpu-manager`, a hardened launcher that strips the inherited environment (`env -i`) and runs `python3 -m iitgpu` with `PYTHONPATH=/opt/iit-gpu`.
3. `iitgpu.__main__` installs signal traps (so `Ctrl-Z`/backgrounding can't expose a shell), logs `session_start` to the audit daemon, fires a login-notification email in the background, and enforces a forced password change if one is pending.
4. The Main Menu (`iitgpu/menu.py`) drives everything from here — see [Using the tool](#using-the-tool--full-menu-reference) below.
5. Job submission (`iitgpu/jobs.py` renders the `sbatch` script, `iitgpu/slurm.py` calls `sbatch`) happens as the **user's own real Linux account** — no shared/sudo identity is needed because every gateway user has a real UID on both nodes (see next section). `sbatch` runs on the login node; the job itself executes on the GPU node via `slurmd`.
6. Every menu action that matters (job submit/cancel, file transfer, admin action, login) calls `iitgpu.auditclient.log()`, which delivers the event to `iit-gpu-audit.service` over a Unix socket (falling back to an on-disk spool if the daemon is briefly unreachable). The daemon is the only process that can write `/var/lib/iit-gpu/audit.db` / `audit.jsonl`.
7. All persistent state — job folders, environments, models, templates, and each user's private files — lives under `/shared`, NFS-mounted read/write on both nodes. Filesystem permissions (not the TUI) are the last line of defence: even a bug in the TUI's jail logic can't let one user read another's files, because the OS itself denies it.

---

## Linux users, groups & access model

This is the part that actually enforces isolation — the TUI's own path-jail logic (`iitgpu/validate.py`) is a *convenience* layer that keeps well-behaved code from wandering outside a user's area; the **real** boundary is standard POSIX permissions plus these three groups, checked identically on both nodes.

### The three groups

| Group | Purpose | Who's in it |
|---|---|---|
| `gpuusers` | Gates SSH access to the gateway. `sshd`'s `Match Group gpuusers` block is what fires `ForceCommand` and locks down forwarding. | Every "tool" and "admin" user (not shell users) |
| `gpuadmins` | Grants the admin panel (`iitgpu.config.is_admin()` just checks membership) **and** read/write access to every user's private folder on disk (folders are mode `2770` owned by that user, group `gpuadmins`). | Admin accounts only |
| `docker` (login node only) | Passwordless `docker` access for "shell" accounts that need it. | Shell users only, opt-in |

A user's group membership is the *entire* access-control mechanism — the TUI binary itself is never copied or modified per user, and there is no per-user config file granting privilege.

### The three account types

Every account is provisioned with `deploy/iit-gpu-adduser.sh` (directly, or via the admin panel's **Provision user**), which creates one of three kinds of account:

| Type | Flag | `gpuusers`? | `gpuadmins`? | Gets on login | Audited? |
|---|---|---|---|---|---|
| **tool** (default) | *(none)* | ✅ | ❌ | Forced into the TUI, no shell | ✅ Yes |
| **admin** | `--admin` | ✅ | ✅ | Forced into the TUI **plus** the Admin menu item | ✅ Yes |
| **shell** | `--shell-user` | ❌ | ❌ | A real `bash` login shell on the login node (`docker` group only) | ❌ **No** — explicitly not audited; use only when the TUI genuinely can't do what's needed |

Regardless of type, every account is also:
- Registered as a real SLURM association (`sacctmgr add user <name> account=<SLURM_ACCOUNT> qos=<SLURM_QOS>`), so SLURM enforces its own per-user resource caps independently of the TUI.
- Given `/shared/users/<username>` and `/shared/jobs/<username>` — see below.

### Why real per-user Linux accounts (not one shared account)

Every gateway user is a **real Linux user with a real UID**, created identically (same UID) on *both* nodes by `iit-gpu-adduser.sh`:

1. Picks a UID that's free on **both** the login node and the GPU node (queries `getent passwd` on each, takes the max, walks up until free on both) — so a file owned by UID 2011 means the same person on either machine.
2. `useradd -u <uid> -g <uid> -m -s /bin/bash <username>` on the login node, then the identical command over SSH on the GPU node.
3. Force-`chown`s the home directory on both nodes (a *pre-existing* home from an earlier account with the same name can be left owned by a stale UID, which silently breaks `conda activate` because the user can't read their own `~/.condarc`).
4. Adds `gpuusers` (and `gpuadmins` if `--admin`) on **both** nodes — admin group membership must exist on the GPU node too, because that's where jobs and notebooks actually run and where the per-user data areas live; login-node-only membership would show the admin panel but leave the admin unable to read anyone's files.
5. Registers the SLURM association.
6. Creates `/shared/users/<username>` and `/shared/jobs/<username>` **on the GPU node** (the real NFS server — the export uses `root_squash`, so a `chown`/`setfacl` issued from the login node over NFS would be squashed to `nobody` and silently fail). Both are `chown <uid>:gpuadmins`, `chmod 2770` (setgid, so anything created inside inherits group `gpuadmins`), plus an explicit recursive + default ACL (`setfacl -m g:gpuadmins:rwX -m d:g:gpuadmins:rwx`) so the guarantee doesn't depend on inheriting the parent directory's own ACL.
7. Symlinks `~/shared` to the user's own area (`/shared/users/<username>` for tool/admin users; the whole `/shared` root for shell users, who need to reach datasets/models/envs directly too).

**Net effect:** owner (rwx) + `gpuadmins` group (rwx) + everyone else (nothing). A regular user can read/write only their own area; admins can reach any user's area for support; nobody outside those two can see anything. This is checked by the kernel on every syscall — it does not depend on the Python code ever running correctly.

### Where things live on `/shared`

| Path | Owner : Group | Mode | Who can access |
|---|---|---|---|
| `/shared/users/<username>` | `<user> : gpuadmins` | `2770` + ACL | Owner + any `gpuadmins` member |
| `/shared/jobs/<username>` | `<user> : gpuadmins` | `2770` + ACL | Owner + any `gpuadmins` member — per-run job folders, `sbatch` scripts, `slurm-<id>.out/.err` |
| `/shared/envs`, `/shared/models`, `/shared/datasets`, `/shared/data`, `/shared/templates` | `public : gpuusers` (typically) | group-writable, setgid | Read/write by any `gpuusers` member — shared assets everyone can use |

The TUI's own jail (`iitgpu/validate.py`) mirrors this: `user_browse_roots()` / `in_user_browse_jail()` limit a regular user's file browser to their own `users/<username>` folder plus the shared read-only asset directories; `in_user_upload_jail()` limits uploads/writes to their own folder only. Admins bypass these per-user jails (they use the global `in_jail()`, scoped only to `NFS_ROOT` itself) — but as noted above, that's a convenience, not the actual security boundary; the filesystem enforces the boundary either way.

### `sudo` scope — exactly what each group can run as root

Nothing runs the TUI itself as root. Two narrow `sudoers.d` files grant exactly the commands each group needs, nothing else (`Defaults ... timestamp_timeout=0` means no `sudo` password caching window):

**`deploy/sudoers-gateway`** (legacy/optional — see note below):
```
%gpuusers ALL=(daham) NOPASSWD: /usr/bin/sbatch, /usr/bin/squeue, /usr/bin/scancel, /usr/bin/sinfo, /usr/bin/sacct
```
> This authorizes running SLURM commands as a single shared identity (`daham`) — a leftover from before real per-user accounts existed. With real accounts (the current, default setup — `GATEWAY_SHARED_USER=0`), every user already has their own SLURM association and runs `sbatch`/`squeue`/etc. directly as themselves via their own login shell, with no `sudo` needed at all. This file only matters if `GATEWAY_SHARED_USER=1` is ever turned on.

**`deploy/sudoers-gateway-admin`**:
```
%gpuadmins ALL=(root) NOPASSWD:
    /usr/bin/scontrol update *, /usr/bin/scontrol reconfigure,
    /usr/local/bin/iit-gpu-adduser, /usr/local/bin/iit-gpu-deluser,
    /usr/sbin/chpasswd, /usr/bin/sacctmgr, /usr/bin/scancel

%gpuadmins ALL=(%gpuusers) NOPASSWD: /usr/local/bin/iit-gpu-manager

%gpuadmins ALL=(slurmadmin) NOPASSWD: /opt/iit-gpu/deploy/resize-pods.sh *
```
The second line is what powers **Log in as user**: an admin can launch the TUI *as* any `gpuusers` member (`sudo -H -u <user> iit-gpu-manager`) to see exactly what that user sees — never a general shell, only ever the TUI launcher, and every action taken is audited under the *target* user's identity (via `SO_PEERCRED` on the audit socket), with the switch itself separately logged.

The third line powers the admin panel's **pod-count resize** (Pods screen): the panel runs `sudo -n -u slurmadmin /opt/iit-gpu/deploy/resize-pods.sh <N>` — as `slurmadmin`, the account that owns `gres.conf`/`slurm.conf` and the `slurmctld`/`slurmd` restart path, never as root (the script refuses to run as anyone else). The grant is the run only; the script itself enforces the empty-queue check, a lockfile, and backup/rollback.

> **Installing this file is a manual step.** `deploy/redeploy-igm.sh` fast-forwards `/opt/iit-gpu` and resyncs a couple of `/usr/local/bin` scripts — it does **not** touch `/etc/sudoers.d`. After any change to `deploy/sudoers-gateway-admin` (including the resize grant above), re-run the install on the login node or the affected admin action fails with a bare `sudo` permission error:
> ```bash
> sudo install -m 0440 -o root -g root deploy/sudoers-gateway-admin /etc/sudoers.d/iit-gpu-admin
> sudo visudo -cf /etc/sudoers.d/iit-gpu-admin
> ```

### `sshd` access rules (`deploy/sshd-gateway.conf`)

```
Match Group gpuadmins
    PermitTTY yes
    X11Forwarding no
    AllowTcpForwarding local          # so admins can also tunnel to JupyterLab/TensorBoard
    PermitOpen 192.168.122.1:*        # only to the compute node, nowhere else

Match Group gpuusers
    ForceCommand /usr/local/bin/iit-gpu-gateway
    PermitTTY yes
    AllowTcpForwarding local          # local (-L) port-forwards only — needed for tunnels
    PermitOpen 192.168.122.1:*        # restricted to the compute node — can't jump elsewhere
    AllowAgentForwarding no
    AllowStreamLocalForwarding no
    X11Forwarding no
    PermitTunnel no
    GatewayPorts no
    PermitUserRC no
```
Shell users (not in `gpuusers`) fall through to sshd's normal default behaviour — an ordinary login shell, no `ForceCommand`.

---

## Using the tool — full menu reference

SSH in (`ssh -p <port> <user>@<host>`) and you land on the **Main Menu** after a splash screen and a live cluster-status line.

### Main Menu

```
1. New Job       (JupyterLab, script, or shell — pick, review, launch)
2. My Workspace  (files, models, environments)
3. Jobs          (queue, history, logs, rerun)
4. Settings      (health check, shell, cluster status, hardware)
5. Admin         (cluster ops, users, audit)     ← only shown to gpuadmins members
```

If an admin has posted a **maintenance notice**, everyone sees a yellow banner with the reason and who set it; non-admins are then blocked from going any further until it's cleared.

### 1. New Job

The whole flow is three questions wide — what you want to do, what it runs on, then one editable **review hub** — not a long form.

**Step 1 — pick an intent:**

| Choice | What it launches |
|---|---|
| **Open JupyterLab** | Interactive JupyterLab session on the GPU node, reached over an SSH tunnel |
| **Run a script or notebook** | Batch job — a `.py`, `.sh`, or `.ipynb` executed end-to-end (notebooks run via `papermill`, streaming each cell's output live and auto-installing any package the cell imports but the environment lacks) |
| **Fine-tune a model** | The same batch pipeline as above, but pre-filled with whole-GPU / `llm-finetune`-env defaults, plus two extra guided questions (base model, dataset) |
| **Open a shell on the GPU node** | A real interactive shell (`srun --pty`) on the compute node |
| **Other: my own `.sbatch` · templates** | Submit a hand-written `.sbatch` file directly, or load a previously **saved template** |

**Step 2** (batch/fine-tune only) — which script/notebook to run, picked from a jailed file browser scoped to your own folder plus shared read-only assets.

**Step 3 — the review hub.** One panel shows exactly what will launch (with live free-GPU-share availability in the title) and a menu that edits any row in place. Only the rows relevant to your chosen intent are shown:

| Row | What you can change |
|---|---|
| **Pods** | How many pods (GPU shares) to claim, 1 up to the node's live pod count — each row shows the CPU/RAM and estimated VRAM that many pods gets you, and is labelled *starts now* or *will queue*, based on live availability. If the cluster can't be read the row says *GPU availability unknown* instead of guessing |
| **Time limit** | Presets up to the cluster's QOS ceiling, or a custom `HH:MM` (rejected if it exceeds the ceiling, since `sbatch` would reject it too) |
| **Environment** | A prebuilt env from `/shared/envs/` (defaults to `data-science` if installed), your own conda env or venv, a jailed `.sif` container image, or plain system Python |
| **Data / model** | Jailed folder browser for data, or a path / HuggingFace repo id for a model *(batch and notebook jobs)* |
| **Packages** | A `requirements.txt` or typed package list, installed before the session/first cell runs *(JupyterLab and `.ipynb` jobs)* |
| **Args** | Extra command-line arguments *(batch jobs)* |
| **Advanced…** | Job array, run-after-job dependency, email notifications on/off, and a preview of the exact `sbatch` script that will be submitted |

From the hub: **Launch**, **Save as template** (reusable from "Other" next time), or back out entirely.

After launch you're dropped straight into the live dashboard for that job. For JupyterLab/TensorBoard, the tool waits for the service's readiness marker and then shows a **Connect card**:
```
1. On YOUR laptop, open a terminal and run:
   ssh -p <port> -N -L <local-port>:<compute-node>:<local-port> <user>@<gateway-host>
   (keeps running; an idle terminal is correct)

2. Then open in your browser:
   http://127.0.0.1:<local-port>/lab?token=<token>
```
The tunnel command and URL are parsed straight out of the job's own stdout, never reconstructed — so what you're shown is guaranteed to be what the job actually bound to.

### 2. My Workspace

A single dashboard screen showing, at a glance:
- **Disk** — free/total space on `/shared`
- **My Files** — item count + size for your `datasets/`, `data/`, `models/`, `scripts/` folders, plus your 5 most recent job folders with a state guess (COMPLETED/FAILED/PENDING from the presence and size of `.err`/`.out`)
- **Environments** — every conda env / venv you've registered (name, kind, path)
- **Downloaded Models** — everything in your local model registry (name, size, source)

Actions:
| Action | What it does |
|---|---|
| **Browse my files** | Full jailed file manager, confined to your own folder (plus shared read-only assets) |
| **Upload data** | Copy local files/folders into `/shared/users/<you>/` (also supports uploading and safely unzipping a `.zip` — the extractor rejects any archive member that would traverse outside the destination folder) |
| **Download a model** | HuggingFace repo id, or an arbitrary URL |
| **Build / manage environments** | Same env-builder and prebuilt-env installer as Settings → Build/Install environment |
| **Delete a model** | Remove a model from your local registry |

### 3. Jobs

```
Live dashboard  (auto-refresh)
View queue
Manage a job  (cancel/hold/release/requeue/details)
View job log
Job history  (filters)
Rerun a job
─────────────────────
Hardware stats
Usage & accounting
My running services
Cluster status
```

| Item | What it does |
|---|---|
| **Live dashboard** | Rich auto-refreshing (every 2s) view of every running/recent job on the cluster. Your own jobs render in green, other users' in cyan (colour = ownership, not job state). A job's status reads `STARTING` instead of `RUNNING` until its readiness marker appears (for JupyterLab/TensorBoard — usually a few seconds). Keys: `↑`/`↓` scroll the selected job's log, `PgUp`/`PgDn` jump 10 lines, `S` switch selected job, `C` cancel the selected job (admins can cancel **anyone's**; everyone else only their own), `E` extend a running JupyterLab job by 2 hours, `T` show the Connect card for your own JupyterLab job, `R` force refresh, `Q` quit. |
| **View queue** | Plain table of your own `squeue` output. |
| **Manage a job** | Pick a job from a list, then Cancel / Hold / Release / Requeue / Details+efficiency (`seff`). Admins see and can act on **every** user's jobs here (owner shown in parentheses); everyone else only sees their own. |
| **View job log** | Jailed browser to tail any `.out`/`.err` file under your own job folders, full-log pager with search. |
| **Job history** | Filter by state (`COMPLETED`/`FAILED`/`CANCELLED`/`TIMEOUT`); admins get an extra prompt to include every user's history, not just their own. |
| **Rerun a job** | Pick a past job folder, re-parses its `job.sbatch`, and relaunches it through the wizard with the same settings pre-filled. |
| **Hardware stats** | Live-refreshing GPU/CPU/RAM utilization panel for the compute node (`Q` to quit). |
| **Usage & accounting** | GPU/CPU-hours per user (30d), fairshare standing, or raw `sreport` output. |
| **My running services** | Lists your active JupyterLab/TensorBoard/interactive sessions with their tunnel command; stop any of them directly. |
| **Cluster status** | Partition table: name, state, node count, GPUs/node. |

### 4. Settings

```
Cluster health check
Build environment
Install prebuilt environment
Run smoke test
Advanced SLURM shell
```

| Item | What it does |
|---|---|
| **Cluster health check** | Runs `sinfo`, verifies `/shared` is writable and `/shared/envs/` exists; reports pass/fail. |
| **Build environment** | Framework picker (PyTorch / TensorFlow / JAX / bare Python, etc.) → version list → `conda create` + `pip install`, with a live per-package progress gauge; registers the new env for future jobs. |
| **Install prebuilt environment** | One-click install of a ready-made shared env (e.g. `data-science`) into `/shared/envs/` — the same environments the wizard offers by default. |
| **Run smoke test** | Submits a tiny SLURM job, attaches to its live output, and reports `✔ CUDA: True` / `✘ CUDA: False`. |
| **Advanced SLURM shell** | A restricted command loop (never `shell=True`) — only `sbatch`, `squeue`, `scancel`, `sinfo`, `sacct`/`tail` execute; every typed line is audit-logged; any path passed must pass the jail check. |

### 5. Admin (admins only)

Shown only if you're in `gpuadmins`. The status line always shows active user count, mail service on/off, and a maintenance flag if set.

```
──  User Management  ──────────────────────────
  Provision user
  Offboard user
  View users
  Log in as user
──  Jobs & Usage  ─────────────────────────────
  All-user job history
  Cluster usage (all users)
  Disk usage by user
  Any user's job output
──  Cluster Control  ──────────────────────────
  Drain node
  Resume node
  QOS / limits
  Maintenance notice
──  Monitoring  ───────────────────────────────
  Audit log
  Service health
  Mail delivery log
  Mail service: ON/OFF
```

| Item | What it does |
|---|---|
| **Provision user** | Prompts: username, type (**tool** / **admin** / **shell** — shell requires an extra confirmation, warning it's unaudited), full name, email, and either a self-chosen or randomly generated initial password (forces a change on first login). Runs `iit-gpu-adduser.sh` under the hood (see [Adding and removing users](#adding-and-removing-users)) and writes a `users.db` row if an email was given, which also triggers the welcome email. |
| **Offboard user** | Runs `iit-gpu-deluser.sh` (removes the account from both nodes; optional `--purge-data` to delete their `/shared` area too). |
| **View users** | Roster table: username, role, email, last login, etc. |
| **Log in as user** | Launches the TUI *as* the chosen `gpuusers` member (`sudo -u <user> iit-gpu-manager`) to see exactly what they see. Audited under the target's identity; the switch itself is separately logged. |
| **All-user job history** | Like Jobs → Job history, but always cluster-wide. |
| **Cluster usage (all users)** | Usage/accounting reports across everyone, not just yourself. |
| **Disk usage by user** | Per-user breakdown of `/shared/jobs/<user>` (and similar) disk consumption. |
| **Any user's job output** | Tail any user's job log directly, without impersonating them. |
| **Drain node** | Prompts for node name + reason; optionally cancels currently-running jobs on it first; puts the node in SLURM `DRAIN` state so no new jobs schedule there. |
| **Resume node** | Clears `DRAIN`/`DOWN` state. |
| **QOS / limits** | View/adjust SLURM QOS settings (time/GPU-share ceilings). |
| **Maintenance notice** | Sets or clears the cluster-wide banner that blocks non-admins from proceeding past the Main Menu. |
| **Audit log** | Browse the audit trail (see [Audit logging](#audit-logging)). |
| **Service health** | Status of `iit-gpu-audit`, `slurmctld`/`slurmd`, mail delivery, etc. |
| **Mail delivery log** | Tails `/var/log/msmtp.log` — the SLURM job-notification mail path. |
| **Mail service: ON/OFF** | A single cluster-wide kill switch (a flag file every mail-sending code path checks) — instantly silences all outgoing mail (welcome/login/job-notification) without touching SMTP config. |

---

## Linux machine setup (before installation)

You need **two machines** (or, for a smaller/demo deployment, everything can technically collapse onto one — the code doesn't assume two nodes, but the current production layout is split): a **login/gateway node** users SSH into, and a **GPU/compute node** SLURM actually schedules jobs onto. Both must be reachable from each other over SSH as root (or a sudo-capable account), and both must mount the same NFS export at the same path.

### Both nodes

- A recent Linux distribution with `systemd`
- Python 3.11+ and `pip`
- OpenSSH server, `sshd_config.d/` support (OpenSSH 8.2+)
- `sudo` + `visudo`
- SLURM installed and configured as one cluster (`slurmd` on the compute node, `slurmctld`+`slurmdbd` on the login node is the layout used here) — `sbatch`, `squeue`, `scancel`, `sinfo`, `sacct`, `scontrol`, `sacctmgr` all on `PATH`
- The NFS export mounted read/write at the same path on both nodes (`/shared` by default) — `nfs-common` (or equivalent) installed
- `acl` package (`setfacl`/`getfacl`) — the per-user ACL guarantees depend on it
- A shared UID range reserved for gateway users on **both** nodes (default `2000`–`60000`, configurable) — don't let local system accounts collide with it

### GPU / compute node additionally needs

- NVIDIA driver + `nvidia-smi` on `PATH`
- GPU sharing configured in `slurm.conf`/`gres.conf` if you want more than one job per GPU at once (see [GPU sharing model](#gpu-sharing-model)) — `SelectType=select/cons_tres`, `GresTypes=gpu,shard`, `Gres=gpu:1,shard:N` on the node line, and a matching `gres.conf` entry
- Conda (Miniconda/Miniforge) available at a shared path both nodes can see, e.g. `/shared/miniforge3` — the installer can install Miniforge for you (see below)
- `docker` installed if you plan to provision any `--shell-user` accounts that need it

### Login node additionally needs

- The Resend (or any transactional-email-compatible) API setup if you want welcome/login/job-notification email — see [Mail / notifications](#mail--notifications). Fully optional; the tool works with mail disabled.
- `msmtp` if you want SLURM's own `MailProg` job-completion emails (separate path from the daemon's transactional mail)

### Before you run the installer

1. Decide your `NFS_ROOT` (default `/shared`) and confirm it's already mounted read/write on both nodes.
2. Decide your SLURM `account`/`qos`/`partition` names — these must already exist in `slurmdbd`/`slurm.conf`.
3. Pick your admin group name (default `gpuadmins`) and gateway group name (default `gpuusers`) if you don't want the defaults.
4. Have root/sudo on both nodes and password-less SSH from the login node to the compute node as a sudo-capable account (`iit-gpu-adduser.sh` needs this to provision the mirrored account on the GPU node).

---

## Installation

Everything below runs **on the login node** unless stated otherwise.

### Step 1 — Clone the repository

```bash
git clone <repo-url> /opt-src/iit-gpu-manager
cd /opt-src/iit-gpu-manager
```

### Step 2 — Configure site settings

```bash
cp deploy/site.env.example deploy/site.env
nano deploy/site.env       # NFS_ROOT, SLURM_ACCOUNT/QOS/PARTITION, GATEWAY_HOST/PORT, cluster name, UID range, resource ceilings
```
`site.env` is git-ignored — safe to fill in real values. It carries **no secrets** (must stay world-readable, `0644`, because every user's TUI process reads it directly to build things like the welcome email's SSH command).

If you want transactional/notification email, also:
```bash
cp deploy/secrets.env.example deploy/secrets.env
nano deploy/secrets.env    # RESEND_API_KEY
sudo chown root:gpusync deploy/secrets.env
sudo chmod 640 deploy/secrets.env
```

### Step 3 — Run the installer (as root)

```bash
sudo bash deploy/install.sh
```

This single script:

1. Installs system packages (`python3`, `git`, `rsync`, `acl`, `nfs-common`, etc.)
2. Creates system group `gpuusers`, system group `auditadmin`, system users `slurmsvc` and `gpusync` (both no-login)
3. Creates the shared directory tree under `NFS_ROOT` (`scripts/`, `jobs/`, `data/`, `envs/`, `models/`, `templates/`) with group-writable, setgid, and (if `acl` is present) default-ACL permissions for `gpuusers`
4. Installs Miniforge (conda) to `CONDA_PREFIX_SHARED` if not already present, and wires `conda.sh` into `/etc/bash.bashrc` so non-interactive `sbatch` scripts pick it up
5. Copies the repo to `/opt/iit-gpu/` (root-owned, `0755` — regular users cannot modify the tool)
6. `pip install`s `requirements.txt` system-wide
7. Adds `gpusync` to `auditadmin` and `gpuusers` (the audit daemon execs code from `/opt/iit-gpu`, so it needs read access there too), creates `/var/lib/iit-gpu` (`0750`, owner `gpusync:auditadmin`)
8. If a `slurm` system account exists, adds it to `gpusync` so SLURM's `MailProg` (running as `SlurmUser`, not root) can read the mail API key, and restarts `slurmctld`
9. Installs the hardened launcher at `/usr/local/bin/iit-gpu-manager` and the ForceCommand wrapper at `/usr/local/bin/iit-gpu-gateway`
10. Installs and starts `iit-gpu-audit.service`
11. Installs `deploy/sshd-gateway.conf` into `/etc/ssh/sshd_config.d/`, validates with `sshd -t` (removes it automatically if invalid), reloads `sshd`
12. Installs `deploy/sudoers-gateway` into `/etc/sudoers.d/`, validates with `visudo -cf` (removes it automatically if invalid)
13. Installs the admin audit-log viewer at `/usr/local/bin/iit-gpu-log`

You'll also want the admin sudoers file and the per-user provisioning scripts, which `install.sh` does not install automatically (they're layered on afterward — see next step):
```bash
sudo install -m 0755 deploy/iit-gpu-adduser.sh /usr/local/bin/iit-gpu-adduser
sudo install -m 0755 deploy/iit-gpu-deluser.sh /usr/local/bin/iit-gpu-deluser
sudo install -m 0440 -o root -g root deploy/sudoers-gateway-admin /etc/sudoers.d/iit-gpu-admin
sudo visudo -cf /etc/sudoers.d/iit-gpu-admin   # verify before trusting it
```

### Step 4 — Set `GPU_HOST_SSH` for the adduser script

`iit-gpu-adduser.sh` needs to know how to reach the compute node as a sudo-capable account:
```bash
echo 'GPU_HOST_SSH=root@192.168.122.1' | sudo tee -a /opt/iit-gpu/deploy/site.env
```
Test password-less SSH works before provisioning anyone:
```bash
ssh root@192.168.122.1 true
```

### Step 5 — Verify the service

```bash
systemctl status iit-gpu-audit
journalctl -u iit-gpu-audit -f
```

### Step 6 — Provision your first admin and test

```bash
sudo /usr/local/bin/iit-gpu-adduser <yourname> --admin
```
Then from your own machine:
```bash
ssh -p <GATEWAY_PORT> <yourname>@<GATEWAY_HOST>
```
You should land directly in the TUI with the Admin menu item visible, and no shell escape.

### Step 7 — GPU sharing (recommended)

By default SLURM gives one job the whole GPU. To let up to 4 smaller jobs share it (JupyterLab/inference workloads), configure `slurm.conf`/`gres.conf` on **both** nodes:
```
# slurm.conf
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory
GresTypes=gpu,shard
NodeName=<node> ... Gres=gpu:1,shard:4

# gres.conf (compute node)
Name=shard Count=4 File=/dev/nvidia0
```
Restart `slurmctld` (login node) then `slurmd` (compute node) — in that order; a `SelectType` change is not hot-reloadable via `scontrol reconfigure`. See [GPU sharing model](#gpu-sharing-model) for how the app side maps to this.

---

## Installing via an AI coding agent (igm-installer skill)

This repo ships an `igm-installer` skill that drives the install above as a
**guided, checkpointed** conversation instead of a blind one-shot script: it
runs the preflight checks from [Linux machine setup](#linux-machine-setup-before-installation),
interviews you for the site-specific values `deploy/site.env` needs, then
walks through `install.sh`, the admin sudoers file, GPU-host SSH, service
verification, first-admin provisioning, and optional GPU sharing — asking
for explicit confirmation before every root/sudo/sshd/sudoers/systemd action.
It never runs the whole install unattended.

### Claude Code

```bash
/plugin marketplace add DahamDissanayake/IIT-Secure-SLURM-Job-Gateway
/plugin install igm-installer@iit-gpu-manager
/igm-installer
```

Then answer its questions and approve each checkpoint as it comes up.

### Any other AI coding agent

Open [`skills/igm-installer/AGENT-INSTRUCTIONS.md`](skills/igm-installer/AGENT-INSTRUCTIONS.md)
and paste it into your assistant of choice (Cursor, Copilot Workspace,
Windsurf, etc.) with this repo open — it describes the same guided,
checkpointed process in agent-agnostic terms.

---

## Adding and removing users

**Always use `iit-gpu-adduser`/`iit-gpu-deluser`, not raw `useradd`/`usermod`** — the scripts are what keep UIDs, home ownership, group membership, SLURM association, and the per-user `/shared` area consistent across *both* nodes. Either run them directly as root, or use the admin panel's **Provision user** / **Offboard user**, which call the same scripts.

### Add a user

```bash
sudo iit-gpu-adduser <username>                # tool user (default) — forced TUI, audited
sudo iit-gpu-adduser <username> --admin        # admin — forced TUI + admin panel
sudo iit-gpu-adduser <username> --shell-user   # real shell, NOT audited, gets docker group
sudo iit-gpu-adduser <username> --dry-run      # preview every command without executing
```
Then set credentials:
```bash
sudo passwd <username>
# or: install ~<username>/.ssh/authorized_keys
```
What it does (full detail in [Linux users, groups & access model](#linux-users-groups--access-model)): finds a UID free on both nodes, creates the account identically on both, fixes home ownership, joins the right groups on both nodes, registers the SLURM association, creates `/shared/users/<user>` and `/shared/jobs/<user>` on the GPU node (the real NFS server) with `2770` + ACL, and symlinks `~/shared`.

### Remove a user

```bash
sudo iit-gpu-deluser <username>                # remove the account from both nodes
sudo iit-gpu-deluser <username> --purge-data   # also delete their /shared area
sudo iit-gpu-deluser <username> --dry-run
```

### Check who has access

```bash
getent group gpuusers     # everyone who can reach the gateway
getent group gpuadmins    # everyone with admin panel + cross-user file access
```
Or, inside the tool: **Admin → View users**.

---

## Configuration reference

All settings are environment variables, layered lowest-to-highest priority: **built-in default → `deploy/site.env` → real process environment variable**. This means the whole tool can be repointed at a different cluster by editing only `site.env` — nothing is hardcoded in the Python source.

### Storage & behaviour

| Variable | Default | Purpose |
|---|---|---|
| `NFS_ROOT` | `/shared` | Root of the shared filesystem and the path jail |
| `JOBS_SUBDIR` | `jobs` | Subdirectory under `NFS_ROOT` for per-user job folders |
| `CONDA_PREFIX_SHARED` | `/shared/miniforge3` | Where conda lives |
| `DEMO_MODE` | `0` | `1` simulates SLURM entirely in memory — no cluster needed |
| `SACCT_ENABLED` | `auto` | Whether to trust `sacct` for authoritative job state (`auto` probes for the binary) |

### Identity / SLURM

| Variable | Default | Purpose |
|---|---|---|
| `GPUUSERS_GROUP` | `gpuusers` | POSIX group gating gateway access |
| `ADMIN_GROUP` | `gpuadmins` | POSIX group granting the admin panel + cross-user file access |
| `SLURM_ACCOUNT` | `default` | SLURM account new users are registered under |
| `SLURM_QOS` | `normal` | SLURM QOS new users are registered under |
| `SLURM_PARTITION` | `gpu` | Default partition for job submission |
| `GATEWAY_SHARED_USER` | `0` | Legacy: `1` runs all SLURM commands as one shared identity instead of each user's own account |
| `GATEWAY_SHARED_USER_NAME` | `daham` | The shared identity, if the above is on |
| `UID_MIN` / `UID_MAX` | `2000` / `60000` | UID range `iit-gpu-adduser` picks new accounts from |

### Gateway / tunnels

| Variable | Default | Purpose |
|---|---|---|
| `GATEWAY_HOST` | `localhost` | Public-facing host users SSH to and tunnel through (shown in welcome/connect messages) |
| `GATEWAY_PORT` | `22` | Public-facing SSH port |

### Resource ceilings

| Variable | Default | Purpose |
|---|---|---|
| `MAX_GPUS` | `8` | Ceiling on *whole* GPUs per job (not used for share-based requests) |
| `MAX_GPU_SHARDS` | `4` | **Fallback** ceiling on GPU shares (pods) per job. The real ceiling is the node's live shard count read from `scontrol show node` (`iitgpu.pods.pod_count`); this value is only used when that reading is unavailable, so set it to the `Count=` in `gres.conf` |
| `MAX_CPUS` | `64` | Ceiling on CPUs per job |
| `MAX_MEM_GB` | `256` | Ceiling on memory (GB) per job |
| `MAX_HOURS` | `72` | Ceiling on job duration |

### Mail / cluster identity

| Variable | Default | Purpose |
|---|---|---|
| `MAIL_FROM` | `GPU Cluster <no-reply@example.com>` | From: address for transactional mail |
| `NOTIFY_MAIL_TYPES` | `BEGIN,END,FAIL,REQUEUE,TIME_LIMIT` | SLURM `--mail-type` value used when a job has `--mail-user` set |
| `CLUSTER_NAME` | `IIT GPU Cluster` | Display name (emails, TUI) |
| `CLUSTER_LOCATION` | `IIT-CityCampus-SpencerBuilding` | Display network label |
| `CLUSTER_TZ_OFFSET` | `+05:30` | UTC offset used when rendering timestamps |
| `RESEND_API_KEY` | *(none)* | **Set only in `deploy/secrets.env`**, never `site.env` — the transactional-mail API key |

### Audit

| Variable | Default | Purpose |
|---|---|---|
| `AUDIT_SOCKET` | `/run/iit-gpu/audit.sock` | Unix socket to the audit daemon |
| `AUDIT_SPOOL` | `/run/iit-gpu/spool` | Fallback spool dir when the socket is unreachable |
| `AUDIT_STATE` | `/var/lib/iit-gpu` | Directory for `audit.db`/`audit.jsonl` (daemon-side only) |

---

## Audit logging

`iit-gpu-audit.service` runs as `gpusync` (mode `0750` state dir — regular users cannot read or write it) and maintains:

| File | Format | Purpose |
|---|---|---|
| `/var/lib/iit-gpu/audit.db` | SQLite (WAL) | Queryable structured log |
| `/var/lib/iit-gpu/audit.jsonl` | Newline-delimited JSON | Append-only human-readable log |

Columns: `ts | user | session | action | detail | job_id | remote`. The **user** field is stamped authoritatively by the daemon via `SO_PEERCRED` on the Unix socket — a client cannot lie about who it is.

Every consequential action (login, job submit/cancel/hold/release/requeue, file transfer, admin action, password change, crash) calls `auditclient.log()`. Job submission specifically calls `log_or_block()`: if neither the socket nor the spool directory is reachable, **the submission is refused** rather than proceeding unlogged.

View logs: **Admin → Audit log** inside the tool, or as root/`gpusync`:
```bash
sudo -u gpusync python3 /opt/iit-gpu/deploy/iit-gpu-log
```

---

## Mail / notifications

Two independent mail paths — don't confuse them:

1. **SLURM job-completion mail** (BEGIN/END/FAIL/REQUEUE/TIME_LIMIT) — SLURM's own `MailProg`, set to `/usr/local/bin/iit-gpu-mailer`, which runs as `SlurmUser` (typically the `slurm` account, **not root**). It sends via the Resend HTTP API if `RESEND_API_KEY` is readable, falling back to local `msmtp` otherwise.
2. **Transactional mail** (welcome email with initial password, login notification with source IP, JupyterLab session extended/expiring warnings) — sent by the TUI client asking the audit daemon (which alone can read `secrets.env`) to relay via the Resend API.

Both paths respect a single cluster-wide kill switch: **Admin → Mail service: ON/OFF**, backed by a flag file every send-path checks before doing anything.

Setup: fill `RESEND_API_KEY` in `deploy/secrets.env` (root:gpusync, `0640`). No further code changes needed — mail activates automatically once the key is present and the service isn't disabled.

---

## GPU sharing model

By default SLURM gives a job the entire GPU it requests. This cluster instead splits the single physical GPU into **pods** (`gres/shard` — currently 4) so JupyterLab/inference-sized jobs can run concurrently:

- **Nothing hardcodes the split.** The pod count is whatever `scontrol show node` currently reports for `gres/shard` (set by `Count=` in `gres.conf`), and per-pod CPU/RAM is floor-divided from the node's real `CPUTot`/`RealMemory` — both computed in `iitgpu/pods.py` (`pod_count()`, `pod_resources()`, `resources_for()`), the one place that math lives. Re-split the GPU in `gres.conf` and the tool follows on the next read, with no code change. On today's node (`CPUTot=32`, `RealMemory=62000`, `shard:4`) that works out to 4 pods of 8 CPU / 14 GB (2 GB reserved for the OS).
- `render_*_sbatch()` (in `iitgpu/jobs.py`) emits `--gres=shard:N`, never `--gres=gpu:N`, for share-based tasks; CPU-only jobs omit `--gres` entirely.
- One-pod tasks (notebook / interactive / inference) get one pod's worth of CPU and memory too — splitting the GPU alone is pointless if 2 jobs still fight over all the RAM.
- `train`/`finetune`/`custom` tasks request every pod on the node by default (`jobs.TASK_POD_DEFAULTS`, the `"all"` sentinel).
- When the live reading is unavailable the tool says so ("GPU availability unknown") and keeps its built-in one-pod defaults, rather than claiming a share it cannot verify.
- Shares are a **scheduling** split only, not VRAM isolation — 4 co-resident jobs share the physical 32 GB and can, in principle, OOM each other. The wizard's review hub shows a VRAM sanity check before launch. True hard isolation would require NVIDIA MPS (not configured) — MIG is not an option on a consumer RTX 5090.
- The QOS enforces `MaxTRESPerUser=gres/gpu=1,gres/shard=4` — one whole GPU's worth of shares, whichever form a job requests.

---

## Demo mode (no SLURM required)

Run the full TUI on any machine, no cluster needed — SLURM partitions, job submission, and the queue are simulated in memory.

```bash
git clone <repo-url> && cd IIT-Secure-SLURM-Job-Gateway
pip install rich questionary prompt_toolkit huggingface_hub
python -m iitgpu --demo
python -m iitgpu --demo --no-splash   # skip the ASCII splash
```

Built-in self-check (jail logic, validators, audit fallback — no interactive prompts):
```bash
python -m iitgpu --selftest
```

---

## Running the test suite

```bash
pip install pytest rich questionary prompt_toolkit huggingface_hub
pytest tests/ -q
```
Run a specific file: `pytest tests/test_jobs.py -v`. The suite (786+ tests as of this writing) covers the path jail, validators, sbatch rendering, the wizard/review hub, the dashboard, the audit client and daemon, admin actions, mail, GPU sharding, and end-to-end demo-mode flows. Deploying via `deploy/redeploy-igm.sh` runs this suite as a hard gate before syncing anything live — **run it as the deploying user directly, not via `sudo`**, or three privilege-dependent tests spuriously fail because they're now running as root.

---

## Project layout

```
iitgpu/
  __main__.py       Hardened entry point: signal traps, session logging, forced pw change, --demo/--selftest
  menu.py            Main menu + Jobs/Settings submenus
  wizard.py          New Job: 3-question intake → review hub → submit
  review.py          The review hub: one editable pre-launch screen
  launchspec.py      LaunchSpec dataclass + pure launch-flow logic (sizes, defaults)
  jobs.py            JobSpec dataclass, sbatch/notebook/tensorboard script renderers, GPU-share directives
  templates.py       Saved job templates
  slurm.py           SLURM wrappers: submit_job, queue, cancel, hold/release/requeue, node stats, history
  dashboard.py       Live job dashboard + hardware-stats view (Rich Live, auto-refresh)
  monitor.py         Queue view, job management (cancel/hold/release/requeue), log tail, history, rerun
  notebooks.py       Running-services list, TensorBoard launch
  connect.py         Post-submit readiness wait + Connect card (tunnel command, URL, token)
  accounting.py      Usage/fairshare reports
  workspace.py        "My Workspace" dashboard (files, envs, models, recent jobs)
  files.py           Jailed file manager
  upload.py          Jailed upload (incl. safe zip extraction, zip-slip guarded)
  models.py          HuggingFace / URL model downloader + local registry
  envs.py             Conda/venv environment registry
  envbuilder.py       conda create + pip install, live progress, prebuilt-env installer
  setup.py            Health check, env setup, smoke test wiring for Settings menu
  containers.py       Apptainer/Singularity container helpers
  shell.py            Restricted SLURM command shell (allowlisted commands only)
  admin.py            Full admin panel: provisioning, node control, usage, audit, mail, maintenance
  config.py           Environment-variable configuration, site.env layering, group/admin checks
  validate.py         Global + per-user path jails, input validators, resource-ceiling clamps
  auditclient.py      Emit audit events to the daemon socket; spool fallback; log_or_block gate
  daemonclient.py      Client for the audit daemon's users.db verbs (create_user, email_for, ...)
  daemoncli.py         CLI entry for daemon-side operations
  mailer.py            Welcome/login/job-extend/warning email templates + Resend send
  notify.py             Small notification helpers
  splash.py             ASCII splash + live cluster-status line
  ui.py                  Rich console helpers (screen, select_menu, ok/err/warn/info/kv)

deploy/
  install.sh                 Root installer (see Installation)
  iit-gpu-adduser.sh          Provision a user on both nodes (see Adding and removing users)
  iit-gpu-deluser.sh          Offboard a user from both nodes
  audit_daemon.py             The iit-gpu-audit service: SQLite WAL + JSONL, users.db verbs
  iit-gpu-audit.service       systemd unit for the audit daemon
  iit-gpu-gateway             ForceCommand entry point: scp/rsync/sftp passthrough or launch the TUI
  iit-gpu-mailer              Standalone SLURM MailProg (Resend, msmtp fallback)
  iit-gpu-stats-writer        GPU/CPU/RAM stats writer (runs on the compute node)
  iit-gpu-log                 Admin audit-log viewer
  redeploy-igm.sh              Maintainer redeploy: test gate → fast-forward /opt/iit-gpu → resync binaries → restart daemon
  sshd-gateway.conf            sshd drop-in (ForceCommand, forwarding rules — see Linux users & access model)
  sudoers-gateway               gpuusers → SLURM commands (legacy shared-user path)
  sudoers-gateway-admin          gpuadmins → node control, adduser/deluser, log-in-as-user
  site.env.example / secrets.env.example   Configuration templates

tests/
  40+ files covering every module above — see "Running the test suite"
```

---

## Security model / bypass-test checklist

Every row below must fail for the deployment to be correctly locked down. Verify after any change to `sshd-gateway.conf`, `sudoers-*`, or the launcher.

| Attack | Defence |
|---|---|
| `ssh user@host bash` | `ForceCommand` in the `gpuusers` `Match` block always runs `iit-gpu-gateway` → the TUI |
| `ssh -R`/remote or `-D` dynamic forwarding | Only `AllowTcpForwarding local` is granted — remote/dynamic forwarding is refused |
| `ssh -L 8080:internal-host:80` to a host *other* than the compute node | `PermitOpen 192.168.122.1:*` restricts local forwards to the compute node only |
| `ssh -A` (agent forwarding) | `AllowAgentForwarding no` |
| X11 forwarding | `X11Forwarding no` |
| Unix-socket forwarding | `AllowStreamLocalForwarding no` |
| `Ctrl-Z` to background into a shell | `SIGTSTP` is ignored in `__main__.py`'s signal handler |
| Symlink inside a job/upload folder pointing outside the jail | `in_jail()`/`in_user_*_jail()` call `Path.resolve()` before comparing — symlinks are followed, not trusted blindly |
| `../../etc` in any file browser | Same `resolve()`-then-compare check rejects the escape |
| Reading/writing another user's `/shared/users/<other>` directly (bypassing the TUI) | Filesystem denies it independently: `2770`, owner-only + `gpuadmins` group — this is enforced by the kernel, not the Python code |
| Typing a raw shell command in the Advanced SLURM shell | Allowlist: only `sbatch`, `squeue`, `scancel`, `sinfo`, `sacct`/`tail` execute; never `shell=True` |
| `sbatch /etc/cron.d/malicious` in the shell | Path must pass the jail check before `sbatch` runs |
| Directly reading/editing the audit DB or JSONL | `/var/lib/iit-gpu` is `0750`, owned by `gpusync:auditadmin` — unprivileged users get `Permission denied` |
| Submitting a job while the audit daemon is down and the spool dir is also gone | `log_or_block()` returns `False` → submission is refused |
| Requesting far more GPU/CPU/mem/time than allowed | `clamp_int`/`clean_time_limit` silently cap to `MAX_*` server-side, regardless of what the client sends |
| Setting `LD_PRELOAD` or other env vars via SSH | `PermitUserEnvironment no` (sshd) + the launcher's `env -i` strips the inherited environment entirely |
| `.bashrc`/`.bash_profile` execution on login | `PermitUserRC no` |
| Crashing the tool to fall through to a shell prompt | `try/except` around the whole menu loop in `__main__.py`; `finally` always logs `session_end` and exits non-zero — there is no code path that drops to a shell |
| Port forwarding / tunnelling to arbitrary hosts | `PermitTunnel no`, `GatewayPorts no`, plus the `PermitOpen` restriction above |
| A non-admin cancelling another user's job | `manage_job()` and the dashboard's cancel key both check `is_admin()` before allowing action on a job that isn't the caller's own |
| A shell user assuming they're audited | They're explicitly told at provisioning time they are **not** — this is a deliberate, narrow escape hatch, not a bug |

---

## Day-2 operations (maintainers)

The canonical source of truth for the running cluster is a dev clone on the login node (default `/home/slurmadmin/IIT-Secure-SLURM-Job-Gateway`); the live install is `/opt/iit-gpu`, kept in sync by:

```bash
cd <dev-clone>
python3 -m pytest -q          # must pass before deploying
bash deploy/redeploy-igm.sh   # run as the owning non-root user, NOT via sudo
```

`redeploy-igm.sh`:
1. Re-runs the full test suite as a hard gate (as **root**, this spuriously fails 3 privilege-dependent tests — run it as a normal sudo-capable user, not `sudo bash ...`)
2. Fast-forwards `/opt/iit-gpu` from the dev clone (not from GitHub directly — see `IIT_GPU_SOURCE` if that ever needs to change)
3. Re-syncs `site.env` permissions (must stay `0644`), `iit-gpu-mailer`/`iit-gpu-adduser`/`iit-gpu-deluser` into `/usr/local/bin`
4. Restarts `iit-gpu-audit.service` so any daemon-side code change actually takes effect — **a source edit alone does nothing until this restart runs**; the daemon loads its Python once at start

If you change `slurm.conf`/`gres.conf` (e.g. GPU sharing), restart order matters: `slurmctld` (login node) **then** `slurmd` (compute node) — `SelectType` changes are not safely hot-reloadable via `scontrol reconfigure`, and restarting in the wrong order can leave the node falsely `DRAIN`ed (`gres/shard count reported lower than configured`); fix with `scontrol update NodeName=<node> State=RESUME` once both are confirmed restarted.
