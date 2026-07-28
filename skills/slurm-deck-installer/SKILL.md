---
name: slurm-deck-installer
description: Use when the user wants to install, set up, or deploy the Slurm Deck on a login node / compute node pair. Runs the real install (deploy/install.sh and the steps around it) as a guided, checkpointed process instead of a blind one-shot script.
---

# slurm-deck-installer

Guided installation of the Slurm Deck onto a fresh (or existing) SLURM
cluster. You are driving a **real install onto someone's infrastructure** —
root packages, `sshd_config.d/`, `sudoers.d/`, a systemd service, and a new
Linux/SLURM account. Treat every state-changing step as a checkpoint: explain
what you're about to run, then get an explicit go-ahead before running it.
Never guess site-specific values (hostnames, account/QOS/partition names,
paths) — ask.

**Ground truth lives in this repo, not in this file.** Before doing anything,
read `README.md`'s "Linux machine setup (before installation)" and
"Installation" sections, and `deploy/install.sh` itself, in the checkout you
are running from. Those are what install.sh actually does *today*; this skill
only adds the interview + checkpoint process around them. If they've drifted
from the summary below, follow the repo, not this file.

## Process

Track this as a checklist (one task per phase) so progress survives context
compaction.

### 1. Preflight (read-only, no checkpoint needed)

Walk the "Both nodes" / "GPU / compute node additionally" / "Login node
additionally" checklists from README's "Linux machine setup" section against
the actual target machine(s) the user gives you (ask for the login-node and
compute-node hostnames/SSH targets if not already provided). For each item,
check don't assume:
- `python3 --version`, `sshd -V`, `sudo -V`
- `sbatch`/`squeue`/`scancel`/`sinfo`/`sacct`/`scontrol`/`sacctmgr` on `PATH`
- The NFS export mounted read/write at the intended path on **both** nodes
- `setfacl`/`getfacl` present (the `acl` package)
- `nvidia-smi` on the compute node
- Whether a shared conda path already exists, or install.sh needs to install
  Miniforge

Report a clear pass/fail table. If there are hard blockers (no SLURM, no NFS
mount, no root/sudo), stop and tell the user what to fix first — don't try to
provision missing infrastructure yourself, only the packages install.sh
itself installs.

### 2. Interview for site.env

Ask for the values `deploy/site.env.example` needs (open that file to see the
current field list — don't hardcode a copy here). At minimum you need:
`NFS_ROOT`, SLURM account/QOS/partition names, admin group name (default
`gpuadmins`), gateway group name (default `gpuusers`), `GATEWAY_HOST` /
`GATEWAY_PORT`, cluster name, UID range. These can be asked together (they're
independent choices, not a step-by-step wizard) with the example file's
defaults shown as suggestions.

Ask separately whether they want transactional/notification email configured
now (Resend API key into `deploy/secrets.env`) or want to skip it — mail is
optional and the tool works with it disabled.

Write `deploy/site.env` (must end up `0644`, no secrets). If they opted into
mail, write `deploy/secrets.env` and set it `0640 root:gpusync`.

### 3. CHECKPOINT — run install.sh

Before running anything, show the user the numbered list of exactly what
`deploy/install.sh` does (read it from the script / from README's
"Installation" step 3, do not paraphrase from stale memory of this skill).
Get an explicit yes, then run:
```
sudo bash deploy/install.sh
```
If it fails, stop and show the real error — don't retry blindly, don't skip
ahead to later steps on a partial failure.

### 4. CHECKPOINT — admin scripts + sudoers

install.sh does **not** install the per-user provisioning scripts or the
admin sudoers file automatically. Show the user the exact commands (from
README's "Installation" step 3, "next step" block) before running them,
particularly:
```
sudo install -m 0440 -o root -g root deploy/sudoers-gateway-admin /etc/sudoers.d/slurm-deck-admin
sudo visudo -cf /etc/sudoers.d/slurm-deck-admin
```
Always run the `visudo -cf` validation and show its output — a bad sudoers
file is a lockout risk.

### 5. GPU host SSH

Ask for the compute node's SSH target (e.g. `root@<compute-ip>`), append
`GPU_HOST_SSH=...` to `deploy/site.env`, then test:
```
ssh <GPU_HOST_SSH> true
```
If this fails, stop here — don't proceed to provisioning a user whose
account creation depends on this working (`slurm-deck-adduser.sh` needs it).

### 6. Verify the service

```
systemctl status slurm-deck-audit
```
Confirm it's active before moving on.

### 7. CHECKPOINT — provision the first admin

Ask for the exact username to provision (a real Linux account will be
created on both nodes). Confirm the username back to the user before
running:
```
sudo /usr/local/bin/slurm-deck-adduser <username> --admin
```
Then suggest they verify by SSHing in as that user:
```
ssh -p <GATEWAY_PORT> <username>@<GATEWAY_HOST>
```

### 8. Optional CHECKPOINT — GPU sharing

This is the highest-blast-radius step: it edits `slurm.conf`/`gres.conf` on
**both** nodes and requires restarting `slurmctld` then `slurmd`, in that
order (not hot-reloadable via `scontrol reconfigure` — see README's "GPU
sharing model"). Ask if they want this now or want to skip it. If yes: show
the exact diff for each node's config file, get confirmation per file before
editing, then get a separate confirmation before each service restart. If
either restart doesn't come back clean, stop and report — don't restart the
other node's service on top of a failure you haven't diagnosed.

### 9. Summary

Tell the user what was done, what was skipped (e.g. GPU sharing deferred),
and point them at README's "Adding and removing users" and "Using the tool —
full menu reference" sections for what comes next.

## Rules

- Never run a root/sudo/sshd/sudoers/systemd/account-creation command
  without first stating what it does and getting an explicit go-ahead.
- Never invent a hostname, account name, or path the user hasn't given you.
- If a step fails, stop and surface the real error instead of improvising a
  workaround — infrastructure installs are not the place to guess.
- Prefer re-reading `README.md` / `deploy/install.sh` over trusting this
  file's summaries if they ever disagree; this file describes the process,
  the repo describes the current install.
