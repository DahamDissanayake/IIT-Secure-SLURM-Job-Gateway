# Installing the IIT GPU Manager — instructions for any AI coding assistant

Paste this whole file into your AI coding assistant (Cursor, Copilot
Workspace, Windsurf, etc.) while it has this repository open, and ask it to
install the IIT GPU Manager for you. It describes the same guided,
checkpointed process as this repo's Claude Code skill (`skills/igm-installer`
— use that instead if you're on Claude Code, via `/igm-installer`).

## Your job as the assistant

You are installing real software onto someone's SLURM cluster: root
packages, `sshd_config.d/`, `sudoers.d/`, a systemd service, and a new Linux
account. Before running ANY command that changes system state (installs
packages, writes to `/etc`, restarts a service, creates an account), explain
exactly what it will do and get the user's explicit "yes, do it" first. Never
invent a hostname, account name, or path the user hasn't told you — ask.

Read `README.md`'s "Linux machine setup (before installation)" and
"Installation" sections, plus `deploy/install.sh` itself, before you start —
they are the current source of truth for what the install actually does;
this file only describes the interview-and-checkpoint process around them.

## Steps

1. **Preflight.** Ask for the login-node and compute-node SSH targets. Check
   each item in README's "Linux machine setup" checklist against the real
   machines (Python version, sshd version, sudo, SLURM binaries on PATH, the
   NFS mount, `acl` package, `nvidia-smi` on the compute node, whether a
   shared conda already exists). Report pass/fail. Stop and tell the user
   what to fix if there's a hard blocker (no SLURM, no NFS mount, no root).

2. **Interview for `deploy/site.env`.** Open `deploy/site.env.example` to see
   the current field list and defaults. Ask the user for their real values:
   `NFS_ROOT`, SLURM account/QOS/partition names, admin group (default
   `gpuadmins`), gateway group (default `gpuusers`), `GATEWAY_HOST`/
   `GATEWAY_PORT`, cluster name, UID range. Ask whether they want
   transactional email (Resend API key in `deploy/secrets.env`) or want to
   skip it. Write `deploy/site.env` (must stay `0644`, no secrets in it); if
   they opted into mail, write `deploy/secrets.env` as `0640 root:gpusync`.

3. **Checkpoint: run the installer.** Show the numbered list of what
   `deploy/install.sh` does (read it from the script/README, don't
   paraphrase from memory). Get explicit confirmation, then run
   `sudo bash deploy/install.sh`. If it fails, stop and show the real error —
   don't retry blindly or continue to later steps on a partial failure.

4. **Checkpoint: admin scripts + sudoers.** `install.sh` doesn't install the
   per-user provisioning scripts or the admin sudoers file automatically.
   Show the exact commands from README's "Installation" step 3 before
   running them, and always run `sudo visudo -cf /etc/sudoers.d/iit-gpu-admin`
   afterward and show its output — a bad sudoers file is a lockout risk.

5. **GPU host SSH.** Ask for the compute node's SSH target, append
   `GPU_HOST_SSH=...` to `deploy/site.env`, then test
   `ssh <GPU_HOST_SSH> true`. Stop here if it fails — provisioning depends on
   it.

6. **Verify the service.** `systemctl status iit-gpu-audit` should be active.

7. **Checkpoint: provision the first admin.** Ask for and confirm the exact
   username, then run `sudo /usr/local/bin/iit-gpu-adduser <username>
   --admin`. Suggest verifying with
   `ssh -p <GATEWAY_PORT> <username>@<GATEWAY_HOST>`.

8. **Optional checkpoint: GPU sharing.** Ask if they want it now. This edits
   `slurm.conf`/`gres.conf` on both nodes and requires restarting
   `slurmctld` then `slurmd`, in that order (see README's "GPU sharing
   model" — not hot-reloadable). Show the exact diff per node, confirm
   before editing, confirm again before each restart. If a restart doesn't
   come back clean, stop and report rather than restarting the other node on
   top of an undiagnosed failure.

9. **Summary.** State what was done and what was skipped, and point the user
   at README's "Adding and removing users" and "Using the tool — full menu
   reference" sections for next steps.

## Rules

- Every root/sudo/sshd/sudoers/systemd/account-creation command gets
  explained and confirmed before it runs. No exceptions.
- If a step fails, stop and show the real error instead of improvising
  around it.
- If this file and the repo's current README/`install.sh` disagree, trust
  the repo — this file describes the process, not a frozen copy of the
  install steps.
