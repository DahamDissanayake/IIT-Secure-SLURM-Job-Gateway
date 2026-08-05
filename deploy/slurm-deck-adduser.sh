#!/usr/bin/env bash
# slurm-deck-adduser.sh — provision a real per-user account across both cluster nodes.
#
# Usage:  sudo slurm-deck-adduser <username> [--dry-run] [--admin] [--shell-user]
#
# Three user types:
#   (default)     → gpuusers; forced-TUI via ForceCommand; audited
#   --admin       → gpuusers + gpuadmins; forced-TUI + admin panel; audited
#   --shell-user  → NO gpuusers / NO gpuadmins; real bash shell; NOT audited
#                   Still gets a SLURM association and /shared/users/<user>.
#
# Every account type joins the `docker` group on BOTH nodes (login node and
# GPU host both run a Docker daemon), so they can `docker build`/`docker run`
# without sudo — needed for JupyterLab/notebook work on the GPU host as much
# as shell sessions on the login node.
#
# --admin and --shell-user are mutually exclusive.
#
# Creates the user on the login node AND (over SSH) the GPU host with a UID free
# on BOTH nodes, registers their SLURM association, and makes their /shared area.
# Group membership is the whole mechanism — the TUI itself is never copied per user.
#
# Site config comes from deploy/site.env (or environment). No hardcoded values.
set -euo pipefail

# ── Load site config ───────────────────────────────────────────────────────────
SITE_ENV="${SD_SITE_ENV:-/opt/slurm-deck/deploy/site.env}"
[ -f "$SITE_ENV" ] && set -a && . "$SITE_ENV" && set +a

GPUUSERS_GROUP="${GPUUSERS_GROUP:-gpuusers}"
ADMIN_GROUP="${ADMIN_GROUP:-gpuadmins}"
DOCKER_GROUP="${DOCKER_GROUP:-docker}"
NFS_ROOT="${NFS_ROOT:-/shared}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-default}"
SLURM_QOS="${SLURM_QOS:-normal}"
GPU_HOST_SSH="${GPU_HOST_SSH:-}"          # e.g. root-daham@192.168.122.1 (required)
GPU_HOST_USER="${GPU_HOST_SSH%%@*}"       # the user on the GPU host (e.g. root-daham)
UID_MIN="${UID_MIN:-2000}"
UID_MAX="${UID_MAX:-60000}"

ok()   { echo "  ✔  $*"; }
warn() { echo "  ⚠  $*"; }
fail() { echo "  ✘  $*" >&2; exit 1; }
step() { echo; echo "==> $*"; }

# ── Args ───────────────────────────────────────────────────────────────────────
USERNAME=""; DRY=0; ADMIN=0; SHELL_USER=0
for a in "$@"; do
    case "$a" in
        --dry-run)    DRY=1 ;;
        --admin)      ADMIN=1 ;;
        --shell-user) SHELL_USER=1 ;;
        -*)           fail "unknown flag: $a" ;;
        *)            USERNAME="$a" ;;
    esac
done
[ -n "$USERNAME" ] || fail "usage: slurm-deck-adduser <username> [--dry-run] [--admin|--shell-user]"
[[ "$USERNAME" =~ ^[a-z_][a-z0-9_-]{0,31}$ ]] || fail "invalid username: $USERNAME"
[ "$ADMIN" = 1 ] && [ "$SHELL_USER" = 1 ] && fail "--admin and --shell-user are mutually exclusive"
[ -n "$GPU_HOST_SSH" ] || fail "GPU_HOST_SSH not set (in $SITE_ENV or environment)"

run() { if [ "$DRY" = 1 ]; then echo "  [dry-run] $*"; else eval "$@"; fi; }

[ "$(id -u)" = 0 ] || [ "$DRY" = 1 ] || fail "must run as root (sudo)"

if [ "$SHELL_USER" = 1 ]; then
    step "Shell user — will NOT be added to $GPUUSERS_GROUP or $ADMIN_GROUP"
    warn "Activity will NOT be audited by the gateway tool"
fi

# ── 1. Pick a UID free on BOTH nodes ───────────────────────────────────────────
step "Finding a UID free on both nodes (>= $UID_MIN) ..."
local_max=$(getent passwd | awk -F: -v lo="$UID_MIN" -v hi="$UID_MAX" '$3>=lo && $3<=hi {print $3}' | sort -n | tail -1)
remote_max=$(ssh "$GPU_HOST_SSH" "getent passwd | awk -F: -v lo=$UID_MIN -v hi=$UID_MAX '\$3>=lo && \$3<=hi {print \$3}' | sort -n | tail -1")
start=$(( ${local_max:-$((UID_MIN-1))} > ${remote_max:-$((UID_MIN-1))} ? ${local_max:-$((UID_MIN-1))} : ${remote_max:-$((UID_MIN-1))} ))
NEW_UID=$(( start < UID_MIN ? UID_MIN : start + 1 ))
# Ensure truly free on both
while getent passwd "$NEW_UID" >/dev/null 2>&1 || ssh "$GPU_HOST_SSH" "getent passwd $NEW_UID >/dev/null 2>&1"; do
    NEW_UID=$((NEW_UID + 1))
done
ok "Chosen UID/GID: $NEW_UID"

# ── 2. Create on login node ────────────────────────────────────────────────────
step "Creating $USERNAME on login node ..."
run "groupadd -g $NEW_UID $USERNAME 2>/dev/null || true"
run "useradd -u $NEW_UID -g $NEW_UID -m -s /bin/bash $USERNAME 2>/dev/null || true"
# A pre-existing home from an earlier incarnation of this name can be left owned
# by a stale UID (useradd -m won't re-chown an existing dir). The user then can't
# read their own dotfiles -- e.g. ~/.config/conda/.condarc -- so `conda activate`
# crashes and notebook/job env activation fails. Force home ownership to match.
run "chown -R $NEW_UID:$NEW_UID /home/$USERNAME"
[ "$SHELL_USER" = 0 ] && run "usermod -aG $GPUUSERS_GROUP $USERNAME"
[ "$ADMIN" = 1 ] && run "getent group $ADMIN_GROUP >/dev/null 2>&1 && usermod -aG $ADMIN_GROUP $USERNAME || true"
# Every account type gets docker access via group rather than sudo, matching
# how gpuusers get GPU access via a group rather than sudo.
run "getent group $DOCKER_GROUP >/dev/null 2>&1 && usermod -aG $DOCKER_GROUP $USERNAME || true"
ok "login: $USERNAME created"

# ── 3. Create on GPU host (same UID) ───────────────────────────────────────────
# --admin must also join $ADMIN_GROUP HERE, not just on the login node: jobs and
# notebooks run on the GPU host, and per-user areas there are group $ADMIN_GROUP
# (2770). An admin who is only in the group on the login node sees the admin
# panel but cannot read any user's area from a job/notebook — the login-node
# membership doesn't reach the host where the data actually lives.
step "Creating $USERNAME on GPU host ($GPU_HOST_SSH) ..."
run "ssh $GPU_HOST_SSH \"sudo groupadd -g $NEW_UID $USERNAME 2>/dev/null || true; \
    sudo useradd -u $NEW_UID -g $NEW_UID -m -s /bin/bash $USERNAME 2>/dev/null || true; \
    sudo chown -R $NEW_UID:$NEW_UID /home/$USERNAME\""
if [ "$SHELL_USER" = 0 ]; then
    run "ssh $GPU_HOST_SSH \"sudo usermod -aG $GPUUSERS_GROUP $USERNAME\""
    if [ "$ADMIN" = 1 ]; then
        run "ssh $GPU_HOST_SSH \"getent group $ADMIN_GROUP >/dev/null 2>&1 && sudo usermod -aG $ADMIN_GROUP $USERNAME || true\""
    fi
fi
# Every account type gets docker access on the GPU host too — JupyterLab and
# notebook jobs run here, so this is where users actually need it most.
run "ssh $GPU_HOST_SSH \"getent group $DOCKER_GROUP >/dev/null 2>&1 && sudo usermod -aG $DOCKER_GROUP $USERNAME || true\""
ok "GPU host: $USERNAME created (UID $NEW_UID)"

# ── 4. SLURM association ────────────────────────────────────────────────────────
step "Registering SLURM association ..."
run "sacctmgr -i add user $USERNAME account=$SLURM_ACCOUNT qos=$SLURM_QOS 2>/dev/null || true"
ok "SLURM: $USERNAME → account=$SLURM_ACCOUNT qos=$SLURM_QOS"

# ── 5. Shared workspace (owner + admins, 2770) + ~/shared convenience symlink ──
# Create + chown ON THE GPU HOST: it is the NFS server, so root is real there.
# With root_squash on the export, an admin chown over NFS from the login node
# would be squashed to nobody and fail. Mode 2770 group $ADMIN_GROUP means the
# owner and admins can reach the area and nobody else can; setgid keeps that
# group on anything created inside. Shell users get the same treatment.
#
# Also provision $NFS_ROOT/jobs/$USERNAME here, same ownership/mode. Nothing
# else creates it ahead of time: make_job_folder() only mkdir(parents=True)s a
# job folder as a side effect, which (absent this) inherits group $GPUUSERS_GROUP
# from the jobs/ parent instead of $ADMIN_GROUP, and the user isn't a member of
# $ADMIN_GROUP to fix it themselves — so their very first job becomes readable
# and writable by every cluster user.
step "Creating $NFS_ROOT/users/$USERNAME and $NFS_ROOT/jobs/$USERNAME on the NFS server (GPU host) ..."
# The chmod above already grants $ADMIN_GROUP real access via ordinary POSIX
# group bits: setgid is inherited normally on this filesystem (see the
# comment at step 5 above), so a 2770 $ADMIN_GROUP directory here means
# anything created inside picks up that group too. (An earlier note here
# claimed setgid was not inherited on this share; that was a measurement
# artifact of this host's `mkdir` binary -- uutils coreutils 0.8.0 --
# mishandling ACL-bearing parent directories. Checked against the raw
# syscall, setgid inherits fine -- see slurmdeck/jobs.py make_job_folder for
# the live comparison. On this host, verify mode-bit behaviour with a raw
# syscall, not a coreutils binary.)
#
# The explicit setfacl below (same as deploy/fix-shared-perms.sh, which
# repairs existing areas) adds a second, independent layer: `stat`/mode
# bits can still mislead a reader once a directory carries an extended ACL
# (verified live: a dir reporting group=gpuadmins mode=660 had ACL entry
# `group::---`), so the ACL is what deploy/check-shared-perms.sh actually
# verifies. The `d:` default entries make that survive everything the new
# user creates afterward. Set it explicitly here rather than relying on
# inheriting the users/ and jobs/ parents' own default ACL -- a brand-new
# area should carry its own guarantee, not depend on the parent alone.
run "ssh $GPU_HOST_SSH \"sudo mkdir -p $NFS_ROOT/users/$USERNAME && \
    sudo chown $NEW_UID:$ADMIN_GROUP $NFS_ROOT/users/$USERNAME && \
    sudo chmod 2770 $NFS_ROOT/users/$USERNAME && \
    sudo setfacl -R -m g:$ADMIN_GROUP:rwX -m d:g:$ADMIN_GROUP:rwx $NFS_ROOT/users/$USERNAME && \
    sudo mkdir -p $NFS_ROOT/jobs/$USERNAME && \
    sudo chown $NEW_UID:$ADMIN_GROUP $NFS_ROOT/jobs/$USERNAME && \
    sudo chmod 2770 $NFS_ROOT/jobs/$USERNAME && \
    sudo setfacl -R -m g:$ADMIN_GROUP:rwX -m d:g:$ADMIN_GROUP:rwx $NFS_ROOT/jobs/$USERNAME\""

# Shell users get ~/shared → NFS root so they can reach datasets, models, envs,
# AND their private area at ~/shared/users/<user> in one place.
# Regular/admin users keep the old narrow link: only their private workspace.
if [ "$SHELL_USER" = 1 ]; then
    _SHARED_LINK="$NFS_ROOT"
else
    _SHARED_LINK="$NFS_ROOT/users/$USERNAME"
fi

run "ln -sfn $_SHARED_LINK /home/$USERNAME/shared 2>/dev/null || true"
# ln is not NOPASSWD on the GPU host. Temporarily chown the home dir to the
# provisioning user (NOPASSWD), create the symlink without sudo, then restore.
run "ssh $GPU_HOST_SSH \"sudo chown $GPU_HOST_USER /home/$USERNAME && \
    ln -sfn $_SHARED_LINK /home/$USERNAME/shared && \
    sudo chown $NEW_UID:$NEW_UID /home/$USERNAME\""
ok "workspace ready (owned $NEW_UID:$NEW_UID, 0700)"

# ── 6. Verify ──────────────────────────────────────────────────────────────────
if [ "$DRY" = 0 ]; then
    step "Verifying ..."
    luid=$(id -u "$USERNAME"); ruid=$(ssh "$GPU_HOST_SSH" "id -u $USERNAME")
    [ "$luid" = "$ruid" ] || fail "UID mismatch: login=$luid gpu=$ruid"
    lho=$(stat -c %u "/home/$USERNAME" 2>/dev/null || echo "?")
    rho=$(ssh "$GPU_HOST_SSH" "stat -c %u /home/$USERNAME 2>/dev/null || echo '?'")
    [ "$lho" = "$luid" ] || fail "login home /home/$USERNAME owned by UID $lho, expected $luid"
    [ "$rho" = "$ruid" ] || fail "gpu home /home/$USERNAME owned by UID $rho, expected $ruid"
    if [ "$SHELL_USER" = 0 ]; then
        id "$USERNAME" | grep -q "$GPUUSERS_GROUP" || fail "$USERNAME not in $GPUUSERS_GROUP"
        ok "UID matched ($luid) · in $GPUUSERS_GROUP · forced-TUI applies via group"
    else
        id "$USERNAME" | grep -qw "$GPUUSERS_GROUP" && fail "shell user must NOT be in $GPUUSERS_GROUP"
        ok "UID matched ($luid) · NOT in $GPUUSERS_GROUP · real shell, cluster-capped by SLURM"
    fi
fi

echo
if [ "$SHELL_USER" = 1 ]; then
    echo "Done (shell user). Set a password or install an SSH key:"
    echo "    sudo passwd $USERNAME"
    echo "NOTE: $USERNAME has a real shell. Their activity is NOT audited by the tool."
    echo "      They are subject to SLURM gres/gpu limits via their association."
else
    echo "Done. Set a password or install an SSH key:"
    echo "    sudo passwd $USERNAME            # or: install ~$USERNAME/.ssh/authorized_keys"
    echo "$USERNAME will land directly in the TUI on next SSH login."
fi
echo "      They are in the '$DOCKER_GROUP' group on both nodes (login + GPU host)."
