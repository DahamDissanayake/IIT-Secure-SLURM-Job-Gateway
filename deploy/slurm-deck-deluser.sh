#!/usr/bin/env bash
# slurm-deck-deluser.sh — offboard a per-user account from both nodes.
#
# Usage:  sudo slurm-deck-deluser <username> [--dry-run] [--purge-data]
#
# Removes the user from login + GPU host and their SLURM association. By default
# their /shared/users/<user> data is KEPT (renamed to <user>.offboarded).
# --purge-data deletes it. Never removes 'public' or the shared service account.
set -euo pipefail

SITE_ENV="${SD_SITE_ENV:-/opt/slurm-deck/deploy/site.env}"
[ -f "$SITE_ENV" ] && set -a && . "$SITE_ENV" && set +a
NFS_ROOT="${NFS_ROOT:-/shared}"
GPU_HOST_SSH="${GPU_HOST_SSH:-}"
SHARED_USER="${GATEWAY_SHARED_USER_NAME:-daham}"
SD_INSTALL_DIR="${SD_INSTALL_DIR:-/opt/slurm-deck}"

ok(){ echo "  ✔  $*"; }; warn(){ echo "  ⚠  $*"; }
fail(){ echo "  ✘  $*" >&2; exit 1; }; step(){ echo; echo "==> $*"; }

USERNAME=""; DRY=0; PURGE=0
for a in "$@"; do
    case "$a" in
        --dry-run)    DRY=1 ;;
        --purge-data) PURGE=1 ;;
        -*)           fail "unknown flag: $a" ;;
        *)            USERNAME="$a" ;;
    esac
done
[ -n "$USERNAME" ] || fail "usage: slurm-deck-deluser <username> [--dry-run] [--purge-data]"
case "$USERNAME" in
    public|"$SHARED_USER"|root|slurm|slurmadmin|root-daham)
        fail "refusing to remove protected account: $USERNAME" ;;
esac
[ -n "$GPU_HOST_SSH" ] || fail "GPU_HOST_SSH not set"
[ "$(id -u)" = 0 ] || [ "$DRY" = 1 ] || fail "must run as root (sudo)"
run() { if [ "$DRY" = 1 ]; then echo "  [dry-run] $*"; else eval "$@"; fi; }

step "Removing SLURM association ..."
run "sacctmgr -i delete user $USERNAME 2>/dev/null || true"; ok "assoc removed"

step "Cancelling any active/queued SLURM jobs ..."
run "scancel -u $USERNAME 2>/dev/null || true"
[ "$DRY" = 1 ] || sleep 2   # give slurmd a moment to reap the job's processes
ok "jobs cancelled"

step "Handling /shared/users data (on the NFS server; root_squash-safe) ..."
USER_DATA="${NFS_ROOT}/users/${USERNAME}"
if [ "$PURGE" = 1 ]; then
    # chown -R is NOPASSWD on GPU host; once root-daham owns all files, rm needs no sudo
    run "ssh $GPU_HOST_SSH \"sudo -n chown -R root-daham:root-daham ${USER_DATA} 2>/dev/null || true; rm -rf ${USER_DATA}\""
    ok "data purged"
else
    # mv within /shared/users/ (owned by root-daham) needs no sudo — just a rename
    ARCHIVED="${USER_DATA}.offboarded"
    run "ssh $GPU_HOST_SSH \"[ -d ${ARCHIVED} ] && rm -rf ${ARCHIVED}; mv ${USER_DATA} ${ARCHIVED} 2>/dev/null || true\""
    ok "data archived as ${USERNAME}.offboarded"
fi

step "Removing user on GPU host ..."
if [ "$DRY" = 1 ]; then
    echo "  [dry-run] userdel -r $USERNAME; groupdel $USERNAME  (on GPU host)"
else
    # userdel silently no-ops on a still-busy account (e.g. a leftover job
    # process) unless we check afterwards — a masked failure here previously
    # left the OS account alive while the tool reported the offboard as done.
    ssh "$GPU_HOST_SSH" "
        sudo userdel -r '$USERNAME' 2>/dev/null
        sudo groupdel '$USERNAME' 2>/dev/null
        ! id '$USERNAME' >/dev/null 2>&1
    " && _gpu_gone=1 || _gpu_gone=0
    [ "$_gpu_gone" = 1 ] || fail "GPU host account $USERNAME still exists after userdel (a leftover job process is likely holding it open — retry, or check 'ps -u $USERNAME' on the GPU host)"
fi
ok "GPU host cleaned"

step "Removing user on login node ..."
if [ "$DRY" = 1 ]; then
    echo "  [dry-run] pkill -u $USERNAME; userdel -r $USERNAME; groupdel $USERNAME  (on login node)"
else
    # Kick any live login/TUI session first — otherwise userdel refuses to
    # remove an account that's "currently used by process N", the exact
    # silent-failure mode that let offboarded users keep logging in.
    pkill -KILL -u "$USERNAME" 2>/dev/null || true
    sleep 1
    userdel -r "$USERNAME" 2>/dev/null || true
    groupdel "$USERNAME" 2>/dev/null || true
    if id "$USERNAME" >/dev/null 2>&1; then
        fail "login account $USERNAME still exists after userdel (an active session/process is likely blocking removal — retry, or check 'ps -u $USERNAME')"
    fi
fi
ok "login cleaned"

# ── Update users.db via daemon (best-effort; daemon must be running) ──────────
step "Updating user database ..."
if [ "$DRY" = 0 ]; then
    PYTHONPATH="$SD_INSTALL_DIR" python3 -m slurmdeck.daemoncli users.offboard \
        --username "$USERNAME" 2>/dev/null \
        && ok "DB row offboarded" \
        || warn "Could not update user DB (daemon may be stopped — offboard manually)"
else
    echo "  [dry-run] would call users.offboard for $USERNAME in user DB"
fi

echo; echo "Offboarded $USERNAME."
