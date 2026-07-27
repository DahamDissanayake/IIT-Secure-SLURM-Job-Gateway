#!/usr/bin/env bash
# Resize the cluster's GPU pod count (gres/shard Count=N) on both nodes.
# MUST be run with an empty job queue (checked by the admin panel before
# this script is ever invoked; --dry-run skips the live squeue check
# entirely so this script is testable without a real cluster).
#
# Usage: resize-pods.sh <new_pod_count> [--dry-run]
#
# DEPLOYMENT PREREQUISITE (manual, one time):
#   The admin panel invokes this as
#       sudo -n -u slurmadmin /opt/iit-gpu/deploy/resize-pods.sh <N>
#   which only works once deploy/sudoers-gateway-admin -- which carries the
#       %gpuadmins ALL=(slurmadmin) NOPASSWD: /opt/iit-gpu/deploy/resize-pods.sh *
#   grant -- has been installed on the login node:
#       sudo install -m 0440 -o root -g root \
#            deploy/sudoers-gateway-admin /etc/sudoers.d/iit-gpu-admin
#       sudo visudo -cf /etc/sudoers.d/iit-gpu-admin
#   deploy/redeploy-igm.sh does NOT install sudoers files; a plain redeploy of
#   /opt/iit-gpu leaves this feature failing with a sudo permission error until
#   the install above is run by hand. See README.md ("sudoers-gateway-admin").
#
# --dry-run: only rewrite the LOCAL $SLURM_CONF_DIR files (gres.conf,
# slurm.conf), skip the squeue check, skip GPU-host SSH, skip restarts.
# Used by tests/test_resize_pods_script.py. Without --dry-run, this
# performs the real cross-node resize and MUST run as slurmadmin on the
# login node.
set -euo pipefail

NEW_N="${1:-}"
DRY_RUN=0
[ "${2:-}" = "--dry-run" ] && DRY_RUN=1

CONF_DIR="${SLURM_CONF_DIR:-/etc/slurm}"
NODE_NAME="${NODE_NAME:-iit-MS-7E06}"
GPU_HOST_SSH="${GPU_HOST_SSH:-root-daham@192.168.122.1}"
# /var/run (-> /run) is root:root 0755, so slurmadmin -- the account this
# script runs as -- cannot create a lockfile there; `exec 9>` would fail and
# set -e would abort the whole resize before it did anything. /tmp is writable
# by slurmadmin and is where this script already stages its sync directories.
LOCK_FILE="${RESIZE_LOCK_FILE:-/tmp/iit-gpu-resize.lock}"

if ! [[ "$NEW_N" =~ ^[0-9]+$ ]] || [ "$NEW_N" -lt 1 ]; then
    echo "ERROR: pod count must be a positive integer, got: '$NEW_N'" >&2
    exit 1
fi

[ -d "$CONF_DIR" ] || { echo "ERROR: $CONF_DIR not found" >&2; exit 1; }
[ -f "$CONF_DIR/gres.conf" ] || { echo "ERROR: $CONF_DIR/gres.conf not found" >&2; exit 1; }
[ -f "$CONF_DIR/slurm.conf" ] || { echo "ERROR: $CONF_DIR/slurm.conf not found" >&2; exit 1; }

if [ "$DRY_RUN" -eq 0 ]; then
    [ "$(id -un)" = "slurmadmin" ] || { echo "ERROR: run as slurmadmin" >&2; exit 1; }

    # Lockfile: refuse a second concurrent resize instead of interleaving.
    exec 9>"$LOCK_FILE"
    flock -n 9 || { echo "ERROR: a resize is already in progress" >&2; exit 1; }

    # Cluster-wide empty-queue check, immediately before touching anything --
    # the admin panel already checked this once at confirm time; this is the
    # atomic re-check right before execution.
    running="$(squeue --noheader --states=RUNNING,PENDING 2>/dev/null | wc -l)"
    if [ "$running" -gt 0 ]; then
        echo "ERROR: $running job(s) still active -- refusing to resize" >&2
        exit 1
    fi
fi

TS="$(date +%Y%m%d%H%M%S)"
cp "$CONF_DIR/gres.conf" "$CONF_DIR/gres.conf.bak.$TS"
cp "$CONF_DIR/slurm.conf" "$CONF_DIR/slurm.conf.bak.$TS"
echo "== backed up gres.conf and slurm.conf ($TS)"

sed -i -E "s/(Name=shard Count=)[0-9]+/\1${NEW_N}/" "$CONF_DIR/gres.conf"
sed -i -E "s/(Gres=gpu:1,shard:)[0-9]+/\1${NEW_N}/" "$CONF_DIR/slurm.conf"
echo "== rewrote local gres.conf/slurm.conf -> Count=$NEW_N"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "== dry-run: stopping before GPU-host sync / restarts"
    exit 0
fi

# Push whatever gres.conf/slurm.conf currently sit in $CONF_DIR to the GPU host
# and restart slurmd there. Staging dir is passed in so the rollback push uses
# its own directory and cannot be confused with the forward one. Every step is
# explicitly `|| return 1`: `set -e` is suppressed inside a function invoked in
# an `if`/`||` context, so without these a failed ssh would silently fall
# through to the next command.
gpu_sync() {
    local dir="$1"
    ssh "$GPU_HOST_SSH" "mkdir -p '$dir'" || return 1
    scp "$CONF_DIR/gres.conf" "$CONF_DIR/slurm.conf" "${GPU_HOST_SSH}:$dir/" || return 1
    ssh "$GPU_HOST_SSH" \
        "sudo cp '$dir/gres.conf' '$dir/slurm.conf' /etc/slurm/ && sudo systemctl restart slurmd" \
        || return 1
}

# Single rollback path, shared by BOTH failure modes (GPU-host sync/restart
# failed outright, and sync succeeded but slurmd -G reports the wrong count).
# It restores the login node from the backups taken above and pushes those same
# restored files back to the GPU host -- and reports honestly if that push
# itself failed, because "login node rolled back, GPU host still on the new
# config" is precisely the half-applied state that must never be reported as a
# clean rollback.
rollback() {
    echo "== rolling back to the pre-resize config" >&2
    cp "$CONF_DIR/gres.conf.bak.$TS" "$CONF_DIR/gres.conf"
    cp "$CONF_DIR/slurm.conf.bak.$TS" "$CONF_DIR/slurm.conf"
    sudo systemctl restart slurmctld
    local rb_rc=0
    gpu_sync "/tmp/iit-resize-rollback-$TS" || rb_rc=$?
    sudo scontrol update NodeName="$NODE_NAME" State=RESUME || true
    if [ "$rb_rc" -ne 0 ]; then
        echo "CRITICAL: rollback of the GPU HOST FAILED (exit $rb_rc)." >&2
        echo "CRITICAL: the login node is back on the pre-resize config but the GPU" >&2
        echo "CRITICAL: host may still hold the NEW one -- the cluster is in a" >&2
        echo "CRITICAL: half-applied state and needs manual repair. Restore" >&2
        echo "CRITICAL: $CONF_DIR/gres.conf + slurm.conf onto ${GPU_HOST_SSH}:/etc/slurm/" >&2
        echo "CRITICAL: and 'systemctl restart slurmd' there before running any job." >&2
        exit 2
    fi
    echo "ERROR: rollback performed -- resize did NOT apply" >&2
    exit 1
}

echo "== restarting slurmctld (login node)"
sudo systemctl restart slurmctld

echo "== syncing gres.conf to the GPU host and restarting slurmd"
if ! gpu_sync "/tmp/iit-resize-$TS"; then
    echo "ERROR: GPU-host sync/restart failed -- rolling back" >&2
    rollback
fi

# slurmctld restarted first above, then slurmd: when the old slurmd
# re-registers with the freshly-restarted controller it can briefly report
# the old shard count, putting the node into an expected transient DRAIN.
# This RESUME clears that -- restarting in the other order (slurmd first)
# skips past this safety net and was the root cause of a prior production
# incident on this cluster.
echo "== resuming node (a GRES change reliably drains it)"
sudo scontrol update NodeName="$NODE_NAME" State=RESUME

echo "== verifying"
reported="$(ssh "$GPU_HOST_SSH" "slurmd -G" 2>/dev/null | grep -o 'shard:[0-9]*' | head -1 || true)"
if [ "$reported" != "shard:$NEW_N" ]; then
    echo "ERROR: GPU host reports '$reported', expected 'shard:$NEW_N' -- rolling back" >&2
    rollback
fi

echo "== resize applied: pod count is now $NEW_N"
