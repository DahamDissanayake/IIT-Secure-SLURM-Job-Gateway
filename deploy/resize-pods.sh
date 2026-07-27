#!/usr/bin/env bash
# Resize the cluster's GPU pod count (gres/shard Count=N) on both nodes.
# MUST be run with an empty job queue (checked by the admin panel before
# this script is ever invoked; --dry-run skips the live squeue check
# entirely so this script is testable without a real cluster).
#
# Usage: resize-pods.sh <new_pod_count> [--dry-run]
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
LOCK_FILE="${RESIZE_LOCK_FILE:-/var/run/iit-gpu-resize.lock}"

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

echo "== syncing gres.conf to the GPU host and restarting slurmd"
scp "$CONF_DIR/gres.conf" "$CONF_DIR/slurm.conf" \
    "${GPU_HOST_SSH}:/tmp/iit-resize-$TS/" 2>/dev/null || {
    mkdir_cmd="mkdir -p /tmp/iit-resize-$TS"
    ssh "$GPU_HOST_SSH" "$mkdir_cmd"
    scp "$CONF_DIR/gres.conf" "$CONF_DIR/slurm.conf" "${GPU_HOST_SSH}:/tmp/iit-resize-$TS/"
}
ssh "$GPU_HOST_SSH" "sudo cp /tmp/iit-resize-$TS/gres.conf /tmp/iit-resize-$TS/slurm.conf /etc/slurm/ && sudo systemctl restart slurmd"

echo "== restarting slurmctld (login node)"
sudo systemctl restart slurmctld

echo "== resuming node (a GRES change reliably drains it)"
sudo scontrol update NodeName="$NODE_NAME" State=RESUME

echo "== verifying"
reported="$(ssh "$GPU_HOST_SSH" "slurmd -G" 2>/dev/null | grep -o 'shard:[0-9]*' | head -1 || true)"
if [ "$reported" != "shard:$NEW_N" ]; then
    echo "ERROR: GPU host reports '$reported', expected 'shard:$NEW_N' -- rolling back" >&2
    cp "$CONF_DIR/gres.conf.bak.$TS" "$CONF_DIR/gres.conf"
    cp "$CONF_DIR/slurm.conf.bak.$TS" "$CONF_DIR/slurm.conf"
    scp "$CONF_DIR/gres.conf" "$CONF_DIR/slurm.conf" "${GPU_HOST_SSH}:/tmp/iit-resize-rollback-$TS/" 2>/dev/null || true
    ssh "$GPU_HOST_SSH" "sudo cp /tmp/iit-resize-rollback-$TS/gres.conf /tmp/iit-resize-rollback-$TS/slurm.conf /etc/slurm/ 2>/dev/null; sudo systemctl restart slurmd" || true
    sudo systemctl restart slurmctld
    sudo scontrol update NodeName="$NODE_NAME" State=RESUME
    echo "ERROR: rollback performed -- resize did NOT apply" >&2
    exit 1
fi

echo "== resize applied: pod count is now $NEW_N"
