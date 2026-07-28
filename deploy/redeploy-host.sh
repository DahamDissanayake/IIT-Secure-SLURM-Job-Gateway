#!/usr/bin/env bash
# redeploy-host.sh — Run from the GPU host (iit-MS-7E06) as root-daham.
#
# This machine has no git. All git/deploy work runs on the login node
# (192.168.122.10) via SSH. This script:
#   1. Triggers the login-node deploy (git pull → tests → /opt/slurm-deck → service restart)
#   2. Ensures the slurm-deck-stats systemd service is active on this host
set -euo pipefail

LOGIN="slurmadmin@192.168.122.10"
STATS_JSON="/shared/.gpu_stats.json"
WRITER_DEST="/usr/local/bin/slurm-deck-stats-writer"
SERVICE_SRC="/tmp/slurm-deck-stats.service"   # synced from login node via SSH before this script

ok()   { echo "  ✔  $*"; }
warn() { echo "  ⚠  $*"; }
fail() { echo "  ✘  $*" >&2; exit 1; }
step() { echo; echo "==> $*"; }

# ── 1. Deploy on login node ───────────────────────────────────────────────────
step "Running deploy on login node (192.168.122.10)..."
ssh "$LOGIN" "bash /home/slurmadmin/slurm-deck/deploy/redeploy-slurm-deck.sh" \
    || fail "Login-node deploy failed"

# ── 2. Sync service files from login node ────────────────────────────────────
step "Syncing stats writer from login node..."
scp "$LOGIN:/home/slurmadmin/slurm-deck/deploy/slurm-deck-stats-writer" \
    "$WRITER_DEST"
chmod +x "$WRITER_DEST"

scp "$LOGIN:/home/slurmadmin/slurm-deck/deploy/slurm-deck-stats.service" \
    /etc/systemd/system/slurm-deck-stats.service

ok "Files synced"

# ── 3. Install / reload systemd service ──────────────────────────────────────
step "Installing slurm-deck-stats systemd service..."

# [GPU-HOST] these commands require root — emit as labeled block if not root
if [ "$(id -u)" -ne 0 ]; then
    echo
    echo "  ┌─────────────────────────────────────────────────────────────────┐"
    echo "  │  [GPU-HOST] run manually as root to install/restart service:    │"
    echo "  │    sudo systemctl daemon-reload                                  │"
    echo "  │    sudo systemctl enable --now slurm-deck-stats                    │"
    echo "  │    sudo systemctl status slurm-deck-stats                          │"
    echo "  └─────────────────────────────────────────────────────────────────┘"
    echo
    warn "Run the above as root to activate the systemd service."
else
    systemctl daemon-reload
    systemctl enable slurm-deck-stats
    systemctl restart slurm-deck-stats
    sleep 3
    if systemctl is-active --quiet slurm-deck-stats; then
        ok "slurm-deck-stats is active ($(systemctl show slurm-deck-stats --property=MainPID --value))"
    else
        warn "Service not active — check: journalctl -u slurm-deck-stats -n 30"
    fi
fi

# ── 4. Verify stats file is fresh ────────────────────────────────────────────
step "Checking stats file freshness..."
sleep 5
if [ -f "$STATS_JSON" ]; then
    AGE=$(( $(date +%s) - $(stat --format=%Y "$STATS_JSON") ))
    if [ "$AGE" -lt 15 ]; then
        ok "Stats file fresh (age ${AGE}s)"
    else
        warn "Stats file stale (age ${AGE}s) — service may not be running"
    fi
else
    warn "Stats file not found yet — service may still be starting"
fi

echo
ok "Done."
