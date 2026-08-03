#!/usr/bin/env bash
# install-wizard.sh — entry point for the Slurm Deck installation wizard.
#
# Usage (from a fresh GPU host or a machine that will orchestrate install):
#   git clone https://github.com/DahamDissanayake/slurm-deck.git
#   cd slurm-deck
#   sudo bash install-wizard.sh
#
# Walks through creating the login-node VM (or pointing at existing
# machines), Linux users/groups, the app install on the login node,
# cross-node permission sync, Resend email setup, and first-admin
# provisioning -- with a confirmation checkpoint before every
# root/sudo/system-mutating step. Safe to re-run if interrupted; see
# deploy/wizard/state.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 is required. Install it first: apt-get install -y python3" >&2
    exit 1
fi

exec python3 "$SCRIPT_DIR/deploy/wizard/main.py" "$@"
