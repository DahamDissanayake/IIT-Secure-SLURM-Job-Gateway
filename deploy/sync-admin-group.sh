#!/usr/bin/env bash
# Ensure the admin group has the same members on this node as on the login node.
# Per-user areas are mode 2770 group gpuadmins, so an admin missing from this
# group on the GPU host cannot reach user data where jobs actually run.
set -euo pipefail

ADMIN_GROUP="${ADMIN_GROUP:-gpuadmins}"
ADMINS="${ADMINS:-slurmadmin dahamadmin indrajith daham}"

[ "$(id -u)" -eq 0 ] || { echo "ERROR: run as root on the GPU host" >&2; exit 1; }
getent group "$ADMIN_GROUP" >/dev/null || { echo "ERROR: group $ADMIN_GROUP missing" >&2; exit 1; }

changed=0
for u in $ADMINS; do
    if ! id -u "$u" >/dev/null 2>&1; then
        echo "  skip $u (no such account on this node)"
        continue
    fi
    if id -nG "$u" | tr ' ' '\n' | grep -qx "$ADMIN_GROUP"; then
        echo "  ok   $u already in $ADMIN_GROUP"
    else
        usermod -aG "$ADMIN_GROUP" "$u"
        echo "  ADD  $u -> $ADMIN_GROUP"
        changed=1
    fi
done

echo "admin group sync complete (changed=$changed)"
