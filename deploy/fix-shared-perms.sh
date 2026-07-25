#!/usr/bin/env bash
# Apply the /shared access model. MUST run as root ON THE GPU HOST: the NFS
# export uses root_squash, so root on the login node cannot change modes here.
# Idempotent.
set -euo pipefail

NFS_ROOT="${NFS_ROOT:-/shared}"
ADMIN_GROUP="${ADMIN_GROUP:-gpuadmins}"
SHARED_DIRS="${SHARED_DIRS:-data datasets envs models templates}"

[ "$(id -u)" -eq 0 ] || { echo "ERROR: run as root" >&2; exit 1; }
[ -d "$NFS_ROOT/users" ] || { echo "ERROR: $NFS_ROOT/users missing — wrong node?" >&2; exit 1; }
getent group "$ADMIN_GROUP" >/dev/null || { echo "ERROR: group $ADMIN_GROUP missing" >&2; exit 1; }

echo "== shared assets -> 2775 (group writable, other read-only) -- top level only,"
echo "   contents underneath are NOT touched (see check-shared-perms.sh header)"
for d in $SHARED_DIRS; do
    p="$NFS_ROOT/$d"
    [ -d "$p" ] || continue
    [ -L "$p" ] && continue
    chmod 2775 "$p"
    echo "  $p"
done

echo "== per-user areas -> 2770 owner:$ADMIN_GROUP"
for base in users jobs; do
    [ -d "$NFS_ROOT/$base" ] || continue
    for p in "$NFS_ROOT/$base"/*; do
        [ -d "$p" ] || continue
        [ -L "$p" ] && continue
        u=$(basename "$p")
        if id -u "$u" >/dev/null 2>&1; then
            chown "$u:$ADMIN_GROUP" "$p"
            echo "  $p"
        else
            # Orphaned area (offboarded account). Keep the numeric owner for
            # forensics, but the group and mode must still be corrected — the
            # data is as private as any live user's.
            chown ":$ADMIN_GROUP" "$p"
            echo "  $p (orphan: no account '$u'; owner left as-is)"
        fi
        chmod 2770 "$p"
    done
done

echo "done"
