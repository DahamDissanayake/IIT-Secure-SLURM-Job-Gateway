#!/usr/bin/env bash
# Read-only audit of /shared access. Safe to run from any node — root_squash
# blocks writes from the login node but stat() works fine.
#
# Rule: per-user areas (users/, jobs/) grant nothing to "other".
#       Shared asset dirs grant no write to "other".
set -uo pipefail

NFS_ROOT="${NFS_ROOT:-/shared}"
ADMIN_GROUP="${ADMIN_GROUP:-gpuadmins}"
SHARED_DIRS="${SHARED_DIRS:-data datasets envs models templates}"
fail=0

for base in users jobs; do
    [ -d "$NFS_ROOT/$base" ] || continue
    for d in "$NFS_ROOT/$base"/*; do
        [ -d "$d" ] || continue
        [ -L "$d" ] && continue
        mode=$(stat -c %a "$d" 2>/dev/null) || continue
        if [ "${mode: -1}" != "0" ]; then
            echo "EXPOSED   $d  mode=$mode  (other must be 0)"
            fail=1
        fi
        grp_name=$(stat -c %G "$d" 2>/dev/null)
        if [ "$grp_name" != "$ADMIN_GROUP" ]; then
            echo "WRONGGROUP $d  group=$grp_name  (per-user areas must be group $ADMIN_GROUP)"
            fail=1
        fi
    done
done

for d in $SHARED_DIRS; do
    p="$NFS_ROOT/$d"
    [ -d "$p" ] || continue
    mode=$(stat -c %a "$p" 2>/dev/null) || continue
    other="${mode: -1}"
    case "$other" in
        2|3|6|7) echo "WRITABLE  $p  mode=$mode  (other must not have write)"; fail=1 ;;
    esac
done

if [ "$fail" -ne 0 ]; then
    echo
    echo "Fix on the GPU HOST (the NFS server — root is squashed on the login node):"
    echo "  ssh root-daham@192.168.122.1 'sudo bash /opt/iit-gpu/deploy/fix-shared-perms.sh'"
    exit 1
fi

echo "shared permissions OK"
