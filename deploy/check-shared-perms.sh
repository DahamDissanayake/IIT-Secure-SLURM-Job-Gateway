#!/usr/bin/env bash
# Read-only audit of /shared access. Safe to run from any node — root_squash
# blocks writes from the login node but stat() works fine.
#
# Rule: NFS_ROOT itself grants no write to "other".
#       Per-user areas (users/, jobs/) grant nothing to "other" -- checked at
#       their own top level (each users/<u>, jobs/<u> directory itself).
#       Shared asset dirs grant no write to "other" -- checked at their TOP
#       LEVEL ONLY (e.g. the envs/ directory entry itself). This does NOT
#       recurse: files and subdirectories underneath a shared asset dir are
#       not inspected, so a world-writable file three levels down inside
#       envs/ or models/ is invisible to this check. That's a known, accepted
#       gap (deliberately not auto-fixed — it would touch envs jobs are
#       actively using) -- do not read a clean run as "verified recursively".
set -uo pipefail

NFS_ROOT="${NFS_ROOT:-/shared}"
ADMIN_GROUP="${ADMIN_GROUP:-gpuadmins}"
SHARED_DIRS="${SHARED_DIRS:-data datasets envs models templates}"
fail=0

# The server is the host where the real path backing /shared exists as an
# actual directory. On the GPU host (the NFS server) that's true; on the
# login node /shared is only an NFS mount, so its attribute cache can lag
# the server by its acdirmin/acdirmax window (up to ~60s here).
if [ -d /mnt/nvme_storage/shared ]; then
    AUTHORITATIVE=1
else
    AUTHORITATIVE=0
    echo "NOTE: reading /shared over NFS — directory attributes may lag the server by"
    echo "      up to ~60s, so a very recent change may not be visible yet. The"
    echo "      authoritative check runs on the GPU host (the NFS server)."
fi

if [ -d "$NFS_ROOT" ]; then
    # -L: NFS_ROOT itself may be a convenience symlink (e.g. on the GPU host,
    # /shared -> /mnt/nvme_storage/shared) rather than a real mountpoint. A
    # symlink's own mode is always effectively 777 and means nothing here --
    # dereference to the real directory's permissions.
    mode=$(stat -L -c %a "$NFS_ROOT" 2>/dev/null) || mode=""
    if [ -n "$mode" ]; then
        other="${mode: -1}"
        case "$other" in
            2|3|6|7) echo "WRITABLE  $NFS_ROOT  mode=$mode  ($NFS_ROOT itself must not be other-writable -- any user could then replace a shared dir with a symlink)"; fail=1 ;;
        esac
    fi
fi

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
    [ -L "$p" ] && continue
    mode=$(stat -c %a "$p" 2>/dev/null) || continue
    other="${mode: -1}"
    case "$other" in
        2|3|6|7) echo "WRITABLE  $p  mode=$mode  (other must not have write -- top-level check only, see header)"; fail=1 ;;
    esac
done

if [ "$fail" -ne 0 ]; then
    echo
    echo "Fix on the GPU HOST (the NFS server — root is squashed on the login node)."
    echo "From a shell on the GPU host:"
    echo "  ssh slurmadmin@192.168.122.10 'cat /opt/iit-gpu/deploy/fix-shared-perms.sh' > /tmp/fix-shared-perms.sh"
    echo "  sudo bash /tmp/fix-shared-perms.sh"
    exit 1
fi

if [ "$AUTHORITATIVE" -eq 1 ]; then
    echo "shared permissions OK (top level only -- see header; contents not inspected)"
else
    echo "shared permissions OK (top level only, as seen over NFS -- see NOTEs above; contents not inspected)"
fi
