#!/usr/bin/env bash
# Read-only audit of /shared access. Safe to run from any node — root_squash
# blocks writes from the login node but stat() works fine.
#
# IMPORTANT — mode bits are not sufficient evidence on this filesystem. A
# per-user area is provisioned 2770 owner:gpuadmins, and setgid IS inherited
# normally here, so `chown user:gpuadmins` + `chmod 2770` does grant the
# admin group real access via ordinary POSIX group bits. (An earlier note
# here claimed setgid inheritance was broken on this share; that was a
# measurement artifact of this host's `mkdir` binary -- uutils coreutils
# 0.8.0 -- mishandling ACL-bearing parent directories. Checked against the
# raw syscall, `python3 -c "import os; os.mkdir(...)"`, setgid inherits
# fine. See slurmdeck/jobs.py make_job_folder for the live comparison. On this
# host, verify mode-bit behaviour with a raw syscall, not a coreutils
# binary.)
#
# Even so, `stat`/mode bits can mislead: once a directory carries an
# extended ACL, the "group" column `stat`/`ls -l` show is the ACL mask, not
# the literal group:: permissions. Verified live: a per-user area reporting
# `group=gpuadmins mode=660` had ACL entry `group::---`. A check that only
# reads mode bits therefore cannot see this class of drift -- it must also
# confirm the `gpuadmins` ACL entry is actually present (see the
# per-user-area loop below), as a second, independent layer on top of the
# ordinary group permissions. Treat mode bits as necessary but NOT
# sufficient.
#
# ACL verification only runs when this script executes on the GPU host (the
# NFS server, where /mnt/nvme_storage/shared is a real local mount). Over
# the NFS(v4) client mount on the login node, `getfacl` does not error and
# does not just return stale data -- it silently fabricates a trivial ACL
# from the mode bits with no named entries at all, indistinguishable from a
# real (but empty) ACL. So on a non-authoritative host this script skips the
# ACL check entirely rather than report a result derived from fabricated
# data; see the AUTHORITATIVE branch below for the exact reasoning.
#
# Rule: NFS_ROOT itself grants no write to "other".
#       Per-user areas (users/, jobs/) grant nothing to "other" -- checked at
#       their own top level (each users/<u>, jobs/<u> directory itself) --
#       AND, when run on the GPU host, must carry an explicit access +
#       default ACL entry for $ADMIN_GROUP as a second, independent layer
#       on top of mode/group -- mode bits alone can look correct while an
#       ACL entry disagrees (see IMPORTANT above).
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

# ACL verification (see header) can ONLY run where AUTHORITATIVE=1. This is
# not the same ~60s staleness caveat as above -- it's worse. Verified live:
# over this NFS(v4) mount, `getfacl` on the login node does not error and
# does not return stale data, it silently SYNTHESIZES a fake trivial ACL
# from the mode bits (no `group:$ADMIN_GROUP:...` entry ever appears, even
# when the real ACL on the server grants it) -- there is no `+` marker or
# any other signal that the output is fabricated. A getfacl-based check run
# from the login node would therefore either always "detect" ACLs as
# missing (false FAIL, permanently blocking deploys) or, if matched
# carelessly, could look clean while genuinely wrong (false PASS) -- neither
# outcome is trustworthy, so on a non-authoritative host we do not attempt
# it at all rather than report a result nobody should trust.
HAVE_GETFACL=0
if [ "$AUTHORITATIVE" -eq 1 ]; then
    if command -v getfacl >/dev/null 2>&1; then
        HAVE_GETFACL=1
    else
        echo "NOACL     getfacl not found (package acl) -- cannot verify the $ADMIN_GROUP ACL entry on the GPU host, mode bits alone are NOT sufficient evidence on this filesystem"
        fail=1
    fi
else
    echo "NOTE: ACL verification skipped on this host -- getfacl over this NFS mount"
    echo "      does not reflect the real ACL (see script header). Run this checker"
    echo "      on the GPU host (the NFS server) for an authoritative ACL result."
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
        # Mode/group above grant $ADMIN_GROUP real access via ordinary POSIX
        # group bits, but the ACL is a second, independent layer and mode
        # bits alone are not sufficient evidence of it (see header). Confirm
        # the access AND default ACL entries for $ADMIN_GROUP are actually
        # present.
        if [ "$HAVE_GETFACL" -eq 1 ]; then
            facl=$(getfacl -p "$d" 2>/dev/null) || facl=""
            if ! printf '%s\n' "$facl" | grep -q "^group:${ADMIN_GROUP}:rwx"; then
                echo "NOACL     $d  missing ACL entry group:${ADMIN_GROUP}:rwx (second layer on top of mode/group; mode bits alone can look correct while this ACL entry disagrees)"
                fail=1
            fi
            if ! printf '%s\n' "$facl" | grep -q "^default:group:${ADMIN_GROUP}:rwx"; then
                echo "NOACL     $d  missing default ACL entry default:group:${ADMIN_GROUP}:rwx (new files/subdirs won't inherit $ADMIN_GROUP access)"
                fail=1
            fi
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
    echo "  ssh slurmadmin@192.168.122.10 'cat /opt/slurm-deck/deploy/fix-shared-perms.sh' > /tmp/fix-shared-perms.sh"
    echo "  sudo bash /tmp/fix-shared-perms.sh"
    exit 1
fi

if [ "$AUTHORITATIVE" -eq 1 ]; then
    echo "shared permissions OK (top level only -- see header; contents not inspected)"
else
    echo "shared permissions OK (top level only, as seen over NFS -- see NOTEs above; contents not inspected)"
fi
