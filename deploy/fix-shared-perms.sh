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
command -v setfacl >/dev/null || { echo "ERROR: setfacl not found (package acl) — required, see comment below" >&2; exit 1; }

echo "== $NFS_ROOT itself -> 2775 (group writable, other read-only)"
# check-shared-perms.sh fails the deploy if $NFS_ROOT itself is writable by
# "other" and tells the operator to run this script. Make that true.
chmod 2775 "$NFS_ROOT"

echo "== shared assets -> 2775 (group writable, other read-only) -- top level only,"
echo "   contents underneath are NOT touched (see check-shared-perms.sh header)"
for d in $SHARED_DIRS; do
    p="$NFS_ROOT/$d"
    [ -d "$p" ] || continue
    [ -L "$p" ] && continue
    chmod 2775 "$p"
    echo "  $p"
done

echo "== per-user areas -> 2770 owner:$ADMIN_GROUP + $ADMIN_GROUP ACL"
# WHY both chown/chmod AND an ACL: setgid IS inherited normally on this NFS
# export -- a 2770 <user>:$ADMIN_GROUP parent produces 2770 children, and
# files written inside them pick up group $ADMIN_GROUP, the same as on any
# local filesystem. (An earlier note here claimed setgid inheritance was
# broken on this share; that was a measurement artifact of this host's
# `mkdir` binary -- uutils coreutils 0.8.0 -- mishandling ACL-bearing parent
# directories. Checked against the raw syscall, `python3 -c "import os;
# os.mkdir(...)"`, setgid inherits fine. See iitgpu/jobs.py make_job_folder
# for the live comparison. On this host, verify mode-bit behaviour with a
# raw syscall, not a coreutils binary.) So `chown ... :$ADMIN_GROUP` +
# `chmod 2770` does grant $ADMIN_GROUP real access via ordinary POSIX group
# bits.
#
# The ACL below is a second, independent layer: `stat`/mode bits can still
# mislead a reader once a directory carries an extended ACL -- verified
# live, a directory reporting `group=gpuadmins mode=660` had ACL entry
# `group::---` (the "group" column `stat` shows in that case is the ACL
# mask, not the literal group:: permissions). The explicit ACL is what
# deploy/check-shared-perms.sh actually verifies, so it catches that class
# of drift even when the mode bits alone look fine. The default (d:) entry
# is what makes the ACL survive future writes into the area. Keep both
# layers -- don't drop the ACL step even though setgid also grants access.
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
        # rwX (capital X): grant admins rwx on directories but not blindly +x
        # on plain data files underneath. -R + d: (default) so both existing
        # content and anything written later are covered. setfacl is
        # idempotent -- safe to re-run every deploy.
        setfacl -R -m "g:$ADMIN_GROUP:rwX" -m "d:g:$ADMIN_GROUP:rwx" "$p"
    done
done

echo "done"
