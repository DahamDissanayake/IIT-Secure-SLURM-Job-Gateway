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
# WHY ACLs, not just chown/chmod: on this NFS export, POSIX group permissions
# do NOT govern access -- default ACLs do, and stat(1)/mode bits actively
# conceal that. Verified live: a directory reporting `group=gpuadmins
# mode=660` had ACL entry `group::---` -- the owning GROUP had NO access,
# only users named explicitly in the ACL could get in. setgid is also NOT
# inherited on this share (same mkdir, same non-admin user: on /tmp a 2770
# parent yields a 2770 child; on /shared it yields 5770/1770, no setgid), so
# there is no umask/chmod trick that reaches the admin group here either.
# `chown ... :$ADMIN_GROUP` + `chmod 2770` alone therefore produces exactly
# the state that LOOKS correct under `stat` while admins are actually locked
# out. The explicit ACL below is what actually grants access, and the
# default (d:) entry is what makes it survive future writes into the area.
# Do NOT "simplify" this back to chown/chmod only -- that regression is the
# whole reason this comment exists.
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
