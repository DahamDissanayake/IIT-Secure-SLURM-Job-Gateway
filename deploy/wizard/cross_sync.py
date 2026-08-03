"""Cross-node consistency stage: admin group membership + /shared permission
model. Both scripts must run on the GPU host (the NFS server) per their own
headers -- root_squash on the NFS export means the login node cannot repair
permissions there.
"""
import os
import subprocess

from prompts import step, ok, warn, fail, confirm_risky
from state import State


def run(state: State, site: dict) -> None:
    if state.is_done("cross_sync"):
        ok("Cross-node sync already done — skipping")
        return

    step("Cross-node consistency")
    env = {**os.environ, "ADMIN_GROUP": site["ADMIN_GROUP"]}

    if confirm_risky("sync gpuadmins group membership across nodes "
                      "(deploy/sync-admin-group.sh)"):
        r = subprocess.run(["sudo", "-E", "bash", "deploy/sync-admin-group.sh"], env=env)
        if r.returncode != 0:
            fail("sync-admin-group.sh failed")
            raise SystemExit(1)
        ok("admin group synced")
    else:
        warn("skipped admin group sync")

    env["NFS_ROOT"] = site["NFS_ROOT"]
    if confirm_risky(f"apply the {site['NFS_ROOT']} access model "
                      f"(deploy/fix-shared-perms.sh)"):
        r = subprocess.run(["sudo", "-E", "bash", "deploy/fix-shared-perms.sh"], env=env)
        if r.returncode != 0:
            fail("fix-shared-perms.sh failed")
            raise SystemExit(1)
        ok("shared permissions applied")
    else:
        warn("skipped shared-perms fix")

    state.mark_done("cross_sync")
    ok("Cross-node sync complete")
