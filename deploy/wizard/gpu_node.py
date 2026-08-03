"""GPU-node setup stage: compute toolchain, groups, NFS export for /shared.

Always runs locally -- the wizard is invoked on the GPU host itself.
"""
import os
import subprocess

from prompts import step, ok, warn, fail, confirm_risky
from state import State


def _env() -> dict:
    return dict(os.environ)


def run(state: State, site: dict) -> None:
    if state.is_done("gpu_node"):
        ok("GPU-node setup already done — skipping")
        return

    step("GPU-node setup")

    if confirm_risky("install build-essential and verify the CUDA/torch toolchain "
                      "(deploy/setup-compute-toolchain.sh)"):
        r = subprocess.run(["sudo", "bash", "deploy/setup-compute-toolchain.sh"])
        if r.returncode != 0:
            fail("setup-compute-toolchain.sh failed — see output above")
            raise SystemExit(1)
        ok("compute toolchain OK")
    else:
        warn("skipped compute toolchain setup")

    gpuusers = site["GPUUSERS_GROUP"]
    gpuadmins = site["ADMIN_GROUP"]
    if confirm_risky(f"create Linux groups '{gpuusers}' and '{gpuadmins}' if missing"):
        for grp in (gpuusers, gpuadmins):
            if subprocess.run(["getent", "group", grp], capture_output=True).returncode == 0:
                ok(f"group {grp} already exists")
            else:
                subprocess.run(["sudo", "groupadd", "--system", grp], check=True)
                ok(f"group {grp} created")
    else:
        warn(f"skipped group creation — create {gpuusers}/{gpuadmins} manually")

    nfs_root = site["NFS_ROOT"]
    export_cidr = site.get("NFS_EXPORT_CIDR", "192.168.122.0/24")
    if confirm_risky(f"export {nfs_root} over NFS to {export_cidr} (/etc/exports)"):
        _setup_nfs_export(nfs_root, export_cidr)
    else:
        warn(f"skipped NFS export — configure {nfs_root} manually in /etc/exports")

    state.mark_done("gpu_node")
    ok("GPU-node setup complete")


def _setup_nfs_export(nfs_root: str, export_cidr: str) -> None:
    subprocess.run(["sudo", "mkdir", "-p", nfs_root], check=True)
    existing = subprocess.run(["sudo", "cat", "/etc/exports"],
                               capture_output=True, text=True)
    if nfs_root in existing.stdout:
        ok(f"{nfs_root} already exported")
        return
    export_line = f"{nfs_root} {export_cidr}(rw,sync,no_subtree_check,root_squash)\n"
    subprocess.run(["sudo", "tee", "-a", "/etc/exports"], input=export_line,
                    text=True, check=True, capture_output=True)
    subprocess.run(["sudo", "apt-get", "install", "-y", "nfs-kernel-server"], check=True)
    subprocess.run(["sudo", "exportfs", "-ra"], check=True)
    subprocess.run(["sudo", "systemctl", "enable", "--now", "nfs-kernel-server"], check=True)
    ok(f"{nfs_root} exported via NFS to {export_cidr}")
