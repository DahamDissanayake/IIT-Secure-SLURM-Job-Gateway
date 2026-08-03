"""Login-node setup stage: canonical clone, site.env, install.sh.

Runs against a Target (SSHTarget for a VM or an existing machine) -- see
ssh_target.py. This stage never branches on VM-vs-existing; main.py already
resolved that into a uniform Target before calling here.
"""
import os
import tempfile

from prompts import step, ok, warn, fail, confirm_risky
from state import State
from ssh_target import Target

REPO_URL = "https://github.com/DahamDissanayake/slurm-deck.git"
INSTALL_DIR = "/opt/slurm-deck"


def render_site_env(site: dict) -> str:
    """Render the site dict as deploy/site.env KEY=VALUE lines. Pure
    function -- no I/O -- so it's directly unit-testable."""
    lines = ["# Written by the slurm-deck installation wizard\n"]
    for key, value in site.items():
        text = str(value)
        if " " in text or text == "":
            lines.append(f'{key}="{text}"\n')
        else:
            lines.append(f"{key}={text}\n")
    return "".join(lines)


def run(state: State, site: dict, target: Target) -> None:
    if state.is_done("login_node"):
        ok("Login-node setup already done — skipping")
        return

    step("Login-node setup")

    if confirm_risky(f"clone {REPO_URL} to {INSTALL_DIR} on the login node"):
        already = target.run(["test", "-d", f"{INSTALL_DIR}/.git"], check=False)
        if already.returncode == 0:
            ok(f"{INSTALL_DIR} already a git clone — skipping clone")
        else:
            target.run(["sudo", "git", "clone", REPO_URL, INSTALL_DIR])
            target.run(["sudo", "chown", "-R",
                        f"slurmadmin:{site['GPUUSERS_GROUP']}", INSTALL_DIR])
            ok(f"cloned to {INSTALL_DIR}")
    else:
        fail("cannot continue without the canonical clone on the login node")
        raise SystemExit(1)

    step("Writing site.env")
    _write_remote_file(target, render_site_env(site), f"{INSTALL_DIR}/deploy/site.env")
    ok("site.env written")

    if confirm_risky("run deploy/install.sh on the login node as root"):
        conda_prefix = site.get("CONDA_PREFIX_SHARED", f"{site['NFS_ROOT']}/miniforge3")
        gateway_user = site.get("GATEWAY_SHARED_USER_NAME", "public")
        env_exports = (
            f"NFS_ROOT={site['NFS_ROOT']} "
            f"CONDA_PREFIX_SHARED={conda_prefix} "
            f"GATEWAY_USER={gateway_user}"
        )
        r = target.run(
            ["sudo", "bash", "-c",
             f"cd {INSTALL_DIR} && {env_exports} bash deploy/install.sh"],
            check=False,
        )
        if r.returncode != 0:
            fail("install.sh failed:\n" + (r.stderr or r.stdout))
            raise SystemExit(1)
        ok("install.sh completed")
    else:
        warn("skipped install.sh — run it manually on the login node")

    state.mark_done("login_node")
    ok("Login-node setup complete")


def _write_remote_file(target: Target, content: str, remote_path: str) -> None:
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".env") as f:
        f.write(content)
        local_tmp = f.name
    try:
        target.copy_to(local_tmp, "/tmp/slurm-deck-site.env")
        target.run(["sudo", "install", "-m", "0644",
                    "/tmp/slurm-deck-site.env", remote_path])
        target.run(["rm", "-f", "/tmp/slurm-deck-site.env"])
    finally:
        os.unlink(local_tmp)
