#!/usr/bin/env python3
"""Slurm Deck installation wizard.

Run from a fresh (or existing) GPU host as root:

    sudo python3 deploy/wizard/main.py

or via the convenience entry point at the repo root:

    sudo bash install-wizard.sh

Walks through: optionally creating the login-node VM, site configuration,
GPU-node setup, login-node setup (over SSH), cross-node sync, Resend email
setup, first-admin provisioning, and final validation. Safe to re-run --
already-completed steps are skipped and earlier answers are reused, see
state.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import admin_provision
import cross_sync
import gpu_node
import login_node
import mail
import validate
import vm
from prompts import choice, fail, ok, step, text
from ssh_target import SSHTarget
from state import State

DEFAULT_SSH_KEY = "/root/.ssh/slurm-deck-wizard"

SITE_DEFAULTS = {
    "NFS_ROOT": "/shared",
    "SLURM_ACCOUNT": "default",
    "SLURM_QOS": "normal",
    "SLURM_PARTITION": "gpu",
    "GPUUSERS_GROUP": "gpuusers",
    "ADMIN_GROUP": "gpuadmins",
    "GATEWAY_PORT": "2225",
    "CLUSTER_NAME": "GPU Cluster",
    "CLUSTER_TZ_OFFSET": "+00:00",
}


def main() -> None:
    if os.geteuid() != 0:
        fail("run as root: sudo python3 deploy/wizard/main.py")
        sys.exit(1)

    print("=" * 60)
    print("  Slurm Deck Installation Wizard")
    print("=" * 60)

    state = State()

    mode = state.get_answer("mode")
    if mode is None:
        idx = choice(
            "How would you like to set this up?",
            [
                "Fresh GPU host — create the login-node VM for me",
                "I already have two reachable machines",
            ],
        )
        mode = "vm" if idx == 0 else "existing"
        state.set_answer("mode", mode)

    site = collect_site_config(state)

    if mode == "vm":
        target = run_vm_path(state, site)
    else:
        target = run_existing_path(state, site)

    gpu_node.run(state, site)
    login_node.run(state, site, target)
    cross_sync.run(state, site)
    mail.run(state, site, target)
    admin_provision.run(state, site, target)
    validate.run(state, site, target)

    ok("Wizard complete.")


def collect_site_config(state: State) -> dict:
    step("Site configuration")
    site = state.get_answer("site") or {}
    for key, default in SITE_DEFAULTS.items():
        if key not in site:
            site[key] = text(key, default=default)
    if "GATEWAY_HOST" not in site:
        site["GATEWAY_HOST"] = text(
            "Gateway host (public IP/hostname users SSH to)")
    if "CLUSTER_LOCATION" not in site:
        site["CLUSTER_LOCATION"] = text("Cluster location label (optional)", default="", allow_empty=True)
    state.set_answer("site", site)
    return site


def run_vm_path(state: State, site: dict) -> SSHTarget:
    if not vm.check_prerequisites():
        fail("install the missing libvirt tooling, then re-run")
        sys.exit(1)

    vm.gen_ssh_keypair(DEFAULT_SSH_KEY)

    if not state.is_done("vm_created"):
        name = text("Login-node VM name", default="login-node")
        vcpus = int(text("vCPUs", default="4"))
        mem_mb = int(text("Memory (MB)", default="8192"))
        disk_gb = int(text("Disk (GB)", default="60"))
        vm.create_login_vm(name, vcpus, mem_mb, disk_gb, DEFAULT_SSH_KEY + ".pub")
        state.set_answer("vm_name", name)
        state.mark_done("vm_created")

    vm_name = state.get_answer("vm_name")
    ip = state.get_answer("vm_ip")
    if not ip:
        ip = vm.wait_for_ip(vm_name)
        state.set_answer("vm_ip", ip)

    vm.wait_for_ssh(ip, DEFAULT_SSH_KEY)

    if not state.is_done("port_forward"):
        vm.setup_port_forward(int(site["GATEWAY_PORT"]), ip)
        state.mark_done("port_forward")

    return SSHTarget(ip, "slurmadmin", DEFAULT_SSH_KEY)


def run_existing_path(state: State, site: dict) -> SSHTarget:
    login_host = state.get_answer("login_host")
    if not login_host:
        login_host = text("Login node hostname or IP")
        state.set_answer("login_host", login_host)
    login_user = state.get_answer("login_user") or text(
        "SSH username on the login node (must have sudo)", default="root")
    key_path = state.get_answer("login_ssh_key") or text(
        "Path to an SSH private key for that account",
        default=os.path.expanduser("~/.ssh/id_ed25519"))
    state.set_answer("login_user", login_user)
    state.set_answer("login_ssh_key", key_path)
    return SSHTarget(login_host, login_user, key_path)


if __name__ == "__main__":
    main()
