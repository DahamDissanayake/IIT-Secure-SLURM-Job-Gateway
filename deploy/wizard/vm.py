"""Login-node VM creation on the "fresh GPU host" wizard path.

Uses virt-install + cloud-init (NoCloud datasource) for an unattended Ubuntu
Server install on the libvirt 'default' NAT network, then a DNAT iptables
rule to port-forward an external port to the VM's SSH port. Skipped
entirely on the "I already have two machines" path -- see main.py.
"""
import os
import subprocess
import time

from prompts import step, ok, warn, fail, confirm_risky

CLOUD_IMAGE_URL = (
    "https://cloud-images.ubuntu.com/releases/22.04/release/"
    "ubuntu-22.04-server-cloudimg-amd64.img"
)
IMAGE_DIR = "/var/lib/libvirt/images"
REQUIRED_TOOLS = ("virsh", "virt-install", "cloud-localds", "qemu-img")


def check_prerequisites() -> bool:
    step("Checking libvirt/KVM prerequisites")
    missing = [t for t in REQUIRED_TOOLS if subprocess.run(
        ["which", t], capture_output=True).returncode != 0]
    if missing:
        fail(f"missing: {', '.join(missing)} — install with: "
             f"apt-get install -y qemu-kvm libvirt-daemon-system virtinst cloud-image-utils")
        return False
    r = subprocess.run(["virsh", "net-list", "--all"], capture_output=True, text=True)
    if "default" not in r.stdout:
        warn("libvirt 'default' network not found — will try to use it anyway; "
             "if VM creation fails, run: virsh net-define /etc/libvirt/qemu/networks/default.xml "
             "&& virsh net-start default && virsh net-autostart default")
    ok("libvirt tooling present")
    return True


def gen_ssh_keypair(key_path: str) -> None:
    if os.path.exists(key_path):
        ok(f"reusing existing wizard SSH key at {key_path}")
        return
    os.makedirs(os.path.dirname(key_path), exist_ok=True)
    subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-N", "", "-f", key_path,
         "-C", "slurm-deck-wizard"],
        check=True, capture_output=True,
    )
    ok(f"generated a new SSH keypair at {key_path}")


def create_login_vm(name: str, vcpus: int, mem_mb: int, disk_gb: int,
                     ssh_pubkey_path: str) -> None:
    step(f"Creating login-node VM '{name}' ({vcpus} vCPU, {mem_mb}MB RAM, {disk_gb}GB disk)")
    os.makedirs(IMAGE_DIR, exist_ok=True)
    image_path = f"{IMAGE_DIR}/{name}.img"
    seed_path = f"{IMAGE_DIR}/{name}-seed.iso"
    base_image = f"{IMAGE_DIR}/ubuntu-22.04-base.img"

    if not os.path.exists(base_image):
        step("Downloading Ubuntu 22.04 cloud image (one-time)")
        subprocess.run(["wget", "-q", "-O", base_image, CLOUD_IMAGE_URL], check=True)
        ok("base image downloaded")
    else:
        ok("base cloud image already present")

    subprocess.run(
        ["qemu-img", "create", "-f", "qcow2", "-F", "qcow2",
         "-b", base_image, image_path, f"{disk_gb}G"],
        check=True, capture_output=True,
    )

    pubkey = open(ssh_pubkey_path).read().strip()
    user_data = (
        "#cloud-config\n"
        "hostname: login-node\n"
        "users:\n"
        "  - name: slurmadmin\n"
        "    sudo: ALL=(ALL) NOPASSWD:ALL\n"
        "    shell: /bin/bash\n"
        "    ssh_authorized_keys:\n"
        f"      - {pubkey}\n"
        "package_update: true\n"
    )
    meta_data = "instance-id: slurm-deck-login\nlocal-hostname: login-node\n"
    user_data_path = f"{IMAGE_DIR}/{name}-user-data"
    meta_data_path = f"{IMAGE_DIR}/{name}-meta-data"
    with open(user_data_path, "w") as f:
        f.write(user_data)
    with open(meta_data_path, "w") as f:
        f.write(meta_data)
    subprocess.run(["cloud-localds", seed_path, user_data_path, meta_data_path], check=True)

    subprocess.run([
        "virt-install",
        "--name", name,
        "--memory", str(mem_mb),
        "--vcpus", str(vcpus),
        "--disk", f"path={image_path},format=qcow2",
        "--disk", f"path={seed_path},device=cdrom",
        "--os-variant", "ubuntu22.04",
        "--network", "network=default,model=virtio",
        "--graphics", "none",
        "--noautoconsole",
        "--import",
    ], check=True)
    ok(f"VM '{name}' created and booting")


def wait_for_ip(name: str, timeout_s: int = 180) -> str:
    step(f"Waiting for '{name}' to get an IP address (up to {timeout_s}s)")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = subprocess.run(["virsh", "domifaddr", name], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 4 and "/" in parts[3]:
                ip = parts[3].split("/")[0]
                ok(f"VM IP: {ip}")
                return ip
        time.sleep(3)
    fail(f"VM '{name}' did not get an IP within {timeout_s}s")
    raise TimeoutError(f"no IP for VM {name}")


def wait_for_ssh(ip: str, ssh_key_path: str, user: str = "slurmadmin",
                  timeout_s: int = 120) -> None:
    step(f"Waiting for SSH on {ip} (up to {timeout_s}s)")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = subprocess.run([
            "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=5", "-i", ssh_key_path, f"{user}@{ip}", "true",
        ], capture_output=True)
        if r.returncode == 0:
            ok(f"SSH reachable at {user}@{ip}")
            return
        time.sleep(3)
    fail(f"SSH not reachable at {user}@{ip} within {timeout_s}s")
    raise TimeoutError(f"ssh not reachable: {ip}")


def setup_port_forward(external_port: int, vm_ip: str, vm_ssh_port: int = 22) -> None:
    """DNAT external_port -> vm_ip:vm_ssh_port so gateway SSH access reaches
    the VM from outside. Persisted with netfilter-persistent if available."""
    step(f"Setting up port-forward {external_port} -> {vm_ip}:{vm_ssh_port}")
    if not confirm_risky(
            f"add an iptables DNAT rule: *:{external_port} -> {vm_ip}:{vm_ssh_port}"):
        warn("skipped — configure port-forwarding manually")
        return
    subprocess.run([
        "iptables", "-t", "nat", "-A", "PREROUTING", "-p", "tcp",
        "--dport", str(external_port), "-j", "DNAT",
        "--to-destination", f"{vm_ip}:{vm_ssh_port}",
    ], check=True)
    subprocess.run([
        "iptables", "-A", "FORWARD", "-p", "tcp", "-d", vm_ip,
        "--dport", str(vm_ssh_port), "-j", "ACCEPT",
    ], check=True)
    ok("DNAT rule added")
    if subprocess.run(["which", "netfilter-persistent"], capture_output=True).returncode == 0:
        subprocess.run(["netfilter-persistent", "save"], check=False)
        ok("rules persisted (netfilter-persistent)")
    else:
        warn("netfilter-persistent not found — this rule will NOT survive a reboot; "
             "install iptables-persistent, or persist it yourself")
