# tests/test_adduser_wrapper.py
"""Tests for the interactive addUser.sh wrapper."""
import subprocess
from pathlib import Path

REPO = Path(__file__).parent.parent
WRAPPER = REPO / "addUser.sh"


def test_wrapper_exists_and_executable():
    assert WRAPPER.exists(), "addUser.sh missing at repo root"
    assert WRAPPER.stat().st_mode & 0o111, "addUser.sh not executable"


def test_wrapper_passes_bash_syntax():
    r = subprocess.run(["bash", "-n", str(WRAPPER)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_wrapper_delegates_to_slurm_deck_adduser():
    text = WRAPPER.read_text()
    assert "slurm-deck-adduser" in text, "wrapper must call the real provisioning script"


def test_wrapper_rejects_invalid_username_then_cancels_on_eof():
    # Feed one invalid username; EOF should cancel without spinning forever.
    r = subprocess.run(["bash", str(WRAPPER)], input="BAD NAME\n",
                       capture_output=True, text=True, timeout=10,
                       env={"PATH": "/usr/bin:/bin", "SD_SITE_ENV": "/dev/null"})
    out = r.stdout + r.stderr
    assert "Invalid" in out
    assert "cancelled" in out.lower()
    assert r.returncode != 0


ADDUSER = REPO / "deploy" / "slurm-deck-adduser.sh"


def test_adduser_script_passes_bash_syntax():
    r = subprocess.run(["bash", "-n", str(ADDUSER)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_adduser_creates_shared_symlink_on_both_nodes():
    """~/shared symlink must be created on BOTH login node and GPU host.
    ln is not NOPASSWD on the GPU host, so the script uses the chown trick:
    temporarily chown the home to the provisioning user, ln, then restore."""
    text = ADDUSER.read_text()
    assert "ln -sfn $_SHARED_LINK /home/$USERNAME/shared" in text, \
        "login-node symlink missing"
    assert "chown $GPU_HOST_USER /home/$USERNAME" in text, \
        "GPU-host chown trick missing — needed to create symlink without sudo ln"


def test_adduser_shell_user_gets_nfs_root_symlink():
    """Shell users' ~/shared must point to NFS_ROOT so they can reach datasets,
    models, envs AND their private ~/shared/users/<user> in one place.
    Regular users keep the narrow link to their private workspace only."""
    text = ADDUSER.read_text()
    assert '_SHARED_LINK="$NFS_ROOT"' in text, \
        "shell-user branch missing: ~/shared should point to NFS root"
    assert '_SHARED_LINK="$NFS_ROOT/users/$USERNAME"' in text, \
        "regular-user branch missing: ~/shared should point to private workspace"


def test_adduser_every_account_type_gets_docker_group_on_both_nodes():
    """Both the login node and the GPU host run a Docker daemon (confirmed
    live), and JupyterLab/notebook jobs run on the GPU host -- so every
    account type (regular, --admin, --shell-user) needs docker access on
    BOTH nodes, not just shell-users on the login node. Granted via group
    membership, matching how gpuusers get GPU access, rather than sudo."""
    text = ADDUSER.read_text()
    assert 'DOCKER_GROUP="${DOCKER_GROUP:-docker}"' in text
    login_node_block = text.split("Creating $USERNAME on GPU host")[0]
    gpu_host_block = text.split("Creating $USERNAME on GPU host")[1].split("SLURM association")[0]
    assert "usermod -aG $DOCKER_GROUP $USERNAME" in login_node_block, \
        "docker group must be granted in the login-node provisioning block"
    assert "usermod -aG $DOCKER_GROUP $USERNAME" in gpu_host_block, \
        "docker group must also be granted in the GPU-host provisioning block"
    # Not gated behind the shell-user check on either node: the docker grant
    # line must sit after (not inside) the SHELL_USER=0 conditional.
    assert login_node_block.index('usermod -aG $GPUUSERS_GROUP $USERNAME') < \
           login_node_block.index('usermod -aG $DOCKER_GROUP $USERNAME')
    assert gpu_host_block.index('usermod -aG $GPUUSERS_GROUP $USERNAME') < \
           gpu_host_block.index('usermod -aG $DOCKER_GROUP $USERNAME')


def test_adduser_enforces_home_ownership_both_nodes():
    """A stale home owned by a prior UID breaks `conda activate` (unreadable
    ~/.config/conda/.condarc). adduser must chown the home to the account UID
    on both the login node and the GPU host, and verify it."""
    text = ADDUSER.read_text()
    # login node (no sudo) + GPU host (sudo over ssh)
    assert "chown -R $NEW_UID:$NEW_UID /home/$USERNAME" in text
    assert "sudo chown -R $NEW_UID:$NEW_UID /home/$USERNAME" in text
    # verify step asserts ownership matches
    assert "expected $luid" in text and "expected $ruid" in text


def test_adduser_creates_user_area_owner_and_admin_only():
    """New user areas must be 2770 group admin, not 0700 owner-only.

    0700 locks admins out; anything looser exposes the area to other users.
    """
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / "deploy" / "slurm-deck-adduser.sh"
    text = script.read_text()
    assert "chmod 2770" in text, "user area must be mode 2770"
    assert "chmod 0700" not in text, "0700 would lock admins out of user areas"
    assert "$ADMIN_GROUP" in text, "group must come from ADMIN_GROUP, not hardcoded"
