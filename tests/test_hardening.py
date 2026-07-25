# tests/test_hardening.py
"""Tests for Phase 7 hardening — job dir permissions, UPGRADE-RUNBOOK.md."""
import grp
import stat
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).parent.parent


# ── Job directory permissions ─────────────────────────────────────────────────

def test_make_job_folder_uses_0o2770(tmp_path):
    """This test constructs a setgid (2770) parent under tmp_path -- a plain
    ext4 directory with no ACLs, the kind kernel setgid-inheritance actually
    works on (e.g. /tmp) -- and asserts the job folder mkdir'd inside it
    comes out 2770 too: setgid inherited by the kernel, not from a chmod
    inside make_job_folder (chmod cannot grant setgid to a non-admin caller;
    see the comment on make_job_folder).

    Production's /shared behaves the same way: jobs/<user> is provisioned
    2770 owner:gpuadmins by iit-gpu-adduser.sh, and a folder mkdir'd inside
    it (via the raw os.mkdir syscall, which is what Path.mkdir uses) comes
    out 2770 too, with files written inside it inheriting group gpuadmins.
    (An earlier version of this docstring claimed production /shared does
    NOT inherit setgid; that was a measurement artifact of this host's
    `mkdir` binary -- uutils coreutils 0.8.0 -- mishandling ACL-bearing
    parent directories. Re-checked against the raw syscall, setgid inherits
    fine. On this host, verify mode-bit behaviour with `python3 -c "import
    os; os.mkdir(...)"`, not with coreutils binaries.) The gpuadmins ACL
    applied by deploy/fix-shared-perms.sh and deploy/iit-gpu-adduser.sh is a
    second, independent layer on top of this, and is what
    deploy/check-shared-perms.sh verifies -- useful because mode bits alone
    can look correct while an ACL entry disagrees.

    This test pins make_job_folder's own behaviour -- given a setgid
    parent, it does not chmod the setgid bit away -- which also matches
    what production job folders actually end up with."""
    from iitgpu.jobs import JobSpec, make_job_folder

    user_dir = tmp_path / "sec_test"
    user_dir.mkdir()
    user_dir.chmod(0o2770)  # mimics jobs/<user> as provisioned in production

    spec = JobSpec(
        job_name="sec_test",
        partition="gpu",
        gpu_shards=1,
        cpus=4,
        mem_gb=8,
        time_limit="00:30:00",
        run_command="echo hi",
        user="sec_test",
    )
    folder = make_job_folder(str(tmp_path), spec)
    st = Path(folder).stat()
    mode = stat.S_IMODE(st.st_mode)
    # Must be 0o2770 (rwxrwx--- + setgid): no world permissions, group inherited
    assert mode == 0o2770, (
        f"Job folder has mode {oct(mode)} — expected 0o2770 (rwxrwx--- + setgid). "
        "Other users should not be able to read each other's job output."
    )


def test_make_job_folder_no_world_readable(tmp_path):
    from iitgpu.jobs import JobSpec, make_job_folder

    spec = JobSpec(
        job_name="sec_test2",
        partition="gpu",
        gpu_shards=1,
        cpus=4,
        mem_gb=8,
        time_limit="",
        run_command="echo hi",
    )
    folder = make_job_folder(str(tmp_path), spec)
    mode = stat.S_IMODE(Path(folder).stat().st_mode)
    # World bits (r=4, w=2, x=1) must all be 0
    world_bits = mode & 0o007
    assert world_bits == 0, (
        f"Job folder has world bits {oct(world_bits)} set — users can read each other's jobs."
    )


# ── UPGRADE-RUNBOOK.md ────────────────────────────────────────────────────────

def test_upgrade_runbook_exists():
    runbook = REPO_ROOT / "deploy" / "UPGRADE-RUNBOOK.md"
    assert runbook.exists(), "deploy/UPGRADE-RUNBOOK.md is missing"


def test_upgrade_runbook_has_all_phases():
    runbook = REPO_ROOT / "deploy" / "UPGRADE-RUNBOOK.md"
    if not runbook.exists():
        pytest.skip("runbook missing")
    content = runbook.read_text()
    for phase in ["Phase 1", "Phase 2", "Phase 3", "Phase 4", "Phase 5", "Phase 6", "Phase 7"]:
        assert phase in content, f"UPGRADE-RUNBOOK.md is missing section for {phase}"


def test_upgrade_runbook_has_gpu_host_markers():
    runbook = REPO_ROOT / "deploy" / "UPGRADE-RUNBOOK.md"
    if not runbook.exists():
        pytest.skip("runbook missing")
    content = runbook.read_text()
    assert "[GPU-HOST]" in content, "UPGRADE-RUNBOOK.md has no [GPU-HOST] markers"


def test_upgrade_runbook_has_final_checklist():
    runbook = REPO_ROOT / "deploy" / "UPGRADE-RUNBOOK.md"
    if not runbook.exists():
        pytest.skip("runbook missing")
    content = runbook.read_text()
    assert "Final Checklist" in content


# ── CHANGES.md ────────────────────────────────────────────────────────────────

def test_changes_md_exists():
    assert (REPO_ROOT / "CHANGES.md").exists(), "CHANGES.md is missing"


def test_make_job_folder_chowns_to_gpuusers(tmp_path):
    """make_job_folder must attempt os.chown(..., -1, gpuusers_gid).
    We verify the call is made with the right GID regardless of whether
    the test runner is a member of gpuusers (it usually is not in CI).
    """
    from unittest.mock import patch, MagicMock
    from iitgpu.jobs import JobSpec, make_job_folder

    spec = JobSpec(
        job_name="chown_test",
        partition="gpu",
        gpu_shards=1,
        cpus=4,
        mem_gb=8,
        time_limit="",
        run_command="echo hi",
    )

    fake_gid = 1500
    chown_calls = []

    with patch("iitgpu.jobs.grp") as mock_grp,          patch("iitgpu.jobs.os.chown", side_effect=lambda *a: chown_calls.append(a)):
        mock_grp.getgrnam.return_value = MagicMock(gr_gid=fake_gid)
        folder = make_job_folder(str(tmp_path), spec)

    assert len(chown_calls) == 1, "os.chown should be called exactly once per job folder"
    path_arg, uid_arg, gid_arg = chown_calls[0]
    assert path_arg == folder, "chown must target the job folder itself"
    assert uid_arg == -1, "chown must leave uid unchanged (-1)"
    assert gid_arg == fake_gid, f"chown must set gid to gpuusers({fake_gid})"


def test_make_job_folder_chown_failure_does_not_raise(tmp_path):
    """If os.chown fails (e.g. process not in gpuadmins -- the normal case for
    a regular submitting user), make_job_folder must not raise — it logs
    best-effort and returns the folder path.

    Like test_make_job_folder_uses_0o2770, this test sets up a setgid (2770)
    parent so it can assert that the folder's pre-existing 2770 mode survives
    a failed chown. That setgid-inheriting setup also matches production
    /shared: jobs/<user> is 2770 owner:gpuadmins, and setgid is inherited
    normally there too (checked against the raw os.mkdir syscall -- an
    earlier version of this docstring claimed otherwise, based on a
    measurement using this host's `mkdir` binary, uutils coreutils 0.8.0,
    which mishandles ACL-bearing parent directories). What this test
    actually proves is that make_job_folder never strips or downgrades a
    folder's existing mode just because os.chown failed."""
    from unittest.mock import patch
    from iitgpu.jobs import JobSpec, make_job_folder

    user_dir = tmp_path / "chown_fail"
    user_dir.mkdir()
    user_dir.chmod(0o2770)  # mimics jobs/<user> as provisioned in production

    spec = JobSpec(
        job_name="chown_fail",
        partition="gpu",
        gpu_shards=1,
        cpus=4,
        mem_gb=8,
        time_limit="",
        run_command="echo hi",
        user="chown_fail",
    )

    with patch("iitgpu.jobs.os.chown", side_effect=PermissionError("not in group")):
        folder = make_job_folder(str(tmp_path), spec)

    assert Path(folder).is_dir(), "folder must be created even if chown fails"
    assert stat.S_IMODE(Path(folder).stat().st_mode) == 0o2770, (
        "setgid must survive a failed chown -- it came from mkdir inheriting "
        "the setgid parent, not from chown or any chmod in make_job_folder"
    )


def test_make_job_folder_no_setgid_parent_still_gets_safe_permissions(tmp_path):
    """Without a setgid ancestor (e.g. a plain tmp dir, unlike production's
    jobs/<user>), make_job_folder cannot manufacture setgid out of nothing --
    chmod() cannot grant it to a non-privileged, non-group-member caller. The
    degraded result must still be safe (owner+group only, no world access),
    just without the setgid bit."""
    from iitgpu.jobs import JobSpec, make_job_folder

    spec = JobSpec(
        job_name="no_setgid_parent",
        partition="gpu",
        gpu_shards=1,
        cpus=4,
        mem_gb=8,
        time_limit="",
        run_command="echo hi",
    )
    folder = make_job_folder(str(tmp_path), spec)  # tmp_path is NOT setgid
    mode = stat.S_IMODE(Path(folder).stat().st_mode)
    assert mode == 0o770, (
        f"Job folder has mode {oct(mode)} — expected 0o770 (no setgid parent "
        "to inherit from, and chmod cannot safely add setgid here)."
    )


# ── Gateway ForceCommand wrapper ──────────────────────────────────────────────

GATEWAY_SCRIPT = REPO_ROOT / "deploy" / "iit-gpu-gateway"
SSHD_CONF      = REPO_ROOT / "deploy" / "sshd-gateway.conf"
INSTALL_SH     = REPO_ROOT / "deploy" / "install.sh"


def test_gateway_wrapper_exists():
    assert GATEWAY_SCRIPT.exists(), "deploy/iit-gpu-gateway is missing"


def test_gateway_wrapper_is_bash_script():
    text = GATEWAY_SCRIPT.read_text()
    assert text.startswith("#!/bin/bash"), "gateway wrapper must start with #!/bin/bash"


def test_gateway_wrapper_routes_scp():
    text = GATEWAY_SCRIPT.read_text()
    assert "scp" in text
    assert "SSH_ORIGINAL_COMMAND" in text


def test_gateway_wrapper_routes_rsync():
    text = GATEWAY_SCRIPT.read_text()
    assert "rsync" in text


def test_gateway_wrapper_has_jail_check():
    text = GATEWAY_SCRIPT.read_text()
    assert "NFS_ROOT" in text, "gateway wrapper must jail-check the transfer path"


def test_gateway_wrapper_falls_through_to_manager():
    text = GATEWAY_SCRIPT.read_text()
    assert "iit-gpu-manager" in text, "wrapper must exec iit-gpu-manager for interactive sessions"


def test_gateway_wrapper_rejects_shell_metacharacters():
    text = GATEWAY_SCRIPT.read_text()
    for meta in [r"\;", r"\`", "&&", "||", r"$("]:
        assert meta in text, f"gateway wrapper must guard against metacharacter: {meta!r}"


def test_gateway_wrapper_rejects_path_traversal():
    text = GATEWAY_SCRIPT.read_text()
    assert ".." in text, "gateway wrapper must guard against path traversal (..)"


def test_gateway_wrapper_has_nfs_root_placeholder():
    text = GATEWAY_SCRIPT.read_text()
    assert "__NFS_ROOT__" in text, (
        "deploy/iit-gpu-gateway must contain __NFS_ROOT__ placeholder "
        "so install.sh substitutes the real path at install time"
    )


def test_sshd_conf_uses_gateway_wrapper_not_manager_directly():
    text = SSHD_CONF.read_text()
    assert "ForceCommand /usr/local/bin/iit-gpu-gateway" in text, (
        "sshd-gateway.conf must use iit-gpu-gateway as ForceCommand"
    )
    assert "ForceCommand /usr/local/bin/iit-gpu-manager" not in text, (
        "ForceCommand must point to the gateway wrapper, not directly to iit-gpu-manager"
    )


def test_install_sh_installs_gateway_wrapper():
    text = INSTALL_SH.read_text()
    assert "iit-gpu-gateway" in text, "install.sh must install the gateway wrapper"


def test_install_sh_substitutes_nfs_root_in_wrapper():
    text = INSTALL_SH.read_text()
    assert "__NFS_ROOT__" in text, "install.sh must substitute __NFS_ROOT__ in the wrapper"
    assert "sed" in text, "install.sh must use sed to substitute NFS_ROOT"


def test_gateway_wrapper_routes_sftp_server():
    """Modern scp (OpenSSH 9+, Windows/Mac default) uses sftp-server internally.
    SSH_ORIGINAL_COMMAND will be '/usr/lib/openssh/sftp-server', not 'scp -t'.
    The wrapper must pass this through or scp always fails with 'message too long'."""
    text = GATEWAY_SCRIPT.read_text()
    assert "sftp-server" in text, (
        "gateway wrapper must handle sftp-server (modern scp on Windows/Mac uses "
        "SFTP protocol internally, not legacy scp -t)"
    )


def test_gateway_wrapper_uses_eval_exec_not_bash_c():
    """eval exec runs the transfer binary in-process without spawning a new bash,
    preventing any ~/.bashrc sourcing that could corrupt the SCP protocol."""
    text = GATEWAY_SCRIPT.read_text()
    assert "eval exec" in text, (
        "wrapper must use 'eval exec' not 'exec /bin/bash -c' to avoid "
        "spawning a new bash that could source ~/.bashrc"
    )
    code_lines = [l for l in text.splitlines() if not l.lstrip().startswith("#")]
    assert "exec /bin/bash -c" not in "\n".join(code_lines), (
        "'exec /bin/bash -c' spawns a new bash that may source ~/.bashrc "
        "and corrupt the transfer protocol"
    )


def test_gateway_wrapper_has_audit_helper():
    text = GATEWAY_SCRIPT.read_text()
    assert "_audit_transfer" in text, \
        "gateway wrapper must define an _audit_transfer helper"
    assert "auditclient" in text, \
        "_audit_transfer must call auditclient.log"


def test_gateway_wrapper_audits_all_three_transfer_branches():
    text = GATEWAY_SCRIPT.read_text()
    # Each branch must call _audit_transfer with its kind string
    for kind in ('"scp"', '"rsync"', '"sftp"'):
        assert f'_audit_transfer {kind}' in text, \
            f"gateway wrapper missing _audit_transfer {kind} call"


def test_gateway_wrapper_audit_is_best_effort():
    """The audit call must not block the transfer — it must be followed by || true
    or equivalent so a daemon failure never prevents a file transfer."""
    text = GATEWAY_SCRIPT.read_text()
    assert "|| true" in text, \
        "gateway audit call must be best-effort (|| true) so daemon failure never blocks transfer"


def test_gateway_wrapper_passes_bash_syntax_check():
    import subprocess
    r = subprocess.run(["bash", "-n", str(GATEWAY_SCRIPT)],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"bash -n failed: {r.stderr}"