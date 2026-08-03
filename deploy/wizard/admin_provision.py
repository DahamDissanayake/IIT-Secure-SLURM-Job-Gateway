"""First-admin provisioning stage.

Creates a real admin account so whoever ran the wizard isn't locked out of
their own cluster once install.sh's ForceCommand TUI is the only way in.
"""
from prompts import step, ok, warn, fail, text, confirm_risky
from state import State
from ssh_target import Target


def run(state: State, site: dict, target: Target) -> None:
    if state.is_done("admin_provision"):
        ok("First admin already provisioned — skipping")
        return

    step("First admin account")
    username = text("Admin username (lowercase, no spaces)")
    email = text("Admin email address")
    full_name = text("Full name", default="")

    if not confirm_risky(f"create Linux+SLURM admin account '{username}' on both nodes"):
        warn("skipped — provision an admin manually later via the Admin panel")
        return

    r = target.run(["sudo", "/usr/local/bin/slurm-deck-adduser", username, "--admin"],
                    check=False)
    if r.returncode != 0:
        fail("slurm-deck-adduser failed:\n" + (r.stderr or r.stdout))
        raise SystemExit(1)
    ok(f"Linux+SLURM account created for {username}")

    # slurm-deck-adduser --admin already added the account to gpuadmins, so
    # the daemon's SO_PEERCRED admin check recognizes this call as coming
    # from an admin even though its own users.db row doesn't exist yet.
    snippet = (
        "import sys; sys.path.insert(0, '/opt/slurm-deck'); "
        "from slurmdeck import daemonclient; "
        f"ok, msg = daemonclient.create_user({username!r}, {email!r}, "
        f"'admin', {full_name!r}, ''); "
        "print(ok, msg)"
    )
    r = target.run(["sudo", "-u", username, "python3", "-c", snippet], check=False)
    if r.returncode != 0 or "True" not in r.stdout:
        warn("could not create the users.db record automatically — "
             "do it from the Admin panel -> Manage Users once logged in")
    else:
        ok(f"users.db record created — {username} can log in as an admin")

    state.mark_done("admin_provision")
