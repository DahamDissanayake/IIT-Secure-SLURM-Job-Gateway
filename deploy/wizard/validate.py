"""Final validation stage: pytest gate + launcher smoke test + summary."""
from prompts import step, ok, fail
from state import State
from ssh_target import Target


def run(state: State, site: dict, target: Target) -> None:
    step("Final validation")

    r = target.run(
        ["bash", "-c",
         "cd /opt/slurm-deck && PYTHONPATH=/opt/slurm-deck "
         "python3 -m pytest tests/ -q --tb=short"],
        check=False,
    )
    if r.returncode == 0:
        ok("pytest gate passed")
    else:
        fail("pytest gate failed:\n" + (r.stdout or r.stderr)[-2000:])

    r = target.run(["test", "-x", "/usr/local/bin/slurm-deck"], check=False)
    if r.returncode == 0:
        ok("launcher installed at /usr/local/bin/slurm-deck")
    else:
        fail("launcher missing at /usr/local/bin/slurm-deck")

    print()
    print("=" * 60)
    print("  Slurm Deck installation complete.")
    print(f"  Users connect with: ssh -p {site['GATEWAY_PORT']} "
          f"<username>@{site['GATEWAY_HOST']}")
    print("  Add more users:  sudo slurm-deck-adduser <username>")
    print("  Admin panel:     log in, then choose '6. Admin'")
    print("=" * 60)
