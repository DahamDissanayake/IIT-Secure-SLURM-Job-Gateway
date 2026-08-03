"""Resend email setup stage.

Collects the sending domain + API key, live-verifies the key against
Resend's API, writes deploy/secrets.env on the login node, walks through
DNS domain verification (inherently manual -- it happens on the admin's DNS
provider, not something this tool can do for them), and records MAIL_FROM
for the rest of the wizard to write into site.env.
"""
import os
import re
import tempfile
import urllib.error
import urllib.request

from prompts import step, ok, warn, fail, text, secret, yesno, confirm_risky
from state import State
from ssh_target import Target

KEY_RE = re.compile(r"^re_[A-Za-z0-9_]{10,}$")


def validate_key_format(key: str) -> str | None:
    """Pure validator: None if key looks like a Resend key, else an error
    string. Matches the `validator` signature prompts.text() expects."""
    if not KEY_RE.match(key):
        return "that doesn't look like a Resend API key (expected re_...)"
    return None


def verify_key(api_key: str, timeout: int = 10) -> bool:
    """Live-check the key against Resend's API. True if it authenticates (or
    the check couldn't be completed, e.g. no network) -- False only on a
    definite auth rejection, so a real network hiccup never blocks setup."""
    req = urllib.request.Request(
        "https://api.resend.com/domains",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as exc:
        return exc.code not in (401, 403)
    except (urllib.error.URLError, TimeoutError):
        warn("network error reaching Resend's API — skipping live verification")
        return True


def run(state: State, site: dict, target: Target) -> None:
    if state.is_done("mail"):
        ok("Email setup already done — skipping")
        return

    step("Email setup (Resend)")
    print("  Resend (https://resend.com) sends welcome/login/job-status mail.")
    print("  You need: a Resend account, a sending domain, and an API key")
    print("  scoped to 'Sending access' only (Resend dashboard -> API Keys).")

    domain = state.get_answer("resend_domain") or text(
        "Sending domain (e.g. slurm-deck.example.edu)")
    from_name = state.get_answer("resend_from_name") or text(
        "From display name", default="Slurm Deck")
    api_key = state.get_answer("resend_api_key") or secret(
        "Resend API key (re_...)")
    while validate_key_format(api_key):
        warn(validate_key_format(api_key))
        api_key = secret("Resend API key (re_...)")

    if verify_key(api_key):
        ok("API key verified against the Resend API")
    elif not yesno("Could not verify the key against Resend's API — continue anyway?",
                    default=False):
        raise SystemExit(1)

    mail_from = f"{from_name} <no-reply@{domain}>"
    site["MAIL_FROM"] = mail_from

    if confirm_risky("write the Resend API key to secrets.env on the login node "
                      "(mode 0640 root:gpusync)"):
        _write_secrets_env(target, api_key, mail_from)
        ok("secrets.env written")
    else:
        warn("skipped — write deploy/secrets.env manually before mail will work")

    print()
    print("  DNS verification (do this in your DNS provider, then continue):")
    print(f"  1. In Resend: Domains -> Add Domain -> {domain}")
    print("  2. Add the SPF/DKIM/DMARC records Resend shows you")
    print("  3. Wait for Resend to show the domain as 'Verified' "
          "(can take a few minutes)")
    while not yesno("Domain verified in Resend?", default=False):
        print("  waiting — verify the domain in the Resend dashboard, "
              "then answer yes")

    state.set_answer("resend_domain", domain)
    state.set_answer("resend_from_name", from_name)
    state.set_answer("resend_api_key", api_key)
    state.mark_done("mail")
    ok("Email setup complete")


def _write_secrets_env(target: Target, api_key: str, mail_from: str) -> None:
    content = (
        "# Written by the slurm-deck installation wizard\n"
        f"RESEND_API_KEY={api_key}\n"
        f"MAIL_FROM={mail_from}\n"
    )
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".env") as f:
        f.write(content)
        local_tmp = f.name
    try:
        target.copy_to(local_tmp, "/tmp/slurm-deck-secrets.env")
        target.run(["sudo", "install", "-o", "root", "-g", "gpusync", "-m", "0640",
                    "/tmp/slurm-deck-secrets.env",
                    "/opt/slurm-deck/deploy/secrets.env"])
        target.run(["rm", "-f", "/tmp/slurm-deck-secrets.env"])
    finally:
        os.unlink(local_tmp)
