# tests/test_wizard_install.py
"""Tests for the standalone installation wizard (deploy/wizard/).

The wizard's modules are flat scripts (not a package under slurmdeck/) that
import each other assuming their own directory is on sys.path -- exactly
how they behave when run directly (`python3 deploy/wizard/main.py`), which
auto-prepends the script's directory to sys.path[0]. Tests replicate that
by inserting the directory before importing, same spirit as
test_daemon_verbs.py loading deploy/audit_daemon.py by path.

Only pure logic is covered here (state resume, key-format validation, live
Resend-key verification against a mocked urllib, site.env rendering) --
the actual libvirt/apt/systemd/SSH orchestration isn't realistically
unit-testable and is verified by running the wizard for real.
"""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

WIZARD_DIR = Path(__file__).parent.parent / "deploy" / "wizard"
sys.path.insert(0, str(WIZARD_DIR))

import state as wizard_state  # noqa: E402
import mail as wizard_mail  # noqa: E402
import login_node  # noqa: E402


# ── state.py resume logic ───────────────────────────────────────────────────

def test_fresh_state_has_nothing_done(tmp_path):
    s = wizard_state.State(str(tmp_path / "state.json"))
    assert not s.is_done("gpu_node")
    assert s.get_answer("mode") is None


def test_mark_done_persists_across_instances(tmp_path):
    path = str(tmp_path / "state.json")
    s1 = wizard_state.State(path)
    s1.mark_done("gpu_node")

    s2 = wizard_state.State(path)
    assert s2.is_done("gpu_node")
    assert not s2.is_done("login_node")


def test_set_answer_persists_across_instances(tmp_path):
    path = str(tmp_path / "state.json")
    s1 = wizard_state.State(path)
    s1.set_answer("site", {"NFS_ROOT": "/shared"})

    s2 = wizard_state.State(path)
    assert s2.get_answer("site") == {"NFS_ROOT": "/shared"}


def test_get_answer_default_when_missing(tmp_path):
    s = wizard_state.State(str(tmp_path / "state.json"))
    assert s.get_answer("nope", "fallback") == "fallback"


def test_corrupt_state_file_does_not_crash(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not valid json")
    s = wizard_state.State(str(path))
    assert not s.is_done("gpu_node")
    assert s.get_answer("mode") is None


# ── mail.py: Resend key format + live verification ─────────────────────────

def test_validate_key_format_accepts_real_looking_key():
    assert wizard_mail.validate_key_format("re_AbCd1234567890") is None


def test_validate_key_format_rejects_wrong_prefix():
    err = wizard_mail.validate_key_format("sk_AbCd1234567890")
    assert err is not None
    assert "re_" in err


def test_validate_key_format_rejects_too_short():
    assert wizard_mail.validate_key_format("re_short") is not None


def test_verify_key_true_on_200():
    fake_resp = MagicMock()
    fake_resp.status = 200
    fake_resp.__enter__.return_value = fake_resp
    with patch("urllib.request.urlopen", return_value=fake_resp):
        assert wizard_mail.verify_key("re_validkey1234567890") is True


def test_verify_key_false_on_401():
    import urllib.error
    err = urllib.error.HTTPError("url", 401, "unauthorized", {}, None)
    with patch("urllib.request.urlopen", side_effect=err):
        assert wizard_mail.verify_key("re_badkey1234567890") is False


def test_verify_key_false_on_403():
    import urllib.error
    err = urllib.error.HTTPError("url", 403, "forbidden", {}, None)
    with patch("urllib.request.urlopen", side_effect=err):
        assert wizard_mail.verify_key("re_badkey1234567890") is False


def test_verify_key_true_on_unrelated_http_error():
    import urllib.error
    err = urllib.error.HTTPError("url", 500, "server error", {}, None)
    with patch("urllib.request.urlopen", side_effect=err):
        assert wizard_mail.verify_key("re_key1234567890") is True


def test_verify_key_true_on_network_error():
    import urllib.error
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("no route")):
        assert wizard_mail.verify_key("re_key1234567890") is True


# ── login_node.py: site.env rendering ───────────────────────────────────────

def test_render_site_env_quotes_values_with_spaces():
    out = login_node.render_site_env({"CLUSTER_NAME": "GPU Cluster"})
    assert 'CLUSTER_NAME="GPU Cluster"\n' in out


def test_render_site_env_leaves_simple_values_unquoted():
    out = login_node.render_site_env({"NFS_ROOT": "/shared"})
    assert "NFS_ROOT=/shared\n" in out
    assert '"' not in out.split("NFS_ROOT=")[1].split("\n")[0]


def test_render_site_env_quotes_empty_values():
    out = login_node.render_site_env({"CLUSTER_LOCATION": ""})
    assert 'CLUSTER_LOCATION=""\n' in out


def test_render_site_env_preserves_key_order():
    out = login_node.render_site_env({"A": "1", "B": "2", "C": "3"})
    assert out.index("A=1") < out.index("B=2") < out.index("C=3")
