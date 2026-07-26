# tests/test_wizard.py
"""Regression tests for wizard.py.

The notebook branch (Phase 5) accidentally re-imported `panel` and `shutil`
locally inside run_wizard(). Python then treated those module-level names as
function-locals across the WHOLE function, so the non-notebook path raised
UnboundLocalError: "cannot access local variable 'panel'...". These tests guard
against that class of bug: a module-level import must never be shadowed by a
function-local of the same name.
"""
import iitgpu.wizard as wizard


# Names imported at module top in wizard.py that must stay global inside functions.
_MODULE_LEVEL_NAMES = {
    "panel", "shutil", "getpass", "questionary",
    "JobSpec", "make_job_folder", "render_sbatch", "resource_defaults",
    "submit_job", "load_config", "jobs_dir",
    "err", "header", "info", "kv", "ok", "warn",
    "clean_run_command", "in_jail", "safe_listdir", "auditclient",
}


def _function_locals(fn) -> set[str]:
    return set(fn.__code__.co_varnames)


def test_run_wizard_does_not_shadow_module_imports():
    shadowed = _function_locals(wizard.run_wizard) & _MODULE_LEVEL_NAMES
    assert not shadowed, (
        f"run_wizard() shadows module-level names as locals: {shadowed}. "
        "Remove the redundant local imports — they cause UnboundLocalError on "
        "code paths that run before the local import line."
    )


def test_panel_is_module_global_in_wizard():
    # panel must be resolvable at module scope (imported at top), not per-branch.
    assert hasattr(wizard, "panel"), "panel should be a module-level import in wizard.py"


def test_shutil_is_module_global_in_wizard():
    assert hasattr(wizard, "shutil"), "shutil should be a module-level import in wizard.py"


def test_wizard_module_compiles_and_imports():
    # Importing the module already ran above; assert the entry point exists.
    assert callable(wizard.run_wizard)


# ─── TUI refactor: data path in the sbatch, rerun parsing ────────────────────

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_data_path_exported_in_sbatch_when_set(tmp_path):
    """render_sbatch() includes 'export DATA_PATH=...' when data_path is set."""
    from iitgpu.jobs import JobSpec, render_sbatch

    spec = JobSpec(
        job_name="test_job",
        partition="gpu",
        gpu_shards=1,
        cpus=4,
        mem_gb=16,
        time_limit="01:00:00",
        run_command="python /shared/testuser/scripts/train.py",
        task_type="train",
        data_path="/shared/testuser/data/20260601_120000_inline.txt",
    )
    script = render_sbatch(spec, str(tmp_path))

    assert "export DATA_PATH=" in script
    assert "/shared/testuser/data/20260601_120000_inline.txt" in script

    # Export must appear before the run_command line
    export_idx = script.index("export DATA_PATH=")
    run_idx = script.index("python /shared/testuser/scripts/train.py")
    assert export_idx < run_idx, "DATA_PATH export must appear before the run command"


def test_data_path_not_in_sbatch_when_not_set(tmp_path):
    """render_sbatch() omits 'export DATA_PATH' when data_path is empty."""
    from iitgpu.jobs import JobSpec, render_sbatch

    spec = JobSpec(
        job_name="test_job",
        partition="gpu",
        gpu_shards=1,
        cpus=4,
        mem_gb=16,
        time_limit="01:00:00",
        run_command="python /shared/testuser/scripts/train.py",
        task_type="train",
        data_path="",  # explicitly empty
    )
    script = render_sbatch(spec, str(tmp_path))
    assert "export DATA_PATH" not in script


def test_rerun_parses_sbatch_fields(tmp_path):
    """_parse_sbatch correctly extracts all common SBATCH fields."""
    from iitgpu.monitor import _parse_sbatch

    sbatch = """\
#!/bin/bash
#SBATCH --job-name=train_20260601_120000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --output=/shared/jobs/testuser/train_20260601_120000/slurm-%j.out
#SBATCH --error=/shared/jobs/testuser/train_20260601_120000/slurm-%j.err
#SBATCH --chdir=/shared/jobs/testuser/train_20260601_120000

_conda_sh="${CONDA_PREFIX_SHARED:-/shared/miniforge3}/etc/profile.d/conda.sh"
[ -f "$_conda_sh" ] && source "$_conda_sh"
conda activate /shared/envs/pytorch-cifar

cd /shared/jobs/testuser/train_20260601_120000
python /shared/testuser/scripts/train.py --epochs 10
"""
    result = _parse_sbatch(sbatch)

    assert result.get("partition") == "gpu"
    assert result.get("gpu_shards") == 1
    assert result.get("cpus") == 16
    assert result.get("mem_gb") == 60
    assert result.get("time_limit") == "08:00:00"
    assert result.get("conda_env") == "/shared/envs/pytorch-cifar"
    assert result.get("script_path") == "/shared/testuser/scripts/train.py"


def test_rerun_parses_container_image_from_sbatch(tmp_path):
    """_parse_sbatch extracts the container image path and leaves conda_env empty."""
    from iitgpu.monitor import _parse_sbatch

    sbatch = """\
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

cd /shared/jobs/testuser/inference_20260601_130000
apptainer exec --nv --bind /shared /shared/images/llm-finetune.sif bash -lc 'python /shared/testuser/scripts/infer.py'
"""
    result = _parse_sbatch(sbatch)

    assert result.get("container_image") == "/shared/images/llm-finetune.sif"
    assert result.get("conda_env", "") == ""


def test_wizard_accepts_prefill_without_error(monkeypatch, tmp_path):
    """run_wizard(prefill=...) must not raise when all prompts are mocked.

    Both prefill shapes the rerun path can produce: without a script_path the
    flow stops at the script intake, with one it goes straight to the hub.
    """
    import iitgpu.wizard as wiz

    # Mock all questionary prompts to bail out immediately
    monkeypatch.setattr(
        "questionary.confirm",
        lambda *a, **kw: MagicMock(ask=lambda: False),
    )
    monkeypatch.setattr(
        "questionary.select",
        lambda *a, **kw: MagicMock(ask=lambda: None),
    )
    monkeypatch.setattr(
        "questionary.text",
        lambda *a, **kw: MagicMock(ask=lambda: ""),
    )
    monkeypatch.setattr(
        "questionary.autocomplete",
        lambda *a, **kw: MagicMock(ask=lambda: None),
    )

    # Should return cleanly (wizard exits when the script intake cancels)
    wiz.run_wizard(prefill={"task_type": "train", "conda_env": "/shared/envs/x"})

    # With a script the intake is skipped — the hub takes over and cancels.
    script = tmp_path / "train.py"
    script.write_text("print(1)\n")
    wiz.run_wizard(prefill={"gpu_shards": 2, "cpus": 8, "mem_gb": 16,
                            "time_limit": "02:00:00", "script_path": str(script)})


# ── Email auto-wire ────────────────────────────────────────────────────────────

def test_mail_user_set_from_users_db_when_mta_present(tmp_path):
    """When MTA is available and users.db has an email, mail_user is auto-populated."""
    from iitgpu.jobs import JobSpec, make_job_folder, render_sbatch

    spec = JobSpec(job_name="j", partition="gpu", gpu_shards=1, cpus=4, mem_gb=8,
                   time_limit="01:00:00", run_command="python x.py")

    with patch("iitgpu.notify.mta_present", return_value=True), \
         patch("iitgpu.daemonclient.email_for", return_value="alice@uni.edu"):
        from iitgpu.notify import mta_present
        from iitgpu import daemonclient
        if mta_present():
            email = daemonclient.email_for("alice")
            if email:
                spec.mail_user = email

    folder = make_job_folder(str(tmp_path), spec)
    sbatch = render_sbatch(spec, folder)
    assert "#SBATCH --mail-user=alice@uni.edu" in sbatch
    assert "--mail-type=" in sbatch


def test_mail_user_not_set_when_mta_absent(tmp_path):
    """When no MTA is present, mail_user stays empty even if users.db has an email."""
    from iitgpu.jobs import JobSpec, make_job_folder, render_sbatch

    spec = JobSpec(job_name="j", partition="gpu", gpu_shards=1, cpus=4, mem_gb=8,
                   time_limit="01:00:00", run_command="python x.py")

    with patch("iitgpu.notify.mta_present", return_value=False):
        from iitgpu.notify import mta_present
        if mta_present():
            spec.mail_user = "should-not-be-set@example.com"

    folder = make_job_folder(str(tmp_path), spec)
    sbatch = render_sbatch(spec, folder)
    assert "--mail-user" not in sbatch


def test_mail_user_not_set_when_no_email_in_db(tmp_path):
    """When MTA is present but user has no email registered, mail_user stays empty."""
    from iitgpu.jobs import JobSpec, make_job_folder, render_sbatch

    spec = JobSpec(job_name="j", partition="gpu", gpu_shards=1, cpus=4, mem_gb=8,
                   time_limit="01:00:00", run_command="python x.py")

    with patch("iitgpu.notify.mta_present", return_value=True), \
         patch("iitgpu.daemonclient.email_for", return_value=None):
        from iitgpu.notify import mta_present
        from iitgpu import daemonclient
        if mta_present():
            email = daemonclient.email_for("newuser")
            if email:
                spec.mail_user = email

    folder = make_job_folder(str(tmp_path), spec)
    sbatch = render_sbatch(spec, folder)
    assert "--mail-user" not in sbatch


# ─── Regression: wizard file browsers honour a per-user jail (issue: data/script
#     picker must start in & stay confined to shared/users/<user>) ─────────────

def _select_returning(value):
    """Build a questionary.select stand-in that returns `value` once then cancels."""
    seq = iter([value, "[cancel]"])
    return lambda *a, **kw: MagicMock(ask=lambda: next(seq))


def test_browse_data_folder_uses_supplied_jail(tmp_path, monkeypatch):
    """A regular user's browse jail must gate selection — picking a folder inside
    their own area is allowed; the same browser must refuse paths outside it."""
    import iitgpu.wizard as wiz
    from iitgpu.validate import in_user_browse_jail

    nfs = str(tmp_path)
    alice_dir = Path(nfs) / "users" / "alice"
    bob_dir = Path(nfs) / "users" / "bob"
    alice_dir.mkdir(parents=True)
    bob_dir.mkdir(parents=True)

    jail = lambda p: in_user_browse_jail(p, nfs, "alice")

    # Selecting alice's own dir → allowed.
    monkeypatch.setattr("questionary.select", _select_returning("[select this folder]"))
    assert wiz._browse_data_folder(str(alice_dir), jail) == str(alice_dir)

    # Selecting bob's dir with alice's jail → denied (returns None).
    monkeypatch.setattr("questionary.select", _select_returning("[select this folder]"))
    assert wiz._browse_data_folder(str(bob_dir), jail) is None


def test_browse_script_uses_supplied_jail(tmp_path, monkeypatch):
    """The script picker must likewise refuse a file outside the user's jail."""
    import iitgpu.wizard as wiz
    from iitgpu.validate import in_user_browse_jail

    nfs = str(tmp_path)
    alice_dir = Path(nfs) / "users" / "alice"
    bob_dir = Path(nfs) / "users" / "bob"
    alice_dir.mkdir(parents=True)
    bob_dir.mkdir(parents=True)
    (alice_dir / "train.py").write_text("print('hi')\n")
    (bob_dir / "secret.py").write_text("print('nope')\n")

    jail = lambda p: in_user_browse_jail(p, nfs, "alice")

    # Pick alice's own script → returned.
    monkeypatch.setattr("questionary.select", _select_returning("train.py"))
    assert wiz._browse_script(str(alice_dir), jail) == str(alice_dir / "train.py")

    # Pick bob's script while jailed to alice → denied.
    monkeypatch.setattr("questionary.select", _select_returning("secret.py"))
    assert wiz._browse_script(str(bob_dir), jail) is None


def test_browse_helpers_default_jail_is_global_in_jail():
    """Default jail param stays the global in_jail so admin callers are unaffected.
    (Identity-free check: other tests may reload modules, which would rebind the
    function object while keeping the same semantics.)"""
    import inspect
    import iitgpu.wizard as wiz

    for fn in (wiz._browse_data_folder, wiz._browse_script):
        default = inspect.signature(fn).parameters["jail"].default
        assert callable(default)
        assert getattr(default, "__name__", "") == "in_jail"


def test_valid_pkg_tokens_keeps_specs_drops_shell_metachars():
    from iitgpu.wizard import _valid_pkg_tokens
    assert _valid_pkg_tokens("tqdm wfdb==4.1 torch>=2.0 scikit-learn[extra]") == \
        ["tqdm", "wfdb==4.1", "torch>=2.0", "scikit-learn[extra]"]
    for bad in ["a;b", "$(x)", "a&&b", "../x", "a|b", "`x`"]:
        assert _valid_pkg_tokens(bad) == [], bad


def test_notebook_deps_prompt_autodetects_requirements(tmp_path, monkeypatch):
    """A requirements.txt in the notebook's project root is auto-detected and,
    when chosen, returned for pip-install before the run."""
    import iitgpu.wizard as wiz
    proj = tmp_path / "proj"
    (proj / "notebooks").mkdir(parents=True)
    nb = proj / "notebooks" / "run.ipynb"
    nb.write_text("{}")
    reqs = proj / "requirements.txt"
    reqs.write_text("tqdm\n")

    auto = f"Install from {reqs}  (auto-detected)"
    monkeypatch.setattr("questionary.select",
                        lambda *a, **k: MagicMock(ask=lambda: auto))
    req, pkgs = wiz._notebook_deps_prompt(str(nb), lambda p: True, str(proj))
    assert req == str(reqs) and pkgs == ""


def test_notebook_deps_prompt_skip_returns_empty(tmp_path, monkeypatch):
    import iitgpu.wizard as wiz
    nb = tmp_path / "run.ipynb"
    nb.write_text("{}")
    monkeypatch.setattr("questionary.select",
                        lambda *a, **k: MagicMock(ask=lambda: "Skip — my environment already has everything"))
    assert wiz._notebook_deps_prompt(str(nb), lambda p: True, str(tmp_path)) == ("", "")


def test_notebook_deps_prompt_no_notebook_skips_autodetect(tmp_path, monkeypatch):
    """For the JupyterLab flow (no notebook path) there is no auto-detect choice;
    typing packages still works."""
    import iitgpu.wizard as wiz
    seen = {}

    def _cap(*a, **k):
        seen["choices"] = k.get("choices", [])
        return MagicMock(ask=lambda: "Type package names (e.g. tqdm wfdb h5py)")

    monkeypatch.setattr("questionary.select", _cap)
    monkeypatch.setattr("questionary.text",
                        lambda *a, **k: MagicMock(ask=lambda: "tensorboard tqdm"))
    req, pkgs = wiz._notebook_deps_prompt("", lambda p: True, str(tmp_path))
    assert req == "" and pkgs == "tensorboard tqdm"
    assert not any("auto-detected" in c for c in seen["choices"])


# ── Step counter and UX fixes ──────────────────────────────────────────────────

def test_notebook_deps_custom_question(tmp_path, monkeypatch):
    import iitgpu.wizard as wiz
    seen = {}
    def _cap_select(*a, **k):
        seen["question"] = a[0] if a else ""
        return MagicMock(ask=lambda: "Skip — my environment already has everything")
    monkeypatch.setattr("questionary.select", _cap_select)
    wiz._notebook_deps_prompt("", lambda p: True, str(tmp_path),
                               question="Optional — Pre-install packages for this session?")
    assert seen["question"] == "Optional — Pre-install packages for this session?"


def test_notebook_deps_default_question(tmp_path, monkeypatch):
    import iitgpu.wizard as wiz
    seen = {}
    def _cap_select(*a, **k):
        seen["question"] = a[0] if a else ""
        return MagicMock(ask=lambda: "Skip — my environment already has everything")
    monkeypatch.setattr("questionary.select", _cap_select)
    wiz._notebook_deps_prompt("", lambda p: True, str(tmp_path))
    assert seen["question"] == "Install Python dependencies first?"


def test_step_counter_increments_per_call():
    _n = [1]
    def _S(label):
        _n[0] += 1
        return f"Step {_n[0]} — {label}"
    assert _S("Environment type:") == "Step 2 — Environment type:"
    assert _S("Your data:")        == "Step 3 — Your data:"
    assert _S("Your model:")       == "Step 4 — Your model:"


def test_own_sbatch_lives_under_other_and_stops_interrupting(monkeypatch):
    """The own-.sbatch bypass used to stop every user mid-flow with a confirm
    they mostly answered "no" to. It now sits under "Other" — two selects for
    the people who want it, invisible to everyone else."""
    import iitgpu.wizard as wiz

    confirms_seen = []
    def _cap_confirm(msg, **kw):
        confirms_seen.append(msg)
        return MagicMock(ask=lambda: False)
    picks = iter([wiz._OTHER_CHOICE, "Submit my own .sbatch"])
    def _cap_select(*a, **kw):
        return MagicMock(ask=lambda: next(picks, None))

    reached = []
    monkeypatch.setattr("questionary.confirm", _cap_confirm)
    monkeypatch.setattr("questionary.select", _cap_select)
    monkeypatch.setattr("questionary.text", lambda *a, **kw: MagicMock(ask=lambda: ""))
    monkeypatch.setattr(wiz, "_run_own_sbatch", lambda *a: reached.append(a))

    wiz.run_wizard()

    assert reached, "Other → Submit my own .sbatch must reach the own-sbatch path"
    assert not any("ready-made" in m for m in confirms_seen), (
        f"the mid-flow bypass confirm must be gone. Confirms: {confirms_seen}"
    )


def test_settings_menu_no_duplicates():
    import iitgpu.menu as m
    import inspect
    src = inspect.getsource(m._settings_menu)
    assert '"Cluster status"' not in src
    assert '"Hardware stats (live)"' not in src
    assert '"Admin panel"' not in src
    assert "Cluster health check" in src
    assert "Build environment" in src
    assert "Run smoke test" in src


# ── VRAM guardrail ────────────────────────────────────────────────────────────

def _make_stats(gpu_mem_used_mb: int = 12288, gpu_mem_total_mb: int = 32768):
    from iitgpu.slurm import NodeStats
    return NodeStats(
        state="ALLOCATED", cpu_load=1.0, cpu_total=32, cpu_alloc=16,
        mem_total_mb=131072, mem_alloc_mb=65536, gpu_total=1, gpu_alloc=1,
        gpu_util=50, gpu_mem_used_mb=gpu_mem_used_mb,
        gpu_mem_total_mb=gpu_mem_total_mb,
        gpu_temp=60, gpu_power_w=200.0, cpu_util=40, cpu_load5=1.0,
        mem_used_mb=40000, live_stats=True,
    )


def test_vram_check_no_longer_blocks_and_never_prompts(monkeypatch):
    """The VRAM gate is gone. It asked for an estimate that bound nobody — VRAM
    is shared between concurrent jobs and SLURM does not enforce it — and then
    refused the job on the strength of that guess. _vram_check now states the
    situation and always proceeds, even with the card nearly full."""
    from unittest.mock import patch
    import iitgpu.wizard as wiz

    # 30 GB in use of 32 GB — the old gate would have blocked here.
    stats = _make_stats(gpu_mem_used_mb=30720, gpu_mem_total_mb=32768)

    def _no_prompt(*a, **kw):
        raise AssertionError("_vram_check must not ask the user anything")

    with patch("iitgpu.wizard.get_node_stats", return_value=stats), \
         patch("questionary.text", side_effect=_no_prompt), \
         patch("questionary.confirm", side_effect=_no_prompt):
        result = wiz._vram_check()

    assert result is True

    # …and with no live stats at all, which used to be its own early return.
    with patch("iitgpu.wizard.get_node_stats", return_value=None), \
         patch("questionary.text", side_effect=_no_prompt), \
         patch("questionary.confirm", side_effect=_no_prompt):
        assert wiz._vram_check() is True


def test_vram_check_reports_the_fair_share_without_asking(capsys):
    """What survives of the check is the useful half: the live reading and what
    one slice's fair share of the card actually is, printed, not asked."""
    from unittest.mock import patch
    import iitgpu.wizard as wiz

    stats = _make_stats(gpu_mem_used_mb=0, gpu_mem_total_mb=32768)
    with patch("iitgpu.wizard.get_node_stats", return_value=stats), \
         patch("questionary.text", side_effect=AssertionError):
        assert wiz._vram_check() is True

    out = " ".join(capsys.readouterr().out.split())   # rich wraps; ignore layout
    assert "about 8 GB" in out
    assert "not enforced" in out


def test_vram_check_present_in_wizard_source():
    """_vram_check must be called in run_wizard for both submission paths."""
    import inspect
    from iitgpu import wizard
    src = inspect.getsource(wizard.run_wizard)
    assert src.count("_vram_check") >= 2, (
        f"Expected _vram_check called at least twice in run_wizard, found {src.count('_vram_check')}"
    )


def test_gpu_share_note_describes_a_partial_card():
    """A one-slice job must say so — resource sizing changed and nothing said."""
    from iitgpu.jobs import SHARDS_PER_GPU, gpu_share_note
    note = gpu_share_note(1)
    assert f"1/{SHARDS_PER_GPU}" in note
    assert "left for others" in note


def test_gpu_share_note_is_used_by_the_wizard():
    """Defined-but-unused is worse than absent: it reads as done."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "iitgpu" / "wizard.py").read_text()
    assert "gpu_share_note" in src, "wizard must show the GPU share to the user"


def test_vram_prompt_states_the_budget_is_shared_and_unenforced():
    """Slices schedule, they do not isolate — two jobs can still OOM each other."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "iitgpu" / "wizard.py").read_text()
    start = src.index("def _vram_check")
    body = src[start:start + 3000]
    assert "shared" in body.lower(), "prompt must say VRAM is shared"
    assert "not enforced" in body.lower(), "prompt must say it is not enforced"


def test_notebook_submit_audits_the_interactive_session():
    """A notebook is a full execution environment; the trail must show it started."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "iitgpu" / "wizard.py").read_text()
    assert "notebook_session_start" in src
    idx = src.index("notebook_session_start")
    window = src[idx - 200:idx + 300]
    assert "gpu_shards" in window, "record the slice the session holds"
    assert "job_id" in window, "record which job the session is"


def test_notebook_post_submit_waits_and_shows_connect_card():
    """After submit the TUI must wait for readiness and print the Connect card
    parsed from the job's own output — not reconstruct tunnel details."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "iitgpu" / "wizard.py").read_text()
    assert "_post_submit_notebook" in src
    i = src.index("def _post_submit_notebook")
    body = src[i:i + 2500]
    assert "wait_ready" in body and "parse_connect" in body and "render_card" in body
    assert "gone" in body and "timeout" in body  # both failure states handled


def test_post_submit_ready_shows_card(monkeypatch, tmp_path):
    """When the job is ready and its .out already parses, show the Connect card
    built from the job's own output — not a reconstructed tunnel hint."""
    from rich.console import Console
    import iitgpu.connect as connect
    import iitgpu.ui as ui
    from iitgpu.wizard import _post_submit_notebook

    sample_out = (
        "=================================================\n"
        "JupyterLab is starting on the GPU node.\n"
        "Token: bc123979da0269048efef70ff6bfb8fffdbb2ef71827937f\n"
        "SSH tunnel — open a NEW terminal on YOUR LAPTOP and run:\n"
        "  ssh -p 2225 -N -L 8930:192.168.122.1:8930 yenuli@10.35.4.100\n"
        "  (-N = tunnel only, no shell opens — terminal sitting idle is correct)\n"
        "Then open in browser: http://127.0.0.1:8930/lab?token="
        "bc123979da0269048efef70ff6bfb8fffdbb2ef71827937f\n"
        "=================================================\n"
    )
    (tmp_path / "slurm-1.out").write_text(sample_out)

    monkeypatch.setattr(connect, "wait_ready", lambda *a, **kw: "ready")
    test_console = Console(record=True, force_terminal=True, width=120)
    monkeypatch.setattr(ui, "console", test_console)

    _post_submit_notebook("123", str(tmp_path))

    rendered = test_console.export_text()
    assert "ssh -p 2225 -N -L 8930:192.168.122.1:8930 yenuli@10.35.4.100" in rendered


def test_post_submit_ready_race_never_says_still_starting(monkeypatch, tmp_path):
    """A ready job whose .out hasn't flushed the connect block yet must never
    be reported as 'still starting' — that's factually wrong. It must point
    the user at the dashboard's Connect card instead."""
    from rich.console import Console
    import iitgpu.connect as connect
    import iitgpu.ui as ui
    from iitgpu.wizard import _post_submit_notebook

    (tmp_path / "slurm-1.out").write_text("JupyterLab is starting on the GPU node.\n")

    monkeypatch.setattr(connect, "wait_ready", lambda *a, **kw: "ready")
    monkeypatch.setattr("time.sleep", lambda *a, **kw: None)
    test_console = Console(record=True, force_terminal=True, width=120)
    monkeypatch.setattr(ui, "console", test_console)

    _post_submit_notebook("123", str(tmp_path))

    rendered = test_console.export_text()
    assert "Still starting" not in rendered
    assert "T" in rendered and "dashboard" in rendered.lower()


def test_post_submit_gone_tails_stderr(monkeypatch, tmp_path):
    """Crash output lands in .err (the launcher is un-redirected there), so a
    'gone' state must tail .err, not just .out."""
    from rich.console import Console
    import iitgpu.connect as connect
    import iitgpu.ui as ui
    from iitgpu.wizard import _post_submit_notebook

    (tmp_path / "slurm-1.out").write_text("")
    (tmp_path / "slurm-1.err").write_text("ERROR: JupyterLab is missing\n")

    monkeypatch.setattr(connect, "wait_ready", lambda *a, **kw: "gone")
    test_console = Console(record=True, force_terminal=True, width=120)
    monkeypatch.setattr(ui, "console", test_console)

    _post_submit_notebook("123", str(tmp_path))

    rendered = test_console.export_text()
    assert "ERROR: JupyterLab is missing" in rendered


# ── Launch-flow rewrite ──────────────────────────────────────────────────────

def test_wizard_offers_three_intents_not_seven_task_types():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "iitgpu" / "wizard.py").read_text()
    assert "Open JupyterLab" in src
    assert "Run a script or notebook" in src
    assert "Open a shell on the GPU node" in src
    assert "Step 1 — What are you doing?" not in src


def test_wizard_no_longer_quizzes_vram_but_keeps_the_wording():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "iitgpu" / "wizard.py").read_text()
    assert "Estimated VRAM your job needs" not in src
    i = src.index("def _vram_check")
    body = src[i:i + 1800]
    assert "shared" in body.lower() and "not enforced" in body.lower()


def test_wizard_hands_off_to_the_hub():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "iitgpu" / "wizard.py").read_text()
    assert "run_hub" in src and "default_spec" in src
    assert "recent_scripts" in src and "autocomplete" in src


def test_parse_sbatch_handles_a_quoted_script_path():
    """The wizard shlex.quotes the script path into the sbatch, so a notebook
    with a space in its name comes back through the rerun parser quoted. A
    whitespace split hands back "'/a/my" — a path that does not exist, offered
    to the user as the thing they are about to re-run."""
    from iitgpu.monitor import _parse_sbatch

    sbatch = (
        "#!/bin/bash\n"
        "#SBATCH --partition=gpu\n"
        "#SBATCH --gres=shard:1\n"
        "\n"
        "cd /shared/jobs/alice/train_20260725_120000\n"
        "python3 '/shared/users/alice/my train.py' --lr 3 --epochs 10\n"
    )
    result = _parse_sbatch(sbatch)

    assert result["script_path"] == "/shared/users/alice/my train.py"
    assert result["extra_args"] == "--lr 3 --epochs 10"


def test_parse_sbatch_unquoted_path_still_works():
    """The common case must not regress: no quotes, no change."""
    from iitgpu.monitor import _parse_sbatch

    sbatch = ("#!/bin/bash\n"
              "cd /shared/jobs/alice/j\n"
              "python3 /shared/users/alice/train.py --epochs 10\n")
    result = _parse_sbatch(sbatch)
    assert result["script_path"] == "/shared/users/alice/train.py"
    assert result["extra_args"] == "--epochs 10"


def test_parse_sbatch_survives_unbalanced_quotes():
    """A hand-edited sbatch must not take the rerun browser down with it."""
    from iitgpu.monitor import _parse_sbatch

    result = _parse_sbatch("#!/bin/bash\ncd /x\npython3 '/broken/quote.py --lr 3\n")
    assert "script_path" not in result       # unparseable, so not offered
    assert result["run_command"].startswith("python3")


def test_rerun_prefill_validates_the_script_before_using_it(tmp_path, monkeypatch):
    """A path lifted out of an old sbatch has had no more validation than a
    typed one. When it no longer resolves, the wizard must fall through to the
    normal intake instead of carrying a dead path into the hub."""
    import iitgpu.wizard as wiz

    monkeypatch.setenv("NFS_ROOT", str(tmp_path))
    monkeypatch.setenv("IIT_SITE_ENV", "/nonexistent")

    asked = []

    def _fake_autocomplete(*a, **kw):
        asked.append(a[0] if a else "")
        return MagicMock(ask=lambda: None)      # user cancels the intake

    monkeypatch.setattr("questionary.autocomplete", _fake_autocomplete)
    monkeypatch.setattr("questionary.select", lambda *a, **kw: MagicMock(ask=lambda: None))
    monkeypatch.setattr("questionary.text", lambda *a, **kw: MagicMock(ask=lambda: ""))
    monkeypatch.setattr("questionary.confirm", lambda *a, **kw: MagicMock(ask=lambda: False))

    reached_hub = []
    monkeypatch.setattr(wiz, "run_hub", lambda *a, **kw: reached_hub.append(a) or None)

    wiz.run_wizard(prefill={"script_path": "/shared/users/ghost/deleted.py",
                            "gpu_shards": 1, "cpus": 8, "mem_gb": 14})

    assert asked, "a vanished prefill script must drop into the script intake"
    assert not reached_hub, "and must not reach the hub carrying the dead path"


def test_rerun_prefill_keeps_a_script_that_is_still_there(tmp_path, monkeypatch):
    """The good case: a script that still exists goes straight to the hub."""
    import iitgpu.wizard as wiz

    monkeypatch.setenv("NFS_ROOT", str(tmp_path))
    monkeypatch.setenv("IIT_SITE_ENV", "/nonexistent")
    import getpass
    udir = tmp_path / "users" / getpass.getuser()
    udir.mkdir(parents=True)
    script = udir / "train.py"
    script.write_text("print(1)\n")

    def _no_intake(*a, **kw):
        raise AssertionError("a usable prefill script must skip the intake")

    monkeypatch.setattr("questionary.autocomplete", _no_intake)
    seen = {}
    monkeypatch.setattr(wiz, "run_hub",
                        lambda ls, *a, **kw: seen.update(script=ls.script) or None)

    wiz.run_wizard(prefill={"script_path": str(script), "gpu_shards": 1})

    assert seen["script"] == str(script)


def test_batch_flow_reaches_the_hub_with_the_picked_script(tmp_path, monkeypatch):
    """End-to-end through the real run_wizard: intent select -> script intake ->
    hub, with every prompt driven. Cancels at the hub so nothing is submitted."""
    import iitgpu.wizard as wiz

    monkeypatch.setenv("NFS_ROOT", str(tmp_path))
    monkeypatch.setenv("IIT_SITE_ENV", "/nonexistent")
    import getpass
    udir = tmp_path / "users" / getpass.getuser()
    udir.mkdir(parents=True)
    script = udir / "train.py"
    script.write_text("print('hi')\n")

    batch_label = next(l for k, l in wiz._INTENTS if k == "batch")
    selects = iter([batch_label, "Cancel"])
    monkeypatch.setattr("questionary.select",
                        lambda *a, **kw: MagicMock(ask=lambda: next(selects, None)))
    monkeypatch.setattr("questionary.autocomplete",
                        lambda *a, **kw: MagicMock(ask=lambda: str(script)))
    monkeypatch.setattr("questionary.text", lambda *a, **kw: MagicMock(ask=lambda: ""))
    monkeypatch.setattr("questionary.confirm", lambda *a, **kw: MagicMock(ask=lambda: False))
    monkeypatch.setattr("iitgpu.review.get_node_stats", lambda *a, **kw: None)

    def _never(*a, **kw):
        raise AssertionError("cancelling at the hub must not submit anything")
    monkeypatch.setattr(wiz, "submit_job", _never)

    seen = {}
    real_hub = wiz.run_hub
    monkeypatch.setattr(wiz, "run_hub",
                        lambda ls, *a, **kw: seen.update(
                            script=ls.script, intent=ls.intent,
                            shards=ls.gpu_shards, time=ls.time_limit)
                        or real_hub(ls, *a, **kw))

    wiz.run_wizard()

    assert seen["intent"] == "batch"
    assert seen["script"] == str(script)
    assert seen["shards"] == 1 and seen["time"] == "04:00:00"   # Standard default


def test_other_back_returns_to_the_intent_list_not_out_of_the_wizard(monkeypatch):
    """"back" that drops you to the main menu is not back, it is cancel. The
    intent question must be asked again."""
    import iitgpu.wizard as wiz

    questions = []

    def _sel(question, choices=None, **kw):
        questions.append(question)
        # 1st: intent -> Other. 2nd: Other submenu -> back. 3rd: intent -> quit.
        answers = [wiz._OTHER_CHOICE, "back", None]
        return MagicMock(ask=lambda: answers[min(len(questions), 3) - 1])

    monkeypatch.setattr("questionary.select", _sel)
    wiz.run_wizard()

    intent_asked = [q for q in questions if q.startswith("What do you want to do?")]
    assert len(intent_asked) == 2, f"intent list should come back. Asked: {questions}"


def test_other_template_cancel_returns_to_the_intent_list(monkeypatch):
    """Same for backing out of the template picker."""
    import iitgpu.wizard as wiz

    questions = []

    def _sel(question, choices=None, **kw):
        questions.append(question)
        answers = [wiz._OTHER_CHOICE, "Load a template", None]
        return MagicMock(ask=lambda: answers[min(len(questions), 3) - 1])

    monkeypatch.setattr("questionary.select", _sel)
    monkeypatch.setattr("iitgpu.templates.pick_template", lambda cfg: None)
    wiz.run_wizard()

    intent_asked = [q for q in questions if q.startswith("What do you want to do?")]
    assert len(intent_asked) == 2, f"intent list should come back. Asked: {questions}"


# ── Final whole-branch review: I4 (sbatch preview) and I2 (default env) ──────

class _Cfg:
    """Minimal config stand-in for the two helpers under test."""
    def __init__(self, nfs_root, partition="gpu",
                 gateway_host="gw.example", gateway_port="22"):
        self.nfs_root = str(nfs_root)
        self.partition = partition
        self.gateway_host = gateway_host
        self.gateway_port = gateway_port


def test_preview_round_trips_a_batch_spec(tmp_path):
    """I4: "view generated sbatch" now renders through the same to_job_spec +
    renderer + run-command builder the submit path uses."""
    from iitgpu.launchspec import default_spec

    ls = default_spec("batch")
    ls.script = "/shared/users/u/train.py"
    ls.args = "--epochs 3"
    text = wizard._preview_sbatch(ls, _Cfg(tmp_path), "u")
    assert "#SBATCH" in text
    assert "train.py" in text
    assert "python3 /shared/users/u/train.py --epochs 3" in text


def test_preview_round_trips_a_notebook_spec(tmp_path):
    from iitgpu.launchspec import default_spec

    ls = default_spec("notebook")
    ls.conda_env = "/shared/envs/data-science"
    text = wizard._preview_sbatch(ls, _Cfg(tmp_path), "u")
    assert "#SBATCH" in text
    assert "jupyter lab" in text
    assert "conda activate /shared/envs/data-science" in text


def test_preview_writes_nothing_to_disk(tmp_path):
    """Display only — the real job folder is assigned at submit."""
    from iitgpu.launchspec import default_spec

    ls = default_spec("batch")
    ls.script = "/shared/users/u/train.py"
    text = wizard._preview_sbatch(ls, _Cfg(tmp_path), "u")
    assert "<job-folder-assigned-at-submit>" in text
    assert list(tmp_path.iterdir()) == []


def test_run_wizard_hands_the_preview_to_the_hub():
    import inspect
    src = inspect.getsource(wizard.run_wizard)
    assert "preview=" in src and "_hub_preview" in src


def test_default_env_points_at_the_prebuilt_data_science_env(tmp_path):
    """I2: default_spec leaves conda_env empty, which rendered as "(not set)"
    and launched JupyterLab on system python for anyone who accepted the
    defaults. The prebuilt env is now the default when it is installed."""
    from iitgpu.launchspec import default_spec

    (tmp_path / "envs" / "data-science").mkdir(parents=True)
    for intent in ("notebook", "batch"):
        ls = default_spec(intent)
        wizard._apply_default_env(ls, _Cfg(tmp_path))
        assert ls.conda_env == str(tmp_path / "envs" / "data-science")
        assert ls.env_kind == "prebuilt"


def test_default_env_is_left_empty_when_the_env_is_not_installed(tmp_path):
    """Absent env must not error — the empty default is still a valid launch."""
    from iitgpu.launchspec import default_spec

    ls = default_spec("notebook")
    wizard._apply_default_env(ls, _Cfg(tmp_path))
    assert ls.conda_env == ""


def test_default_env_skips_a_shell_and_never_overwrites_a_choice(tmp_path):
    from iitgpu.launchspec import default_spec

    (tmp_path / "envs" / "data-science").mkdir(parents=True)

    shell = default_spec("shell")          # renders no sbatch at all
    wizard._apply_default_env(shell, _Cfg(tmp_path))
    assert shell.conda_env == ""

    from_template = default_spec("batch")  # already carries an environment
    from_template.env_kind, from_template.venv_path = "venv", "/shared/users/u/.venv"
    wizard._apply_default_env(from_template, _Cfg(tmp_path))
    assert from_template.venv_path == "/shared/users/u/.venv"
    assert from_template.conda_env == ""


def test_run_wizard_applies_the_default_env_to_a_fresh_spec():
    """Source-level, per the project's standing trade-off: the helper is wired
    into the intent branch, not merely defined."""
    import inspect
    src = inspect.getsource(wizard.run_wizard)
    assert "_apply_default_env(ls, cfg)" in src


def test_post_submit_wait_is_interruptible_and_cannot_crash_the_session():
    """A plain 90s time.sleep() loop with no terminal input handling once
    swallowed 'q' and made Ctrl-C propagate as an uncaught KeyboardInterrupt,
    crashing the whole TUI process — which is the user's login shell, so it
    took the SSH session down with it. The wait must be wired through
    should_stop, and any interrupt that slips past it must still be caught
    rather than propagate out of this function."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "iitgpu" / "wizard.py").read_text()
    i = src.index("def _post_submit_notebook")
    body = src[i:i + 2000]
    assert "should_stop=" in body
    assert "except KeyboardInterrupt" in body
    assert '"cancelled"' in body


def test_wait_or_keypress_returns_bool_without_a_tty(monkeypatch):
    """Non-interactive contexts (piped input, CI) must degrade to a plain
    sleep rather than raising, matching splash.py's established pattern."""
    import sys
    import iitgpu.wizard as W
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert W._wait_or_keypress(0.01) is False


def test_own_sbatch_auto_opens_the_dashboard_no_confirm(monkeypatch, tmp_path):
    """Every job type auto-directs to the live dashboard after submit — no
    'watch now?' gate. Deliberately does NOT mock questionary.confirm: if a
    confirm gate were still in the code path, calling .ask() on the real
    (un-mocked) prompt in a non-interactive test run raises, so this proves
    the gate is gone structurally, not just that its old text is absent."""
    import iitgpu.wizard as wiz

    monkeypatch.setattr(wiz, "_tier3_own_script", lambda user, cfg: "#!/bin/bash\necho hi\n")
    monkeypatch.setattr(wiz.auditclient, "log_or_block", lambda *a, **kw: True)
    monkeypatch.setattr(wiz.auditclient, "log", lambda *a, **kw: None)
    monkeypatch.setattr(wiz, "submit_job", lambda path: (True, "999"))

    seen = {}
    monkeypatch.setattr(wiz, "make_job_folder", lambda jdir, spec: str(tmp_path))

    class _FakeDashboardModule:
        @staticmethod
        def run_dashboard(job_id=None):
            seen["job_id"] = job_id

    monkeypatch.setitem(__import__("sys").modules, "iitgpu.dashboard", _FakeDashboardModule)

    cfg = type("FakeCfg", (), {"partition": "gpu"})()
    wiz._run_own_sbatch(cfg, "u", str(tmp_path))

    assert seen.get("job_id") == "999"


def test_batch_submit_success_reaches_dashboard_with_no_confirm_between():
    """The audit-success line and the dashboard launch must be adjacent, with
    no questionary.confirm() gate in between — a structural check, not a
    grep for the old prompt's wording."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "iitgpu" / "wizard.py").read_text()
    i = src.index('auditclient.log("job_submitted_ok", detail=job_name, job_id=result)')
    j = src.index("run_dashboard(job_id=result)", i)
    between = src[i:j]
    assert "questionary.confirm" not in between
    assert "poll_until_done" not in between


def test_notebook_submit_success_reaches_dashboard_with_no_confirm_between():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "iitgpu" / "wizard.py").read_text()
    i = src.index("_post_submit_notebook(result, folder)")
    j = src.index("run_dashboard(job_id=result)", i)
    between = src[i:j]
    assert "questionary.confirm" not in between


def test_finetune_choice_offered_between_batch_and_shell():
    """A guided shortcut, not a fourth intent — the menu shows it, but it must
    not be a new value ls.intent can take (that would need every ls.intent ==
    "batch" check in this file updated, and none of them were)."""
    import inspect
    src = inspect.getsource(wizard.run_wizard)
    assert "_FINETUNE_CHOICE" in src
    assert '"finetune"' not in src


def test_finetune_env_prefers_llm_finetune_over_data_science(tmp_path):
    (tmp_path / "envs" / "data-science").mkdir(parents=True)
    (tmp_path / "envs" / "llm-finetune").mkdir(parents=True)
    from iitgpu.launchspec import default_spec

    ls = default_spec("batch")
    wizard._apply_default_env(ls, _Cfg(tmp_path), prefer="llm-finetune")
    assert ls.conda_env == str(tmp_path / "envs" / "llm-finetune")


def test_finetune_env_falls_back_when_the_preferred_env_is_missing(tmp_path):
    """Only data-science installed — prefer= must not leave conda_env empty."""
    (tmp_path / "envs" / "data-science").mkdir(parents=True)
    from iitgpu.launchspec import default_spec

    ls = default_spec("batch")
    wizard._apply_default_env(ls, _Cfg(tmp_path), prefer="llm-finetune")
    assert ls.conda_env == str(tmp_path / "envs" / "data-science")


def test_default_env_without_prefer_is_unchanged(tmp_path):
    """prefer=None (every existing call site) must behave exactly as before."""
    (tmp_path / "envs" / "data-science").mkdir(parents=True)
    from iitgpu.launchspec import default_spec

    ls = default_spec("batch")
    wizard._apply_default_env(ls, _Cfg(tmp_path))
    assert ls.conda_env == str(tmp_path / "envs" / "data-science")


def test_pick_finetune_model_registry(monkeypatch):
    from iitgpu.models import ModelEntry

    entry = ModelEntry(name="mistral", source="huggingface",
                       path="/shared/models/mistral", added_at="2026-01-01",
                       added_by="u", size_mb=1.0)
    monkeypatch.setattr("questionary.select", _select_returning("Pick from registry"))
    monkeypatch.setattr("iitgpu.models.pick_model", lambda cfg: entry)
    assert wizard._pick_finetune_model(object()) == "/shared/models/mistral"


def test_pick_finetune_model_registry_cancelled_returns_empty(monkeypatch):
    monkeypatch.setattr("questionary.select", _select_returning("Pick from registry"))
    monkeypatch.setattr("iitgpu.models.pick_model", lambda cfg: None)
    assert wizard._pick_finetune_model(object()) == ""


def test_pick_finetune_model_huggingface_download(monkeypatch):
    def _text(question, **kw):
        return type("A", (), {"ask": lambda self: "mistralai/Mistral-7B-v0.1"})()

    monkeypatch.setattr("questionary.select", _select_returning("Download from HuggingFace"))
    monkeypatch.setattr("questionary.text", _text)
    monkeypatch.setattr("iitgpu.models.download_hf",
                        lambda cfg, repo_id: (True, "/shared/models/Mistral-7B-v0.1"))
    assert wizard._pick_finetune_model(object()) == "/shared/models/Mistral-7B-v0.1"


def test_pick_finetune_model_path_rejects_outside_jail(monkeypatch, capsys):
    def _text(question, **kw):
        return type("A", (), {"ask": lambda self: "/etc/passwd"})()

    monkeypatch.setattr("questionary.select", _select_returning("Enter a path or HF repo id"))
    monkeypatch.setattr("questionary.text", _text)
    assert wizard._pick_finetune_model(object()) == ""
    assert "outside the allowed jail" in capsys.readouterr().out


def test_pick_finetune_model_skip_returns_empty(monkeypatch):
    monkeypatch.setattr("questionary.select", _select_returning("Skip for now"))
    assert wizard._pick_finetune_model(object()) == ""


def test_finetune_branch_sets_whole_gpu_and_prefers_its_env():
    """Structural: the finetune branch must call apply_size(ls, "whole") and
    thread prefer=_FINETUNE_ENV_NAME into _apply_default_env — the two things
    that make it launch differently from a plain batch job."""
    import inspect
    src = inspect.getsource(wizard.run_wizard)
    i = src.index("_FINETUNE_CHOICE:")
    j = src.index("break", i)
    branch = src[i:j]
    assert 'apply_size(ls, "whole")' in branch
    assert "prefer=_FINETUNE_ENV_NAME" in branch
    assert "_pick_batch_script(" in branch
    assert "_pick_finetune_model(cfg)" in branch
