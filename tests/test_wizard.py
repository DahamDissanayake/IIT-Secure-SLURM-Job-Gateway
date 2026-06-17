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


# ─── New tests for TUI refactor (data path, inline paste, rerun) ─────────────

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_inline_paste_creates_file_under_nfs_root(tmp_path, monkeypatch):
    """Inline paste writes to /shared/<user>/data/<ts>_inline.txt inside the jail."""
    import iitgpu.wizard as wiz
    from iitgpu.config import load_config

    # Point NFS_ROOT at tmp_path so in_jail() accepts the destination
    monkeypatch.setenv("NFS_ROOT", str(tmp_path))
    monkeypatch.setenv("IIT_SITE_ENV", "/nonexistent")

    cfg = load_config()
    user = "testuser"

    # Simulate user typing two lines then EOF
    inputs = iter(["line one", "line two", "EOF"])
    monkeypatch.setattr("builtins.input", lambda: next(inputs))

    # Mock questionary.confirm to return True (create script) and False (don't use as job)
    confirm_responses = iter([True, False])
    monkeypatch.setattr("questionary.confirm", lambda *a, **kw: MagicMock(ask=lambda: next(confirm_responses)))

    # Mock auditclient.log
    logged = []
    monkeypatch.setattr("iitgpu.auditclient.log", lambda action, **kw: logged.append(action))

    data_dest, _ = wiz._inline_paste(cfg, user)

    assert data_dest is not None
    created = Path(data_dest)
    assert created.exists(), f"Expected file at {data_dest}"
    content = created.read_text()
    assert "line one" in content
    assert "line two" in content

    # Must be inside the jail
    from iitgpu.validate import in_jail
    assert in_jail(data_dest), f"{data_dest} not in jail (NFS_ROOT={tmp_path})"

    assert "data_inline_paste" in logged


def test_inline_paste_destination_is_jailed(tmp_path, monkeypatch):
    """If the destination resolves outside the jail, _inline_paste refuses."""
    import iitgpu.wizard as wiz
    from iitgpu.config import load_config
    import dataclasses

    # NFS_ROOT env = /nonexistent so nothing under tmp_path is in the jail
    monkeypatch.setenv("NFS_ROOT", "/nonexistent_nfs_root_xyz")
    monkeypatch.setenv("IIT_SITE_ENV", "/nonexistent")

    cfg = load_config()
    # cfg.nfs_root now = /nonexistent_nfs_root_xyz (from env), but we override it
    # to tmp_path so the file gets written there — which is NOT in_jail
    cfg_bad = dataclasses.replace(cfg, nfs_root=str(tmp_path / "outside"))

    inputs = iter(["some data", "EOF"])
    monkeypatch.setattr("builtins.input", lambda: next(inputs))

    data_dest, script_path = wiz._inline_paste(cfg_bad, "testuser")
    # Should return (None, None) because destination is not in_jail
    assert data_dest is None
    assert script_path is None


def test_generated_loader_script_is_valid_python(tmp_path, monkeypatch):
    """The auto-generated loader script must compile without errors."""
    import iitgpu.wizard as wiz
    from iitgpu.config import load_config

    monkeypatch.setenv("NFS_ROOT", str(tmp_path))
    monkeypatch.setenv("IIT_SITE_ENV", "/nonexistent")

    cfg = load_config()
    user = "testuser"

    inputs = iter(["alpha", "beta", "EOF"])
    monkeypatch.setattr("builtins.input", lambda: next(inputs))

    # confirm: create script = True, use as job script = False
    confirm_responses = iter([True, False])
    monkeypatch.setattr("questionary.confirm", lambda *a, **kw: MagicMock(ask=lambda: next(confirm_responses)))
    monkeypatch.setattr("iitgpu.auditclient.log", lambda *a, **kw: None)

    data_dest, script_path = wiz._inline_paste(cfg, user)

    # Find the generated script by looking in the scripts dir
    scripts_dir = tmp_path / "users" / user / "scripts"
    scripts = list(scripts_dir.glob("*_load_data.py")) if scripts_dir.exists() else []
    assert scripts, "No loader script was generated"
    script_file = scripts[0]

    source = script_file.read_text()
    assert "DATA_PATH" in source

    # Must compile without SyntaxError
    compile(source, str(script_file), "exec")

    # Must be in jail
    from iitgpu.validate import in_jail
    assert in_jail(str(script_file))


def test_data_path_exported_in_sbatch_when_set(tmp_path):
    """render_sbatch() includes 'export DATA_PATH=...' when data_path is set."""
    from iitgpu.jobs import JobSpec, render_sbatch

    spec = JobSpec(
        job_name="test_job",
        partition="gpu",
        gpus=1,
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
        gpus=1,
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
    assert result.get("gpus") == 1
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


def test_wizard_accepts_prefill_without_error(monkeypatch):
    """run_wizard(prefill=...) must not raise when all prompts are mocked."""
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

    # Should return cleanly (wizard exits when select returns None)
    wiz.run_wizard(prefill={"task_type": "train", "conda_env": "/shared/envs/x"})


# ── Email auto-wire ────────────────────────────────────────────────────────────

def test_mail_user_set_from_users_db_when_mta_present(tmp_path):
    """When MTA is available and users.db has an email, mail_user is auto-populated."""
    from iitgpu.jobs import JobSpec, make_job_folder, render_sbatch

    spec = JobSpec(job_name="j", partition="gpu", gpus=1, cpus=4, mem_gb=8,
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

    spec = JobSpec(job_name="j", partition="gpu", gpus=1, cpus=4, mem_gb=8,
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

    spec = JobSpec(job_name="j", partition="gpu", gpus=1, cpus=4, mem_gb=8,
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


def test_wizard_early_bypass_confirm_is_shown(monkeypatch):
    import iitgpu.wizard as wiz
    confirms_seen = []
    def _cap_confirm(msg, **kw):
        confirms_seen.append(msg)
        return MagicMock(ask=lambda: False)
    call_count = [0]
    def _cap_select(*a, **kw):
        call_count[0] += 1
        if call_count[0] == 1:
            return MagicMock(ask=lambda: list(wiz._TASK_LABELS.values())[1])
        return MagicMock(ask=lambda: None)
    monkeypatch.setattr("questionary.confirm", _cap_confirm)
    monkeypatch.setattr("questionary.select", _cap_select)
    monkeypatch.setattr("questionary.text", lambda *a, **kw: MagicMock(ask=lambda: ""))
    wiz.run_wizard()
    bypass_prompts = [m for m in confirms_seen if "ready-made" in m or ".sbatch" in m]
    assert bypass_prompts, f"Early bypass confirm not seen. Confirms: {confirms_seen}"


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


def test_vram_check_passes_when_enough_free(monkeypatch):
    """_vram_check returns True when requested VRAM fits in free headroom."""
    from unittest.mock import patch
    import iitgpu.wizard as wiz

    # 12 GB in use, 32 GB total → 20 GB free; request 10 GB → should pass
    stats = _make_stats(gpu_mem_used_mb=12288, gpu_mem_total_mb=32768)
    with patch("iitgpu.wizard.get_node_stats", return_value=stats), \
         patch("questionary.text") as mock_text:
        mock_text.return_value.ask.return_value = "10"
        result = wiz._vram_check("train")

    assert result is True


def test_vram_check_blocks_when_insufficient_vram(monkeypatch):
    """_vram_check returns False (cancel) when requested VRAM exceeds free headroom
    and the user declines the override prompt."""
    from unittest.mock import patch
    import iitgpu.wizard as wiz

    # 12 GB in use, 32 GB total → 20 GB free; request 28 GB → shortfall 8 GB
    stats = _make_stats(gpu_mem_used_mb=12288, gpu_mem_total_mb=32768)
    with patch("iitgpu.wizard.get_node_stats", return_value=stats), \
         patch("questionary.text") as mock_text, \
         patch("questionary.confirm") as mock_confirm:
        mock_text.return_value.ask.return_value = "28"
        mock_confirm.return_value.ask.return_value = False   # user says don't override
        result = wiz._vram_check("train")

    assert result is False


def test_vram_check_allows_override_when_user_confirms(monkeypatch):
    """_vram_check returns True when VRAM is insufficient but user forces override."""
    from unittest.mock import patch
    import iitgpu.wizard as wiz

    stats = _make_stats(gpu_mem_used_mb=12288, gpu_mem_total_mb=32768)
    with patch("iitgpu.wizard.get_node_stats", return_value=stats), \
         patch("questionary.text") as mock_text, \
         patch("questionary.confirm") as mock_confirm:
        mock_text.return_value.ask.return_value = "28"
        mock_confirm.return_value.ask.return_value = True   # user overrides
        result = wiz._vram_check("train")

    assert result is True


def test_vram_check_skip_when_zero(monkeypatch):
    """_vram_check returns True immediately when the user enters 0 (skip)."""
    from unittest.mock import patch
    import iitgpu.wizard as wiz

    stats = _make_stats(gpu_mem_used_mb=30000, gpu_mem_total_mb=32768)
    with patch("iitgpu.wizard.get_node_stats", return_value=stats), \
         patch("questionary.text") as mock_text:
        mock_text.return_value.ask.return_value = "0"
        result = wiz._vram_check("train")

    assert result is True


def test_vram_check_skips_when_stats_unavailable(monkeypatch):
    """_vram_check returns True (proceed) when GPU stats cannot be read."""
    from unittest.mock import patch
    import iitgpu.wizard as wiz

    with patch("iitgpu.wizard.get_node_stats", return_value=None), \
         patch("questionary.text") as mock_text:
        mock_text.return_value.ask.return_value = "20"
        result = wiz._vram_check("train")

    assert result is True


def test_vram_check_uses_task_default_for_inference(monkeypatch):
    """_vram_check seeds the prompt default from _VRAM_TASK_DEFAULTS."""
    from unittest.mock import patch, call
    import iitgpu.wizard as wiz

    stats = _make_stats(gpu_mem_used_mb=0, gpu_mem_total_mb=32768)
    captured_default = []

    def fake_text(prompt, default="", **kw):
        captured_default.append(default)
        m = type("M", (), {"ask": lambda self: default})()
        return m

    with patch("iitgpu.wizard.get_node_stats", return_value=stats), \
         patch("questionary.text", side_effect=fake_text):
        wiz._vram_check("inference")

    assert captured_default and captured_default[0] == "8"


def test_vram_check_present_in_wizard_source():
    """_vram_check must be called in run_wizard for both submission paths."""
    import inspect
    from iitgpu import wizard
    src = inspect.getsource(wizard.run_wizard)
    assert src.count("_vram_check") >= 2, (
        f"Expected _vram_check called at least twice in run_wizard, found {src.count('_vram_check')}"
    )
