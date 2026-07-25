"""The review hub: one editable screen, availability at the decision point."""
import re

from rich.console import Console

from iitgpu.jobs import SHARDS_PER_GPU
from iitgpu.launchspec import default_spec
from iitgpu.review import render_hub, run_hub
from iitgpu.slurm import NodeStats


def _stats(free=3):
    return NodeStats(state="MIXED", cpu_load=0.0, cpu_total=32, cpu_alloc=0,
                     mem_total_mb=62000, mem_alloc_mb=0, gpu_total=1,
                     gpu_alloc=0, shard_total=SHARDS_PER_GPU,
                     shard_alloc=SHARDS_PER_GPU - free)


def _plain(renderable) -> str:
    con = Console(force_terminal=True, width=120)
    with con.capture() as cap:
        con.print(renderable)
    return re.sub(r"\x1b\[[0-9;]*m", "", cap.get())


def test_hub_shows_every_field_and_availability():
    ls = default_spec("batch")
    ls.script = "/shared/users/u/train.py"
    ls.conda_env = "/shared/envs/data-science"
    out = _plain(render_hub(ls, _stats(3)))
    assert "train.py" in out
    assert "data-science" in out
    assert "Standard" in out and "8 CPU" in out
    assert "3/4 slices free" in out
    assert "4h" in out or "04:00:00" in out


def test_hub_states_vram_is_shared_and_not_enforced():
    """The deleted VRAM quiz is replaced by a passive fact, same pinned wording."""
    out = _plain(render_hub(default_spec("notebook"), _stats(2)))
    assert "shared" in out.lower() and "not enforced" in out.lower()


def test_hub_shows_gpu_share_note():
    from iitgpu.jobs import gpu_share_note
    ls = default_spec("batch")
    out = _plain(render_hub(ls, None))
    assert gpu_share_note(ls.gpu_shards).split("(")[0].strip() in out


def test_hub_availability_unknown_degrades():
    out = _plain(render_hub(default_spec("batch"), None))
    assert "availability unknown" in out.lower()


def test_run_hub_launch_and_cancel(monkeypatch):
    import iitgpu.review as R

    class _Ask:
        def __init__(self, answers): self.answers = list(answers)
        def __call__(self, *a, **kw): return self
        def ask(self): return self.answers.pop(0)

    ls = default_spec("batch"); ls.script = "/x/y.py"
    monkeypatch.setattr(R, "get_node_stats", lambda: None)
    monkeypatch.setattr(R.questionary, "select", _Ask(["🚀 Launch"]))
    assert run_hub(ls, cfg=None, user="u",
                   browse_script=lambda: None, browse_data=lambda: None) == "launch"

    monkeypatch.setattr(R.questionary, "select", _Ask(["Cancel"]))
    assert run_hub(ls, cfg=None, user="u",
                   browse_script=lambda: None, browse_data=lambda: None) is None


def test_run_hub_size_editor_applies_choice(monkeypatch):
    """Fallback per brief note: the scripted _Sel sequence proved brittle
    against the real call pattern (run_hub's post-edit script-guard consumes
    an extra select() call the fixture didn't account for). Drive _edit_size
    directly instead — the wiring under test is apply_size, not the hub loop."""
    import iitgpu.review as R

    class _Ask:
        def __init__(self, answer): self.answer = answer
        def __call__(self, *a, **kw): return self
        def ask(self): return self.answer

    ls = default_spec("batch")
    choice = "Whole GPU — 4/4 GPU · 16 CPU · 60 GB  — availability unknown"
    monkeypatch.setattr(R.questionary, "select", _Ask(choice))
    R._edit_size(ls, None)
    assert ls.gpu_shards == SHARDS_PER_GPU and ls.cpus == 16 and ls.mem_gb == 60


# ── Fix round 1: review findings ─────────────────────────────────────────────

def test_hub_escapes_user_supplied_markup():
    """The one screen whose job is 'show exactly what will launch' must not let
    a filename or --arg containing Rich markup swallow text or apply color."""
    ls = default_spec("batch")
    ls.script = "/u/[experiment]_v2.py"
    ls.args = "--x [red]INJ[/red]"
    out = _plain(render_hub(ls, None))
    assert "[experiment]_v2.py" in out
    # Unescaped, Rich would colourise INJ and the literal "[red]" tag would
    # vanish from the plain capture. Escaped, the raw tag text survives.
    assert "[red]INJ[/red]" in out


class _Fixed:
    """questionary select()/text() stand-in that always answers the same value."""
    def __init__(self, val): self.val = val
    def __call__(self, *a, **kw): return self
    def ask(self): return self.val


def test_edit_time_custom_rejects_minutes_over_59(monkeypatch):
    import iitgpu.review as R
    ls = default_spec("batch")
    before = ls.time_limit
    monkeypatch.setattr(R.questionary, "select", _Fixed("custom (HH:MM)"))
    monkeypatch.setattr(R.questionary, "text", _Fixed("25:99"))
    R._edit_time(ls)
    assert ls.time_limit == before


def test_edit_time_custom_rejects_zero_duration(monkeypatch):
    import iitgpu.review as R
    ls = default_spec("batch")
    before = ls.time_limit
    monkeypatch.setattr(R.questionary, "select", _Fixed("custom (HH:MM)"))
    monkeypatch.setattr(R.questionary, "text", _Fixed("0:00"))
    R._edit_time(ls)
    assert ls.time_limit == before


def test_edit_time_custom_rejects_over_cluster_max(monkeypatch):
    import iitgpu.review as R
    ls = default_spec("batch")
    before = ls.time_limit
    monkeypatch.setattr(R.questionary, "select", _Fixed("custom (HH:MM)"))
    monkeypatch.setattr(R.questionary, "text", _Fixed("09:00"))
    R._edit_time(ls)
    assert ls.time_limit == before


def test_edit_time_custom_accepts_valid_value(monkeypatch):
    import iitgpu.review as R
    ls = default_spec("batch")
    monkeypatch.setattr(R.questionary, "select", _Fixed("custom (HH:MM)"))
    monkeypatch.setattr(R.questionary, "text", _Fixed("03:30"))
    R._edit_time(ls)
    assert ls.time_limit == "03:30:00"


def test_edit_env_handles_permission_error_on_iterdir(monkeypatch):
    """A permission error while listing prebuilt envs must not crash the
    wizard — it should degrade to an empty prebuilt list."""
    import iitgpu.review as R
    from pathlib import Path

    monkeypatch.setattr(Path, "is_dir", lambda self: True)
    monkeypatch.setattr(Path, "iterdir", lambda self: (_ for _ in ()).throw(PermissionError("nope")))
    monkeypatch.setattr(R.questionary, "select", _Fixed(None))  # user backs out
    ls = default_spec("batch")
    R._edit_env(ls, cfg=None)   # must not raise


def test_run_hub_refuses_launch_without_script(monkeypatch, capsys):
    import iitgpu.review as R

    class _Ask:
        def __init__(self, answers): self.answers = list(answers)
        def __call__(self, *a, **kw): return self
        def ask(self): return self.answers.pop(0)

    ls = default_spec("batch")   # script left unset
    monkeypatch.setattr(R, "get_node_stats", lambda: None)
    monkeypatch.setattr(R.questionary, "select", _Ask(["🚀 Launch", "Cancel"]))
    result = run_hub(ls, cfg=None, user="u",
                     browse_script=lambda: None, browse_data=lambda: None)
    assert result is None
    assert "Pick a script first" in capsys.readouterr().out


# ── The hub adapts to the intent ─────────────────────────────────────────────

def _captured_choices(monkeypatch, question_prefix):
    """Record the choices offered for the first select whose question starts
    with *question_prefix*, then cancel it."""
    seen = {}

    def _sel(question, choices=None, **kw):
        if question.startswith(question_prefix) and "choices" not in seen:
            seen["choices"] = list(choices or [])
        return type("A", (), {"ask": lambda self: None})()

    monkeypatch.setattr("questionary.select", _sel)
    return seen


def test_deps_option_offered_for_a_notebook_submitted_as_a_batch_job(monkeypatch):
    """A .ipynb batch job installs its deps before the first cell runs, exactly
    as a JupyterLab session does. Gating the option on intent alone made it
    unreachable for the one flow that renders a pip-install block."""
    import iitgpu.review as R
    from iitgpu.launchspec import default_spec

    ls = default_spec("batch")
    ls.script = "/shared/users/u/analysis.ipynb"
    seen = _captured_choices(monkeypatch, "Data / model")
    R._edit_data_model(ls, lambda: None, lambda: ("", ""))

    assert any("packages" in c for c in seen["choices"])


def test_deps_option_hidden_for_a_plain_script(monkeypatch):
    import iitgpu.review as R
    from iitgpu.launchspec import default_spec

    ls = default_spec("batch")
    ls.script = "/shared/users/u/train.py"
    seen = _captured_choices(monkeypatch, "Data / model")
    R._edit_data_model(ls, lambda: None, lambda: ("", ""))

    assert not any("packages" in c for c in seen["choices"])


def test_deps_option_offered_for_a_jupyterlab_session(monkeypatch):
    import iitgpu.review as R
    from iitgpu.launchspec import default_spec

    seen = _captured_choices(monkeypatch, "Data / model")
    R._edit_data_model(default_spec("notebook"), lambda: None, lambda: ("", ""))

    assert any("packages" in c for c in seen["choices"])


def test_hub_hides_script_and_args_for_non_batch_intents(monkeypatch):
    """A JupyterLab session and a shell have no command line, so a script or an
    argument row would be a setting that silently does nothing."""
    import iitgpu.review as R
    from iitgpu.launchspec import default_spec

    seen = _captured_choices(monkeypatch, "Select:")
    monkeypatch.setattr(R, "get_node_stats", lambda *a, **kw: None)
    assert R.run_hub(default_spec("shell"), None, "u",
                     browse_script=lambda: None, browse_data=lambda: None) is None

    assert "Change args" not in seen["choices"]
    assert "Change script" not in seen["choices"]
    assert "🚀 Launch" in seen["choices"]


def test_hub_keeps_script_and_args_for_a_batch_job(monkeypatch):
    import iitgpu.review as R
    from iitgpu.launchspec import default_spec

    seen = _captured_choices(monkeypatch, "Select:")
    monkeypatch.setattr(R, "get_node_stats", lambda *a, **kw: None)
    ls = default_spec("batch")
    ls.script = "/shared/users/u/train.py"
    assert R.run_hub(ls, None, "u", browse_script=lambda: None,
                     browse_data=lambda: None) is None

    assert "Change args" in seen["choices"]
    assert "Change script" in seen["choices"]


def test_hub_omits_the_args_row_for_a_session():
    """And the summary panel agrees with the menu."""
    from iitgpu.launchspec import default_spec

    assert "Args" not in _plain(render_hub(default_spec("notebook"), None))
    batch = default_spec("batch")
    batch.script = "/s/u/train.py"
    assert "Args" in _plain(render_hub(batch, None))


# ── Final whole-branch review: I1, I3, I4, I5, M1 ────────────────────────────

def _live_stats(total_mb=32768, used_mb=4096, free=3):
    s = _stats(free)
    s.gpu_mem_total_mb = total_mb
    s.gpu_mem_used_mb = used_mb
    s.live_stats = True
    return s


def test_hub_vram_share_scales_with_the_requested_shards():
    """I1: the line used to read "about 8 GB of 32" for every job, so a
    Whole-GPU launch was told it owned the card and got an eighth of it in the
    same panel. The number now comes from the live reading and the shard count."""
    from iitgpu.launchspec import apply_size, default_spec

    one = default_spec("batch")            # Standard == 1 shard
    out_one = _plain(render_hub(one, _live_stats()))
    assert "8 GB of 32" in out_one

    whole = default_spec("batch")
    apply_size(whole, "whole")
    out_whole = _plain(render_hub(whole, _live_stats()))
    assert "32 GB of 32" in out_whole
    # The contradiction the finding was about: whole-GPU must not repeat the
    # one-shard sentence.
    assert "8 GB of 32" not in out_whole
    assert "the whole GPU" in out_whole


def test_hub_vram_line_claims_no_number_without_live_stats():
    """I1: no live reading, no invented figure — but the pinned wording stays."""
    out = _plain(render_hub(default_spec("batch"), _stats(3)))
    assert "shared" in out.lower() and "not enforced" in out.lower()
    assert "GB of" not in out
    assert "8 GB" not in out
