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
