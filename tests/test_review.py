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
    monkeypatch.setattr(R.questionary, "select", _Ask(["Launch"]))
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
    monkeypatch.setattr(R.questionary, "select", _Ask(["Launch", "Cancel"]))
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
    R._edit_data_model(ls, None, lambda: None, lambda: ("", ""))

    assert any("packages" in c for c in seen["choices"])


def test_deps_option_hidden_for_a_plain_script(monkeypatch):
    import iitgpu.review as R
    from iitgpu.launchspec import default_spec

    ls = default_spec("batch")
    ls.script = "/shared/users/u/train.py"
    seen = _captured_choices(monkeypatch, "Data / model")
    R._edit_data_model(ls, None, lambda: None, lambda: ("", ""))

    assert not any("packages" in c for c in seen["choices"])


def test_deps_option_offered_for_a_jupyterlab_session(monkeypatch):
    import iitgpu.review as R
    from iitgpu.launchspec import default_spec

    seen = _captured_choices(monkeypatch, "Data / model")
    R._edit_data_model(default_spec("notebook"), None, lambda: None, lambda: ("", ""))

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
    assert "Launch" in seen["choices"]


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


def test_edit_time_rejects_eight_thirty(monkeypatch):
    """I5: 8:30 is 500 minutes. The old hours>8 check let it through and the
    QOS (MaxWallDurationPerJob=08:00:00, DenyOnLimit) rejected it at sbatch."""
    import iitgpu.review as R
    ls = default_spec("batch")
    before = ls.time_limit
    monkeypatch.setattr(R.questionary, "select", _Fixed("custom (HH:MM)"))
    monkeypatch.setattr(R.questionary, "text", _Fixed("8:30"))
    R._edit_time(ls)
    assert ls.time_limit == before


def test_edit_time_accepts_exactly_the_cluster_maximum(monkeypatch):
    import iitgpu.review as R
    ls = default_spec("batch")
    monkeypatch.setattr(R.questionary, "select", _Fixed("custom (HH:MM)"))
    monkeypatch.setattr(R.questionary, "text", _Fixed("8:00"))
    R._edit_time(ls)
    assert ls.time_limit == "08:00:00"


def test_edit_time_accepts_just_under_the_maximum(monkeypatch):
    import iitgpu.review as R
    ls = default_spec("batch")
    monkeypatch.setattr(R.questionary, "select", _Fixed("custom (HH:MM)"))
    monkeypatch.setattr(R.questionary, "text", _Fixed("7:59"))
    R._edit_time(ls)
    assert ls.time_limit == "07:59:00"


def test_hub_hides_every_no_op_row_for_a_shell(monkeypatch):
    """I3: build_interactive_cmd() consumes partition/cpus/mem/gres/time and
    nothing else, so environment, data/model and the whole Advanced menu were
    fully-functional-looking rows the submit path threw away. Spec §1: a shell
    is size + time."""
    import iitgpu.review as R

    seen = _captured_choices(monkeypatch, "Select:")
    monkeypatch.setattr(R, "get_node_stats", lambda *a, **kw: None)
    assert R.run_hub(default_spec("shell"), None, "u", browse_script=lambda: None,
                     browse_data=lambda: None) is None

    for gone in ("Change environment", "Change data / model", "Advanced…",
                 "Change script", "Change args"):
        assert gone not in seen["choices"], gone
    assert seen["choices"] == ["Launch", "Change size", "Change time limit",
                               "Save as template", "Cancel"]

    panel = _plain(render_hub(default_spec("shell"), None))
    for gone in ("Environment", "Data / model", "Advanced", "Args", "Script"):
        assert gone not in panel, gone
    assert "Size" in panel and "Time limit" in panel


def test_hub_hides_data_model_for_a_session_but_keeps_env_and_packages(monkeypatch):
    """I3: render_notebook_sbatch() never emits data_path/model_path, but it does
    consume the environment and the pip-install block — so those stay, and the
    packages question gets its own row instead of hiding behind Data / model."""
    import iitgpu.review as R

    seen = _captured_choices(monkeypatch, "Select:")
    monkeypatch.setattr(R, "get_node_stats", lambda *a, **kw: None)
    assert R.run_hub(default_spec("notebook"), None, "u", browse_script=lambda: None,
                     browse_data=lambda: None, deps_prompt=lambda: ("", "")) is None

    assert "Change data / model" not in seen["choices"]
    assert "Change environment" in seen["choices"]
    assert "Advanced…" in seen["choices"]
    assert R._PKG_CHOICE in seen["choices"]

    panel = _plain(render_hub(default_spec("notebook"), None))
    assert "Data / model" not in panel
    assert "Environment" in panel and "Packages" in panel and "Advanced" in panel


def test_hub_keeps_every_row_for_a_batch_job(monkeypatch):
    import iitgpu.review as R

    seen = _captured_choices(monkeypatch, "Select:")
    monkeypatch.setattr(R, "get_node_stats", lambda *a, **kw: None)
    ls = default_spec("batch")
    ls.script = "/shared/users/u/train.py"
    assert R.run_hub(ls, None, "u", browse_script=lambda: None,
                     browse_data=lambda: None) is None
    assert seen["choices"] == R._HUB_CHOICES


def test_advanced_hides_array_and_dependency_for_a_session(monkeypatch):
    """I3: neither directive is rendered for a notebook job."""
    import iitgpu.review as R

    seen = _captured_choices(monkeypatch, "Advanced:")
    R._edit_advanced(default_spec("notebook"))
    joined = " ".join(seen["choices"])
    assert "job array" not in joined and "run after job" not in joined
    assert "email notifications" in joined and "view generated sbatch" in joined


def test_advanced_keeps_array_and_dependency_for_a_batch_job(monkeypatch):
    import iitgpu.review as R

    seen = _captured_choices(monkeypatch, "Advanced:")
    R._edit_advanced(default_spec("batch"))
    joined = " ".join(seen["choices"])
    assert "job array" in joined and "run after job" in joined


def test_view_generated_sbatch_prints_the_script(monkeypatch, capsys):
    """I4: the menu item used to print a promise and show nothing."""
    import iitgpu.review as R

    answers = ["view generated sbatch", "back"]

    def _sel(question, choices=None, **kw):
        return type("A", (), {"ask": lambda self: answers.pop(0)})()

    monkeypatch.setattr(R.questionary, "select", _sel)
    R._edit_advanced(default_spec("batch"),
                     lambda ls: "#!/bin/bash\n#SBATCH --job-name=demo\npython3 x.py\n")
    out = capsys.readouterr().out
    assert "#SBATCH --job-name=demo" in out


def test_view_generated_sbatch_without_a_preview_says_so(monkeypatch, capsys):
    import iitgpu.review as R

    answers = ["view generated sbatch", "back"]

    def _sel(question, choices=None, **kw):
        return type("A", (), {"ask": lambda self: answers.pop(0)})()

    monkeypatch.setattr(R.questionary, "select", _sel)
    R._edit_advanced(default_spec("batch"), None)
    out = capsys.readouterr().out
    assert "No sbatch preview is available" in out
    assert "after Launch builds it" not in out


def test_run_hub_passes_the_preview_through_to_advanced(monkeypatch):
    """The callback must actually reach _edit_advanced — the wiring, not the menu."""
    import iitgpu.review as R

    got = {}
    monkeypatch.setattr(R, "get_node_stats", lambda *a, **kw: None)
    monkeypatch.setattr(R, "_edit_advanced",
                        lambda ls, preview=None: got.setdefault("preview", preview))
    answers = ["Advanced…", "Cancel"]
    monkeypatch.setattr(R.questionary, "select", lambda *a, **kw: type(
        "A", (), {"ask": lambda self: answers.pop(0)})())

    def _preview(ls):
        return "#SBATCH"

    ls = default_spec("batch"); ls.script = "/x/y.py"
    R.run_hub(ls, None, "u", browse_script=lambda: None,
              browse_data=lambda: None, preview=_preview)
    assert got["preview"] is _preview


def test_model_path_outside_the_jail_is_rejected(monkeypatch, capsys):
    """M1: the model path is exported verbatim into the job script."""
    import iitgpu.review as R

    ls = default_spec("batch")
    ls.model_path = "/shared/models/keepme"
    monkeypatch.setattr(R.questionary, "select", _Fixed("model: enter a path or HF repo id"))
    monkeypatch.setattr(R.questionary, "text", _Fixed("/etc/passwd"))
    R._edit_data_model(ls, None, lambda: None, None)
    assert ls.model_path == "/shared/models/keepme"
    assert "outside the allowed jail" in capsys.readouterr().out


def test_model_path_inside_the_jail_is_accepted(monkeypatch):
    import iitgpu.review as R

    ls = default_spec("batch")
    monkeypatch.setattr(R.questionary, "select", _Fixed("model: enter a path or HF repo id"))
    monkeypatch.setattr(R.questionary, "text", _Fixed("/shared/models/llama"))
    R._edit_data_model(ls, None, lambda: None, None)
    assert ls.model_path == "/shared/models/llama"


def test_model_path_accepts_a_hugging_face_repo_id(monkeypatch):
    """Not a path, so the jail does not apply — same as the old wizard."""
    import iitgpu.review as R

    ls = default_spec("batch")
    monkeypatch.setattr(R.questionary, "select", _Fixed("model: enter a path or HF repo id"))
    monkeypatch.setattr(R.questionary, "text", _Fixed("mistralai/Mistral-7B-v0.1"))
    R._edit_data_model(ls, None, lambda: None, None)
    assert ls.model_path == "mistralai/Mistral-7B-v0.1"


def test_model_registry_pick_sets_the_model_path(monkeypatch):
    """The registry option restores what the old wizard's option (a) did:
    pick an already-downloaded model without retyping its path."""
    import iitgpu.review as R
    from iitgpu.models import ModelEntry

    ls = default_spec("batch")
    monkeypatch.setattr(R.questionary, "select", _Fixed("model: pick from registry"))
    entry = ModelEntry(name="llama", source="huggingface",
                       path="/shared/models/llama", added_at="2026-01-01",
                       added_by="u", size_mb=1.0)
    monkeypatch.setattr("iitgpu.models.pick_model", lambda cfg: entry)
    R._edit_data_model(ls, object(), lambda: None, None)
    assert ls.model_path == "/shared/models/llama"


def test_model_registry_pick_cancelled_leaves_model_path_unchanged(monkeypatch):
    import iitgpu.review as R

    ls = default_spec("batch")
    ls.model_path = "/shared/models/keepme"
    monkeypatch.setattr(R.questionary, "select", _Fixed("model: pick from registry"))
    monkeypatch.setattr("iitgpu.models.pick_model", lambda cfg: None)
    R._edit_data_model(ls, object(), lambda: None, None)
    assert ls.model_path == "/shared/models/keepme"


def test_model_hf_download_sets_the_model_path_on_success(monkeypatch):
    """Restores the old wizard's option (b): download from HuggingFace inline."""
    import iitgpu.review as R

    ls = default_spec("batch")

    def _text(question, **kw):
        return type("A", (), {"ask": lambda self: "mistralai/Mistral-7B-v0.1"})()

    monkeypatch.setattr(R.questionary, "select", _Fixed("model: download from HuggingFace"))
    monkeypatch.setattr(R.questionary, "text", _text)
    monkeypatch.setattr("iitgpu.models.download_hf",
                        lambda cfg, repo_id: (True, "/shared/models/Mistral-7B-v0.1"))
    R._edit_data_model(ls, object(), lambda: None, None)
    assert ls.model_path == "/shared/models/Mistral-7B-v0.1"


def test_model_hf_download_failure_leaves_model_path_unchanged(monkeypatch):
    import iitgpu.review as R

    ls = default_spec("batch")
    ls.model_path = "/shared/models/keepme"

    def _text(question, **kw):
        return type("A", (), {"ask": lambda self: "bad/repo"})()

    monkeypatch.setattr(R.questionary, "select", _Fixed("model: download from HuggingFace"))
    monkeypatch.setattr(R.questionary, "text", _text)
    monkeypatch.setattr("iitgpu.models.download_hf", lambda cfg, repo_id: (False, ""))
    R._edit_data_model(ls, object(), lambda: None, None)
    assert ls.model_path == "/shared/models/keepme"


def test_container_image_must_be_a_jailed_sif(monkeypatch, capsys):
    """M1: the image path lands unquoted in the apptainer exec line."""
    import iitgpu.review as R
    from pathlib import Path

    monkeypatch.setattr(Path, "is_dir", lambda self: False)   # no prebuilt envs
    ls = default_spec("batch")
    ls.env_kind, ls.conda_env = "prebuilt", "/shared/envs/data-science"
    monkeypatch.setattr(R.questionary, "select", _Fixed("container image (.sif)"))
    monkeypatch.setattr(R.questionary, "text", _Fixed("/etc/evil.sif"))
    R._edit_env(ls, cfg=None)
    assert ls.container_image == ""
    assert ls.conda_env == "/shared/envs/data-science"   # left untouched
    assert ls.env_kind == "prebuilt"
    assert "not a .sif" in capsys.readouterr().out


def test_container_image_valid_sif_is_accepted(monkeypatch):
    import iitgpu.review as R
    from pathlib import Path

    monkeypatch.setattr(Path, "is_dir", lambda self: False)
    ls = default_spec("batch")
    ls.env_kind, ls.conda_env = "prebuilt", "/shared/envs/data-science"
    monkeypatch.setattr(R.questionary, "select", _Fixed("container image (.sif)"))
    monkeypatch.setattr(R.questionary, "text", _Fixed("/shared/images/torch.sif"))
    R._edit_env(ls, cfg=None)
    assert ls.container_image == "/shared/images/torch.sif"
    assert ls.env_kind == "container" and ls.conda_env == ""
