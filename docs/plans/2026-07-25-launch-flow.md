# Launch Flow Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 12–16-prompt linear wizard with a 3-intent intake plus one editable review hub, and make notebooks connectable (readiness marker + Connect card).

**Architecture:** Three new focused modules hold the logic — `iitgpu/launchspec.py` (pure data: sizes, spec, availability, recents, template mapping), `iitgpu/review.py` (hub renderer + field editors), `iitgpu/connect.py` (readiness wait, `.out` parser, Connect card). `iitgpu/wizard.py` slims to intent → intake → hub → existing submit pipeline. Rollout: pure logic first, connect second, the visible flow change last.

**Tech Stack:** Python 3.14, questionary 2.1.1 (`autocomplete` confirmed available), Rich, pytest. SLURM semantics unchanged.

## Global Constraints

- Repo: `~/IIT-Secure-SLURM-Job-Gateway` on the **login node** (`ssh -o BatchMode=yes slurmadmin@192.168.122.10`). All edit/test/commit there. Edit by read-over-ssh → local edit → `cat local | ssh … 'cat > repo/path'`. **No heredocs containing `$(...)`, backticks, `!` or `$VAR`** through the outer shell — write a local file and pipe it.
- Work on branch **`launch-flow`** (created in Task 1). Never commit to `main`.
- Test gate: `python3 -m pytest -q` as `slurmadmin` on the login node. Baseline **663 passing**; each task states its expected count.
- Do not run `deploy/redeploy-igm.sh` and do not submit SLURM jobs, except where Task 8 says so explicitly.
- Spec: `docs/specs/2026-07-25-launch-flow-design.md`. Sizes exactly: Small 1 shard/4 cpu/8 GB/2h · Standard 1/8/14/4h · Whole GPU `SHARDS_PER_GPU`/16/60/8h. Defaults: notebook Standard 6h, batch Standard 4h, shell Small 2h. 8h is the QOS ceiling — label it.
- Preserved invariants (tests pin them): audit action names/order (`log_or_block("job_submit")`, `notebook_submit`, `notebook_submitted_ok`, `notebook_session_start` with `job_id` + `gpu_shards` in `wizard.py`), `$IIT_PORT`/`$IIT_USER_ROOT`/symlink-guard in the notebook renderer, VRAM wording "shared … not enforced", `gpu_share_note` shown to the user.
- `JobSpec` field is `gpu_shards`. `config.admin_group` etc. via `load_config()`; never hardcode paths that `cfg.nfs_root` provides — exception: tests may use literal `/shared` fixtures as existing tests do.
- If a pre-existing test outside the files a task names contradicts the change, **stop and ask the controller** — do not edit it unilaterally. (Task 7 pre-authorizes a specific named list.)

---

### Task 1: `launchspec.py` — sizes, spec, availability, recents, template mapping

**Files:**
- Create: `iitgpu/launchspec.py`
- Test: `tests/test_launchspec.py`
- Also: create the branch first.

**Interfaces:**
- Consumes: `iitgpu.jobs.SHARDS_PER_GPU`, `iitgpu.jobs.JobSpec`, `iitgpu.slurm.NodeStats` (fields `shard_total`, `shard_alloc`).
- Produces (later tasks rely on these exact names):
  - `@dataclass Size(name: str, gpu_shards: int, cpus: int, mem_gb: int, default_time: str)`
  - `SIZES: dict[str, Size]` keys `"small" | "standard" | "whole"`
  - `@dataclass LaunchSpec` — fields: `intent: str` (`"notebook"|"batch"|"shell"`), `script: str=""`, `env_kind: str="prebuilt"` (`"prebuilt"|"conda"|"venv"|"container"|"none"`), `conda_env: str=""`, `venv_path: str=""`, `container_image: str=""`, `gpu_shards: int=1`, `cpus: int=8`, `mem_gb: int=14`, `time_limit: str="04:00:00"`, `data_path: str=""`, `model_path: str=""`, `args: str=""`, `array: str=""`, `dependency: str=""`, `mail: bool=True`, `requirements: str=""`, `packages: str=""`, `port: int=8888`
  - `default_spec(intent: str) -> LaunchSpec`
  - `apply_size(ls: LaunchSpec, name: str) -> None` (sets the triple + `time_limit` to the size default)
  - `size_name_for(ls: LaunchSpec) -> str | None` (exact triple match else `None` = custom)
  - `size_label(ls: LaunchSpec) -> str` — e.g. `"Standard — ¼ GPU · 8 CPU · 14 GB"`, whole: `"Whole GPU — 4/4 GPU · 16 CPU · 60 GB"`, custom: `"Custom — 1/4 GPU · 6 CPU · 20 GB"`
  - `availability_line(stats) -> str` — `"GPU now: 3/4 slices free"` / `"GPU availability unknown"`
  - `size_availability(gpu_shards: int, stats) -> str` — `"— starts now (3 slices free)"` / `"— will queue (needs 4 free, 3 free)"` / `"— availability unknown"`
  - `to_job_spec(ls: LaunchSpec, *, user: str, partition: str, job_name: str, task_type: str, run_command: str = "") -> JobSpec`
  - `recent_scripts(jobs_dir: str, user: str, limit: int = 5) -> list[str]`
  - `from_template(tdata: dict) -> LaunchSpec` and `from_rerun(parsed: dict, script: str) -> LaunchSpec`

- [ ] **Step 1: Create the branch**

```bash
cd ~/IIT-Secure-SLURM-Job-Gateway && git checkout -b launch-flow && git branch --show-current
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_launchspec.py`:

```python
"""Launch flow pure logic: sizes, availability wording, spec derivation."""
from pathlib import Path

import pytest

from iitgpu.jobs import SHARDS_PER_GPU
from iitgpu.launchspec import (
    SIZES, LaunchSpec, apply_size, availability_line, default_spec,
    from_rerun, from_template, recent_scripts, size_availability,
    size_label, size_name_for, to_job_spec,
)
from iitgpu.slurm import NodeStats

NODE_CPUS = 32
NODE_MEM_GB = 60


def _stats(free: int) -> NodeStats:
    return NodeStats(state="MIXED", cpu_load=0.0, cpu_total=32, cpu_alloc=0,
                     mem_total_mb=62000, mem_alloc_mb=0,
                     gpu_total=1, gpu_alloc=0,
                     shard_total=SHARDS_PER_GPU,
                     shard_alloc=SHARDS_PER_GPU - free)


def test_sizes_table_matches_spec():
    assert SIZES["small"].gpu_shards == 1 and SIZES["small"].cpus == 4 and SIZES["small"].mem_gb == 8
    assert SIZES["standard"].cpus == 8 and SIZES["standard"].mem_gb == 14
    assert SIZES["whole"].gpu_shards == SHARDS_PER_GPU and SIZES["whole"].mem_gb == 60


def test_a_full_cards_worth_of_standard_jobs_fits_the_node():
    s = SIZES["standard"]
    n = SHARDS_PER_GPU // s.gpu_shards
    assert s.cpus * n <= NODE_CPUS and s.mem_gb * n <= NODE_MEM_GB


def test_default_specs_per_intent():
    assert default_spec("notebook").time_limit == "06:00:00"
    assert default_spec("batch").time_limit == "04:00:00"
    sh = default_spec("shell")
    assert (sh.gpu_shards, sh.cpus, sh.mem_gb, sh.time_limit) == (1, 4, 8, "02:00:00")


def test_apply_size_and_roundtrip_name():
    ls = default_spec("batch")
    apply_size(ls, "whole")
    assert (ls.gpu_shards, ls.cpus, ls.mem_gb) == (SHARDS_PER_GPU, 16, 60)
    assert size_name_for(ls) == "whole"
    ls.cpus = 6
    assert size_name_for(ls) is None
    assert size_label(ls).startswith("Custom")


def test_availability_wording():
    assert availability_line(_stats(3)) == "GPU now: 3/4 slices free"
    assert availability_line(None) == "GPU availability unknown"
    assert size_availability(1, _stats(3)) == "— starts now (3 slices free)"
    assert size_availability(4, _stats(3)) == "— will queue (needs 4 free, 3 free)"
    assert size_availability(1, None) == "— availability unknown"


def test_to_job_spec_maps_fields():
    ls = default_spec("batch")
    ls.conda_env = "/shared/envs/data-science"; ls.data_path = "/shared/data/cifar10"
    ls.args = "--epochs 5"; ls.array = "0-3"; ls.dependency = "afterok:12"
    js = to_job_spec(ls, user="u", partition="gpu", job_name="train",
                     task_type="custom", run_command="python3 x.py")
    assert js.gpu_shards == ls.gpu_shards and js.cpus == ls.cpus and js.mem_gb == ls.mem_gb
    assert js.conda_env == ls.conda_env and js.data_path == ls.data_path
    assert js.array == "0-3" and js.dependency == "afterok:12"
    assert js.time_limit == "04:00:00" and js.user == "u"


def test_recent_scripts_excludes_scripts_that_no_longer_exist(tmp_path):
    """A recent entry pointing at a deleted file would 404 the user at intake."""
    jobs = tmp_path / "jobs" / "u" / "a_1"
    jobs.mkdir(parents=True)
    (jobs / "job.sbatch").write_text("#!/bin/bash\npython3 /data/proj/deleted.py\n")
    assert recent_scripts(str(tmp_path / "jobs"), "u", limit=5) == []


def test_recent_scripts_returns_existing_newest_first(tmp_path):
    proj = tmp_path / "proj"; proj.mkdir()
    s1 = proj / "one.py"; s2 = proj / "two.ipynb"
    s1.write_text("x"); s2.write_text("y")
    jobs = tmp_path / "jobs" / "u"
    import os, time
    for i, script in enumerate((s1, s2)):
        d = jobs / f"run_{i}"; d.mkdir(parents=True)
        (d / "job.sbatch").write_text(f"python3 {script}\n")
        t = time.time() + i
        os.utime(d, (t, t))
    got = recent_scripts(str(tmp_path / "jobs"), "u", limit=5)
    assert got == [str(s2), str(s1)]


def test_from_template_maps_task_type_and_nearest_size():
    ls = from_template({"task_type": "notebook", "gpu_shards": 1, "cpus": 8,
                        "mem_gb": 14, "time_limit": "08:00:00",
                        "conda_env": "/shared/envs/data-science"})
    assert ls.intent == "notebook" and size_name_for(ls) == "standard"
    assert ls.time_limit == "08:00:00" and ls.conda_env.endswith("data-science")
    ls2 = from_template({"task_type": "train", "gpu_shards": 4, "cpus": 16, "mem_gb": 60})
    assert ls2.intent == "batch" and size_name_for(ls2) == "whole"
    ls3 = from_template({"task_type": "custom", "gpu_shards": 1, "cpus": 6, "mem_gb": 20})
    assert size_name_for(ls3) is None and ls3.cpus == 6


def test_from_rerun_uses_parsed_resources():
    ls = from_rerun({"gpu_shards": 1, "cpus": 4, "mem_gb": 8,
                     "time_limit": "01:00:00", "array": "0-9"}, script="/p/s.py")
    assert ls.intent == "batch" and ls.script == "/p/s.py"
    assert ls.cpus == 4 and ls.array == "0-9"
```

- [ ] **Step 3: Run to verify they fail**

Run: `python3 -m pytest tests/test_launchspec.py -q`
Expected: collection error — `No module named 'iitgpu.launchspec'`.

- [ ] **Step 4: Implement `iitgpu/launchspec.py`**

```python
# iitgpu/launchspec.py — pure launch-flow logic: no prompts, no I/O beyond
# the recents scan. The review hub (review.py) renders what this produces.
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from iitgpu.jobs import SHARDS_PER_GPU, JobSpec


@dataclass(frozen=True)
class Size:
    name: str
    gpu_shards: int
    cpus: int
    mem_gb: int
    default_time: str


SIZES: dict[str, Size] = {
    "small":    Size("Small", 1, 4, 8, "02:00:00"),
    "standard": Size("Standard", 1, 8, 14, "04:00:00"),
    "whole":    Size("Whole GPU", SHARDS_PER_GPU, 16, 60, "08:00:00"),
}

_INTENT_DEFAULTS = {  # (size key, time override or None)
    "notebook": ("standard", "06:00:00"),
    "batch":    ("standard", None),
    "shell":    ("small", None),
}


@dataclass
class LaunchSpec:
    intent: str
    script: str = ""
    env_kind: str = "prebuilt"      # prebuilt | conda | venv | container | none
    conda_env: str = ""
    venv_path: str = ""
    container_image: str = ""
    gpu_shards: int = 1
    cpus: int = 8
    mem_gb: int = 14
    time_limit: str = "04:00:00"
    data_path: str = ""
    model_path: str = ""
    args: str = ""
    array: str = ""
    dependency: str = ""
    mail: bool = True
    requirements: str = ""
    packages: str = ""
    port: int = 8888


def apply_size(ls: LaunchSpec, name: str) -> None:
    s = SIZES[name]
    ls.gpu_shards, ls.cpus, ls.mem_gb = s.gpu_shards, s.cpus, s.mem_gb
    ls.time_limit = s.default_time


def default_spec(intent: str) -> LaunchSpec:
    size_key, time_override = _INTENT_DEFAULTS.get(intent, ("standard", None))
    ls = LaunchSpec(intent=intent)
    apply_size(ls, size_key)
    if time_override:
        ls.time_limit = time_override
    return ls


def size_name_for(ls: LaunchSpec) -> str | None:
    for key, s in SIZES.items():
        if (ls.gpu_shards, ls.cpus, ls.mem_gb) == (s.gpu_shards, s.cpus, s.mem_gb):
            return key
    return None


def _frac(shards: int) -> str:
    if shards >= SHARDS_PER_GPU:
        return f"{SHARDS_PER_GPU}/{SHARDS_PER_GPU} GPU"
    if shards == 1 and SHARDS_PER_GPU == 4:
        return "¼ GPU"
    return f"{shards}/{SHARDS_PER_GPU} GPU"


def size_label(ls: LaunchSpec) -> str:
    key = size_name_for(ls)
    name = SIZES[key].name if key else "Custom"
    return f"{name} — {_frac(ls.gpu_shards)} · {ls.cpus} CPU · {ls.mem_gb} GB"


def _slices_free(stats) -> int | None:
    if stats and getattr(stats, "shard_total", 0):
        return max(0, stats.shard_total - stats.shard_alloc)
    return None


def availability_line(stats) -> str:
    free = _slices_free(stats)
    if free is None:
        return "GPU availability unknown"
    return f"GPU now: {free}/{stats.shard_total} slices free"


def size_availability(gpu_shards: int, stats) -> str:
    free = _slices_free(stats)
    if free is None:
        return "— availability unknown"
    if gpu_shards <= free:
        return f"— starts now ({free} slices free)"
    return f"— will queue (needs {gpu_shards} free, {free} free)"


def to_job_spec(ls: LaunchSpec, *, user: str, partition: str, job_name: str,
                task_type: str, run_command: str = "") -> JobSpec:
    return JobSpec(
        job_name=job_name, partition=partition,
        gpu_shards=ls.gpu_shards, cpus=ls.cpus, mem_gb=ls.mem_gb,
        time_limit=ls.time_limit, run_command=run_command, user=user,
        model_path=ls.model_path, conda_env=ls.conda_env, venv_path=ls.venv_path,
        task_type=task_type, container_image=ls.container_image,
        array=ls.array, dependency=ls.dependency, data_path=ls.data_path,
    )


_SCRIPT_RE = re.compile(r"(/[^\s\"']+\.(?:py|ipynb|sh))\b")


def recent_scripts(jobs_dir: str, user: str, limit: int = 5) -> list[str]:
    """Last distinct scripts referenced by the user's past jobs, newest first.

    Scans <jobs_dir>/<user>/*/job.sbatch for absolute paths ending .py/.ipynb/.sh
    and keeps only those that still exist on disk.
    """
    base = Path(jobs_dir) / user
    if not base.is_dir():
        return []
    seen: list[str] = []
    folders = sorted((d for d in base.iterdir() if d.is_dir()),
                     key=lambda d: d.stat().st_mtime, reverse=True)
    for d in folders:
        sb = d / "job.sbatch"
        if not sb.is_file():
            continue
        try:
            text = sb.read_text()
        except OSError:
            continue
        for m in _SCRIPT_RE.findall(text):
            if m not in seen and Path(m).is_file():
                seen.append(m)
                if len(seen) >= limit:
                    return seen
    return seen


def from_template(tdata: dict) -> LaunchSpec:
    intent = "notebook" if tdata.get("task_type") == "notebook" else "batch"
    ls = default_spec(intent)
    for f_ in ("conda_env", "venv_path", "container_image", "data_path",
               "model_path", "array", "dependency"):
        if tdata.get(f_):
            setattr(ls, f_, tdata[f_])
    for f_ in ("gpu_shards", "cpus", "mem_gb"):
        if tdata.get(f_) is not None:
            setattr(ls, f_, int(tdata[f_]))
    if tdata.get("time_limit"):
        ls.time_limit = tdata["time_limit"]
    if tdata.get("extra_args"):
        ls.args = tdata["extra_args"]
    if tdata.get("run_command") and not ls.args:
        ls.args = ""
    return ls


def from_rerun(parsed: dict, script: str) -> LaunchSpec:
    ls = default_spec("batch")
    ls.script = script
    for f_ in ("gpu_shards", "cpus", "mem_gb"):
        if parsed.get(f_) is not None:
            setattr(ls, f_, int(parsed[f_]))
    for f_ in ("time_limit", "array", "dependency"):
        if parsed.get(f_):
            setattr(ls, f_, parsed[f_])
    return ls
```

- [ ] **Step 5: Run the new tests**

Run: `python3 -m pytest tests/test_launchspec.py -q`
Expected: all pass (10).

- [ ] **Step 6: Full suite**

Run: `python3 -m pytest -q` — Expected: **673 passed** (663 + 10).

- [ ] **Step 7: Commit**

```bash
git add iitgpu/launchspec.py tests/test_launchspec.py
git commit -m "feat(launch): pure launch-flow logic — sizes, availability, recents, compat mapping"
```

---

### Task 2: `connect.py` — `.out` parser, Connect card, readiness wait

**Files:**
- Create: `iitgpu/connect.py`
- Test: `tests/test_connect.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `@dataclass ConnectInfo(port: int, token: str, tunnel: str, url: str)`
  - `parse_connect(out_text: str) -> ConnectInfo | None`
  - `render_card(info: ConnectInfo) -> Panel` (Rich)
  - `marker_path(folder: str) -> Path` → `<folder>/.iit-ready`
  - `wait_ready(folder: str, is_alive, timeout: float = 90.0, poll: float = 2.0) -> str` returning `"ready" | "timeout" | "gone"`; `is_alive: Callable[[], bool]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_connect.py`:

```python
"""Connect card: parse the job's own output — the only authoritative source."""
from pathlib import Path

from rich.console import Console

from iitgpu.connect import ConnectInfo, marker_path, parse_connect, render_card, wait_ready

SAMPLE_OUT = """=================================================
JupyterLab is starting on the GPU node.
Token: bc123979da0269048efef70ff6bfb8fffdbb2ef71827937f
SSH tunnel — open a NEW terminal on YOUR LAPTOP and run:
  ssh -p 2225 -N -L 8930:192.168.122.1:8930 yenuli@10.35.4.100
  (-N = tunnel only, no shell opens — terminal sitting idle is correct)
Then open in browser: http://127.0.0.1:8930/lab?token=bc123979da0269048efef70ff6bfb8fffdbb2ef71827937f
=================================================
"""


def test_parse_connect_extracts_all_fields():
    info = parse_connect(SAMPLE_OUT)
    assert info is not None
    assert info.port == 8930
    assert info.token.startswith("bc1239")
    assert info.tunnel == "ssh -p 2225 -N -L 8930:192.168.122.1:8930 yenuli@10.35.4.100"
    assert info.url == "http://127.0.0.1:8930/lab?token=" + info.token


def test_parse_connect_none_when_not_started_yet():
    assert parse_connect("slurm queued...\n") is None
    assert parse_connect("") is None


def test_render_card_shows_both_steps():
    info = parse_connect(SAMPLE_OUT)
    con = Console(force_terminal=True, width=100)
    with con.capture() as cap:
        con.print(render_card(info))
    out = cap.get()
    assert "ssh -p 2225 -N -L 8930" in out
    assert "http://127.0.0.1:8930/lab?token=" in out
    assert "YOUR laptop" in out


def test_wait_ready_states(tmp_path):
    # ready: marker exists already
    marker_path(str(tmp_path)).touch()
    assert wait_ready(str(tmp_path), is_alive=lambda: True, timeout=1, poll=0.01) == "ready"
    marker_path(str(tmp_path)).unlink()
    # gone: job left RUNNING before marker appeared
    assert wait_ready(str(tmp_path), is_alive=lambda: False, timeout=1, poll=0.01) == "gone"
    # timeout: alive but never ready
    assert wait_ready(str(tmp_path), is_alive=lambda: True, timeout=0.05, poll=0.01) == "timeout"
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_connect.py -q` — Expected: import error, no module.

- [ ] **Step 3: Implement `iitgpu/connect.py`**

```python
# iitgpu/connect.py — post-submit notebook connection: readiness marker wait,
# authoritative parse of the job's own stdout, and the Connect card.
#
# The tunnel line and URL are parsed from the job's .out rather than being
# reconstructed, so this can never repeat the advertised-vs-bound port bug:
# whatever the job printed is what the user gets.
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from rich.panel import Panel

_TUNNEL_RE = re.compile(r"^\s*(ssh -p \d+ -N -L \d+:[\d.]+:\d+ \S+@\S+)\s*$", re.M)
_URL_RE = re.compile(r"(http://127\.0\.0\.1:(\d+)/lab\?token=([0-9a-f]+))")


@dataclass(frozen=True)
class ConnectInfo:
    port: int
    token: str
    tunnel: str
    url: str


def parse_connect(out_text: str) -> ConnectInfo | None:
    mt = _TUNNEL_RE.search(out_text or "")
    mu = _URL_RE.search(out_text or "")
    if not (mt and mu):
        return None
    return ConnectInfo(port=int(mu.group(2)), token=mu.group(3),
                       tunnel=mt.group(1), url=mu.group(1))


def render_card(info: ConnectInfo) -> Panel:
    body = (
        "\n  [bold]1.[/] On [bold]YOUR laptop[/], open a terminal and run:\n"
        f"     [bold cyan]{info.tunnel}[/]\n"
        "     [dim](keeps running; an idle terminal is correct)[/]\n\n"
        "  [bold]2.[/] Then open in your browser:\n"
        f"     [bold green]{info.url}[/]\n"
    )
    return Panel(body, title="[bold] Connect to your JupyterLab [/bold]",
                 border_style="green")


def marker_path(folder: str) -> Path:
    return Path(folder) / ".iit-ready"


def wait_ready(folder: str, is_alive: Callable[[], bool],
               timeout: float = 90.0, poll: float = 2.0) -> str:
    """Wait for the job's readiness marker.

    "ready"   — marker appeared
    "gone"    — is_alive() went False first (job failed/cancelled/finished)
    "timeout" — still alive but no marker within timeout
    """
    deadline = time.monotonic() + timeout
    mp = marker_path(folder)
    while time.monotonic() < deadline:
        if mp.exists():
            return "ready"
        if not is_alive():
            return "gone"
        time.sleep(poll)
    return "ready" if mp.exists() else "timeout"
```

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest tests/test_connect.py -q` — Expected: 4 passed.

- [ ] **Step 5: Full suite** — `python3 -m pytest -q` — Expected: **677 passed**.

- [ ] **Step 6: Commit**

```bash
git add iitgpu/connect.py tests/test_connect.py
git commit -m "feat(connect): parse the job's own output into a Connect card; readiness wait"
```

---

### Task 3: Readiness marker in the notebook renderer

**Files:**
- Modify: `iitgpu/jobs.py` (notebook renderer; the launcher lines are at ~458 and ~497, snippet composition at ~501 — confirm current numbers)
- Test: `tests/test_service_ports.py` (append)

**Interfaces:**
- Consumes: `_free_port_snippet` pattern (existing).
- Produces: generated notebook scripts contain a background `/dev/tcp` watcher that touches `<folder>/.iit-ready`; `render_notebook_sbatch` unchanged signature.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_service_ports.py`)

```python
# ── Readiness marker ─────────────────────────────────────────────────────────

def test_notebook_script_writes_ready_marker_when_port_answers(tmp_path):
    """RUNNING is not "ready": the TUI shows STARTING until the server accepts
    connections. The job itself is the only thing positioned to know, so it
    writes .iit-ready once the port answers — pure bash /dev/tcp, no deps."""
    script = _notebook(tmp_path)
    assert "/dev/tcp/127.0.0.1/$IIT_PORT" in script
    assert f"{tmp_path}/.iit-ready" in script
    # watcher must be backgrounded BEFORE the (blocking) jupyter line
    assert script.index("/dev/tcp") < script.index("jupyter lab --no-browser")


def test_ready_watcher_present_in_container_path_too(tmp_path):
    from iitgpu.jobs import JobSpec, render_notebook_sbatch
    spec = JobSpec(job_name="nb", partition="gpu", gpu_shards=1, cpus=8,
                   mem_gb=14, time_limit="08:00:00", run_command="",
                   task_type="notebook", container_image="/shared/images/x.sif")
    s = render_notebook_sbatch(spec, str(tmp_path), port=8888,
                               gateway_host="gw.edu", gateway_port=2225)
    assert "/dev/tcp/127.0.0.1/$IIT_PORT" in s and ".iit-ready" in s
```

- [ ] **Step 2: Run to verify they fail** — `python3 -m pytest tests/test_service_ports.py -q` — Expected: 2 FAIL (`/dev/tcp` absent).

- [ ] **Step 3: Implement**

In `iitgpu/jobs.py`, next to `_free_port_snippet`, add:

```python
# Background watcher that marks the moment JupyterLab actually accepts
# connections. squeue says RUNNING as soon as the sbatch starts, minutes
# before the server is reachable; the TUI shows STARTING until this marker
# exists. Pure bash /dev/tcp — no curl/nc dependency on the compute node.
def _ready_marker_snippet(folder: str) -> list[str]:
    return [
        "( for _i in $(seq 1 150); do",
        '    if (exec 3<>"/dev/tcp/127.0.0.1/$IIT_PORT") 2>/dev/null; then',
        f'        touch "{folder}/.iit-ready"; break',
        "    fi",
        "    sleep 2",
        "  done ) &",
        "",
    ]
```

In `render_notebook_sbatch`, change the composition line (~501)

```python
    lines += _NODE_ADDR_SNIPPET + _free_port_snippet(port) + _user_home_snippet()
```

to

```python
    lines += (_NODE_ADDR_SNIPPET + _free_port_snippet(port)
              + _user_home_snippet() + _ready_marker_snippet(folder))
```

Note the loop is bounded (150×2s = 5 min) so a never-starting server does not leave a stray subshell for the job's whole lifetime. Do **not** touch the TensorBoard renderer.

- [ ] **Step 4: Verify** — `python3 -m pytest tests/test_service_ports.py tests/test_sharding.py tests/test_notebook.py -q` then `bash -n` a rendered script:

```bash
cd ~/IIT-Secure-SLURM-Job-Gateway && PYTHONPATH=. python3 -c "
from iitgpu.jobs import JobSpec, render_notebook_sbatch
s=JobSpec(job_name='nb',partition='gpu',gpu_shards=1,cpus=8,mem_gb=14,
          time_limit='08:00:00',run_command='',task_type='notebook',
          conda_env='/shared/envs/data-science')
open('/tmp/nbtest.sh','w').write(render_notebook_sbatch(s,'/tmp/x',port=8888,gateway_host='gw',gateway_port=2225))
" && bash -n /tmp/nbtest.sh && echo SYNTAX-OK && rm /tmp/nbtest.sh
```

- [ ] **Step 5: Full suite** — Expected: **679 passed**.

- [ ] **Step 6: Commit**

```bash
git add iitgpu/jobs.py tests/test_service_ports.py
git commit -m "feat(notebook): job writes .iit-ready once the port answers"
```

---

### Task 4: Dashboard — STARTING state and `t` Connect key

**Files:**
- Modify: `iitgpu/dashboard.py` (`_is_jupyter_job` area ~118, `_build_jobs_table` RUNNING branch ~208, key loop `elif key ==` block ~624-691, footer hints ~342-348)
- Modify: `docs/specs/2026-07-25-launch-flow-design.md` (one line: key is `t`, not `C` — `c` is taken by cancel at dashboard.py:640)
- Test: `tests/test_dashboard.py` (append)

**Interfaces:**
- Consumes: `connect.parse_connect`, `connect.render_card`, `connect.marker_path`.
- Produces: `_is_ready_job(job_id: str, jdir: str) -> bool`; jupyter jobs without marker render `STARTING`; key `t` prints the Connect card for a selected jupyter job.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_dashboard.py`)

```python
# ── STARTING state + Connect key ─────────────────────────────────────────────

def test_jupyter_job_without_marker_shows_starting(tmp_path):
    """RUNNING is a scheduler fact, not a usability fact. Until .iit-ready
    exists the user cannot connect, so the table must say STARTING."""
    import re as _re
    from rich.console import Console
    from iitgpu.dashboard import _build_jobs_table
    from iitgpu.slurm import QueueEntry

    j = QueueEntry("21", "notebook", "RUNNING", "gpu", "0:10", 1,
                   user="alice", time_limit="6:00:00")
    table = _build_jobs_table([j], 0, "alice", jupyter_ready={"21": False})
    con = Console(force_terminal=True, width=160)
    with con.capture() as cap:
        con.print(table)
    plain = _re.sub(r"\x1b\[[0-9;]*m", "", cap.get())
    assert "STARTING" in plain and "RUNNING" not in plain


def test_jupyter_job_with_marker_shows_running(tmp_path):
    import re as _re
    from rich.console import Console
    from iitgpu.dashboard import _build_jobs_table
    from iitgpu.slurm import QueueEntry

    j = QueueEntry("22", "notebook", "RUNNING", "gpu", "0:10", 1, user="alice")
    table = _build_jobs_table([j], 0, "alice", jupyter_ready={"22": True})
    con = Console(force_terminal=True, width=160)
    with con.capture() as cap:
        con.print(table)
    plain = _re.sub(r"\x1b\[[0-9;]*m", "", cap.get())
    assert "RUNNING" in plain and "STARTING" not in plain


def test_dashboard_offers_connect_key_for_jupyter_jobs():
    """The connect ritual must be reachable from the dashboard (key t —
    c is taken by cancel)."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "iitgpu" / "dashboard.py").read_text()
    assert 'key == "t"' in src
    assert "parse_connect" in src and "render_card" in src
```

- [ ] **Step 2: Run to verify they fail** — Expected: `_build_jobs_table` has no `jupyter_ready` kwarg; source greps fail.

- [ ] **Step 3: Implement**

In `iitgpu/dashboard.py`:

1. Next to `_is_jupyter_job` add:

```python
def _is_ready_job(job_id: str, jdir: str) -> bool:
    """True when the job's folder contains the .iit-ready marker."""
    log_path = _find_job_log(job_id, jdir)
    if log_path is None:
        return False
    return (Path(log_path).parent / ".iit-ready").exists()
```

2. `_build_jobs_table(jobs, selected_idx, current_user, jupyter_ready=None)` — new optional `jupyter_ready: dict[str, bool] | None`. In the RUNNING/COMPLETING branch, before computing `label`:

```python
            label = "RUNNING" if j.state == "RUNNING" else "FINISHING"
            if jupyter_ready is not None and j.job_id in jupyter_ready:
                if j.state == "RUNNING" and not jupyter_ready[j.job_id]:
                    label = "STARTING"
```

3. In `run_dashboard`'s refresh section (where `_is_jupyter[0]` is set, ~577): also build the dict for visible jupyter jobs:

```python
            _jready = {}
            for _j in jobs:
                if _j.state == "RUNNING" and _is_jupyter_job(_j.job_id, str(Path(jdir) / _j.user)):
                    _jready[_j.job_id] = _is_ready_job(_j.job_id, str(Path(jdir) / _j.user))
```

pass `jupyter_ready=_jready` through `_build_layout` → `_build_jobs_table` (add the kwarg to `_build_layout`'s signature and call).

4. Key handler, after the `elif key == "s"` block:

```python
                elif key == "t" and jobs:
                    sel = jobs[selected_idx] if selected_idx < len(jobs) else None
                    if sel and _is_jupyter[0]:
                        from iitgpu.connect import parse_connect, render_card
                        _log = _find_job_log(sel.job_id, str(Path(jdir) / sel.user))
                        _info = parse_connect(Path(_log).read_text()) if _log else None
                        live.stop()
                        if _info:
                            console.print(render_card(_info))
                        else:
                            console.print("[yellow]Not ready yet — no tunnel info in the job output.[/]")
                        input("Press Enter to return to the dashboard…")
                        live.start()
```

(Match the surrounding code's actual live/console handling — the `e`-key block at ~664 shows the established pattern for pausing the Live display; mirror it exactly.)

5. Footer hint (~345): add `T=connect` next to the existing hints when `_is_jupyter[0]`.

6. In the spec file, §5, change "new key `C` on a selected jupyter job" to "new key `t` on a selected jupyter job (`c` was already cancel)".

- [ ] **Step 4: Run the dashboard tests** — `python3 -m pytest tests/test_dashboard.py -q` — Expected: all pass.

- [ ] **Step 5: Full suite** — Expected: **682 passed**.

- [ ] **Step 6: Commit**

```bash
git add iitgpu/dashboard.py tests/test_dashboard.py docs/specs/2026-07-25-launch-flow-design.md
git commit -m "feat(dashboard): STARTING until .iit-ready; t shows the Connect card"
```

---

### Task 5: Post-submit wait + Connect card in the wizard

**Files:**
- Modify: `iitgpu/wizard.py` (notebook submit success branch — the block containing `notebook_submitted_ok` / `notebook_session_start`, ~917-940)
- Test: `tests/test_wizard.py` (append)

**Interfaces:**
- Consumes: `connect.wait_ready`, `connect.parse_connect`, `connect.render_card`, `connect.marker_path`; `iitgpu.slurm.queue`.
- Produces: helper `_post_submit_notebook(job_id: str, folder: str) -> None` in `wizard.py` (called from the success branch; later kept by Task 7's rewrite).

- [ ] **Step 1: Write the failing test** (append to `tests/test_wizard.py`)

```python
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
```

- [ ] **Step 2: Run to verify it fails** — grep-based, fails on missing helper.

- [ ] **Step 3: Implement** — add to `wizard.py` (module level, near the notebook flow):

```python
def _post_submit_notebook(job_id: str, folder: str) -> None:
    """Wait for the job's readiness marker, then show the Connect card.

    The card is parsed from the job's own stdout — the authoritative source —
    so it cannot disagree with what the server actually bound.
    """
    from iitgpu.connect import parse_connect, render_card, wait_ready
    from iitgpu.slurm import queue as _q
    from iitgpu.ui import console as _con

    def _alive() -> bool:
        return any(e.job_id == str(job_id) and e.state in ("PENDING", "RUNNING")
                   for e in _q(all_users=True))

    info("Starting JupyterLab… (this can take a minute on first launch)")
    state = wait_ready(folder, is_alive=_alive, timeout=90)
    outs = sorted(Path(folder).glob("slurm-*.out"))
    out_text = outs[-1].read_text() if outs else ""
    cinfo = parse_connect(out_text)
    if state == "ready" and cinfo:
        _con.print(render_card(cinfo))
        return
    if state == "gone":
        err("The job ended before JupyterLab came up. Last output:")
        for line in out_text.splitlines()[-15:]:
            info(f"  {line}")
        return
    warn("Still starting after 90s (large envs can be slow).")
    if cinfo:
        _con.print(render_card(cinfo))
        info("The tunnel may not answer until the dashboard shows RUNNING.")
    else:
        info("Watch it in the dashboard — press T on the job for the Connect card.")
```

In the notebook success branch, replace the current four `info(...)` tunnel-shape lines (keep both audit calls exactly as they are) with:

```python
            _post_submit_notebook(result, folder)
            if questionary.confirm(
                "Watch job output now?", default=False, style=_STYLE
            ).ask():
                try:
                    from iitgpu.dashboard import run_dashboard
                    run_dashboard(job_id=result)
                except ImportError:
                    info("Live dashboard not available.")
```

- [ ] **Step 4: Targeted tests** — `python3 -m pytest tests/test_wizard.py -q` — Expected: all pass.

- [ ] **Step 5: Full suite** — Expected: **683 passed**.

- [ ] **Step 6: Commit**

```bash
git add iitgpu/wizard.py tests/test_wizard.py
git commit -m "feat(wizard): post-submit wait + Connect card for notebooks"
```

---

### Task 6: `review.py` — hub renderer and field editors

**Files:**
- Create: `iitgpu/review.py`
- Test: `tests/test_review.py`

**Interfaces:**
- Consumes: `launchspec` (everything from Task 1), `iitgpu.slurm.get_node_stats`, `iitgpu.jobs.gpu_share_note`, wizard helpers passed in as callables (no import of `wizard.py` — avoids a cycle).
- Produces:
  - `render_hub(ls: LaunchSpec, stats) -> Panel` — pure, testable
  - `run_hub(ls: LaunchSpec, cfg, user: str, *, browse_script, browse_data, deps_prompt=None) -> str | None` — loop returning `"launch"`, `"template"`, or `None` (cancel). Mutates `ls` in place. `browse_script`/`browse_data` are the wizard's existing browser callables.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_review.py`:

```python
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
    import iitgpu.review as R

    seq = ["Change size", None, "🚀 Launch"]      # select() calls: hub, size, hub
    size_answers = ["Whole GPU — 4/4 GPU · 16 CPU · 60 GB  — availability unknown"]

    class _Sel:
        def __init__(self): self.n = 0
        def __call__(self, message=None, choices=None, **kw):
            self._current = seq[0] if "launch" not in str(choices).lower() or True else None
            return self
        def ask(self):
            val = seq.pop(0)
            if val is None:               # size editor select
                return size_answers.pop(0)
            return val

    ls = default_spec("batch")
    monkeypatch.setattr(R, "get_node_stats", lambda: None)
    monkeypatch.setattr(R.questionary, "select", _Sel())
    assert run_hub(ls, cfg=None, user="u",
                   browse_script=lambda: None, browse_data=lambda: None) == "launch"
    assert ls.gpu_shards == SHARDS_PER_GPU and ls.cpus == 16 and ls.mem_gb == 60
```

- [ ] **Step 2: Run to verify they fail** — no module `iitgpu.review`.

- [ ] **Step 3: Implement `iitgpu/review.py`**

```python
# iitgpu/review.py — the launch review hub. One screen, every field editable,
# availability shown where the decision is made (RunPod's pattern). Pure
# rendering is split from the loop so tests can capture it.
from __future__ import annotations

import questionary
from rich.panel import Panel

from iitgpu.jobs import gpu_share_note
from iitgpu.launchspec import (SIZES, LaunchSpec, apply_size, availability_line,
                               size_availability, size_label, size_name_for)
from iitgpu.slurm import get_node_stats
from iitgpu.ui import console, info, warn

_STYLE = None  # set by wizard at import-time hookup if desired; questionary default otherwise

_TIME_PRESETS = [("1h", "01:00:00"), ("2h", "02:00:00"),
                 ("4h", "04:00:00"), ("8h (cluster max)", "08:00:00")]


def _fmt_time(t: str) -> str:
    for label, val in _TIME_PRESETS:
        if val == t:
            return label.split()[0]
    return t


def _env_display(ls: LaunchSpec) -> str:
    return (ls.container_image or ls.conda_env or ls.venv_path
            or ("(none — system python)" if ls.env_kind == "none" else "(not set)"))


def render_hub(ls: LaunchSpec, stats) -> Panel:
    rows = []
    if ls.intent == "batch":
        from pathlib import Path
        p = Path(ls.script)
        rows.append(("Script", f"{p.name}   ({p.parent})" if ls.script else "(not set)"))
    rows += [
        ("Environment", _env_display(ls)),
        ("Size", f"{size_label(ls)}   [dim]{size_availability(ls.gpu_shards, stats)}[/]"),
        ("Time limit", _fmt_time(ls.time_limit)),
        ("Data / model", ls.data_path or ls.model_path or "(none)"),
        ("Args", ls.args or "(none)"),
        ("Advanced", "on" if (ls.array or ls.dependency or not ls.mail) else "off"),
    ]
    body = "\n".join(f"  [bold]{k:<12}[/] {v}" for k, v in rows)
    share = gpu_share_note(ls.gpu_shards)
    vram = ("GPU memory is shared between jobs and not enforced — "
            "your fair share is about 8 GB of 32.")
    body += f"\n\n  [dim]{share}[/]\n  [dim]{vram}[/]"
    return Panel(body, title=f"[bold] Ready to launch ─ {availability_line(stats)} [/bold]",
                 border_style="cyan")


def _edit_size(ls: LaunchSpec, stats) -> None:
    choices, mapping = [], {}
    for key in ("standard", "small", "whole"):
        s = SIZES[key]
        probe = LaunchSpec(intent=ls.intent, gpu_shards=s.gpu_shards,
                           cpus=s.cpus, mem_gb=s.mem_gb)
        label = f"{size_label(probe)}  {size_availability(s.gpu_shards, stats)}"
        choices.append(label); mapping[label] = key
    sel = questionary.select("Size:", choices=choices).ask()
    if sel:
        apply_size(ls, mapping[sel])


def _edit_time(ls: LaunchSpec) -> None:
    labels = [l for l, _ in _TIME_PRESETS] + ["custom (HH:MM)"]
    sel = questionary.select("Time limit:", choices=labels).ask()
    if sel is None:
        return
    for label, val in _TIME_PRESETS:
        if sel == label:
            ls.time_limit = val
            return
    raw = questionary.text("Time limit (HH:MM):").ask() or ""
    import re
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", raw.strip())
    if m:
        ls.time_limit = f"{int(m.group(1)):02d}:{m.group(2)}:00"
    else:
        warn("Not HH:MM — keeping the current limit.")


def _edit_env(ls: LaunchSpec, cfg) -> None:
    from pathlib import Path
    envs_dir = Path(getattr(cfg, "nfs_root", "/shared")) / "envs"
    prebuilt = sorted(str(p) for p in envs_dir.iterdir() if p.is_dir()) if envs_dir.is_dir() else []
    choices = [f"prebuilt: {Path(p).name}" for p in prebuilt] + [
        "own conda env (path)", "own venv (path)", "container image (.sif)", "none (system python)"]
    sel = questionary.select("Environment:", choices=choices).ask()
    if sel is None:
        return
    ls.conda_env = ls.venv_path = ls.container_image = ""
    if sel.startswith("prebuilt: "):
        name = sel.split(": ", 1)[1]
        ls.env_kind, ls.conda_env = "prebuilt", str(envs_dir / name)
    elif sel.startswith("own conda"):
        ls.env_kind = "conda"
        ls.conda_env = questionary.text("Conda env path:").ask() or ""
    elif sel.startswith("own venv"):
        ls.env_kind = "venv"
        ls.venv_path = questionary.text("Venv path:").ask() or ""
    elif sel.startswith("container"):
        ls.env_kind = "container"
        ls.container_image = questionary.text("Full path to .sif image:").ask() or ""
    else:
        ls.env_kind = "none"


def _edit_data_model(ls: LaunchSpec, browse_data, deps_prompt) -> None:
    opts = ["data folder (browse)", "clear data", "model path (text)", "clear model"]
    if ls.intent == "notebook" and deps_prompt is not None:
        opts.append("python packages for this session")
    sel = questionary.select("Data / model:", choices=opts + ["back"]).ask()
    if sel == "data folder (browse)":
        picked = browse_data()
        if picked:
            ls.data_path = picked
    elif sel == "clear data":
        ls.data_path = ""
    elif sel == "model path (text)":
        ls.model_path = questionary.text("Model path (or HF repo id):").ask() or ls.model_path
    elif sel == "clear model":
        ls.model_path = ""
    elif sel == "python packages for this session":
        req, pkgs = deps_prompt()
        ls.requirements, ls.packages = req, pkgs


def _edit_args(ls: LaunchSpec) -> None:
    from iitgpu.validate import clean_run_command
    raw = questionary.text("Extra arguments (blank = none):", default=ls.args).ask()
    if raw is not None:
        ls.args = clean_run_command(raw) if raw.strip() else ""


def _edit_advanced(ls: LaunchSpec) -> None:
    from iitgpu.validate import clean_array_spec, clean_dependency
    while True:
        sel = questionary.select("Advanced:", choices=[
            f"job array [{ls.array or 'off'}]",
            f"run after job [{ls.dependency or 'off'}]",
            f"email notifications [{'on' if ls.mail else 'off'}]",
            "view generated sbatch",
            "back",
        ]).ask()
        if sel is None or sel == "back":
            return
        if sel.startswith("job array"):
            raw = questionary.text("Array spec (e.g. 0-9 or 1-100%4, blank = off):",
                                   default=ls.array).ask() or ""
            ls.array = clean_array_spec(raw) or ""
        elif sel.startswith("run after"):
            raw = questionary.text("Parent job ID (blank = off):").ask() or ""
            ls.dependency = (clean_dependency(f"afterok:{raw.strip()}") or ""
                             if raw.strip().isdigit() else "")
        elif sel.startswith("email"):
            ls.mail = not ls.mail
        elif sel.startswith("view generated"):
            info("The final script is shown here after Launch builds it; "
                 "fields above fully determine it.")


_HUB_CHOICES = ["🚀 Launch", "Change script", "Change size", "Change time limit",
                "Change environment", "Change data / model", "Change args",
                "Advanced…", "Save as template", "Cancel"]


def run_hub(ls: LaunchSpec, cfg, user: str, *, browse_script, browse_data,
            deps_prompt=None) -> str | None:
    """Loop until Launch / Save as template / Cancel. Mutates ls in place."""
    while True:
        stats = None
        try:
            stats = get_node_stats()
        except Exception:
            pass
        console.print(render_hub(ls, stats))
        choices = [c for c in _HUB_CHOICES
                   if not (c == "Change script" and ls.intent != "batch")]
        sel = questionary.select("Select:", choices=choices).ask()
        if sel is None or sel == "Cancel":
            return None
        if sel == "🚀 Launch":
            if ls.intent == "batch" and not ls.script:
                warn("Pick a script first (Change script).")
                continue
            return "launch"
        if sel == "Save as template":
            return "template"
        if sel == "Change script":
            picked = browse_script()
            if picked:
                ls.script = picked
        elif sel == "Change size":
            _edit_size(ls, stats)
        elif sel == "Change time limit":
            _edit_time(ls)
        elif sel == "Change environment":
            _edit_env(ls, cfg)
        elif sel == "Change data / model":
            _edit_data_model(ls, browse_data, deps_prompt)
        elif sel == "Change args":
            _edit_args(ls)
        elif sel == "Advanced…":
            _edit_advanced(ls)
```

- [ ] **Step 4: Run the tests** — `python3 -m pytest tests/test_review.py -q` — Expected: 7 passed. (If the `_Sel` scripted test proves brittle against the real call pattern, simplify it to drive `R._edit_size` directly with a patched `questionary.select` — the assertion that matters is `apply_size` wiring.)

- [ ] **Step 5: Full suite** — Expected: **690 passed**.

- [ ] **Step 6: Commit**

```bash
git add iitgpu/review.py tests/test_review.py
git commit -m "feat(review): editable launch hub with availability at the decision point"
```

---

### Task 7: Wizard rewrite — intent → intake → hub → submit

**Files:**
- Modify: `iitgpu/wizard.py` (`run_wizard` ~481-1210 replaced; helpers `_browse_script`, `_browse_data_folder`, `_notebook_deps_prompt`, `_tier3_own_script`, `_vram_check`, `_post_submit_notebook` kept)
- Modify: `tests/test_wizard.py` (named updates below — pre-authorized)
- Test: new assertions in `tests/test_wizard.py`

**Interfaces:**
- Consumes: everything above. `templates.pick_template(cfg) -> dict | None`, `templates.save_template(cfg, name, spec) -> bool`, `monitor._parse_sbatch(text) -> dict`, `questionary.autocomplete`.
- Produces: `run_wizard(prefill: dict | None = None) -> None` — same name/signature; new flow.

**Flow to implement (complete behaviour spec):**

1. Intent select:
   ```
   What do you want to do?
     Open JupyterLab            — interactive notebook on the GPU
     Run a script or notebook   — batch job (.py or .ipynb)
     Open a shell on the GPU node
     ──────────
     Other: my own .sbatch · templates
   ```
   Use `questionary.select` with a `questionary.Separator()` before Other.
2. **Other** → sub-select `["Submit my own .sbatch", "Load a template", "back"]`. Own-sbatch runs the existing bypass block (the `_tier3_own_script` path, moved verbatim into a helper `_run_own_sbatch(cfg, user, jdir)`). Template → `pick_template(cfg)` → `from_template(tdata)` → continue to the hub (batch/notebook intent from mapping).
3. `prefill` (rerun): when given, build `from_rerun(prefill, prefill.get("script_path",""))` and jump straight to the hub.
4. **Batch intake**: `questionary.autocomplete("Script or notebook (.py/.ipynb/.sh) — type a path or pick:", choices=recent_scripts(jdir_base, user) + ["[browse…]"])`. `[browse…]` → `_browse_script` with the existing jail/start logic (reuse the current `_browse_jail` construction). Validate extension against `(".py", ".sh", ".ipynb")`; re-prompt on anything else with the allowed list. `.ipynb` ⇒ internal task_type `notebook-script`, else `custom`.
5. Build `ls = default_spec(intent)`; carry script; for notebook intent set `ls.port = 8888`.
6. `run_hub(ls, cfg, user, browse_script=…, browse_data=…, deps_prompt=…)`:
   - `"launch"` → build and submit (step 7).
   - `"template"` → `questionary.text` name → `save_template(cfg, name, to_job_spec(...))` → back into the hub loop (re-call `run_hub`).
   - `None` → `info("Cancelled."); return`.
7. Submit paths (all reuse existing pipeline code — move, don't rewrite):
   - **shell**: `to_job_spec` → `build_interactive_cmd` → existing confirm + audit (`interactive_start`) + `subprocess.run`.
   - **notebook**: existing notebook block: mail auto-wire (`mta_present`/`email_for`), `make_job_folder`, `.iit-jupyter` marker, `render_notebook_sbatch(spec, folder, port=ls.port, …, requirements=ls.requirements, packages=ls.packages)`, `log_or_block("notebook_submit")`, `submit_job`, `notebook_submitted_ok` + `notebook_session_start` audits, then `_post_submit_notebook(result, folder)`.
   - **batch**: run_command exactly as today (`notebook_run_command(script)` for `.ipynb` incl. auto-install flow defaults; `python3 {shlex.quote(script)} {args}` for `.py`; `bash` for `.sh` — lift the existing construction), mail auto-wire, `render_sbatch`, `log_or_block("job_submit")`, `submit_job`, audits, then the existing dashboard offer.
   - `gpu_share_note` line printed with the submit confirmation (keeps the pinned call site in `wizard.py`).
8. `_vram_check` is **repurposed, name kept** (a test greps `def _vram_check`): it becomes the passive info line used in the pre-submit summary — body must retain the strings "shared" and "not enforced", ask nothing, return `True` always. The hub already shows the same fact; this keeps the wording test green without a prompt.

- [ ] **Step 1: Write the new flow tests first** (append to `tests/test_wizard.py`)

```python
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
```

- [ ] **Step 2: Run to verify they fail**, then **implement the flow above**. Keep every audit call name. Delete the linear Step-1..6 code paths that the hub replaces; keep and reuse the helpers listed in Files.

- [ ] **Step 3: Update the pre-authorized existing tests** (these pinned the linear flow; update ONLY these, exactly as described):
  - `test_gpu_share_note_is_used_by_the_wizard` — keep; must still pass (share note printed at submit).
  - `test_vram_prompt_states_the_budget_is_shared_and_unenforced` — keep; passes via the repurposed `_vram_check`.
  - `test_notebook_submit_audits_the_interactive_session` — keep; the audit block is preserved verbatim.
  - Any test that patches or asserts the removed prompts (`"Step 1 — What are you doing?"`, template-confirm-first, own-sbatch mid-flow confirm): update its setup to drive the new flow, or, where it tested a removed prompt's existence, invert to assert absence. If a failing test is NOT in this category, **stop and ask the controller** — do not adapt submit-pipeline or renderer tests to make them pass.

- [ ] **Step 4: Full suite** — Expected: **≥ 693 passed**, 0 failed. Record the exact count.

- [ ] **Step 5: Commit**

```bash
git add iitgpu/wizard.py tests/test_wizard.py
git commit -m "feat(wizard): 3-intent intake + review hub replace the linear interrogation"
```

---

### Task 8: End-to-end verification (no deploy)

**Files:** none — verification only. Do not merge; do not run `redeploy-igm.sh` (the controller handles merge + deploy via the finishing skill).

- [ ] **Step 1: Full suite on the branch** — `python3 -m pytest -q`, record count.
- [ ] **Step 2: Render + syntax-check both notebook script paths** (conda + container) as in Task 3 Step 4; confirm `.iit-ready` watcher ordering before `jupyter lab`.
- [ ] **Step 3: Hub smoke test without a TTY** — drive `render_hub` against live `get_node_stats()` on the login node:

```bash
cd ~/IIT-Secure-SLURM-Job-Gateway && PYTHONPATH=. python3 -c "
from iitgpu.launchspec import default_spec
from iitgpu.review import render_hub
from iitgpu.slurm import get_node_stats
from rich.console import Console
ls = default_spec('batch'); ls.script='/shared/users/yenuli/train.py'
Console(force_terminal=True, width=100).print(render_hub(ls, get_node_stats()))
"
```

Expected: panel renders with a real availability line.
- [ ] **Step 4: Live notebook acceptance from BRANCH code** (this step MAY submit one job): render a notebook sbatch via branch code as user `yenuli` (the Task 11 pattern from the previous plan: generate as slurmadmin with `PYTHONPATH=~/IIT-Secure-SLURM-Job-Gateway`, write into a `make_job_folder`-created folder, `sbatch` as yenuli), then verify in order: `.iit-ready` appears after the server logs its URL; `parse_connect` on the real `.out` returns the working tunnel/URL; `curl http://192.168.122.1:<port>/lab?token=…` → HTTP 200 from the login node. `scancel` the job and remove its folder afterwards.
- [ ] **Step 5: Report** — counts, acceptance evidence, and any deviations, to the controller.

---

## Self-review record (kept in-plan)

- Spec coverage: §1 intents → T7; §2 intake → T7; §3 hub → T6+T7; §4 sizes → T1; §5 readiness/card/dashboard → T3/T2+T5/T4; §6 modules → T1/T2/T6; §7 compat → T1 (`from_template`/`from_rerun`) + T7 wiring; §8 exclusions respected. Dashboard key deviation (`t` for `C`) documented in T4 and amended in the spec.
- Type consistency: `LaunchSpec` field names match between T1/T6/T7; `run_hub` returns `"launch" | "template" | None` in both T6 and T7; `wait_ready` returns `"ready" | "timeout" | "gone"` in T2 and is handled as such in T5.
- Known risk, called out for the executor: T6's scripted `_Sel` test and T7's test updates are the fragile spots; both carry explicit fallback instructions rather than silent adaptation.
