# GPU Pod Admin Resize (Plan B of 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an admin change the cluster's pod count N from the admin panel (gated on an empty queue), see which running job holds how many pods, and cap how many pods any one user can hold at once — all built on Plan A's live-derived `iitgpu/pods.py`.

**Architecture:** Three additions to `iitgpu/admin.py` (QOS-merge bugfix + per-user pod cap, a read-only pod occupancy grid, then a resize action) plus one new cross-node shell script, `deploy/resize-pods.sh`, that is the only thing that ever touches `gres.conf`/`slurm.conf`. Nothing in the app stores the pod count anywhere — after a resize, every other screen picks up the new number automatically because `iitgpu/pods.py` (Plan A) always reads it live.

**Tech Stack:** Python 3.14, pytest, `bash` (strict mode), existing `sacctmgr`/`scontrol`/`sudo -n` admin-action conventions in `iitgpu/admin.py`.

## Global Constraints

- **Depends on Plan A** (`2026-07-27-gpu-pod-derivation.md`) being merged first — this plan imports `iitgpu.pods.pod_count`/`fits_new_pod_count`.
- Repo: `/home/slurmadmin/IIT-Secure-SLURM-Job-Gateway` on the login node (`ssh slurmadmin@192.168.122.10`), branch `main`.
- Full pytest gate (`python3 -m pytest -q`) must pass before every commit.
- Deploy only via `bash deploy/redeploy-igm.sh` run **as `slurmadmin`, not sudo**.
- Commit author: Daham only, no co-author line. Push to `origin main` without asking, clean fast-forward only.
- `slurmadmin` has passwordless `sudo` on the login node (`(ALL) NOPASSWD: ALL`) and passwordless SSH to the GPU host (`192.168.122.1`) for the handful of GPU-host-only operations (matches the pattern already used for home-directory repairs elsewhere in this project).
- Live config files touched by this plan: `/etc/slurm/gres.conf` and `/etc/slurm/slurm.conf`'s `NodeName=iit-MS-7E06 ... Gres=gpu:1,shard:N` line, on **both** nodes (login node runs `slurmctld`, GPU host runs `slurmd` — both must agree or the node drains).
- A resize must **never run while any job is queued or running**, cluster-wide — this is enforced twice (confirm-time and immediately pre-execution), not just once.
- Existing sudoers convention: `deploy/sudoers-gateway-admin` grants `%gpuadmins ALL=(root) NOPASSWD: /usr/bin/scontrol update *, /usr/bin/scontrol reconfigure, ..., /usr/bin/sacctmgr, /usr/bin/scancel` — this plan adds one more line to that same file for the new resize script, following the exact same pattern (not a new mechanism).

---

### Task 1: Fix the `MaxTRESPerUser` clobber bug + add a per-user pod cap

**Files:**
- Modify: `iitgpu/admin.py:343-388` (`list_qos`, `set_qos_maxgpu`)
- Modify: `iitgpu/admin.py:394-480` (`_qos_menu`, add a "Max pods per user" field)
- Modify: `tests/test_admin.py:230-282` (QOS test block)

**Interfaces:**
- Produces (used by Task 3 indirectly, and standalone in the admin UI): `set_qos_merge_tres(qos_name: str, **tres_updates: int | None) -> tuple[bool, str]`, `set_qos_maxgpu(qos_name, max_gpu: int | None) -> tuple[bool, str]` (same public name/signature as before, now bug-fixed), `set_qos_max_pods_per_user(qos_name: str, max_pods: int | None) -> tuple[bool, str]` (new). `list_qos()` rows gain a `"max_pods"` key.

- [ ] **Step 1: Write the failing tests**

In `tests/test_admin.py`, replace the `# ── QOS ──` section's existing tests (keep `test_set_qos_maxwall_*` and `test_set_qos_priority` unchanged) and add:

```python
_QOS_OUTPUT = "normal|08:00:00|gres/gpu=10,gres/shard=4|0\nlong|7-00:00:00||0\n"


def test_list_qos_parses_sacctmgr_output():
    with patch("subprocess.run", return_value=_proc(out=_QOS_OUTPUT)):
        rows = admin.list_qos()
    assert len(rows) == 2
    normal = rows[0]
    assert normal["name"] == "normal"
    assert normal["max_wall"] == "08:00:00"
    assert normal["max_gpu"] == "10"
    assert normal["max_pods"] == "4"
    assert normal["priority"] == "0"
    long_qos = rows[1]
    assert long_qos["max_wall"] == "7-00:00:00"
    assert long_qos["max_gpu"] == "unlimited"
    assert long_qos["max_pods"] == "unlimited"


def test_set_qos_maxgpu_sets_tres():
    with patch("subprocess.run", return_value=_proc(out="Modified")) as r:
        ok, _ = admin.set_qos_maxgpu("normal", 2)
    write_cmd = r.call_args[0][0]
    assert write_cmd[:3] == ["sudo", "-n", "sacctmgr"]
    assert "MaxTRESPerUser=gres/gpu=2" in write_cmd
    assert ok


def test_set_qos_maxgpu_none_clears_limit():
    with patch("subprocess.run", return_value=_proc(out="Modified")) as r:
        ok, _ = admin.set_qos_maxgpu("long", None)
    write_cmd = r.call_args[0][0]
    assert "MaxTRESPerUser=" in write_cmd
    assert ok


def test_set_qos_maxgpu_preserves_existing_shard_cap():
    """Regression test for the clobber bug: setting the GPU cap must not wipe
    an existing shard (pod) cap already present in MaxTRESPerUser."""
    read_proc = _proc(out="gres/gpu=10,gres/shard=4\n")
    write_proc = _proc(out="Modified")
    with patch("subprocess.run", side_effect=[read_proc, write_proc]) as r:
        ok, _ = admin.set_qos_maxgpu("normal", 2)
    assert ok
    write_cmd = r.call_args_list[-1][0][0]
    tres_arg = next(a for a in write_cmd if a.startswith("MaxTRESPerUser="))
    assert "gres/gpu=2" in tres_arg
    assert "gres/shard=4" in tres_arg


def test_set_qos_max_pods_per_user_preserves_existing_gpu_cap():
    read_proc = _proc(out="gres/gpu=10,gres/shard=4\n")
    write_proc = _proc(out="Modified")
    with patch("subprocess.run", side_effect=[read_proc, write_proc]) as r:
        ok, _ = admin.set_qos_max_pods_per_user("normal", 2)
    assert ok
    write_cmd = r.call_args_list[-1][0][0]
    tres_arg = next(a for a in write_cmd if a.startswith("MaxTRESPerUser="))
    assert "gres/shard=2" in tres_arg
    assert "gres/gpu=10" in tres_arg


def test_set_qos_max_pods_per_user_none_clears_only_shard():
    read_proc = _proc(out="gres/gpu=10,gres/shard=4\n")
    write_proc = _proc(out="Modified")
    with patch("subprocess.run", side_effect=[read_proc, write_proc]) as r:
        ok, _ = admin.set_qos_max_pods_per_user("normal", None)
    assert ok
    write_cmd = r.call_args_list[-1][0][0]
    tres_arg = next(a for a in write_cmd if a.startswith("MaxTRESPerUser="))
    assert "gres/shard" not in tres_arg
    assert "gres/gpu=10" in tres_arg
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/IIT-Secure-SLURM-Job-Gateway && python3 -m pytest tests/test_admin.py -k qos -v`
Expected: `test_set_qos_maxgpu_preserves_existing_shard_cap` and the two `max_pods_per_user` tests FAIL (`AttributeError: module 'iitgpu.admin' has no attribute 'set_qos_max_pods_per_user'`); `test_list_qos_parses_sacctmgr_output` FAILS on the new `max_pods` key.

- [ ] **Step 3: Implement the `iitgpu/admin.py` change**

Replace `list_qos` (lines 343-364) with:

```python
def list_qos() -> list[dict]:
    rc, out, _ = _run(["sacctmgr", "-n", "--parsable2", "show", "qos",
                        "format=Name,MaxWall,MaxTRESPerUser,Priority"])
    rows: list[dict] = []
    for line in out.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 4 or not parts[0].strip():
            continue
        name = parts[0].strip()
        max_wall = parts[1].strip() or "unlimited"
        tres = parts[2].strip()
        max_gpu = "unlimited"
        max_pods = "unlimited"
        for item in tres.split(","):
            if item.startswith("gres/gpu="):
                max_gpu = item.split("=", 2)[-1]
            elif item.startswith("gres/shard="):
                max_pods = item.split("=", 2)[-1]
        rows.append({
            "name": name,
            "max_wall": max_wall,
            "max_gpu": max_gpu,
            "max_pods": max_pods,
            "priority": parts[3].strip() or "0",
        })
    return rows
```

Replace `set_qos_maxgpu` (lines ~375-380) with:

```python
def set_qos_merge_tres(qos_name: str, **tres_updates) -> tuple[bool, str]:
    """Modify one or more components of MaxTRESPerUser without clobbering the
    others. A value of None clears that component; a positive int sets it.

    Fixes a latent bug: the old set_qos_maxgpu() overwrote the WHOLE
    MaxTRESPerUser string, so setting the GPU cap silently dropped any
    existing shard (pod) cap, and vice versa."""
    rc, out, err = _run(["sacctmgr", "-n", "--parsable2", "show", "qos",
                        qos_name, "format=MaxTRESPerUser"])
    if rc != 0:
        return False, (err.strip() or "could not read current MaxTRESPerUser")
    parts: dict[str, str] = {}
    first_line = out.strip().splitlines()[0] if out.strip() else ""
    for item in first_line.split(","):
        if "=" in item:
            k, _, v = item.partition("=")
            parts[k.strip()] = v.strip()
    for key, val in tres_updates.items():
        if val is None:
            parts.pop(key, None)
        else:
            parts[key] = str(val)
    tres = ",".join(f"{k}={v}" for k, v in parts.items())
    rc, out, err = _run(
        ["sudo", "-n", "sacctmgr", "-i", "modify", "qos", qos_name,
         "set", f"MaxTRESPerUser={tres}"], timeout=20)
    auditclient.log("admin_qos_modify", detail=f"{qos_name}:MaxTRESPerUser={tres!r}")
    return (rc == 0), (out.strip() or "updated") if rc == 0 else (err.strip() or "failed")


def set_qos_maxgpu(qos_name: str, max_gpu: int | None) -> tuple[bool, str]:
    return set_qos_merge_tres(qos_name, **{"gres/gpu": max_gpu})


def set_qos_max_pods_per_user(qos_name: str, max_pods: int | None) -> tuple[bool, str]:
    """Caps how many pods (GPU shards) one user can hold across their
    concurrently running jobs on this QOS -- enforced by SLURM itself
    (sacctmgr/slurmctld), not custom application logic."""
    return set_qos_merge_tres(qos_name, **{"gres/shard": max_pods})
```

In `_qos_menu`, extend the field list and table to show/edit the new cap. Change:

```python
        t.add_column("QOS", style="magenta")
        t.add_column("Max Wall Time")
        t.add_column("Max GPUs / User")
        t.add_column("Priority")
        for r in rows:
            t.add_row(r["name"], r["max_wall"], str(r["max_gpu"]), r["priority"])
```

to:

```python
        t.add_column("QOS", style="magenta")
        t.add_column("Max Wall Time")
        t.add_column("Max GPUs / User")
        t.add_column("Max Pods / User")
        t.add_column("Priority")
        for r in rows:
            t.add_row(r["name"], r["max_wall"], str(r["max_gpu"]),
                     str(r["max_pods"]), r["priority"])
```

and the field-selection menu:

```python
        field = select_menu(
            "Field to change:",
            ["Max Wall Time", "Max GPUs per user", "Max Pods per user", "Priority"])
```

adding a new branch alongside the existing `elif field == "Max GPUs per user":` block:

```python
        elif field == "Max Pods per user":
            info(f"  Current: [magenta]{current.get('max_pods', '?')}[/]")
            val = questionary.text(
                "New max pods per user (positive integer; blank = unlimited):",
                style=style).ask()
            if val is None:
                continue
            val = val.strip()
            pods_val: int | None = None
            if val:
                try:
                    pods_val = int(val)
                    if pods_val <= 0:
                        raise ValueError
                except ValueError:
                    err("Enter a positive integer or leave blank."); continue
            if questionary.confirm(
                    f"Set [magenta]{qname}[/] Max Pods per user to "
                    f"[magenta]{pods_val if pods_val is not None else 'unlimited'}[/]?",
                    default=True, style=style).ask():
                good, msg = set_qos_max_pods_per_user(qname, pods_val)
                (ok if good else err)(msg)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_admin.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add iitgpu/admin.py tests/test_admin.py
git -c user.name="Daham Dissanayake" -c user.email="dahamdissanayake05@gmail.com" commit -m "fix(admin): QOS MaxTRESPerUser edits no longer clobber each other

set_qos_maxgpu() used to overwrite the whole MaxTRESPerUser string,
silently dropping any existing shard cap. set_qos_merge_tres() reads
current TRES first and merges. Adds set_qos_max_pods_per_user() and a
'Max Pods per user' field in the QOS admin screen."
```

---

### Task 2: Pod occupancy grid (read-only admin screen)

**Files:**
- Modify: `iitgpu/admin.py` (new `pod_occupancy()` function; new `_pods_menu()` screen; wire into `admin_menu()`'s choice list under "Cluster Control")
- Modify: `tests/test_admin.py`

**Interfaces:**
- Consumes: `iitgpu.pods.{pod_count, pod_resources}` (Plan A).
- Produces (used by Task 4): `pod_occupancy(node: str, total_pods: int) -> list[dict | None]` — a list of length `total_pods`; each cell is `None` (free) or `{"id", "user", "name"}`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_admin.py`:

```python
def test_pod_occupancy_assigns_cells_by_start_time():
    out = ("42|public|train|2026-07-27T08:00:00|gres/shard:2\n"
           "99|daham|nb|2026-07-27T09:00:00|gres/shard:1\n")
    with patch("subprocess.run", return_value=_proc(out=out)):
        cells = admin.pod_occupancy("iit-MS-7E06", total_pods=4)
    assert len(cells) == 4
    assert cells[0]["id"] == "42" and cells[1]["id"] == "42"
    assert cells[2]["id"] == "99"
    assert cells[3] is None


def test_pod_occupancy_all_free_when_no_jobs():
    with patch("subprocess.run", return_value=_proc(out="")):
        cells = admin.pod_occupancy("iit-MS-7E06", total_pods=4)
    assert cells == [None, None, None, None]


def test_pod_occupancy_ignores_jobs_with_no_gpu(): 
    out = "7|public|cpujob|2026-07-27T08:00:00|\n"
    with patch("subprocess.run", return_value=_proc(out=out)):
        cells = admin.pod_occupancy("iit-MS-7E06", total_pods=4)
    assert cells == [None, None, None, None]


def test_pod_occupancy_caps_at_total_pods_if_over_allocated():
    """Defensive: never index past the grid even if SLURM briefly reports
    more shards allocated than the grid has cells for (e.g. mid-resize)."""
    out = "1|a|j|2026-07-27T08:00:00|gres/shard:9\n"
    with patch("subprocess.run", return_value=_proc(out=out)):
        cells = admin.pod_occupancy("iit-MS-7E06", total_pods=4)
    assert len(cells) == 4
    assert all(c is not None and c["id"] == "1" for c in cells)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_admin.py -k pod_occupancy -v`
Expected: FAIL — `AttributeError: module 'iitgpu.admin' has no attribute 'pod_occupancy'`.

- [ ] **Step 3: Implement `pod_occupancy()` and the `_pods_menu()` screen**

Add to `iitgpu/admin.py`, near `get_jobs_on_node`:

```python
def pod_occupancy(node: str, total_pods: int) -> list[dict | None]:
    """Which running job holds each of the node's pod cells, oldest job
    first. Purely a rendering convention: SLURM only tracks a shard COUNT
    per job, not a specific pod identity, so cell assignment is recomputed
    fresh on every call, never persisted."""
    rc, out, _ = _run(["squeue", "--noheader", "--states=RUNNING",
                        "--format=%i|%u|%j|%S|%b", f"--nodelist={node}"])
    cells: list[dict | None] = [None] * total_pods
    if rc != 0 or not out.strip():
        return cells
    jobs = []
    for line in out.strip().splitlines():
        parts = line.split("|")
        if len(parts) != 5:
            continue
        jid, user, name, start, gres = (p.strip() for p in parts)
        shards = 0
        for item in gres.split(","):
            item = item.strip()
            if item.startswith("gres/shard:") or item.startswith("shard:"):
                try:
                    shards = int(item.split(":")[-1].split("(")[0])
                except ValueError:
                    shards = 0
        if shards > 0:
            jobs.append({"id": jid, "user": user, "name": name,
                        "start": start, "shards": shards})
    jobs.sort(key=lambda j: j["start"])
    idx = 0
    for j in jobs:
        for _ in range(j["shards"]):
            if idx >= total_pods:
                break
            cells[idx] = {"id": j["id"], "user": j["user"], "name": j["name"]}
            idx += 1
    return cells
```

Add the screen (near `_qos_menu`):

```python
def _pods_menu(style, node: str = "iit-MS-7E06") -> None:
    from rich.table import Table
    from iitgpu.pods import pod_count, pod_resources
    from iitgpu.slurm import get_node_stats
    from iitgpu.ui import console, info, screen

    stats = get_node_stats(node)
    n = pod_count(stats)
    size = pod_resources(stats)
    cells = pod_occupancy(node, n)

    t = Table(show_header=True, header_style="bold cyan")
    t.add_column("Pod")
    t.add_column("Status")
    t.add_column("Job")
    seen: set[str] = set()
    for i, cell in enumerate(cells, start=1):
        if cell is None:
            t.add_row(str(i), "[dim]free[/]", "")
        else:
            label = "" if cell["id"] in seen else f"{cell['id']} {cell['user']} ({cell['name']})"
            seen.add(cell["id"])
            t.add_row(str(i), "[green]in use[/]", label)
    screen("Pods", status=f"{n} pod(s) configured — {size.cpus} CPU / "
                          f"{size.mem_gb} GB RAM each")
    console.print(t)
    info("Resizing pod count requires zero jobs running cluster-wide (Task 4).")
```

Add `"  Pods (GPU slots)"` to `admin_menu()`'s choice list right after `"  QOS / limits"`, and dispatch it:

```python
        elif choice == "Pods (GPU slots)":
            _pods_menu(style)
            questionary.press_any_key_to_continue("").ask()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_admin.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add iitgpu/admin.py tests/test_admin.py
git -c user.name="Daham Dissanayake" -c user.email="dahamdissanayake05@gmail.com" commit -m "feat(admin): read-only pod occupancy grid in a new Pods screen

Shows the live pod count, per-pod CPU/RAM, and which running job holds
which cells -- assignment is a rendering convention recomputed every
draw, since SLURM only tracks a shard count per job, not per-cell
identity."
```

---

### Task 3: `deploy/resize-pods.sh` — the cross-node resize script

**Files:**
- Create: `deploy/resize-pods.sh`
- Create: `tests/test_resize_pods_script.py` (drives the script via `subprocess` in `--dry-run` mode against a temp directory standing in for `/etc/slurm` — this is the only way to test shell-script logic in this project's pytest suite; matches how other deploy scripts in this repo are exercised indirectly, and is more direct here since the logic is self-contained)
- Modify: `deploy/sudoers-gateway-admin` (one new NOPASSWD line)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_resize_pods_script.py
"""deploy/resize-pods.sh in --dry-run mode: parses/validates without
touching real system files or restarting anything. Full cross-node
execution is verified live (see Plan B Task 5), not in this unit test."""
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "deploy" / "resize-pods.sh"


def _run(args, env):
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True,
                          text=True, env=env, timeout=10)


def _fake_conf_dir(tmp_path):
    d = tmp_path / "slurm"
    d.mkdir()
    (d / "gres.conf").write_text(
        "Name=gpu File=/dev/nvidia0\nName=shard Count=4 File=/dev/nvidia0\n")
    (d / "slurm.conf").write_text(
        "SelectType=select/cons_tres\n"
        "NodeName=iit-MS-7E06 NodeAddr=192.168.122.1 CPUs=32 RealMemory=62000 "
        "Gres=gpu:1,shard:4 State=UNKNOWN\n"
        "PartitionName=gpu Nodes=iit-MS-7E06 Default=YES MaxTime=1-00:00:00 State=UP\n")
    return d


def test_dry_run_rewrites_gres_conf_count_in_place(tmp_path, monkeypatch):
    conf = _fake_conf_dir(tmp_path)
    env = {**dict(**{}), "SLURM_CONF_DIR": str(conf), "PATH": "/usr/bin:/bin"}
    result = _run(["5", "--dry-run"], env)
    assert result.returncode == 0, result.stderr
    assert "Name=shard Count=5" in (conf / "gres.conf").read_text()
    assert "Gres=gpu:1,shard:5" in (conf / "slurm.conf").read_text()


def test_dry_run_backs_up_originals(tmp_path):
    conf = _fake_conf_dir(tmp_path)
    env = {"SLURM_CONF_DIR": str(conf), "PATH": "/usr/bin:/bin"}
    _run(["5", "--dry-run"], env)
    backups = list(conf.glob("gres.conf.bak.*"))
    assert len(backups) == 1


def test_rejects_non_positive_pod_count(tmp_path):
    conf = _fake_conf_dir(tmp_path)
    env = {"SLURM_CONF_DIR": str(conf), "PATH": "/usr/bin:/bin"}
    result = _run(["0", "--dry-run"], env)
    assert result.returncode != 0
    assert "positive" in (result.stderr + result.stdout).lower()


def test_missing_conf_dir_fails_loudly(tmp_path):
    env = {"SLURM_CONF_DIR": str(tmp_path / "nope"), "PATH": "/usr/bin:/bin"}
    result = _run(["5", "--dry-run"], env)
    assert result.returncode != 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_resize_pods_script.py -v`
Expected: FAIL — `deploy/resize-pods.sh` doesn't exist yet (`bash: .../resize-pods.sh: No such file or directory`).

- [ ] **Step 3: Implement `deploy/resize-pods.sh`**

```bash
#!/usr/bin/env bash
# Resize the cluster's GPU pod count (gres/shard Count=N) on both nodes.
# MUST be run with an empty job queue (checked by the admin panel before
# this script is ever invoked; --dry-run skips the live squeue check
# entirely so this script is testable without a real cluster).
#
# Usage: resize-pods.sh <new_pod_count> [--dry-run]
#
# --dry-run: only rewrite the LOCAL $SLURM_CONF_DIR files (gres.conf,
# slurm.conf), skip the squeue check, skip GPU-host SSH, skip restarts.
# Used by tests/test_resize_pods_script.py. Without --dry-run, this
# performs the real cross-node resize and MUST run as slurmadmin on the
# login node.
set -euo pipefail

NEW_N="${1:-}"
DRY_RUN=0
[ "${2:-}" = "--dry-run" ] && DRY_RUN=1

CONF_DIR="${SLURM_CONF_DIR:-/etc/slurm}"
NODE_NAME="${NODE_NAME:-iit-MS-7E06}"
GPU_HOST_SSH="${GPU_HOST_SSH:-root-daham@192.168.122.1}"
LOCK_FILE="${RESIZE_LOCK_FILE:-/var/run/iit-gpu-resize.lock}"

if ! [[ "$NEW_N" =~ ^[0-9]+$ ]] || [ "$NEW_N" -lt 1 ]; then
    echo "ERROR: pod count must be a positive integer, got: '$NEW_N'" >&2
    exit 1
fi

[ -d "$CONF_DIR" ] || { echo "ERROR: $CONF_DIR not found" >&2; exit 1; }
[ -f "$CONF_DIR/gres.conf" ] || { echo "ERROR: $CONF_DIR/gres.conf not found" >&2; exit 1; }
[ -f "$CONF_DIR/slurm.conf" ] || { echo "ERROR: $CONF_DIR/slurm.conf not found" >&2; exit 1; }

if [ "$DRY_RUN" -eq 0 ]; then
    [ "$(id -un)" = "slurmadmin" ] || { echo "ERROR: run as slurmadmin" >&2; exit 1; }

    # Lockfile: refuse a second concurrent resize instead of interleaving.
    exec 9>"$LOCK_FILE"
    flock -n 9 || { echo "ERROR: a resize is already in progress" >&2; exit 1; }

    # Cluster-wide empty-queue check, immediately before touching anything --
    # the admin panel already checked this once at confirm time; this is the
    # atomic re-check right before execution.
    running="$(squeue --noheader --states=RUNNING,PENDING 2>/dev/null | wc -l)"
    if [ "$running" -gt 0 ]; then
        echo "ERROR: $running job(s) still active -- refusing to resize" >&2
        exit 1
    fi
fi

TS="$(date +%Y%m%d%H%M%S)"
cp "$CONF_DIR/gres.conf" "$CONF_DIR/gres.conf.bak.$TS"
cp "$CONF_DIR/slurm.conf" "$CONF_DIR/slurm.conf.bak.$TS"
echo "== backed up gres.conf and slurm.conf ($TS)"

sed -i -E "s/(Name=shard Count=)[0-9]+/\1${NEW_N}/" "$CONF_DIR/gres.conf"
sed -i -E "s/(Gres=gpu:1,shard:)[0-9]+/\1${NEW_N}/" "$CONF_DIR/slurm.conf"
echo "== rewrote local gres.conf/slurm.conf -> Count=$NEW_N"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "== dry-run: stopping before GPU-host sync / restarts"
    exit 0
fi

echo "== syncing gres.conf to the GPU host and restarting slurmd"
scp "$CONF_DIR/gres.conf" "$CONF_DIR/slurm.conf" \
    "${GPU_HOST_SSH}:/tmp/iit-resize-$TS/" 2>/dev/null || {
    mkdir_cmd="mkdir -p /tmp/iit-resize-$TS"
    ssh "$GPU_HOST_SSH" "$mkdir_cmd"
    scp "$CONF_DIR/gres.conf" "$CONF_DIR/slurm.conf" "${GPU_HOST_SSH}:/tmp/iit-resize-$TS/"
}
ssh "$GPU_HOST_SSH" "sudo cp /tmp/iit-resize-$TS/gres.conf /tmp/iit-resize-$TS/slurm.conf /etc/slurm/ && sudo systemctl restart slurmd"

echo "== restarting slurmctld (login node)"
sudo systemctl restart slurmctld

echo "== resuming node (a GRES change reliably drains it)"
sudo scontrol update NodeName="$NODE_NAME" State=RESUME

echo "== verifying"
reported="$(ssh "$GPU_HOST_SSH" "slurmd -G" 2>/dev/null | grep -o 'shard:[0-9]*' | head -1 || true)"
if [ "$reported" != "shard:$NEW_N" ]; then
    echo "ERROR: GPU host reports '$reported', expected 'shard:$NEW_N' -- rolling back" >&2
    cp "$CONF_DIR/gres.conf.bak.$TS" "$CONF_DIR/gres.conf"
    cp "$CONF_DIR/slurm.conf.bak.$TS" "$CONF_DIR/slurm.conf"
    scp "$CONF_DIR/gres.conf" "$CONF_DIR/slurm.conf" "${GPU_HOST_SSH}:/tmp/iit-resize-rollback-$TS/" 2>/dev/null || true
    ssh "$GPU_HOST_SSH" "sudo cp /tmp/iit-resize-rollback-$TS/gres.conf /tmp/iit-resize-rollback-$TS/slurm.conf /etc/slurm/ 2>/dev/null; sudo systemctl restart slurmd" || true
    sudo systemctl restart slurmctld
    sudo scontrol update NodeName="$NODE_NAME" State=RESUME
    echo "ERROR: rollback performed -- resize did NOT apply" >&2
    exit 1
fi

echo "== resize applied: pod count is now $NEW_N"
```

Make it executable: `chmod +x deploy/resize-pods.sh`.

Add one line to `deploy/sudoers-gateway-admin` (following the exact existing pattern in that file):

```
%gpuadmins ALL=(slurmadmin) NOPASSWD: /opt/iit-gpu/deploy/resize-pods.sh *
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_resize_pods_script.py -v`
Expected: all PASS (dry-run mode only — no real SLURM/SSH involved).

- [ ] **Step 5: Commit**

```bash
git add deploy/resize-pods.sh deploy/sudoers-gateway-admin tests/test_resize_pods_script.py
git -c user.name="Daham Dissanayake" -c user.email="dahamdissanayake05@gmail.com" commit -m "feat(deploy): add resize-pods.sh, the cross-node pod-count resize script

Backs up gres.conf/slurm.conf on both nodes before rewriting, restarts
slurmctld then slurmd in that order, resumes the node, and auto-rolls
back both nodes if the GPU host doesn't report the new count. --dry-run
mode makes the parsing/rewrite logic unit-testable without a real
cluster."
```

---

### Task 4: Wire the resize action into the admin Pods screen

**Files:**
- Modify: `iitgpu/admin.py` (`_pods_menu()` gains a resize action; new `resize_pod_count()` function)
- Modify: `tests/test_admin.py`

**Interfaces:**
- Consumes: `iitgpu.pods.fits_new_pod_count` (Plan A), `deploy/resize-pods.sh` (Task 3).
- Produces: `resize_pod_count(new_n: int, node: str = "iit-MS-7E06") -> tuple[bool, str]`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_admin.py`:

```python
def test_resize_pod_count_refuses_when_jobs_running():
    out = "42|public|train|RUNNING\n"
    with patch("subprocess.run", return_value=_proc(out=out)):
        ok, msg = admin.resize_pod_count(5)
    assert not ok
    assert "running" in msg.lower() or "active" in msg.lower()


def test_resize_pod_count_refuses_degenerate_n(monkeypatch):
    from iitgpu.slurm import NodeStats
    stats = NodeStats(state="MIXED", cpu_load=0.0, cpu_total=32, cpu_alloc=0,
                      mem_total_mb=62000, mem_alloc_mb=0, gpu_total=1, gpu_alloc=0,
                      shard_total=4, shard_alloc=0)
    with patch("subprocess.run", return_value=_proc(out="")), \
         patch("iitgpu.admin.get_node_stats", return_value=stats):
        ok, msg = admin.resize_pod_count(40)
    assert not ok
    assert "0 CPU" in msg


def test_resize_pod_count_runs_the_script_when_clear(monkeypatch):
    from iitgpu.slurm import NodeStats
    stats = NodeStats(state="MIXED", cpu_load=0.0, cpu_total=32, cpu_alloc=0,
                      mem_total_mb=62000, mem_alloc_mb=0, gpu_total=1, gpu_alloc=0,
                      shard_total=4, shard_alloc=0)

    def fake_run(cmd, timeout=15, stdin_data=None):
        if cmd[0] == "squeue":
            return 0, "", ""
        assert cmd[-1] == "5"
        assert "resize-pods.sh" in cmd[-2]
        return 0, "resize applied: pod count is now 5", ""

    with patch("iitgpu.admin._run", side_effect=fake_run), \
         patch("iitgpu.admin.get_node_stats", return_value=stats):
        ok, msg = admin.resize_pod_count(5)
    assert ok
    assert "5" in msg
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_admin.py -k resize_pod_count -v`
Expected: FAIL — `AttributeError: module 'iitgpu.admin' has no attribute 'resize_pod_count'`.

- [ ] **Step 3: Implement `resize_pod_count()` and wire it into `_pods_menu()`**

Add to `iitgpu/admin.py` (needs `from iitgpu.slurm import get_node_stats` already imported near the top of the file for other functions — add it if not already present):

```python
def resize_pod_count(new_n: int, node: str = "iit-MS-7E06") -> tuple[bool, str]:
    """Admin action: change the cluster's pod count. Refuses if any job is
    running/queued anywhere, or if new_n would floor CPU/mem per pod to
    zero. Otherwise shells out to resize-pods.sh, which does the actual
    cross-node config rewrite + restart + verify."""
    rc, out, _ = _run(["squeue", "--noheader"])
    if rc == 0 and out.strip():
        n_jobs = len(out.strip().splitlines())
        return False, f"{n_jobs} job(s) still active cluster-wide -- refusing to resize"

    stats = get_node_stats(node)
    if stats is None:
        return False, "Cannot read live node stats -- refusing to resize blind"

    from iitgpu.pods import fits_new_pod_count
    fits, msg = fits_new_pod_count(new_n, stats)
    if not fits:
        return False, msg

    rc, out, err = _run(
        ["sudo", "-n", "/opt/iit-gpu/deploy/resize-pods.sh", str(new_n)], timeout=120)
    auditclient.log("admin_pod_resize", detail=f"new_n={new_n} rc={rc}")
    if rc != 0:
        return False, (err.strip() or out.strip() or "resize failed")
    return True, (out.strip() or f"resize applied: pod count is now {new_n}")
```

In `_pods_menu`, replace the trailing `info("Resizing pod count requires zero jobs running cluster-wide (Task 4).")` line (added as a placeholder in Task 2) with a real resize prompt:

```python
    if questionary.confirm("Resize pod count?", default=False, style=style).ask():
        val = questionary.text(f"New pod count (currently {n}):", style=style).ask()
        try:
            new_n = int((val or "").strip())
        except ValueError:
            err("Enter a whole number."); return
        good, msg = resize_pod_count(new_n)
        (ok if good else err)(msg)
```

(`questionary`, `ok`, `err` are already imported at the top of `admin_menu`'s enclosing scope per the existing file convention — import them the same way inside `_pods_menu` if not already in scope, matching how `_qos_menu` does it.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_admin.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add iitgpu/admin.py tests/test_admin.py
git -c user.name="Daham Dissanayake" -c user.email="dahamdissanayake05@gmail.com" commit -m "feat(admin): wire pod-count resize into the Pods screen

resize_pod_count() gates on an empty cluster-wide queue and a sanity
check (pods.fits_new_pod_count) before shelling out to
deploy/resize-pods.sh. Nothing else needs to know a resize happened --
every other screen re-reads pod count live on its next draw."
```

---

### Task 5: Full regression, deploy, and live resize verification

**Files:** none (verification-only task)

- [ ] **Step 1: Run the full pytest gate**

`cd ~/IIT-Secure-SLURM-Job-Gateway && python3 -m pytest -q`
Expected: all tests pass, including everything from Plan A and this plan.

- [ ] **Step 2: Deploy**

`bash deploy/redeploy-igm.sh` (as `slurmadmin`, not sudo). Confirm `deploy/resize-pods.sh` landed at `/opt/iit-gpu/deploy/resize-pods.sh` and is executable, and that `/etc/sudoers.d/iit-gpu-admin` (the live installed copy of `sudoers-gateway-admin`) picked up the new line: `sudo cat /etc/sudoers.d/iit-gpu-admin | grep resize-pods`.

- [ ] **Step 3: Live-verify a real resize with the queue empty**

As an admin (`dahamadmin`), from the Pods screen (or directly): confirm `squeue` is empty first, then resize 4 → 5, and verify:

```bash
squeue                      # must show nothing
# (perform the resize via the admin panel, or run the script directly as slurmadmin:)
sudo -u slurmadmin bash /opt/iit-gpu/deploy/resize-pods.sh 5
scontrol show node iit-MS-7E06 | grep -o "Gres=[^ ]*"
ssh root-daham@192.168.122.1 "slurmd -G"
```

Expected: both report `shard:5`; node `State` is not `DRAIN`.

- [ ] **Step 4: Live-verify jobs pick up the new pod count with no code changes**

As real provisioned users, submit two jobs each requesting 2 of the new 5 pods and confirm sizing reflects the new split (not the old 4-way numbers):

```bash
sudo -u dahamadmin sbatch --wrap="sleep 60" --gres=shard:2 --partition=gpu -J pod5-a
sudo -u yenuli   sbatch --wrap="sleep 60" --gres=shard:2 --partition=gpu -J pod5-b
squeue -o "%.8i %.20j %.8u %.2t %b"
```

Expected: both `ST=R`, `AllocTRES` sums to `gres/shard=4` (2+2, out of 5 available) with no queuing.

- [ ] **Step 5: Resize back to 4 and confirm rollback path is clean**

```bash
sudo -u slurmadmin bash /opt/iit-gpu/deploy/resize-pods.sh 4
scontrol show node iit-MS-7E06 | grep -o "Gres=[^ ]*"
```

Expected: back to `shard:4`, matching the cluster's normal operating state before this plan started.

---

**Plan A and Plan B together deliver the full feature described in `docs/specs/2026-07-27-gpu-pod-allocation-design.md`.**
