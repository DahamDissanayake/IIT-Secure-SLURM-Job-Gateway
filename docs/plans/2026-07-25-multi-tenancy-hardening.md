# Multi-Tenancy Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `/shared` enforce one access model — everything shared except per-user areas, which belong to their owner and to admins — and stop the notebook and wizard misdescribing what a user gets.

**Architecture:** Permissions are applied by two new shell scripts: a read-only checker that runs anywhere, and a fixer that must run on the GPU host. Application code changes are small and local: `make_job_folder` creates admin-accessible job folders, `render_notebook_sbatch` roots JupyterLab at the user's own folder with symlinked shared assets, and the wizard gains a share note, corrected VRAM wording and an audit event.

**Tech Stack:** Python 3.14, pytest, Rich/questionary TUI, bash deploy scripts, SLURM 25.11.2, NFSv4.

## Global Constraints

- Repo lives at `~/IIT-Secure-SLURM-Job-Gateway` on the **login node** (`ssh slurmadmin@192.168.122.10`). All editing, testing and committing happens there.
- **`root_squash` is set on the NFS export.** Root on the login node cannot modify `/shared`. Every `chown`/`chmod` must run **on the GPU host** (`192.168.122.1`, the NFS server, where `/shared` is a symlink to `/mnt/nvme_storage/shared`). Reads and `stat` work fine from either node.
- Test gate: `python3 -m pytest -q` on the login node **as `slurmadmin`**. Baseline is 648 passing.
- Deploy with `bash deploy/redeploy-igm.sh` **as `slurmadmin`, never via sudo** — sudo breaks the self-reexec guard and skips post-gate steps.
- Per-user areas are mode `2770`, group `gpuadmins`. Shared asset dirs are mode `2775`. `other` never gets write, and never gets anything on per-user areas.
- `public` is **not** exempt — it is a normal account under this model.
- Admin group name comes from `config.admin_group` (env `ADMIN_GROUP`, default `gpuadmins`). Do not hardcode it in Python.
- Never run live job tests as `root-daham`; it is not a provisioned SLURM user and `sbatch` fails with `fetch_identity()`. Use a real account (`sudo -u dahamadmin`).

---

### Task 1: Sync `gpuadmins` membership on the GPU host

`gpuadmins` currently has four members on the login node but only `daham` on the GPU host. Jobs and notebooks run on the GPU host, so `dahamadmin` would be locked out of user areas exactly where it matters. Everything else in this plan depends on this.

**Files:**
- Create: `deploy/sync-admin-group.sh`

**Interfaces:**
- Consumes: nothing.
- Produces: `deploy/sync-admin-group.sh`, run as root on the GPU host; idempotent.

- [ ] **Step 1: Write the verification command and confirm it currently fails**

Run on the GPU host:

```bash
for u in slurmadmin dahamadmin indrajith; do
  printf "%-12s " "$u"; id -nG "$u" 2>/dev/null | tr ' ' '\n' | grep -qx gpuadmins && echo in || echo MISSING
done
```

Expected now: all three print `MISSING`. Note `slurmadmin` has no account on
the GPU host at all — the script skips it, which is correct; only `dahamadmin`,
`indrajith` and `daham` can be members there.

- [ ] **Step 2: Write the sync script**

Create `deploy/sync-admin-group.sh`:

```bash
#!/usr/bin/env bash
# Ensure the admin group has the same members on this node as on the login node.
# Per-user areas are mode 2770 group gpuadmins, so an admin missing from this
# group on the GPU host cannot reach user data where jobs actually run.
set -euo pipefail

ADMIN_GROUP="${ADMIN_GROUP:-gpuadmins}"
ADMINS="${ADMINS:-slurmadmin dahamadmin indrajith daham}"

[ "$(id -u)" -eq 0 ] || { echo "ERROR: run as root on the GPU host" >&2; exit 1; }
getent group "$ADMIN_GROUP" >/dev/null || { echo "ERROR: group $ADMIN_GROUP missing" >&2; exit 1; }

changed=0
for u in $ADMINS; do
    if ! id -u "$u" >/dev/null 2>&1; then
        echo "  skip $u (no such account on this node)"
        continue
    fi
    if id -nG "$u" | tr ' ' '\n' | grep -qx "$ADMIN_GROUP"; then
        echo "  ok   $u already in $ADMIN_GROUP"
    else
        usermod -aG "$ADMIN_GROUP" "$u"
        echo "  ADD  $u -> $ADMIN_GROUP"
        changed=1
    fi
done

echo "admin group sync complete (changed=$changed)"
```

- [ ] **Step 3: Copy to the GPU host and run it**

From the login node:

```bash
scp deploy/sync-admin-group.sh root-daham@192.168.122.1:/tmp/
ssh root-daham@192.168.122.1 "sudo bash /tmp/sync-admin-group.sh"
```

Expected: `ADD slurmadmin`, `ADD dahamadmin`, `ADD indrajith`, `ok daham`.

- [ ] **Step 4: Re-run the verification from Step 1**

Expected: all three now print `in`.

- [ ] **Step 5: Confirm it is idempotent**

Run the script a second time. Expected: every line says `ok`, `changed=0`.

- [ ] **Step 6: Commit**

```bash
git add deploy/sync-admin-group.sh
git commit -m "feat(deploy): sync admin group membership on the GPU host

gpuadmins had only daham on the GPU host but four members on the login node,
so dahamadmin was not an admin on the node where jobs and notebooks actually
run. Per-user areas are group gpuadmins, so this must match on both nodes."
```

---

### Task 2: Job folders become owner + admin only

**Files:**
- Modify: `iitgpu/jobs.py:107-121` (`make_job_folder`)
- Test: `tests/test_jobs.py`

**Interfaces:**
- Consumes: `config.admin_group` (already exists, default `gpuadmins`).
- Produces: `make_job_folder(jobs_dir: str, spec: JobSpec) -> str` — unchanged signature; now creates mode `0o2770` owned group `admin_group`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_jobs.py`:

```python
def _folder_spec():
    from iitgpu.jobs import JobSpec
    return JobSpec(job_name="j", partition="gpu", gpu_shards=1, cpus=1,
                   mem_gb=1, time_limit="", run_command="", user="tester")


def test_job_folder_is_owner_and_admin_only(tmp_path):
    """Other users must not reach a job folder: it holds the user's scripts."""
    import os
    from iitgpu.jobs import make_job_folder
    folder = make_job_folder(str(tmp_path), _folder_spec())
    assert os.stat(folder).st_mode & 0o7777 == 0o2770


def test_job_folder_group_is_the_admin_group(tmp_path, monkeypatch):
    """Group must be the admin group so admins can support users' jobs."""
    import iitgpu.jobs as J
    seen = {}

    class _Grp:
        gr_gid = 4242

    def _fake_getgrnam(name):
        seen["group"] = name
        return _Grp

    monkeypatch.setattr(J.grp, "getgrnam", _fake_getgrnam)
    monkeypatch.setattr(J.os, "chown", lambda p, uid, gid: seen.__setitem__("gid", gid))
    make_job_folder(str(tmp_path), _folder_spec())
    assert seen["group"] == "gpuadmins"
    assert seen["gid"] == 4242
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python3 -m pytest tests/test_jobs.py -k "job_folder" -q`
Expected: FAIL — mode is `0o770` not `0o2770`, and group is `gpuusers` not `gpuadmins`.

- [ ] **Step 3: Change the implementation**

In `iitgpu/jobs.py`, replace lines 111-118 (the comment block through `os.chown`):

```python
    # 2770: owner + admin group only. A job folder holds the user's scripts and
    # output — the same private content as their home — so other users get
    # nothing. setgid keeps the group on anything created inside, so admin
    # access still works for files written later by the job itself.
    folder.chmod(0o2770)
    try:
        from iitgpu.config import load_config
        gid = grp.getgrnam(load_config().admin_group).gr_gid
        os.chown(str(folder), -1, gid)
    except (KeyError, PermissionError, OSError):
        pass   # best-effort; sbatch will fail with a clear error if still blocked
```

Note: the group was previously `gpuusers` so a shared submit account could read the script under `gateway_shared_user` mode. That mode is off here (`shared_user_mode=False`); if it is ever enabled this needs revisiting.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_jobs.py -k "job_folder" -q`
Expected: 2 passed.

- [ ] **Step 5: Run the full suite**

Run: `python3 -m pytest -q`
Expected: 650 passed (648 baseline + 2).

- [ ] **Step 6: Commit**

```bash
git add iitgpu/jobs.py tests/test_jobs.py
git commit -m "feat(jobs): job folders are owner + admin only

A job folder holds the user's scripts and output — the same private content as
their home — but was group gpuusers, readable by every other user. Now 2770
group gpuadmins, with setgid so files the job writes later stay admin-readable."
```

---

### Task 3: Provisioning creates admin-accessible user areas

**Files:**
- Modify: `deploy/iit-gpu-adduser.sh:124-127`
- Test: `tests/test_adduser_wrapper.py`

**Interfaces:**
- Consumes: `ADMIN_GROUP` from `deploy/site.env` (already present, value `gpuadmins`).
- Produces: new accounts get `/shared/users/<user>` at mode `2770` group `gpuadmins`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_adduser_wrapper.py`:

```python
def test_adduser_creates_user_area_owner_and_admin_only():
    """New user areas must be 2770 group admin, not 0700 owner-only.

    0700 locks admins out; anything looser exposes the area to other users.
    """
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / "deploy" / "iit-gpu-adduser.sh"
    text = script.read_text()
    assert "chmod 2770" in text, "user area must be mode 2770"
    assert "chmod 0700" not in text, "0700 would lock admins out of user areas"
    assert "$ADMIN_GROUP" in text, "group must come from ADMIN_GROUP, not hardcoded"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python3 -m pytest tests/test_adduser_wrapper.py -k user_area -q`
Expected: FAIL — the script still contains `chmod 0700`.

- [ ] **Step 3: Update the provisioning script**

In `deploy/iit-gpu-adduser.sh`, add near the other defaults around line 30:

```bash
ADMIN_GROUP="${ADMIN_GROUP:-gpuadmins}"
```

Then replace the block at lines 124-127 with:

```bash
step "Creating $NFS_ROOT/users/$USERNAME on the NFS server (GPU host) ..."
run "ssh $GPU_HOST_SSH \"sudo mkdir -p $NFS_ROOT/users/$USERNAME && \
    sudo chown $NEW_UID:$ADMIN_GROUP $NFS_ROOT/users/$USERNAME && \
    sudo chmod 2770 $NFS_ROOT/users/$USERNAME\""
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 -m pytest tests/test_adduser_wrapper.py -k user_area -q`
Expected: 1 passed.

- [ ] **Step 5: Run the full suite**

Run: `python3 -m pytest -q`
Expected: 651 passed.

- [ ] **Step 6: Commit**

```bash
git add deploy/iit-gpu-adduser.sh tests/test_adduser_wrapper.py
git commit -m "feat(provisioning): user areas are 2770 group admin

0700 locked admins out of the areas they support. Group now comes from
ADMIN_GROUP so the value is not duplicated across scripts."
```

---

### Task 4: Permission checker and fixer for existing areas

The checker is read-only and runs anywhere. The fixer must run on the GPU host because of `root_squash`. The checker doubles as the test for the fixer.

**Files:**
- Create: `deploy/check-shared-perms.sh`
- Create: `deploy/fix-shared-perms.sh`

**Interfaces:**
- Consumes: `NFS_ROOT` (default `/shared`), `ADMIN_GROUP` (default `gpuadmins`).
- Produces: `deploy/check-shared-perms.sh` exits `0` when clean and `1` when any area is exposed — Task 5 wires this into the deploy.

- [ ] **Step 1: Write the checker**

Create `deploy/check-shared-perms.sh`:

```bash
#!/usr/bin/env bash
# Read-only audit of /shared access. Safe to run from any node — root_squash
# blocks writes from the login node but stat() works fine.
#
# Rule: per-user areas (users/, jobs/) grant nothing to "other".
#       Shared asset dirs grant no write to "other".
set -uo pipefail

NFS_ROOT="${NFS_ROOT:-/shared}"
SHARED_DIRS="${SHARED_DIRS:-data datasets envs models templates}"
fail=0

for base in users jobs; do
    [ -d "$NFS_ROOT/$base" ] || continue
    for d in "$NFS_ROOT/$base"/*; do
        [ -d "$d" ] || continue
        [ -L "$d" ] && continue
        mode=$(stat -c %a "$d" 2>/dev/null) || continue
        if [ "${mode: -1}" != "0" ]; then
            echo "EXPOSED   $d  mode=$mode  (other must be 0)"
            fail=1
        fi
    done
done

for d in $SHARED_DIRS; do
    p="$NFS_ROOT/$d"
    [ -d "$p" ] || continue
    mode=$(stat -c %a "$p" 2>/dev/null) || continue
    other="${mode: -1}"
    case "$other" in
        2|3|6|7) echo "WRITABLE  $p  mode=$mode  (other must not have write)"; fail=1 ;;
    esac
done

if [ "$fail" -ne 0 ]; then
    echo
    echo "Fix on the GPU HOST (the NFS server — root is squashed on the login node):"
    echo "  ssh root-daham@192.168.122.1 'sudo bash /opt/iit-gpu/deploy/fix-shared-perms.sh'"
    exit 1
fi

echo "shared permissions OK"
```

- [ ] **Step 2: Run the checker and confirm it reports the known exposures**

```bash
bash deploy/check-shared-perms.sh; echo "exit=$?"
```

Expected: `EXPOSED` lines for `users/dahamadmin`, `users/hassan`, `users/public`, `users/daham`, `WRITABLE` lines for `data`, `envs`, `models`, `templates`, and `exit=1`.

- [ ] **Step 3: Write the fixer**

Create `deploy/fix-shared-perms.sh`:

```bash
#!/usr/bin/env bash
# Apply the /shared access model. MUST run as root ON THE GPU HOST: the NFS
# export uses root_squash, so root on the login node cannot change modes here.
# Idempotent.
set -euo pipefail

NFS_ROOT="${NFS_ROOT:-/shared}"
ADMIN_GROUP="${ADMIN_GROUP:-gpuadmins}"
SHARED_DIRS="${SHARED_DIRS:-data datasets envs models templates}"

[ "$(id -u)" -eq 0 ] || { echo "ERROR: run as root" >&2; exit 1; }
[ -d "$NFS_ROOT/users" ] || { echo "ERROR: $NFS_ROOT/users missing — wrong node?" >&2; exit 1; }
getent group "$ADMIN_GROUP" >/dev/null || { echo "ERROR: group $ADMIN_GROUP missing" >&2; exit 1; }

echo "== shared assets -> 2775 (group writable, other read-only)"
for d in $SHARED_DIRS; do
    p="$NFS_ROOT/$d"
    [ -d "$p" ] || continue
    chmod 2775 "$p"
    echo "  $p"
done

echo "== per-user areas -> 2770 owner:$ADMIN_GROUP"
for base in users jobs; do
    [ -d "$NFS_ROOT/$base" ] || continue
    for p in "$NFS_ROOT/$base"/*; do
        [ -d "$p" ] || continue
        [ -L "$p" ] && continue
        u=$(basename "$p")
        if ! id -u "$u" >/dev/null 2>&1; then
            echo "  skip $p (no account named $u)"
            continue
        fi
        chown "$u:$ADMIN_GROUP" "$p"
        chmod 2770 "$p"
        echo "  $p"
    done
done

echo "done"
```

- [ ] **Step 4: Run the fixer on the GPU host**

**Transport note:** `ssh` from the login node **to** the GPU host is not available
(`Permission denied (publickey,password)`). Only GPU host → login node works.
Your shell already runs on the GPU host, so **pull** files across; never push.

```bash
ssh -o BatchMode=yes slurmadmin@192.168.122.10 \
  'cat ~/IIT-Secure-SLURM-Job-Gateway/deploy/fix-shared-perms.sh' > /tmp/fix-shared-perms.sh
sudo bash /tmp/fix-shared-perms.sh
```

Expected: every shared dir and every user/job area listed, no errors.

- [ ] **Step 5: Re-run the checker to verify it now passes**

```bash
bash deploy/check-shared-perms.sh; echo "exit=$?"
```

Expected: `shared permissions OK`, `exit=0`.

- [ ] **Step 6: Verify the actual access rule live**

```bash
# a non-admin must be denied both areas of another user
sudo -u yenuli ls /shared/users/dahamadmin  2>&1 | tail -1
sudo -u yenuli ls /shared/jobs/dahamadmin   2>&1 | tail -1
# an admin must still get in — run these two ON THE GPU HOST (your shell is there)
sudo -u dahamadmin ls /shared/users/daham >/dev/null && echo 'admin OK'
```

Expected: two `Permission denied` lines, then `admin OK`.

- [ ] **Step 7: Confirm the fixer is idempotent**

Run it a second time, then re-run the checker. Expected: same output, still `exit=0`.

- [ ] **Step 8: Commit**

```bash
git add deploy/check-shared-perms.sh deploy/fix-shared-perms.sh
git commit -m "feat(deploy): checker and fixer for the /shared access model

Per-user areas grant nothing to other; shared assets grant no write to other.
The checker is read-only and runs anywhere; the fixer must run on the GPU host
because the NFS export uses root_squash and root is squashed on the login node."
```

---

### Task 5: Fail the deploy on permission drift

**Files:**
- Modify: `deploy/redeploy-igm.sh`

**Interfaces:**
- Consumes: `deploy/check-shared-perms.sh` from Task 4.
- Produces: deploy aborts before syncing anything when `/shared` permissions have drifted.

- [ ] **Step 1: Confirm the insertion point**

```bash
grep -n "Running test suite" deploy/redeploy-igm.sh
```

Expected: `56:step "Running test suite ..."`. Insert the new block immediately **before** line 56 so a drifted cluster fails before anything is synced.

The script already defines these helpers at lines 39-42 — use them as-is:
`ok() { echo "  ✔  $*"; }`, `warn()`, `fail() { echo "  ✘  $*" >&2; exit 1; }`, `step()`.

- [ ] **Step 2: Add the drift gate**

Insert immediately before line 56:

```bash
step "Checking /shared permissions ..."
if bash "$(dirname "$0")/check-shared-perms.sh"; then
    ok "shared permissions OK"
else
    fail "shared permission drift detected (see above). This node cannot repair it — the NFS export uses root_squash. Fix on the GPU host, then re-run this deploy."
fi
```

`fail()` already exits 1, so no extra guard is needed.

- [ ] **Step 3: Verify the gate passes on a clean cluster**

```bash
bash deploy/redeploy-igm.sh 2>&1 | grep -A2 "Checking /shared permissions"
```

Expected: `shared permissions OK`, deploy continues.

- [ ] **Step 4: Verify the gate actually catches drift**

```bash
# chmod runs on the GPU host; the checker runs on the login node
sudo chmod 0777 /shared/users/tuser
ssh -o BatchMode=yes slurmadmin@192.168.122.10 'cd ~/IIT-Secure-SLURM-Job-Gateway && bash deploy/check-shared-perms.sh; echo "exit=$?"'
sudo chmod 2770 /shared/users/tuser
ssh -o BatchMode=yes slurmadmin@192.168.122.10 'cd ~/IIT-Secure-SLURM-Job-Gateway && bash deploy/check-shared-perms.sh; echo "exit=$?"'
```

Expected: `EXPOSED .../tuser mode=777` and `exit=1`, then `shared permissions OK` and `exit=0`.

- [ ] **Step 5: Commit**

```bash
git add deploy/redeploy-igm.sh
git commit -m "feat(deploy): abort on /shared permission drift

Detection only — root_squash means this node cannot repair permissions, so the
gate prints the GPU-host command instead of trying and failing."
```

---

### Task 6: Tell users how much of the GPU they get

`gpu_share_note()` exists in `jobs.py` with zero call sites.

**Files:**
- Modify: `iitgpu/wizard.py` (Job Summary panel ~line 1085, notebook confirm ~line 884)
- Test: `tests/test_wizard.py`

**Interfaces:**
- Consumes: `jobs.gpu_share_note(gpu_shards: int) -> str`.
- Produces: no new API.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_wizard.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_wizard.py -k gpu_share_note -q`
Expected: the first test passes, the second FAILS — `wizard.py` never calls it.

- [ ] **Step 3: Wire it into the batch Job Summary**

In `iitgpu/wizard.py`, change the `summary_lines` block (~line 1079) to:

```python
    from iitgpu.jobs import gpu_share_note
    summary_lines = (
        f"  GPU share  : {gpu_share_note(spec.gpu_shards)}\n"
        f"  Data path  : {data_path or 'not set'}\n"
        f"  Model path : {model_path or 'not set'}\n"
        f"  Environment: {_env_display}\n"
        f"  Script     : {script_path or '(none)'}"
    )
    panel("Job Summary", summary_lines)
```

- [ ] **Step 4: Wire it into the notebook confirm**

Immediately before `panel("Generated notebook sbatch script", script_text)` (~line 884):

```python
        from iitgpu.jobs import gpu_share_note
        info(f"GPU share: {gpu_share_note(spec.gpu_shards)}")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_wizard.py -k gpu_share_note -q`
Expected: 2 passed.

- [ ] **Step 6: Run the full suite**

Run: `python3 -m pytest -q`
Expected: 653 passed.

- [ ] **Step 7: Commit**

```bash
git add iitgpu/wizard.py tests/test_wizard.py
git commit -m "feat(wizard): show how much of the GPU a job reserves

Sizing changed materially when the card was split four ways and nothing in the
interface said so. gpu_share_note() existed but had no call sites."
```

---

### Task 7: VRAM guidance reflects a shared card

The check compares the estimate against *total* free VRAM. With four tenants that is not the user's to spend, and shards do not cap VRAM.

**Files:**
- Modify: `iitgpu/wizard.py:436-441` (`_vram_check` prompt)
- Test: `tests/test_wizard.py`

**Interfaces:**
- Consumes: `jobs.SHARDS_PER_GPU`.
- Produces: no new API.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_wizard.py`:

```python
def test_vram_prompt_states_the_budget_is_shared_and_unenforced():
    """Slices schedule, they do not isolate — two jobs can still OOM each other."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "iitgpu" / "wizard.py").read_text()
    start = src.index("def _vram_check")
    body = src[start:start + 3000]
    assert "shared" in body.lower(), "prompt must say VRAM is shared"
    assert "not enforced" in body.lower(), "prompt must say it is not enforced"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_wizard.py -k vram_prompt -q`
Expected: FAIL — neither phrase appears.

- [ ] **Step 3: Reword the prompt**

Replace lines 436-441 of `iitgpu/wizard.py` with:

```python
    default_vram = _VRAM_TASK_DEFAULTS.get(task_type, 0)
    from iitgpu.jobs import SHARDS_PER_GPU
    if free_gb is not None:
        _slice_gb = (stats.gpu_mem_total_mb / 1024) / SHARDS_PER_GPU
        info(f"Your slice's fair share is about {_slice_gb:.0f} GB. "
             f"VRAM is shared between concurrent jobs and is not enforced, so "
             f"this is a budget, not a guarantee — going over can OOM someone else.")
    raw = questionary.text(
        "Estimated VRAM your job needs (GB, 0 = skip check):",
        default=str(default_vram),
        style=_STYLE,
    ).ask()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 -m pytest tests/test_wizard.py -k vram_prompt -q`
Expected: 1 passed.

- [ ] **Step 5: Run the full suite**

Run: `python3 -m pytest -q`
Expected: 654 passed.

- [ ] **Step 6: Commit**

```bash
git add iitgpu/wizard.py tests/test_wizard.py
git commit -m "fix(wizard): VRAM guidance reflects a shared card

Comparing against total free VRAM implied the whole card was the user's to
spend. Shards do not cap VRAM, so the figure is a courtesy budget."
```

---

### Task 8: Widen the file-manager jail to shared datasets

The notebook will show `data` and `datasets`; the file manager must agree, or the two surfaces enforce different boundaries again.

**Files:**
- Modify: `iitgpu/validate.py:99-107` (`user_browse_roots`)
- Test: `tests/test_validate.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `user_browse_roots(nfs_root: str, username: str) -> list[str]` — same signature, now 5 entries.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_validate.py`:

```python
def test_browse_roots_include_shared_datasets():
    """File manager and notebook must expose the same boundary."""
    from iitgpu.validate import user_browse_roots
    roots = user_browse_roots("/shared", "yenuli")
    assert "/shared/users/yenuli" in roots
    for shared in ("/shared/models", "/shared/envs", "/shared/data", "/shared/datasets"):
        assert shared in roots, f"{shared} must be browsable"


def test_browse_roots_exclude_other_users_and_jobs():
    from iitgpu.validate import user_browse_roots
    roots = user_browse_roots("/shared", "yenuli")
    assert "/shared/users" not in roots
    assert "/shared/jobs" not in roots
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_validate.py -k browse_roots -q`
Expected: FAIL — `/shared/data` missing.

- [ ] **Step 3: Widen the roots**

Replace lines 103-107 of `iitgpu/validate.py`:

```python
    return [
        str(Path(base) / "users" / username),
        str(Path(base) / "models"),
        str(Path(base) / "envs"),
        str(Path(base) / "data"),
        str(Path(base) / "datasets"),
    ]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_validate.py -k browse_roots -q`
Expected: 2 passed.

- [ ] **Step 5: Run the full suite**

Run: `python3 -m pytest -q`
Expected: 656 passed.

- [ ] **Step 6: Commit**

```bash
git add iitgpu/validate.py tests/test_validate.py
git commit -m "feat(validate): file manager can browse shared datasets

The jail blocked data/ and datasets/ — the datasets people actually train on.
Matches what the notebook will expose, so both surfaces agree."
```

---

### Task 9: Audit that an interactive session was started

The job cannot reach the audit socket (it is login-node tmpfs), so this is logged client-side at submit.

**Files:**
- Modify: `iitgpu/wizard.py` (notebook submit path, after `submit_job` succeeds)
- Test: `tests/test_wizard.py`

**Interfaces:**
- Consumes: `auditclient.log(action: str, detail: str = "", job_id: str = "", meta: dict | None = None) -> bool`.
- Produces: audit action name `notebook_session_start`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_wizard.py`:

```python
def test_notebook_submit_audits_the_interactive_session():
    """A notebook is a full execution environment; the trail must show it started."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "iitgpu" / "wizard.py").read_text()
    assert "notebook_session_start" in src
    idx = src.index("notebook_session_start")
    window = src[idx - 200:idx + 300]
    assert "gpu_shards" in window, "record the slice the session holds"
    assert "job_id" in window, "record which job the session is"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_wizard.py -k audits_the_interactive -q`
Expected: FAIL — `notebook_session_start` not present.

- [ ] **Step 3: Confirm the notebook submit site**

```bash
grep -n "notebook_submitted_ok" iitgpu/wizard.py
```

Expected: one hit around line 917, inside `if success:` after `submit_job(sbatch_path)`. The job id is in the local variable `result`.

- [ ] **Step 4: Emit the event beside the existing one**

Immediately **after** the existing line:

```python
            auditclient.log("notebook_submitted_ok", detail=job_name, job_id=result)
```

add:

```python
            auditclient.log(
                "notebook_session_start",
                detail=job_name,
                job_id=result,
                meta={"env": spec.conda_env or spec.container_image or "system",
                      "gpu_shards": spec.gpu_shards},
            )
```

`notebook_submitted_ok` records that submission succeeded; this records what the
interactive session actually holds, which is what the trail was missing.

- [ ] **Step 5: Run the test to verify it passes**

Run: `python3 -m pytest tests/test_wizard.py -k audits_the_interactive -q`
Expected: 1 passed.

- [ ] **Step 6: Run the full suite**

Run: `python3 -m pytest -q`
Expected: 657 passed.

- [ ] **Step 7: Commit**

```bash
git add iitgpu/wizard.py tests/test_wizard.py
git commit -m "feat(audit): record that an interactive notebook session started

Logged client-side at submit: the job runs on the GPU host, which has no
/run/iit-gpu, so it cannot reach the daemon socket. Records the accountable
act without claiming per-command capture."
```

---

### Task 10: Root the notebook at the user's own folder

Ships last — it is the only change users see.

**Files:**
- Modify: `iitgpu/jobs.py:329-334` (docstring), `iitgpu/jobs.py:370` and `:409` (launcher), plus a new snippet helper near `_free_port_snippet`
- Test: `tests/test_sharding.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `_user_home_snippet() -> list[str]` — bash lines defining `$IIT_USER_ROOT` and creating the shared-asset symlinks.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sharding.py`:

```python
def _nb_script(tmp_path):
    from iitgpu.jobs import JobSpec, render_notebook_sbatch
    spec = JobSpec(job_name="notebook", partition="gpu", gpu_shards=1, cpus=8,
                   mem_gb=14, time_limit="08:00:00", run_command="",
                   task_type="notebook", conda_env="/shared/envs/data-science")
    return render_notebook_sbatch(spec, str(tmp_path), port=8888,
                                  gateway_host="gw.edu", gateway_port=2225)


def test_notebook_is_rooted_at_the_users_own_folder(tmp_path):
    """Serving /shared made the per-user jail meaningless in the notebook."""
    script = _nb_script(tmp_path)
    assert "--ServerApp.root_dir=$IIT_USER_ROOT" in script
    assert "--notebook-dir=/shared" not in script


def test_notebook_exposes_shared_assets_by_symlink(tmp_path):
    """Users still need the datasets they train on."""
    script = _nb_script(tmp_path)
    for asset in ("models", "envs", "data", "datasets"):
        assert f"ln -sfn /shared/{asset}" in script, f"{asset} must be reachable"


def test_symlink_creation_is_idempotent(tmp_path):
    """The job reruns on every launch; -sfn must not fail on an existing link."""
    script = _nb_script(tmp_path)
    assert "ln -s " not in script.replace("ln -sfn ", "")


def test_notebook_docstring_does_not_claim_loopback_only(tmp_path):
    """It binds the routable NodeAddr; a false security comment is a trap."""
    from iitgpu.jobs import render_notebook_sbatch
    assert "127.0.0.1 only" not in (render_notebook_sbatch.__doc__ or "")
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_sharding.py -k "rooted or symlink or docstring" -q`
Expected: FAIL — script still has `--notebook-dir=/shared`, no symlinks, docstring still claims loopback.

- [ ] **Step 3: Add the user-root snippet**

In `iitgpu/jobs.py`, directly after `_free_port_snippet`:

```python
# Bash that defines $IIT_USER_ROOT and exposes the shared assets inside it.
#
# JupyterLab used to be rooted at /shared, so its file browser ignored the
# per-user jail the file manager enforces. It is now rooted at the user's own
# folder, with the shared read-only assets symlinked in so people can still
# reach the datasets they train on. Jupyter follows symlinks pointing outside
# root_dir, which is what makes this work.
#
# ln -sfn is idempotent: the job reruns this on every launch.
def _user_home_snippet() -> list[str]:
    lines = ['IIT_USER_ROOT="/shared/users/$USER"', 'mkdir -p "$IIT_USER_ROOT"']
    for asset in ("models", "envs", "data", "datasets"):
        lines.append(f'ln -sfn /shared/{asset} "$IIT_USER_ROOT/{asset}"')
    lines.append("")
    return lines
```

- [ ] **Step 4: Emit the snippet and switch the launcher**

Find where the notebook renderer adds `_NODE_ADDR_SNIPPET`:

```bash
grep -n "_NODE_ADDR_SNIPPET + _free_port_snippet" iitgpu/jobs.py
```

In the **notebook** renderer only (not TensorBoard), change that line to:

```python
    lines += _NODE_ADDR_SNIPPET + _free_port_snippet(port) + _user_home_snippet()
```

Then replace both launcher lines (`:370` and `:409`):

```python
            f"--ServerApp.root_dir=$IIT_USER_ROOT --IdentityProvider.token=\"$JUPYTER_TOKEN\""
```

- [ ] **Step 5: Correct the docstring**

Replace the false line in the `render_notebook_sbatch` docstring:

```python
    - Binds JupyterLab to the node's SLURM NodeAddr — reachable from the gateway
      network but not the public interface — gated by a per-job random token
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_sharding.py -q`
Expected: all pass.

- [ ] **Step 7: Run the full suite**

Run: `python3 -m pytest -q`
Expected: 661 passed.

- [ ] **Step 8: Verify the generated script is valid bash**

```bash
python3 -c "
from iitgpu.jobs import JobSpec, render_notebook_sbatch
s = JobSpec(job_name='notebook', partition='gpu', gpu_shards=1, cpus=8, mem_gb=14,
            time_limit='08:00:00', run_command='', task_type='notebook',
            conda_env='/shared/envs/data-science')
open('/tmp/nb.sh','w').write(render_notebook_sbatch(s, '/tmp', port=8888,
    gateway_host='10.35.4.100', gateway_port=2225))
"
bash -n /tmp/nb.sh && echo "SYNTAX OK"
```

Expected: `SYNTAX OK`.

- [ ] **Step 9: Commit**

```bash
git add iitgpu/jobs.py tests/test_sharding.py
git commit -m "feat(notebook): root JupyterLab at the user's own folder

Serving /shared meant the file browser ignored the per-user jail the file
manager enforces. Shared assets are symlinked in so people keep the datasets
they train on. Also corrects a docstring claiming loopback-only binding, which
has been false since the NodeAddr tunnel fix."
```

---

### Task 11: Deploy and verify end to end

**Files:** none — verification only.

- [ ] **Step 1: Deploy**

```bash
cd ~/IIT-Secure-SLURM-Job-Gateway
bash deploy/redeploy-igm.sh 2>&1 | tail -25
```

Expected: permission gate passes, 661 tests pass, `Deploy complete`.

- [ ] **Step 2: Verify the access rule live**

```bash
# all three run ON THE GPU HOST
sudo -u yenuli ls /shared/users/dahamadmin 2>&1 | tail -1
sudo -u yenuli ls /shared/jobs/dahamadmin  2>&1 | tail -1
sudo -u dahamadmin ls /shared/users/daham >/dev/null && echo 'admin OK'
```

Expected: two `Permission denied`, then `admin OK`.

- [ ] **Step 3: Launch a notebook and check the jail**

```bash
sudo -u yenuli env PYTHONPATH=/opt/iit-gpu python3 - <<'PY'
import pathlib, subprocess
from iitgpu.jobs import JobSpec, render_notebook_sbatch, resource_defaults, make_job_folder
from iitgpu.config import load_config
cfg = load_config(); d = resource_defaults("notebook")
spec = JobSpec(job_name="notebook", partition=cfg.partition, gpu_shards=d.gpu_shards,
               cpus=d.cpus, mem_gb=d.mem_gb, time_limit="00:10:00", run_command="",
               task_type="notebook", user="yenuli", conda_env="/shared/envs/data-science")
f = make_job_folder(f"{cfg.nfs_root}/jobs", spec)
p = pathlib.Path(f) / "job.sbatch"
p.write_text(render_notebook_sbatch(spec, f, port=8888,
    gateway_host=cfg.gateway_host, gateway_port=int(cfg.gateway_port)))
p.chmod(0o644)
print(subprocess.run(["sbatch", str(p)], capture_output=True, text=True).stdout)
PY
```

Wait ~30s, then read the job's `.out` for the port and token and confirm the browser root:

```bash
curl -s "http://192.168.122.1:<PORT>/api/contents?token=<TOKEN>" \
  | python3 -c "import json,sys; print([c['name'] for c in json.load(sys.stdin)['content']])"
```

Expected: only the user's own entries plus `models`, `envs`, `data`, `datasets`. **No `users`, no `jobs`.**

- [ ] **Step 4: Confirm a dataset is readable through the symlink**

```bash
curl -s "http://192.168.122.1:<PORT>/api/contents/data?token=<TOKEN>" \
  | python3 -c "import json,sys; print([c['name'] for c in json.load(sys.stdin)['content']])"
```

Expected: `cifar10`, `downloads`, `imagenette2`.

- [ ] **Step 5: Cancel the test job**

```bash
ssh slurmadmin@192.168.122.10 "sudo scancel -u yenuli"
```

- [ ] **Step 6: Tag the release**

```bash
printf '__version__ = "1.1.0"\n' > iitgpu/__init__.py
git add iitgpu/__init__.py
git commit -m "chore: bump version to 1.1.0"
git tag -a v1.1.0 -m "v1.1.0 - one access model for /shared, notebook jail, session audit"
git push origin main && git push origin v1.1.0
bash deploy/redeploy-igm.sh 2>&1 | tail -5
```

Minor bump rather than patch: the access model and the notebook's visible root both change behaviour.

---

## Rollback

Each task is independently revertible.

- Tasks 1, 4 — on the GPU host: `sudo chmod 0777 /shared/users/<u>` restores an area; `sudo gpasswd -d <user> gpuadmins` undoes the group sync.
- Tasks 2, 3, 5–10 — `git revert <sha>` then redeploy.
- Task 10 is the only user-visible change; reverting restores `--notebook-dir=/shared`.
