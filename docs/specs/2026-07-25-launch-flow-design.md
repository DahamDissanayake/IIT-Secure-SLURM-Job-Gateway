# IIT GPU Manager — Launch Flow Redesign

Status: approved, ready for implementation planning
Date: 2026-07-25
Baseline: `main` @ fcc274f (v1.1.0 + NFS-root ACL hotfix), 663 tests passing

## Goal

Setting up a job takes 12–16 prompts, interrogates students with questions the
tool should answer itself, and offers no way back: a wrong answer at any step
means Discard and restart. Connecting to a notebook after submit is a further
five manual steps against a job that says RUNNING before it is ready.

This spec replaces the linear interrogation with a short intake plus one
editable review hub (RunPod's pattern), and makes notebooks connectable
(Kaggle treats infrastructure as a setting, not a quiz — so do we).

## Problems being fixed (walked in the live flow)

- Template confirm before a first-timer knows what a template is (`wizard.py:515`).
- Taxonomy first: 7 task types whose only real effect is resource defaults.
- "Submit your own .sbatch?" asked of everyone, mid-flow (`wizard.py:579`).
- File browser is one `select` per directory level; no typed path, no recents.
- Two sequential Advanced confirms (array, dependency) for every user, every run.
- "Estimated VRAM your job needs (GB)" — students cannot answer this (`wizard.py:443`).
- Review is a raw sbatch dump; nothing editable; wrong choice ⇒ restart.
- Resources (GPU share, CPUs, RAM, time) never shown or offered before the summary.
- No availability at decision time, though `get_node_stats()` has it.
- Post-submit: RUNNING ≠ ready; tunnel info only in the job log; 5-step ritual.

## UX sources

- **RunPod**: template first → GPU picker with live availability → one review
  screen, every field editable, deploy from there. Only touch what you change.
- **Kaggle**: start from the artifact (notebook); accelerator is a dropdown with
  quota visible; datasets attached, not path-typed; versioned re-runs.

## Design

### 1. Entry — three intents

```
What do you want to do?
❯ Open JupyterLab            interactive notebook on the GPU
  Run a script or notebook   batch job — .py or .ipynb
  Open a shell on the GPU node
  ──────────
  Other: my own .sbatch · templates
```

- Old task types (`train`, `finetune`, `inference`, `test`, `custom`,
  `notebook-script`) disappear as questions. They remain internally as the size
  table and for template/rerun compatibility.
- "Other" submenu hosts the own-`.sbatch` bypass (unchanged behaviour) and the
  template picker. Neither interrupts the main flow any more.
- Shell intent goes straight to the hub (size + time only).

### 2. Artifact intake (batch only)

One prompt: `questionary.autocomplete` over paths (typed path with completion),
seeded with a **Recent** list — the last 5 distinct scripts/notebooks found in
the user's own job folders (`<nfs_root>/jobs/<user>/*/job.sbatch` parsed, most
recent first). A `[browse…]` choice falls back to the existing browser
(`_browse_script`), jail rules unchanged. Extension decides `.py`/`.sh` script
vs `.ipynb` notebook-batch; anything else is rejected with the allowed list.

### 3. Review hub — the core

After intake the user lands on the hub and never leaves it except into one
focused editor at a time. Rendered with Rich panel + one questionary select:

```
╭─ Ready to launch ─────────── GPU now: 3/4 slices free ─╮
│  Script       train.py   (physionet_project/)          │
│  Environment  data-science                             │
│  Size         Standard — ¼ GPU · 8 CPU · 14 GB         │
│  Time limit   4h                                       │
│  Data / model (none)                                   │
│  Args         (none)                                   │
│  Advanced     off                                      │
╰────────────────────────────────────────────────────────╯
❯ Launch
  Change size / time / environment / data / args / advanced
  Save as template
  Cancel
```

Field editors (each returns to the hub):

- **Size** — the availability picker. Options computed from the size table and
  `get_node_stats()` (`shard_total - shard_alloc`):

  ```
  ❯ Standard    ¼ GPU · 8 CPU · 14 GB   — starts now (3 slices free)
    Small       ¼ GPU · 4 CPU · 8 GB    — starts now
    Whole GPU   4/4 GPU · 16 CPU · 60 GB — will queue (needs 4 free, 3 free)
  ```

  Stats unavailable ⇒ suffixes read `— availability unknown` and nothing blocks.
- **Time limit** — presets `1h / 2h / 4h / 8h (cluster max)` + custom `HH:MM`.
  8h labelled as the QOS ceiling.
- **Environment** — prebuilt list from `/shared/envs` (default: `data-science`),
  own conda path, container image, none. Same sources as today, one screen.
- **Data / model** — data folder picker (jail rules unchanged) and, in one
  place, the model options that used to be a separate step (browse
  `/shared/models`, HF repo id, manual path). Notebook deps prompt
  (`_notebook_deps_prompt`) also lives here for notebook intent.
- **Args** — the free-text extra-args prompt, unchanged.
- **Advanced** — one submenu: job array, dependency (existing pickers), mail
  on/off for this job, and **View generated sbatch** (the only place the raw
  script is shown).

Hub facts, not questions:

- Availability line in the panel title, refreshed on every return to the hub.
- When live stats exist, a passive line: `GPU memory: 28 GB free of 32 —
  shared between jobs, not enforced`. **The VRAM estimate prompt is deleted.**
  The existing wording strings that tests pin ("shared", "not enforced",
  `gpu_share_note`) remain in the rendered hub/summary.
- Launch runs the existing pipeline unchanged: `log_or_block("job_submit")` /
  notebook path audit events, `make_job_folder`, renderers, `submit_job`.

### 4. Sizes

| Size | gpu_shards | cpus | mem_gb | default time |
|---|---|---|---|---|
| Small | 1 | 4 | 8 | 2h |
| Standard | 1 | 8 | 14 | 4h |
| Whole GPU | 4 (`SHARDS_PER_GPU`) | 16 | 60 | 8h |

Defaults: notebook ⇒ Standard/6h; batch ⇒ Standard/4h; shell ⇒ Small/2h.
Derived from, and kept consistent with, `TASK_DEFAULTS` — one source of truth
in the new size table, with a test asserting a full card's worth of Standard
jobs fits the node (as `test_sharding` does today).

### 5. Post-submit: notebooks become connectable

- `render_notebook_sbatch` gains a readiness block: immediately **before** the
  `jupyter lab` line (which blocks for the job's lifetime), spawn a watcher:

  ```bash
  ( until (exec 3<>/dev/tcp/127.0.0.1/$IIT_PORT) 2>/dev/null; do sleep 2; done
    touch "<folder>/.iit-ready" ) &
  ```

  Pure bash `/dev/tcp`, no new dependencies. Marker sits next to `.iit-jupyter`.
- After submit, the TUI shows `Starting JupyterLab… job <id>` and polls the
  marker (with the job's state as a liveness check). On ready it prints the
  **Connect card**, parsed from the job's own `.out` (the authoritative source
  — this cannot reintroduce the advertised-vs-bound port bug):

  ```
  ── Connect ─────────────────────────────────────
  1. On YOUR laptop:  ssh -p 2225 -N -L 8930:192.168.122.1:8930 yenuli@10.35.4.100
  2. Open:            http://127.0.0.1:8930/lab?token=…
  ────────────────────────────────────────────────
  ```

  Timeout: after 90s of no marker with the job still RUNNING, show the log tail
  and keep offering to wait; if the job left RUNNING, show the error tail.
- Dashboard: jupyter jobs show `STARTING` until `.iit-ready` exists, then
  `RUNNING`; new key `t` on a selected jupyter job re-prints the Connect card
  (`c` was already cancel).

### 6. Structure

- `iitgpu/launchspec.py` — `LaunchSpec` dataclass (intent, script, env fields,
  size, time, data, model, args, array, dependency, mail), the size table,
  availability probe (`slices_free()` etc.), `to_job_spec()` derivation, recent
  scripts scan. Pure logic, no prompts. Unit-tested directly.
- `iitgpu/review.py` — `run_hub(spec) -> LaunchSpec | None` loop, per-field
  editors, hub renderer (pure function `render_hub(spec, stats) -> Panel` so
  Console-capture tests work, like `splash.py`).
- `iitgpu/connect.py` — marker wait, `.out` parser
  (`parse_connect(text) -> ConnectInfo | None`), Connect card renderer. Used by
  wizard post-submit and dashboard `C`.
- `iitgpu/wizard.py` — slims to: intent select → intake → build default
  `LaunchSpec` → `run_hub` → submit pipeline → post-submit. Existing helpers
  (`_browse_script`, `_browse_data_folder`, deps prompt, tier-3 own-script,
  template save/load) are reused, not rewritten.

### 7. Compatibility

- **Templates**: old saved templates carry `task_type` + `gpu_shards/cpus/mem_gb`.
  Loader maps them to intent (`notebook`→notebook, else batch) and the nearest
  size (exact triple match, else Custom shown as its literal numbers). Saving
  writes the same `asdict(JobSpec)` schema as today, so old and new tool
  versions read each other's templates.
- **Rerun prefill** (`monitor.py` sbatch parse) feeds the hub's initial spec.
- **Interactive srun path** (shell intent) keeps its early-return behaviour,
  now with size+time from the hub.
- All existing audit actions keep their names and order.
- Existing test-pinned strings survive: `gpu_share_note` call sites, VRAM
  "shared"/"not enforced" wording, notebook renderer invariants
  (`$IIT_PORT`, `$IIT_USER_ROOT`, symlink guard) are untouched by this work.

### 8. Out of scope

- Main menu and file-manager redesign; admin panel.
- The dashboard beyond STARTING state + `C` key.
- Env content probing ("which env has torch") — env list stays names-only.
- P1-1 (privileged extend) — separate item.

## Testing

- `launchspec`: size table vs node capacity (4× Standard fits 32 CPU/60 GB);
  `to_job_spec` field mapping incl. template/rerun compat mapping; recent-scan
  on a fixture tree; availability text for free/queue/unknown.
- `review`: `render_hub` Console-capture asserts fields, availability line,
  shared-VRAM wording; editors return-to-hub loop with scripted questionary
  (patched `.ask()` sequences, as existing wizard tests do).
- `connect`: `.out` parser on a real captured sample; marker block asserted in
  the generated notebook script (test_service_ports style); dashboard STARTING
  logic on a tmp job folder with/without marker.
- Full suite stays green; count only rises.

## Rollout

1. `launchspec` + `connect` (pure logic, no UI change shipped).
2. Notebook readiness marker + post-submit Connect card + dashboard STARTING/`C`.
3. The hub + new entry flow replacing the linear wizard, template compat last.
Each step independently deployable; step 3 is the only visible flow change.
