# iitgpu/jobs.py
from __future__ import annotations
import getpass
import grp
import os
import shlex
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta


def _cluster_tz():
    try:
        from iitgpu.config import cluster_tz
        return cluster_tz()
    except Exception:
        return timezone(timedelta(hours=5, minutes=30))
from pathlib import Path


# The cluster's single GPU is split into SHARDS_PER_GPU schedulable slices
# (gres.conf: "Name=shard Count=4 File=/dev/nvidia0"), so several jobs can run
# on it concurrently instead of one job locking the whole card. Jobs therefore
# request --gres=shard:N, never --gres=gpu:N — asking for a whole gpu takes the
# device *and* all its shards, which is exactly the blocking we removed.
# Keep this in sync with gres.conf on every node.
SHARDS_PER_GPU = 4


@dataclass(frozen=True)
class TaskDefaults:
    gpu_shards: int   # GPU slices to request; SHARDS_PER_GPU == the whole card
    cpus: int
    mem_gb: int
    time_limit: str  # "" means no time limit (SLURM INFINITE)


# Interactive/light work takes one slice so several users fit on the card at
# once; training-shaped work still takes the whole GPU because it wants all the
# VRAM. Shards are a *scheduling* split, not a VRAM cap — see gpu_share_note().
#
# CPU and RAM must be sized to match the split, or they become the new
# bottleneck: a one-slice job asking for 32G RAM still blocks a second one on a
# 60G node, and the GPU sits mostly idle exactly as it did before sharding.
# A single-slice job therefore gets about a quarter of the node.
_SLICE_CPUS = 8    # 32 cores / 4 slices
_SLICE_MEM_GB = 14  # ~60 GiB / 4 slices, with headroom so four really fit

TASK_DEFAULTS: dict[str, TaskDefaults] = {
    "train":     TaskDefaults(gpu_shards=SHARDS_PER_GPU, cpus=16, mem_gb=60, time_limit=""),
    "finetune":  TaskDefaults(gpu_shards=SHARDS_PER_GPU, cpus=16, mem_gb=60, time_limit=""),
    "inference": TaskDefaults(gpu_shards=1, cpus=_SLICE_CPUS, mem_gb=_SLICE_MEM_GB,
                              time_limit="04:00:00"),
    "test":      TaskDefaults(gpu_shards=1, cpus=4, mem_gb=8, time_limit="00:30:00"),
    "notebook":  TaskDefaults(gpu_shards=1, cpus=_SLICE_CPUS, mem_gb=_SLICE_MEM_GB,
                              time_limit="08:00:00"),
    # An interactive shell mostly sits idle at a prompt — one slice, so it never
    # parks on the whole card while someone reads their scrollback.
    "interactive": TaskDefaults(gpu_shards=1, cpus=_SLICE_CPUS, mem_gb=_SLICE_MEM_GB,
                                time_limit="02:00:00"),
    "custom":    TaskDefaults(gpu_shards=SHARDS_PER_GPU, cpus=16, mem_gb=60, time_limit=""),
}


def resource_defaults(task_type: str) -> TaskDefaults:
    return TASK_DEFAULTS.get(task_type, TASK_DEFAULTS["custom"])


def gres_directive(gpu_shards: int) -> str:
    """SLURM --gres value for a shard request, or "" when the job needs no GPU."""
    return f"shard:{gpu_shards}" if gpu_shards > 0 else ""


def gpu_share_note(gpu_shards: int) -> str:
    """Plain-language description of how much of the GPU a request reserves."""
    if gpu_shards <= 0:
        return "no GPU (CPU-only)"
    if gpu_shards >= SHARDS_PER_GPU:
        return "the whole GPU (no one else can use it)"
    return (f"{gpu_shards}/{SHARDS_PER_GPU} of the GPU "
            f"({SHARDS_PER_GPU - gpu_shards}/{SHARDS_PER_GPU} left for others)")


@dataclass
class JobSpec:
    job_name: str
    partition: str
    gpu_shards: int
    cpus: int
    mem_gb: int
    time_limit: str
    run_command: str
    modules: list[str] = field(default_factory=list)
    uploads: list[str] = field(default_factory=list)
    user: str = field(default_factory=getpass.getuser)
    model_path: str = ""
    conda_env: str = ""
    venv_path: str = ""
    task_type: str = "custom"
    container_image: str = ""  # path to .sif — when set, skips conda/venv
    array: str = ""            # SLURM --array spec, e.g. "0-9" or "1-100%4"
    dependency: str = ""       # SLURM --dependency, e.g. "afterok:12345"
    data_path: str = ""        # path exported as DATA_PATH in the sbatch script
    mail_user: str = ""        # email for SLURM mail directives (if MTA present)


def make_job_folder(jobs_dir: str, spec: JobSpec) -> str:
    timestamp = datetime.now(_cluster_tz()).strftime("%Y%m%d_%H%M%S")
    folder = Path(jobs_dir) / spec.user / f"{spec.job_name}_{timestamp}"
    # Owner + admin group only, no world access. A job folder holds the
    # user's scripts and output — the same private content as their home —
    # so other users get nothing.
    #
    # How "other" is kept out: force a 0o007 umask around mkdir. That strips
    # "other" access on every directory it creates (the leaf, and any
    # missing intermediate dirs), yielding 0770 with no privilege required.
    #
    # How admin access actually works: setgid IS inherited normally on this
    # filesystem. jobs/<user> is provisioned 2770 owner:gpuadmins (see
    # deploy/iit-gpu-adduser.sh / deploy/fix-shared-perms.sh), and mkdir
    # under a setgid parent yields a setgid child with the parent's group —
    # here that means this timestamped job folder comes out 2770 gpuadmins,
    # and files written inside it inherit group gpuadmins too, the same as
    # it would on any other filesystem (e.g. /tmp).
    #
    # (An earlier version of this comment claimed the opposite — that setgid
    # does NOT propagate on this NFS share, based on a live mkdir comparison
    # showing a 2770 parent yielding a 5770/1770 child. That measurement used
    # this host's `/usr/bin/mkdir`, which is uutils coreutils 0.8.0 and
    # mishandles ACL-bearing parent directories. Re-checked against the raw
    # syscall — `python3 -c "import os; os.mkdir(...)"` — under the identical
    # parent, same non-admin user: the child comes out 2770 with setgid set,
    # as expected. `Path.mkdir` below goes through `os.mkdir`, i.e. the raw
    # syscall, not the coreutils binary, so production job folders do get
    # setgid. On this host, verify mode-bit behaviour with a raw syscall, not
    # a coreutils binary — the same `mkdir` bug, `getfacl` fabricating ACLs
    # over the NFS mount, and a missing `stat --cached=never` have each
    # produced a wrong answer here before.)
    #
    # A `gpuadmins` ACL is also applied to every per-user area (jobs/<user>
    # and users/<user>) by deploy/fix-shared-perms.sh and
    # deploy/iit-gpu-adduser.sh, as a second, independent layer — it's what
    # deploy/check-shared-perms.sh actually verifies, since traditional
    # owner/group/mode bits can look correct while an ACL entry disagrees
    # (`stat` won't show that; check with `getfacl` on the GPU host).
    #
    # We still never chmod(0o2770) here to "add" setgid: POSIX chmod()
    # silently clears S_ISGID whenever the caller isn't privileged or a
    # member of the file's current group, so a trailing chmod by a regular
    # submitting user (not in gpuadmins) would strip any setgid bit the
    # mkdir above already produced, for zero benefit. The umask trick still
    # can't grant setgid on its own; absent a setgid ancestor (e.g. a plain
    # tmp dir in a unit test) mkdir just yields a safe 770 with no setgid —
    # a lesser but still private outcome, not a security hole.
    prev_umask = os.umask(0o007)
    try:
        folder.mkdir(parents=True, exist_ok=True)
    finally:
        os.umask(prev_umask)
    try:
        from iitgpu.config import load_config
        # Best-effort: try to set the group explicitly too, as a second line
        # of defense alongside the setgid inheritance and gpuadmins ACL
        # above. This still requires the caller to be a member of
        # admin_group, which a regular submitting user is not, so it
        # routinely no-ops for them — that's fine; setgid inheritance from
        # the jobs/<user> parent already covers the common case.
        # Previously group gpuusers, so a shared submit account could read the
        # script under gateway_shared_user mode. That mode is off here
        # (shared_user_mode=False); if it's ever enabled this needs revisiting.
        gid = grp.getgrnam(load_config().admin_group).gr_gid
        os.chown(str(folder), -1, gid)
    except (KeyError, PermissionError, OSError):
        pass   # best-effort; sbatch will fail with a clear error if still blocked
    return str(folder)


def save_upload(src: str, folder: str) -> str:
    dest = Path(folder) / Path(src).name
    shutil.copy2(src, dest)
    return str(dest)


def render_sbatch(spec: JobSpec, folder: str) -> str:
    lines = [
        "#!/bin/bash",
        # Use the full job-folder name (e.g. finetune_20260601_045303) as the
        # SLURM job name so queue/sacct/log listings are unambiguous per run,
        # instead of every finetune showing up as just "finetune".
        f"#SBATCH --job-name={Path(folder).name}",
        f"#SBATCH --partition={spec.partition}",
        f"#SBATCH --cpus-per-task={spec.cpus}",
        f"#SBATCH --mem={spec.mem_gb}G",
    ]
    if spec.gpu_shards > 0:
        lines.append(f"#SBATCH --gres={gres_directive(spec.gpu_shards)}")
    if spec.time_limit:
        lines.append(f"#SBATCH --time={spec.time_limit}")
    if spec.array:
        lines.append(f"#SBATCH --array={spec.array}")
    if spec.dependency:
        lines.append(f"#SBATCH --dependency={spec.dependency}")
    if spec.mail_user:
        from iitgpu.config import load_config
        mail_types = load_config().notify_mail_types
        lines.append(f"#SBATCH --mail-user={spec.mail_user}")
        lines.append(f"#SBATCH --mail-type={mail_types}")
    lines += [
        (f"#SBATCH --output={folder}/slurm-%A_%a.out"
         if spec.array else f"#SBATCH --output={folder}/slurm-%j.out"),
        (f"#SBATCH --error={folder}/slurm-%A_%a.err"
         if spec.array else f"#SBATCH --error={folder}/slurm-%j.err"),
        f"#SBATCH --chdir={folder}",
        "",
    ]

    for mod in spec.modules:
        lines.append(f"module load {mod}")
    if spec.modules:
        lines.append("")

    if spec.container_image:
        # Container mode: wrap run_command in apptainer exec; skip conda/venv
        lines.append(f"cd {folder}")
        lines.append(
            f"apptainer exec --nv --bind /shared {spec.container_image} "
            f"bash -lc {spec.run_command!r}"
        )
        return "\n".join(lines) + "\n"

    if spec.conda_env:
        # Source conda.sh before activating — required in non-interactive bash
        # (sbatch scripts). Without this, `conda activate` silently fails and the
        # job runs in the base Python environment instead of the requested env.
        # CONDA_PREFIX_SHARED is set by the launcher; fallback covers direct sbatch.
        lines += [
            '_conda_sh="${CONDA_PREFIX_SHARED:-/shared/miniforge3}/etc/profile.d/conda.sh"',
            '[ -f "$_conda_sh" ] && source "$_conda_sh"',
            f"conda activate {spec.conda_env}",
            "",
        ]
    elif spec.venv_path:
        lines.append(f"source {spec.venv_path}/bin/activate")
        lines.append("")

    if spec.model_path:
        lines.append(f"export MODEL_PATH={spec.model_path}")
        lines.append(f"export HF_HOME={spec.model_path}")
        lines.append("")

    if spec.data_path:
        lines.append(f"export DATA_PATH={spec.data_path}")
        lines.append("")

    lines.append(f"cd {folder}")
    lines.append(spec.run_command)
    return "\n".join(lines) + "\n"


def write_sbatch(spec: JobSpec, folder: str) -> str:
    path = Path(folder) / "job.sbatch"
    path.write_text(render_sbatch(spec, folder))
    path.chmod(0o644)
    return str(path)



def pip_install_block(requirements: str = "", packages: str = "") -> str:
    """Bash that pip-installs declared deps into the user site (~/.local) BEFORE a
    run. --user keeps shared prebuilt envs unpolluted while staying importable
    from any conda env (user site is on sys.path unless PYTHONNOUSERSITE)."""
    if requirements:
        req = shlex.quote(requirements)
        return (
            f'echo "Installing dependencies from {Path(requirements).name} ..."\n'
            f"python3 -m pip install --user --no-warn-script-location -r {req} \\\n"
            '    || { echo "Dependency install FAILED - see pip output above." >&2; exit 1; }\n'
            'export PATH="$HOME/.local/bin:$PATH"\n'
        )
    if packages:
        toks = " ".join(shlex.quote(t) for t in packages.split())
        return (
            f'echo "Installing dependencies: {packages} ..."\n'
            f"python3 -m pip install --user --no-warn-script-location {toks} \\\n"
            '    || { echo "Dependency install FAILED - see pip output above." >&2; exit 1; }\n'
            'export PATH="$HOME/.local/bin:$PATH"\n'
        )
    return ""



def build_interactive_cmd(spec: "JobSpec", partition: str = "gpu") -> list[str]:
    """Build an `srun --pty` interactive GPU session command.

    Runs a real shell ON the compute node inside a SLURM allocation. It is bound
    to the allocation (exits when the job ends / time limit hits) and is NOT a
    host login shell — the user only reaches the node through SLURM.
    """
    cmd = [
        "srun",
        f"--partition={spec.partition or partition}",
        f"--cpus-per-task={spec.cpus}",
        f"--mem={spec.mem_gb}G",
    ]
    if spec.gpu_shards > 0:
        cmd.insert(2, f"--gres={gres_directive(spec.gpu_shards)}")
    if spec.time_limit:
        cmd.append(f"--time={spec.time_limit}")
    cmd += ["--pty", "bash", "-l"]
    return cmd


# Bash that resolves the compute node's gateway-reachable address at runtime.
#
# Interactive services (JupyterLab / TensorBoard) run on a GPU *compute* node,
# but users only ever SSH into the *login/gateway* node — a different host. If
# the service binds to 127.0.0.1 it lives on the compute node's loopback, which
# the gateway cannot reach, so the user's `-L <port>:localhost:<port>` tunnel
# (terminating on the login node) connects to nothing. The job therefore looks
# "started" yet is unreachable, and users give up and cancel it.
#
# Fix: bind to the node's SLURM NodeAddr — routable from the gateway over the
# cluster network but NOT the public-facing interface — and forward the tunnel
# to that same address. The per-job random token still gates access. We resolve
# NodeAddr via scontrol (authoritative), falling back to the first local IP and
# finally loopback so the script is always well-defined.
_NODE_ADDR_SNIPPET = [
    'IIT_NODE_ADDR=$(scontrol show node "${SLURMD_NODENAME:-$(hostname -s)}" 2>/dev/null'
    ' | sed -n "s/.*NodeAddr=\\([^ ]*\\).*/\\1/p")',
    '[ -z "$IIT_NODE_ADDR" ] && IIT_NODE_ADDR=$(hostname -I | awk "{print \\$1}")',
    '[ -z "$IIT_NODE_ADDR" ] && IIT_NODE_ADDR=127.0.0.1',
    "",
]


# Bash that resolves a free TCP port into $IIT_PORT before anything is printed.
#
# The GPU is shared, so several notebooks/TensorBoards now run at once — and a
# fixed port collides. JupyterLab reacts by silently moving to the next free
# port ("The port 8888 is already in use, trying another port") while the tunnel
# instructions printed above it still name the ORIGINAL port. The user then
# forwards to a port nothing is listening on and the notebook looks broken.
#
# So the port must be chosen *before* the instructions are written, and the
# service pinned to it. Each job starts scanning from an offset derived from its
# job ID, so two jobs starting at the same moment rarely probe the same port;
# the scan then wraps to cover the whole range.
def _free_port_snippet(base_port: int) -> list[str]:
    # Quoted heredoc ('PYEOF') so the shell expands nothing inside the Python.
    return [
        "IIT_PORT=$(python3 <<'PYEOF'",
        "import os, socket, sys",
        f"base = {base_port}",
        'start = base + (int(os.environ.get("SLURM_JOB_ID", "0")) % 50)',
        "for p in list(range(start, base + 100)) + list(range(base, start)):",
        "    s = socket.socket()",
        "    try:",
        '        s.bind(("", p))',
        "    except OSError:",
        "        continue",
        "    finally:",
        "        s.close()",
        "    print(p)",
        "    break",
        "else:",
        "    sys.exit(1)",
        "PYEOF",
        ")",
        '[ -z "$IIT_PORT" ] && { echo "ERROR: no free port on this node — try again." >&2; exit 1; }',
        "",
    ]


# Bash that defines $IIT_USER_ROOT and exposes the shared assets inside it.
#
# JupyterLab used to be rooted at /shared, so its file browser ignored the
# per-user jail the file manager enforces. It is now rooted at the user's own
# folder, with the shared read-only assets symlinked in so people can still
# reach the datasets they train on. Jupyter follows symlinks pointing outside
# root_dir, which is what makes this work.
#
# ln -sfn is idempotent: the job reruns this on every launch. But -sfn only
# replaces an existing SYMLINK at the destination -- if the destination is
# already a real file or directory (e.g. users/hassan2/envs, users/public/data
# are real dirs some users have from before this existed), ln does not
# clobber it: it creates the link *inside* it instead (.../envs/envs), so the
# user never gets the shared asset at the documented path and a stray link
# piles up on every relaunch. Guard each one: only link when the destination
# is absent or is already a symlink (any target -- ln -sfn safely repoints an
# existing symlink); otherwise skip it and warn so the user knows why that
# asset is missing rather than silently getting a nested stray link.
def _user_home_snippet() -> list[str]:
    lines = ['IIT_USER_ROOT="/shared/users/$USER"', 'mkdir -p "$IIT_USER_ROOT"']
    for asset in ("models", "envs", "data", "datasets"):
        target = f'"$IIT_USER_ROOT/{asset}"'
        lines += [
            f'_iit_target={target}',
            f'if [ -L "$_iit_target" ] || [ ! -e "$_iit_target" ]; then',
            f'    ln -sfn /shared/{asset} "$_iit_target"',
            "else",
            f'    echo "WARNING: $_iit_target already exists and is not a symlink -- skipping the shared {asset} asset (it will not be reachable at that path)." >&2',
            "fi",
        ]
    lines.append("")
    return lines


def render_notebook_sbatch(
    spec: "JobSpec",
    folder: str,
    port: int = 8888,
    gateway_host: str = "localhost",
    gateway_port: int = 22,
    requirements: str = "",
    packages: str = "",
) -> str:
    """Generate an sbatch script that launches JupyterLab on the GPU node.

    The script:
    - Binds JupyterLab to the node's SLURM NodeAddr — reachable from the gateway
      network but not the public interface — gated by a per-job random token
    - Optionally pip-installs *requirements*/*packages* into ~/.local first, so the
      interactive session starts with the user's deps ready (no first-cell import
      failures)
    - Writes port + token to the job's stdout
    - Prints the exact SSH tunnel command for the user's laptop
    - Shuts JupyterLab down when the job's time limit is reached
    Token is per-job and appears only in the job's stdout (never logged).
    """
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={spec.job_name}",
        f"#SBATCH --partition={spec.partition}",
        f"#SBATCH --cpus-per-task={spec.cpus}",
        f"#SBATCH --mem={spec.mem_gb}G",
    ]
    if spec.gpu_shards > 0:
        lines.append(f"#SBATCH --gres={gres_directive(spec.gpu_shards)}")
    if spec.time_limit:
        lines.append(f"#SBATCH --time={spec.time_limit}")
    if spec.mail_user:
        from iitgpu.config import load_config
        mail_types = load_config().notify_mail_types
        lines.append(f"#SBATCH --mail-user={spec.mail_user}")
        lines.append(f"#SBATCH --mail-type={mail_types}")
    lines += [
        f"#SBATCH --output={folder}/slurm-%j.out",
        f"#SBATCH --error={folder}/slurm-%j.err",
        f"#SBATCH --chdir={folder}",
        "",
    ]

    # Environment activation (container or conda/venv)
    if spec.container_image:
        # Notebook inside container — wrap the entire jupyter launch
        launcher = (
            f"apptainer exec --nv --bind /shared {spec.container_image} "
            f"jupyter lab --no-browser --ip=$IIT_NODE_ADDR --port=$IIT_PORT --port-retries=0 "
            f"--ServerApp.root_dir=$IIT_USER_ROOT --IdentityProvider.token=\"$JUPYTER_TOKEN\""
        )
    else:
        if spec.conda_env:
            lines += [
                '_conda_sh="${CONDA_PREFIX_SHARED:-/shared/miniforge3}/etc/profile.d/conda.sh"',
                '[ -f "$_conda_sh" ] && source "$_conda_sh"',
                f"conda activate {spec.conda_env}",
                "",
            ]
        elif spec.venv_path:
            lines.append(f"source {spec.venv_path}/bin/activate")
            lines.append("")

        # JupyterLab must be on PATH for the launch below. Not every environment
        # ships it (a plain PyTorch env, say), which previously left the job
        # dying with "jupyter: command not found". Install it into the user's
        # site (~/.local) on the fly when missing so the notebook still comes up;
        # fail loudly with guidance if even that cannot be done.
        lines += [
            "if ! command -v jupyter >/dev/null 2>&1; then",
            '    echo "JupyterLab not found in this environment - installing it (one-time)..."',
            '    python3 -m pip install --user --quiet --no-warn-script-location jupyterlab \\',
            '        || { echo "ERROR: JupyterLab is missing and could not be installed automatically." >&2; \\',
            '             echo "       Use an environment that includes JupyterLab (e.g. the data-science prebuilt env)." >&2; exit 1; }',
            '    export PATH="$HOME/.local/bin:$PATH"',
            "fi",
            "",
        ]

        # Optional: pre-install the user's deps so the interactive session is
        # ready (e.g. tensorboard/tqdm) instead of failing on the first cell.
        _deps = pip_install_block(requirements, packages)
        if _deps:
            lines.append(_deps.rstrip("\n"))
            lines.append("")

        launcher = (
            f"jupyter lab --no-browser --ip=$IIT_NODE_ADDR --port=$IIT_PORT --port-retries=0 "
            f"--ServerApp.root_dir=$IIT_USER_ROOT --IdentityProvider.token=\"$JUPYTER_TOKEN\""
        )

    lines += _NODE_ADDR_SNIPPET + _free_port_snippet(port) + _user_home_snippet()
    lines += [
        "# Generate a random per-job token (not logged beyond this script's stdout)",
        'JUPYTER_TOKEN=$(python3 -c "import secrets; print(secrets.token_hex(24))")',
        "",
        "echo '================================================='",
        "echo 'JupyterLab is starting on the GPU node.'",
        'echo "Token: $JUPYTER_TOKEN"',
        "echo 'SSH tunnel — open a NEW terminal on YOUR LAPTOP and run:'",
        f'echo "  ssh -p {gateway_port} -N -L $IIT_PORT:$IIT_NODE_ADDR:$IIT_PORT $USER@{gateway_host}"',
        "echo '  (-N = tunnel only, no shell opens — terminal sitting idle is correct)'",
        f'echo "Then open in browser: http://127.0.0.1:$IIT_PORT/lab?token=$JUPYTER_TOKEN"',
        "echo '================================================='",
        "",
        launcher,
    ]
    return "\n".join(lines) + "\n"


def write_notebook_sbatch(
    spec: "JobSpec",
    folder: str,
    port: int = 8888,
) -> str:
    """Write the notebook sbatch script to folder/job.sbatch."""
    path = Path(folder) / "job.sbatch"
    path.write_text(render_notebook_sbatch(spec, folder, port=port))
    path.chmod(0o644)
    return str(path)


def render_tensorboard_sbatch(spec, folder, logdir, port=6006,
                              gateway_host="localhost", gateway_port=22):
    """sbatch that launches TensorBoard bound to 127.0.0.1 and prints the
    SSH tunnel command. Reuses conda/container activation like notebooks."""
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=tensorboard",
        f"#SBATCH --partition={spec.partition}",
        f"#SBATCH --cpus-per-task={spec.cpus}",
        f"#SBATCH --mem={spec.mem_gb}G",
    ]
    if spec.time_limit:
        lines.append(f"#SBATCH --time={spec.time_limit}")
    if spec.mail_user:
        from iitgpu.config import load_config
        mail_types = load_config().notify_mail_types
        lines.append(f"#SBATCH --mail-user={spec.mail_user}")
        lines.append(f"#SBATCH --mail-type={mail_types}")
    lines += [
        f"#SBATCH --output={folder}/slurm-%j.out",
        f"#SBATCH --error={folder}/slurm-%j.err",
        f"#SBATCH --chdir={folder}",
        "",
    ]
    if spec.container_image:
        launcher = (
            f"apptainer exec --bind /shared {spec.container_image} "
            f"tensorboard --logdir {logdir} --port $IIT_PORT --host $IIT_NODE_ADDR"
        )
    else:
        if spec.conda_env:
            lines += [
                '_conda_sh="${CONDA_PREFIX_SHARED:-/shared/miniforge3}/etc/profile.d/conda.sh"',
                '[ -f "$_conda_sh" ] && source "$_conda_sh"',
                f"conda activate {spec.conda_env}",
                "",
            ]
        launcher = f"tensorboard --logdir {logdir} --port $IIT_PORT --host $IIT_NODE_ADDR"
    lines += _NODE_ADDR_SNIPPET + _free_port_snippet(port)
    lines += [
        "echo \'=================================================\'",
        "echo \'TensorBoard starting.\'",
        "echo \'SSH tunnel — open a NEW terminal on YOUR LAPTOP and run:\'",
        f'echo "  ssh -p {gateway_port} -N -L $IIT_PORT:$IIT_NODE_ADDR:$IIT_PORT $USER@{gateway_host}"',
        "echo \'  (-N = tunnel only, no shell opens — terminal sitting idle is correct)\'",
        f'echo "Then open: http://127.0.0.1:$IIT_PORT"',
        "echo \'=================================================\'",
        "",
        launcher,
    ]
    return "\n".join(lines) + "\n"
