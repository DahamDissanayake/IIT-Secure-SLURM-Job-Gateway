"""Interactive services must advertise the port they actually listen on.

Regression guard for the bug that shipped with GPU sharing: the script printed
"ssh -L 8888:..." and then ran "jupyter lab --port=8888", but with several
notebooks now sharing the GPU the port was often taken. JupyterLab silently
moved to 8889 ("The port 8888 is already in use, trying another port") while
the printed instructions still said 8888, so the user's tunnel reached nothing
and the notebook looked broken.

The port is therefore resolved at job runtime into $IIT_PORT, and every place
that names a port uses that one variable.
"""
import re

import pytest

from iitgpu.jobs import (
    JobSpec, render_notebook_sbatch, render_tensorboard_sbatch,
)


def _spec(**kw):
    base = dict(
        job_name="notebook", partition="gpu", gpu_shards=1, cpus=8, mem_gb=14,
        time_limit="08:00:00", run_command="", task_type="notebook",
        conda_env="/shared/envs/data-science",
    )
    base.update(kw)
    return JobSpec(**base)


def _notebook(tmp_path, **kw):
    return render_notebook_sbatch(
        _spec(), str(tmp_path), gateway_host="gw.edu", gateway_port=2225, **kw)


def _tensorboard(tmp_path, **kw):
    return render_tensorboard_sbatch(
        _spec(job_name="tensorboard"), str(tmp_path), "/shared/u/logs",
        gateway_host="gw.edu", gateway_port=2225, **kw)


# ── The port is resolved before anything is printed ───────────────────────────

@pytest.mark.parametrize("render", [_notebook, _tensorboard])
def test_port_is_resolved_at_runtime(tmp_path, render):
    script = render(tmp_path)
    assert "IIT_PORT=$(python3" in script
    # ...and the resolver must come before the instructions that quote it.
    assert script.index("IIT_PORT=$(python3") < script.index("$IIT_PORT:$IIT_NODE_ADDR")


@pytest.mark.parametrize("render", [_notebook, _tensorboard])
def test_tunnel_and_service_agree_on_the_port(tmp_path, render):
    script = render(tmp_path)
    assert "-L $IIT_PORT:$IIT_NODE_ADDR:$IIT_PORT" in script


def test_jupyter_binds_the_advertised_port_and_never_drifts(tmp_path):
    script = _notebook(tmp_path)
    assert "--port=$IIT_PORT" in script
    # Without this, a lost race silently moves Jupyter to another port and the
    # printed tunnel is wrong again — the original bug.
    assert "--port-retries=0" in script
    assert "http://127.0.0.1:$IIT_PORT/lab?token=$JUPYTER_TOKEN" in script


def test_tensorboard_serves_the_advertised_port(tmp_path):
    script = _tensorboard(tmp_path)
    assert "--port $IIT_PORT" in script
    assert "http://127.0.0.1:$IIT_PORT" in script


# ── No stale hardcoded port may survive anywhere that names one ───────────────

@pytest.mark.parametrize("render,base", [(_notebook, 8888), (_tensorboard, 6006)])
def test_no_hardcoded_port_in_user_facing_lines(tmp_path, render, base):
    """The base port may appear only in the resolver's scan range."""
    for line in render(tmp_path).splitlines():
        if line.startswith("base = "):
            continue
        if str(base) in line:
            pytest.fail(f"hardcoded port {base} leaked into: {line!r}")


@pytest.mark.parametrize("render", [_notebook, _tensorboard])
def test_custom_base_port_flows_into_the_scan(tmp_path, render):
    assert "base = 9123" in render(tmp_path, port=9123)


# ── The resolver itself behaves ───────────────────────────────────────────────

def _run_resolver(script: str, job_id: str, occupied: list[int]) -> int:
    """Run the generated resolver exactly as the job would, in a subprocess."""
    import os
    import socket
    import subprocess
    import sys

    body = re.search(r"IIT_PORT=\$\(python3 <<'PYEOF'\n(.*?)\nPYEOF",
                     script, re.S).group(1)
    held = []
    try:
        for p in occupied:
            s_ = socket.socket()
            s_.bind(("", p))
            s_.listen(1)
            held.append(s_)
        out = subprocess.run(
            [sys.executable, "-c", body],
            capture_output=True, text=True,
            env=dict(os.environ, SLURM_JOB_ID=job_id),
        )
        assert out.returncode == 0, out.stderr
        return int(out.stdout.strip())
    finally:
        for s_ in held:
            s_.close()


def test_resolver_skips_a_busy_port(tmp_path):
    script = _notebook(tmp_path)
    chosen = _run_resolver(script, "0", occupied=[8888, 8889])
    assert chosen not in (8888, 8889)


def test_resolver_spreads_concurrent_jobs_across_ports(tmp_path):
    """Two jobs submitted together must not race for the same port."""
    script = _notebook(tmp_path)
    a = _run_resolver(script, "331", occupied=[])
    b = _run_resolver(script, "332", occupied=[])
    assert a != b, f"both jobs targeted port {a}"
