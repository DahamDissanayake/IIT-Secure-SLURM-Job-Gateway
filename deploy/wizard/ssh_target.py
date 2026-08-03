"""Uniform local-or-remote command execution.

The same stage code (gpu_node, login_node, mail, admin_provision, validate)
runs its commands against a Target -- either the local machine (the GPU
host the wizard is invoked on) or the login node over SSH (a freshly
created VM, or an existing machine). Stage code never branches on which.
"""
import shlex
import subprocess


class Target:
    def run(self, cmd: list[str], check: bool = True,
            input_text: str | None = None) -> subprocess.CompletedProcess:
        raise NotImplementedError

    def copy_to(self, local_path: str, remote_path: str) -> None:
        raise NotImplementedError


class LocalTarget(Target):
    def run(self, cmd, check=True, input_text=None):
        return subprocess.run(cmd, check=check, input=input_text, text=True,
                               capture_output=True)

    def copy_to(self, local_path, remote_path):
        subprocess.run(["cp", local_path, remote_path], check=True)


class SSHTarget(Target):
    def __init__(self, host: str, user: str, key_path: str):
        self.host = host
        self.user = user
        self.key_path = key_path

    def _ssh_base(self) -> list[str]:
        return [
            "ssh", "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-i", self.key_path, f"{self.user}@{self.host}",
        ]

    def run(self, cmd, check=True, input_text=None):
        remote_cmd = " ".join(shlex.quote(c) for c in cmd)
        return subprocess.run(self._ssh_base() + [remote_cmd], check=check,
                               input=input_text, text=True, capture_output=True)

    def copy_to(self, local_path, remote_path):
        subprocess.run([
            "scp", "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-i", self.key_path, local_path,
            f"{self.user}@{self.host}:{remote_path}",
        ], check=True)
