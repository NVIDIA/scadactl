# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manage global NVIDIA MPS via ``nvidia-cuda-mps-control``.

* ``start``  -- ensure the pipe/log directories exist, then
  ``nvidia-cuda-mps-control -d`` with optional ``-multiuser-server``.
  Reuse an existing daemon only when its mode matches.
* ``stop``   -- ``quit`` sent to the running control daemon.
* ``status`` -- ``get_server_list`` against the running control daemon.

The control binary defaults to ``nvidia-cuda-mps-control`` (on PATH), overridable
with ``SCADA_MPS_CONTROL_BIN``. Pipe/log directories follow the standard
``CUDA_MPS_PIPE_DIRECTORY`` / ``CUDA_MPS_LOG_DIRECTORY`` env vars, defaulting to
``/tmp/nvidia-mps`` and ``/tmp/nvidia-log``.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Optional

DEFAULT_CONTROL_BIN = "nvidia-cuda-mps-control"
DEFAULT_PIPE_DIR = "/tmp/nvidia-mps"
DEFAULT_LOG_DIR = "/tmp/nvidia-log"


def control_binary() -> str:
    return os.environ.get("SCADA_MPS_CONTROL_BIN") or DEFAULT_CONTROL_BIN


def pipe_directory() -> str:
    return os.environ.get("CUDA_MPS_PIPE_DIRECTORY") or DEFAULT_PIPE_DIR


def log_directory() -> str:
    return os.environ.get("CUDA_MPS_LOG_DIRECTORY") or DEFAULT_LOG_DIR


@dataclass
class MpsResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def _mps_env() -> dict:
    """Daemon/control environment with the resolved pipe/log directories."""
    env = dict(os.environ)
    env["CUDA_MPS_PIPE_DIRECTORY"] = pipe_directory()
    env["CUDA_MPS_LOG_DIRECTORY"] = log_directory()
    return env


def _proc_pipe_dir(pid: str) -> Optional[str]:
    """The ``CUDA_MPS_PIPE_DIRECTORY`` a process was started with (or the MPS
    default when it set none), read from ``/proc``; ``None`` if unreadable."""
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            entries = f.read().split(b"\x00")
    except OSError:
        return None
    for e in entries:
        if e.startswith(b"CUDA_MPS_PIPE_DIRECTORY="):
            return e.split(b"=", 1)[1].decode("utf-8", "replace")
    return DEFAULT_PIPE_DIR


def _proc_args(pid: str) -> Optional[list[str]]:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().decode("utf-8", "replace").strip("\x00").split("\x00")
    except OSError:
        return None


def _is_control(args: list[str]) -> bool:
    return bool(args) and os.path.basename(args[0]) == "nvidia-cuda-mps-control"


def is_running() -> bool:
    """True if an MPS control daemon is running for the *resolved* pipe dir.

    MPS control/server pairs are keyed by ``CUDA_MPS_PIPE_DIRECTORY``, so a
    daemon serving a different pipe directory is a different instance. Matching
    any ``nvidia-cuda-mps-control`` pid (as a bare ``pgrep -f`` does) would
    wrongly conclude ours is already up and skip starting it, leaving a custom
    pipe-dir deployment disconnected. Confirm a candidate pid actually uses our
    pipe directory via its ``/proc`` environ.
    """
    want = pipe_directory()
    try:
        proc = subprocess.run(
            ["pgrep", "-f", "nvidia-cuda-mps-control"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False
    if proc.returncode != 0:
        return False
    return any(_proc_pipe_dir(pid) == want and _is_control(_proc_args(pid) or []) for pid in proc.stdout.split())


def start_command(multiuser: bool = False) -> list[str]:
    return [control_binary(), "-d"] + (["-multiuser-server"] if multiuser else [])


def compatibility_error(multiuser: bool = False) -> Optional[str]:
    """Check an existing daemon without restarting it or changing host state."""
    if multiuser and os.geteuid() != 0:
        return "multiuser MPS must be started as root"
    try:
        proc = subprocess.run(["pgrep", "-f", "nvidia-cuda-mps-control"], capture_output=True, text=True, check=False)
    except OSError as exc:
        return f"cannot inspect MPS daemons with pgrep: {exc}; install procps and retry"
    if proc.returncode not in (0, 1):
        return f"cannot inspect MPS daemons with pgrep: {proc.stderr.strip() or f'exit {proc.returncode}'}"
    for pid in proc.stdout.split():
        if _proc_pipe_dir(pid) != pipe_directory():
            continue
        args = _proc_args(pid)
        if args is None:
            return f"cannot determine MPS daemon {pid} mode; inspect it before applying"
        # pgrep -f may also match a shell command mentioning the control binary.
        if not _is_control(args):
            continue
        if not any(a in args for a in ("-d", "--daemon", "-f", "--forground")):
            return f"cannot determine MPS daemon {pid} mode; inspect it before applying"
        actual = any(a in args for a in ("-M", "--multiuser-server", "-multiuser-server"))
        if actual != multiuser:
            return (
                f"MPS daemon {pid} at {pipe_directory()} uses "
                f"{'multiuser' if actual else 'single-user'} mode; requested "
                f"{'multiuser' if multiuser else 'single-user'}. Stop its workloads and daemon "
                "before changing mode."
            )
    return None


def _control(stdin: str, env: dict) -> MpsResult:
    """Send a command to the MPS control daemon over stdin."""
    proc = subprocess.run(
        [control_binary()],
        input=stdin,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    return MpsResult(proc.returncode, proc.stdout, proc.stderr)


def start(dry_run: bool = False, multiuser: bool = False) -> MpsResult:
    """Start the global MPS control daemon (idempotent)."""
    env = _mps_env()
    if dry_run:
        return MpsResult(
            returncode=0,
            stdout=(
                f"{' '.join(start_command(multiuser))} "
                f"(pipe={env['CUDA_MPS_PIPE_DIRECTORY']}, log={env['CUDA_MPS_LOG_DIRECTORY']})"
            ),
        )
    error = compatibility_error(multiuser)
    if error:
        return MpsResult(returncode=1, stderr=error)
    if is_running():
        return MpsResult(returncode=0, stdout="MPS control daemon already running")
    os.makedirs(env["CUDA_MPS_PIPE_DIRECTORY"], exist_ok=True)
    os.makedirs(env["CUDA_MPS_LOG_DIRECTORY"], exist_ok=True)
    proc = subprocess.run(
        start_command(multiuser),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    return MpsResult(proc.returncode, proc.stdout, proc.stderr)


def stop(dry_run: bool = False) -> MpsResult:
    """Stop the global MPS control daemon."""
    if dry_run:
        return MpsResult(returncode=0, stdout=f"echo quit | {control_binary()}")
    if not is_running():
        return MpsResult(returncode=0, stdout="MPS control daemon not running")
    return _control("quit\n", _mps_env())


def status(dry_run: bool = False) -> MpsResult:
    """Query the MPS control daemon's server list."""
    if dry_run:
        return MpsResult(
            returncode=0, stdout=f"echo get_server_list | {control_binary()}"
        )
    if not is_running():
        return MpsResult(returncode=1, stdout="MPS control daemon not running")
    return _control("get_server_list\n", _mps_env())
