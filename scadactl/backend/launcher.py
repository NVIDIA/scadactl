# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Invoke ``manage-scada-domains`` once per domain.

One invocation per domain allows each domain to receive its own ``server_env``.
Each invocation gets:

* a single-line generated config (``<gpu_index>  <unvme_dev> [<unvme_dev>...]``)
* a per-domain ``--log-dir`` under ``/var/log/scada/<config>/``
* an environment carrying that domain's resolved ``server_env`` AND
  ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` (so the computed host CUDA index
  deterministically selects the intended physical GPU).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from urllib.parse import quote

MANAGE_SCRIPT = "/usr/sbin/manage-scada-domains"
LOG_ROOT = "/var/log/scada"
PIDFILE_NAME = "scada-swap-domains.pidfile"


def domain_binary(storage_backend: str) -> str:
    """Return the binary supplied by either mutually-exclusive server package."""
    del storage_backend
    return os.environ.get("SCADA_DOMAIN_BINARY", "scada-swap-domain")


def manage_script_path() -> str:
    """Return the system-installed manager or an explicit absolute override."""
    value = os.environ.get("SCADA_MANAGE_SCRIPT") or MANAGE_SCRIPT
    if not os.path.isabs(value):
        raise ValueError(f"SCADA_MANAGE_SCRIPT must be an absolute path: {value!r}")
    return value


def _safe_segment(value: str, fallback: str) -> str:
    """Reduce an arbitrary config/domain name to a single, safe path segment.

    Config and domain names are unconstrained strings in the model (``apply -f``
    does not charset-validate them), so percent-encode anything outside a safe
    filename allowlist. Unlike ``basename`` this is *injective*: distinct names
    map to distinct segments (``foo/bar`` -> ``foo%2Fbar`` != ``bar``), so two
    domains never collide onto one log/pidfile directory, while the encoded result
    still contains no ``/`` and so cannot escape ``LOG_ROOT``. Deterministic, so
    ``launch`` and ``stop`` agree on the path.
    """
    # quote() keeps [A-Za-z0-9_.-~] and encodes everything else (incl. "/") as %XX.
    seg = quote((value or "").strip(), safe="")
    # "." and ".." survive quote() unencoded, so guard them (and the empty string).
    return seg if seg and seg not in (".", "..") else fallback


def domain_log_dir(config: str, domain: str) -> str:
    """Per-domain log/pidfile directory: ``/var/log/scada/<config>/<domain>``.

    Each domain gets its own directory so the per-domain ``manage-scada-domains``
    invocations don't share (and truncate) one another's pidfile. This keeps
    ``stop`` granular for multi-domain configs. ``config``/``domain`` are
    normalized to a single path segment so a name containing ``/`` or ``..``
    cannot escape ``LOG_ROOT`` when the directory is created (as root) at launch.
    """
    return os.path.join(
        LOG_ROOT, _safe_segment(config, "config"), _safe_segment(domain, "domain")
    )


@dataclass
class LaunchResult:
    returncode: int
    config_line: str = ""
    log_dir: str = ""
    cmd: List[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    pid: Optional[int] = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def read_launched_pid(log_dir: str, gpu_index: str) -> Optional[int]:
    """Read the pid the manager recorded for ``gpu_index`` from its pidfile.

    The pidfile holds tab-separated ``<gpu>\\t<pid>\\t<exe>\\t<start>`` lines plus
    leading ``#`` comment lines. Returns None if the file/line is absent or
    unparsable. Read immediately after launch, before any sibling domain's
    invocation rewrites the shared pidfile.
    """
    path = os.path.join(log_dir, PIDFILE_NAME)
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 2 and parts[0] == str(gpu_index):
                    try:
                        return int(parts[1])
                    except ValueError:
                        return None
    except OSError:
        return None
    return None


def build_config_line(host_cuda_index: str, unvme_devices: List[str]) -> str:
    """Build the single-line manager config for one domain."""
    return " ".join([str(host_cuda_index), *unvme_devices])


def build_domain_env(resolved_server_env: Dict[str, str]) -> Dict[str, str]:
    """Compose the invocation environment for one domain.

    Carries the resolved server_env plus the deterministic CUDA device order.
    """
    env = dict(os.environ)
    env.update({k: str(v) for k, v in resolved_server_env.items()})
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return env


def launch_domain(
    config: str,
    domain: str,
    host_cuda_index: str,
    unvme_devices: List[str],
    resolved_server_env: Dict[str, str],
    socket_group: Optional[str] = None,
    profiler_socket: str = "",
    dry_run: bool = False,
    storage_backend: str = "kernel",
) -> LaunchResult:
    """Launch a single domain via one ``manage-scada-domains launch`` call.

    ``socket_group`` (the domain's ``socket_group``) is forwarded as ``--group``
    so the server sets a group-owned, mode-0660 socket; without it the socket is
    root-owned and unprivileged clients cannot connect.
    """
    config_line = build_config_line(host_cuda_index, unvme_devices)
    log_dir = domain_log_dir(config, domain)
    env = build_domain_env(resolved_server_env)
    group_args = ["--group", socket_group] if socket_group else []
    profiler_args = ["--profiler-socket", profiler_socket] if profiler_socket else []
    backend_args = [
        "--storage-backend", storage_backend,
        "--domain-binary", domain_binary(storage_backend),
    ]

    if dry_run:
        cmd = [
            manage_script_path(),
            "launch",
            "--config",
            "<generated>",
            "--log-dir",
            log_dir,
            *backend_args,
            *profiler_args,
            *group_args,
        ]
        return LaunchResult(
            returncode=0, config_line=config_line, log_dir=log_dir, cmd=cmd
        )

    os.makedirs(log_dir, exist_ok=True)
    fd, cfg_path = tempfile.mkstemp(prefix="scada-domain-", suffix=".cfg")
    try:
        os.write(fd, (config_line + "\n").encode("utf-8"))
        os.close(fd)
        cmd = [
            manage_script_path(),
            "launch",
            "--config",
            cfg_path,
            "--log-dir",
            log_dir,
            *backend_args,
            *profiler_args,
            *group_args,
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=env, check=False
        )
        pid = (
            read_launched_pid(log_dir, str(host_cuda_index))
            if proc.returncode == 0
            else None
        )
        return LaunchResult(
            returncode=proc.returncode,
            config_line=config_line,
            log_dir=log_dir,
            cmd=cmd,
            stdout=proc.stdout,
            stderr=proc.stderr,
            pid=pid,
        )
    finally:
        try:
            os.unlink(cfg_path)
        except OSError:
            pass


def stop_domain(
    config: str, domain: str, storage_backend: str = "kernel", dry_run: bool = False
) -> LaunchResult:
    """Stop one domain launched under its per-domain log dir (pidfile-verified)."""
    log_dir = domain_log_dir(config, domain)
    cmd = [
        manage_script_path(), "stop", "--log-dir", log_dir,
        "--domain-binary", domain_binary(storage_backend),
    ]
    if dry_run:
        return LaunchResult(returncode=0, log_dir=log_dir, cmd=cmd)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return LaunchResult(
        returncode=proc.returncode,
        log_dir=log_dir,
        cmd=cmd,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )
