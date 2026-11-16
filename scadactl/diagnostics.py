# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diagnostics: status, doctor, logs.

``status`` summarizes the active layout from the registry. ``doctor`` runs
host + domain checks, cross-checks registry vs live process state, and verifies
the live unvme mapping via the sysfs identity when present. ``logs`` collects
per-domain server logs.

All checks degrade gracefully: a missing tool/permission yields an actionable
message rather than a crash.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from scadactl import display
from scadactl import identity as identity_mod
from scadactl import registry as registry_mod
from scadactl.backend import launcher, lender
from scadactl.models import Inventory, Registry


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
def render_status(reg: Registry, details: bool = False) -> str:
    lines: List[str] = []
    lines.append(f"Configs: {', '.join(reg.active_configs) or '(none)'}")
    lines.append(f"Registry: {registry_mod.registry_path()}")
    lines.append("")
    if not reg.domains:
        lines.append("No active domains.")
        return "\n".join(lines)
    columns = [("domain", 14), ("backend", 8), ("gpu", 6), ("gpu_uuid", 16),
               ("socket", 28), ("profiler", 12), ("drives", 16), ("status", 8)]
    lines.extend(display.render_table(columns, []))
    for d in reg.domains:
        drives = ",".join(
            f"{os.path.basename(drv.unvme_device or drv.pci_bus_id or '')}/{drv.serial or ''}".strip("/")
            for drv in d.drives
        )
        status = "running" if d.healthy else "unknown"
        row = (d.name, d.storage_backend or "kernel", d.gpu_alias, d.gpu_uuid,
               d.socket, d.profiler_endpoint_mode, drives, status)
        lines.extend(display.render_table(columns, [row])[1:])
        if details:
            lines.append(f"    socket_group={d.socket_group or '(root only)'}")
            lines.append(
                f"    profiler={d.profiler_endpoint_mode} {d.profiler_socket}"
            )
            for k, v in d.server_env.items():
                lines.append(f"    env {k}={v}")
            lines.append(f"    config={d.config} pid={d.pid} cvd={d.host_cuda_visible_devices}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #
@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class DoctorReport:
    host_checks: List[Check] = field(default_factory=list)
    domain_checks: List[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.host_checks + self.domain_checks)


def run_doctor(
    reg: Optional[Registry] = None, inventory: Optional[Inventory] = None
) -> DoctorReport:
    """Run host + domain checks; cross-check registry vs live process state."""
    reg = reg if reg is not None else registry_mod.read_registry()
    report = DoctorReport()

    # Host checks.
    report.host_checks.append(_check_nvidia_smi())
    if any(d.storage_backend != "lender" for d in reg.domains):
        report.host_checks.append(_check_unvme_module())
    for socket in sorted({
        d.server_env.get("STORAGE_LENDER_SOCKET", "/run/storage-lender/api.sock")
        for d in reg.domains if d.storage_backend == "lender"
    }):
        report.host_checks.append(_check_lender_socket(socket))
    report.host_checks.append(_check_runtime_dir())
    report.host_checks.append(_check_registry_present())

    # Domain checks (unvme verification reads the driver-published sysfs
    # identity directly, so no inventory is required).
    for d in reg.domains:
        report.domain_checks.extend(_check_domain(d))
    return report


def _check_nvidia_smi() -> Check:
    from shutil import which

    if which("nvidia-smi"):
        return Check("nvidia-smi", True, "on PATH")
    return Check("nvidia-smi", False, "not found on PATH; install the NVIDIA driver")


def _check_unvme_module() -> Check:
    if os.path.isdir("/sys/class/unvme") or os.path.exists("/sys/module/unvme"):
        return Check("UNVMe module", True, "loaded")
    return Check(
        "UNVMe module",
        False,
        "not loaded; run apply or scada-load-modules to bind drives",
    )


def _check_lender_socket(socket: str) -> Check:
    if lender.is_unix_socket(socket):
        return Check("Storage Lender socket", True, socket)
    return Check(
        "Storage Lender socket", False,
        f"{socket} missing or not a Unix socket; start Storage Lender or set STORAGE_LENDER_SOCKET",
    )


def _check_runtime_dir() -> Check:
    rt = registry_mod.runtime_dir()
    if os.path.isdir(rt):
        return Check(f"{rt} permissions", True, "present")
    return Check(f"{rt} permissions", False, f"{rt} missing; created on apply")


def _check_registry_present() -> Check:
    p = registry_mod.registry_path()
    if os.path.exists(p):
        return Check("registry.json", True, p)
    return Check("registry.json", False, f"{p} absent; run apply")


def _check_domain(d) -> List[Check]:
    checks: List[Check] = []

    # Socket reachability.
    sock_ok = os.path.exists(d.socket)
    checks.append(
        Check(
            f"{d.name}: socket",
            sock_ok,
            "reachable" if sock_ok else f"missing socket {d.socket}",
        )
    )

    # Cross-check registry pid vs live process state.
    if d.pid:
        alive = _pid_alive(d.pid)
        checks.append(
            Check(
                f"{d.name}: process",
                alive,
                f"pid {d.pid} running" if alive else f"pid {d.pid} not running (stale)",
            )
        )

    # Verify the live unvme mapping by reading the driver-published sysfs
    # identity on the bound device and comparing to the recorded registry
    # identity. This needs no pre-bind inventory -- once a drive is bound to
    # unvme it no longer appears in ``nvme list``, so a fresh inventory cannot
    # supply its serial; the sysfs ``serial``/``address`` attributes can.
    for drv in d.drives:
        if not drv.unvme_device:
            continue
        label = f"{d.name}: unvme {drv.unvme_device}"
        name = os.path.basename(drv.unvme_device)  # /dev/unvme1 -> unvme1
        ident = identity_mod.read_unvme_sysfs_identity(name)
        if not ident:
            checks.append(
                Check(label, False,
                      "no sysfs identity on device (driver attributes absent?)")
            )
            continue
        live_serial = ident.get("serial")
        live_bdf = ident.get("pci_bdf")
        if drv.serial and live_serial:
            ok = live_serial == drv.serial
            checks.append(Check(
                label, ok,
                "mapping verified" if ok
                else f"serial mismatch: registry {drv.serial} vs live {live_serial}",
            ))
        elif drv.pci_bus_id and live_bdf:
            ok = live_bdf == drv.pci_bus_id
            checks.append(Check(
                label, ok,
                "mapping verified (by pci_bdf)" if ok
                else f"pci_bdf mismatch: registry {drv.pci_bus_id} vs live {live_bdf}",
            ))
        else:
            checks.append(
                Check(label, False,
                      "could not verify live mapping via sysfs identity")
            )
    return checks


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we lack permission to signal it.
        return True
    except OSError:
        return False


def render_doctor(report: DoctorReport) -> str:
    lines: List[str] = []
    lines.append("Host checks:")
    for c in report.host_checks:
        mark = "ok" if c.ok else "FAIL"
        lines.append(f"  {c.name}: {mark}" + (f" ({c.detail})" if c.detail else ""))
    lines.append("")
    lines.append("Domain checks:")
    if not report.domain_checks:
        lines.append("  (no active domains)")
    for c in report.domain_checks:
        mark = "ok" if c.ok else "FAIL"
        lines.append(f"  {c.name}: {mark}" + (f" ({c.detail})" if c.detail else ""))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# logs
# --------------------------------------------------------------------------- #
def collect_logs(reg: Registry, domain_names: Optional[List[str]] = None) -> dict:
    """Return a map of domain -> list of discovered per-domain log file paths."""
    out: dict = {}
    targets = reg.domains
    if domain_names:
        wanted = set(domain_names)
        targets = [d for d in reg.domains if d.name in wanted]
    for d in targets:
        log_dir = launcher.domain_log_dir(d.config, d.name)
        files: List[str] = []
        if os.path.isdir(log_dir):
            paths = [os.path.join(log_dir, fn) for fn in os.listdir(log_dir)]
            # Newest run first: log filenames are timestamped, so by mtime
            # (descending) the current run's logs lead the list.
            files = sorted(paths, key=lambda p: _safe_mtime(p), reverse=True)
        out[d.name] = files
    return out


def _safe_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0
