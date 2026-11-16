# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wrap ``scada-load-modules`` / ``scada-unload-modules`` (drive bind/unbind).

These scripts already encode validated safety behavior (mount / RAID / LVM /
swap exclusion, rebind bookkeeping under ``/var/lib/scada/unbound-devices``).
This wrapper calls them; it never reimplements that logic and never modifies the
scripts.

The installed scripts provide these interfaces:

* ``scada-load-modules --non-interactive`` to bind unattended/idempotently,
  plus machine-readable output of the resulting bind set.
* ``scada-unload-modules`` accepting an explicit set of PCI addresses for a
  targeted unbind (so ``stop --config`` releases one config's drives while
  others keep running), only ``rmmod`` when no managed devices remain.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional

LOAD_SCRIPT = "/usr/sbin/scada-load-modules"
UNLOAD_SCRIPT = "/usr/sbin/scada-unload-modules"


def _resolve_script(default: str, env_override: Optional[str]) -> str:
    """Return an installed helper path or an explicit absolute override.

    ``scadactl`` is distributed independently of the SCADA source tree. A
    relative override would therefore be ambiguous and, for commands commonly
    run as root, unnecessarily risky.
    """
    value = os.environ.get(env_override) if env_override else None
    value = value or default
    if not os.path.isabs(value):
        name = env_override or "helper path"
        raise ValueError(f"{name} must be an absolute path: {value!r}")
    return value


def load_script_path() -> str:
    return _resolve_script(LOAD_SCRIPT, "SCADA_LOAD_SCRIPT")


def unload_script_path() -> str:
    return _resolve_script(UNLOAD_SCRIPT, "SCADA_UNLOAD_SCRIPT")


@dataclass
class BindResult:
    returncode: int
    bound_pci: List[str] = field(default_factory=list)
    unvme_by_pci: Dict[str, str] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def _run(cmd: List[str], dry_run: bool) -> subprocess.CompletedProcess:
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def bind_drives(
    pci_addresses: List[str],
    dry_run: bool = False,
    non_interactive: bool = True,
) -> BindResult:
    """Bind the given PCI addresses to the unvme driver.

    Uses ``scada-load-modules --non-interactive --print-result`` so apply runs
    unattended and idempotently, preserving the script's safety checks, and so
    the script emits the machine-readable ``BOUND <pci> <unvmeN>`` lines this
    wrapper parses. ``dry_run`` returns the command that would run without
    executing it.
    """
    cmd = [load_script_path()]
    if non_interactive:
        cmd.append("--non-interactive")
    cmd.append("--print-result")
    cmd.extend(pci_addresses)
    if dry_run:
        return BindResult(returncode=0, bound_pci=list(pci_addresses), stdout=" ".join(cmd))
    proc = _run(cmd, dry_run=False)
    bound, unvme_by_pci = _parse_bound_set(proc.stdout)
    return BindResult(
        returncode=proc.returncode,
        # Report exactly what the script confirmed BOUND -- do NOT fall back to
        # the requested set. An empty bound set (the script emitted no BOUND
        # lines) must stay empty so apply's completeness check can distinguish
        # "nothing confirmed" from "all confirmed" (see apply.apply_config).
        bound_pci=bound,
        unvme_by_pci=unvme_by_pci,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def unbind_drives(
    pci_addresses: List[str],
    dry_run: bool = False,
) -> BindResult:
    """Targeted unbind of the given PCI addresses via ``scada-unload-modules``.

    Passing an explicit device set keeps other configs' bound drives intact;
    the script only ``rmmod``s when no managed devices remain.
    """
    cmd = [unload_script_path()]
    cmd.extend(pci_addresses)
    if dry_run:
        return BindResult(returncode=0, bound_pci=[], stdout=" ".join(cmd))
    proc = _run(cmd, dry_run=False)
    return BindResult(
        returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr
    )


def unbind_all(dry_run: bool = False) -> BindResult:
    """Full teardown + module unload (``stop --all`` path)."""
    cmd = [unload_script_path()]
    if dry_run:
        return BindResult(returncode=0, bound_pci=[], stdout=" ".join(cmd))
    proc = _run(cmd, dry_run=False)
    return BindResult(
        returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr
    )


def _parse_bound_set(stdout: str) -> "tuple[List[str], Dict[str, str]]":
    """Parse the machine-readable bind set from scada-load-modules output.

    With ``--print-result``, the script emits ``BOUND <pci> <unvmeN>`` lines
    (the node given bare, e.g. ``unvme3``). We parse those defensively into
    both the bound-pci list and a pci -> ``/dev/<node>`` map. Both remain empty
    if no bindings are confirmed; callers must not assume the request succeeded.
    """
    bound: List[str] = []
    unvme_by_pci: Dict[str, str] = {}
    for line in stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0] == "BOUND":
            pci = parts[1]
            bound.append(pci)
            if len(parts) >= 3:
                node = parts[2]
                unvme_by_pci[pci] = node if node.startswith("/dev/") else f"/dev/{node}"
    return bound, unvme_by_pci
