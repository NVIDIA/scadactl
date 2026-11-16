# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host device policy: allow/block lists for GPUs and NVMe drives.

A host-wide, admin-owned policy that marks physical devices as off-limits
(block list) or confines SCADA to a specific set (allow list). It is advisory
in listings and config authoring, and enforced at ``apply``. An empty or absent
policy is a complete no-op, so default behavior is unchanged.

Semantics (per device class): the block list always wins; if an allow list is
non-empty, only devices on it are permitted (everything else is implicitly
blocked). Devices are keyed by stable identity (GPU UUID / drive serial, plus
PCI bus id), matching the config store's convention.

Persisted at ``/etc/scada/policy.yaml`` (world-readable, root-writable). Reads
are unprivileged; writes need root on a default install. Override the path with
``SCADA_POLICY_FILE`` (used by tests).
"""

from __future__ import annotations

import os
import tempfile
from typing import List, Optional, Tuple

import yaml

from scadactl.models import Policy, PolicyEntry

DEFAULT_POLICY_FILE = "/etc/scada/policy.yaml"


class PolicyError(RuntimeError):
    """Raised for policy I/O problems (permission denied, malformed file, …)."""


def policy_file() -> str:
    return os.environ.get("SCADA_POLICY_FILE") or DEFAULT_POLICY_FILE


def load() -> Policy:
    """Load the host policy, or an empty policy when the file is absent."""
    path = policy_file()
    if not os.path.isfile(path):
        return Policy()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as exc:
        raise PolicyError(f"cannot read policy {path}: {exc}") from exc
    if data is None:
        return Policy()
    if not isinstance(data, dict):
        raise PolicyError(f"policy {path}: top-level YAML must be a mapping")
    return Policy.from_dict(data)


def save(policy: Policy) -> str:
    """Atomically write the policy (0644). Returns the path. Needs root by default."""
    path = policy_file()
    d = os.path.dirname(path) or "."
    try:
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".policy.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                yaml.safe_dump(policy.to_dict(), f, sort_keys=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            os.chmod(path, 0o644)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    except PermissionError as exc:
        raise PolicyError(f"cannot write policy {path} (need root?): {exc}") from exc
    except OSError as exc:
        raise PolicyError(f"cannot write policy {path}: {exc}") from exc
    return path


# --------------------------------------------------------------------------- #
# Decisions
# --------------------------------------------------------------------------- #
def _decide(
    block: List[PolicyEntry],
    allow: List[PolicyEntry],
    primary: Optional[str],
    pci: Optional[str],
) -> Tuple[bool, Optional[str]]:
    for e in block:
        if e.matches(primary, pci):
            return True, (e.reason or "blocked by policy")
    if allow and not any(e.matches(primary, pci) for e in allow):
        return True, "not in allow-list"
    return False, None


def gpu_decision(
    policy: Policy, uuid: Optional[str], pci: Optional[str]
) -> Tuple[bool, Optional[str]]:
    """Return ``(blocked, reason)`` for a GPU under the given policy."""
    return _decide(policy.gpu_block, policy.gpu_allow, uuid, pci)


def drive_decision(
    policy: Policy, serial: Optional[str], pci: Optional[str]
) -> Tuple[bool, Optional[str]]:
    """Return ``(blocked, reason)`` for a drive under the given policy."""
    return _decide(policy.drive_block, policy.drive_allow, serial, pci)


# --------------------------------------------------------------------------- #
# List mutation helpers (used by the CLI)
# --------------------------------------------------------------------------- #
def upsert(entries: List[PolicyEntry], entry: PolicyEntry) -> str:
    """Add or update an entry (matched by key/PCI). Returns 'added' or 'updated'."""
    for i, e in enumerate(entries):
        if e.matches(entry.key, entry.pci_bus_id):
            entry.added = entry.added or e.added  # preserve original timestamp
            entries[i] = entry
            return "updated"
    entries.append(entry)
    return "added"


def remove(
    entries: List[PolicyEntry], primary: Optional[str], pci: Optional[str]
) -> bool:
    """Remove entries identifying the device. Returns True if any were removed."""
    before = len(entries)
    entries[:] = [e for e in entries if not e.matches(primary, pci)]
    return len(entries) != before
