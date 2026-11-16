# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reconciliation planning for ``apply`` -- compute the delta, don't replay setup.

Given a desired config, the host inventory, and the current registry, classify
each desired domain into the action ``apply`` should take:

* **noop** -- already set up and owned by this config in the registry, matching
  the desired shape; leave it running (an unchanged re-apply does nothing).
* **self_heal** -- owned by this config and matching the record, but the process
  has died; rebuild only the missing pieces.
* **build** -- target resources are free; build fresh.
* **reclaim** -- target drives are bound to ``unvme`` but *unmanaged* (no registry
  owner) and not in identifiable active use; free them, then build.

Two conditions block the whole apply (returned as ``plan.errors``, no mutation):

* a target resource is managed by a *different active* config (conflict), or
* this config is already active with a *different* definition than the registry
  records (incompatible -- an active config is immutable in place; stop then apply),
* a target ``unvme`` drive is held by a running process with no registry record,
  or its usage cannot be established (no ``fuser`` / probe error) -- refuse rather
  than free a possibly-in-use drive (we have no device-targeted backend kill and
  reclaim is destructive; fail closed).

This module is pure classification: it reads inventory/registry state and probes
process/device liveness through injectable callables, but performs no host
mutation. ``apply`` executes the plan.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

from scadactl.models import Config, Inventory, Registry, RegistryDomain


class DomainAction(str, Enum):
    NOOP = "noop"
    SELF_HEAL = "self_heal"
    BUILD = "build"
    RECLAIM = "reclaim"


@dataclass
class DomainPlan:
    """The action ``apply`` should take for one desired domain."""

    name: str
    action: DomainAction
    bind_pci: List[str] = field(default_factory=list)     # nvme drives to bind
    reclaim_pci: List[str] = field(default_factory=list)  # idle unmanaged unvme to free+rebind
    reasons: List[str] = field(default_factory=list)


@dataclass
class ReconcilePlan:
    config: str
    active: bool
    errors: List[str] = field(default_factory=list)   # config-level blockers (no mutation)
    domains: List[DomainPlan] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when the plan can be executed (no blocking conflict/incompatibility)."""
        return not self.errors

    def actionable(self) -> List[DomainPlan]:
        """Domains that require work (everything except a pure no-op)."""
        return [d for d in self.domains if d.action != DomainAction.NOOP]

    def bind_pci(self) -> List[str]:
        """Every drive to bind this apply (deduped, order-preserving)."""
        return _dedup(p for d in self.domains for p in d.bind_pci)

    def reclaim_pci(self) -> List[str]:
        """Every unmanaged drive to free before rebuilding (deduped)."""
        return _dedup(p for d in self.domains for p in d.reclaim_pci)


def _dedup(items) -> List[str]:
    seen: List[str] = []
    for it in items:
        if it and it not in seen:
            seen.append(it)
    return seen


# --------------------------------------------------------------------------- #
# Liveness / usage probes (injectable so planning is testable off-host)
# --------------------------------------------------------------------------- #
def _domain_alive(rd: RegistryDomain) -> bool:
    """True if the registry domain's recorded process is still running."""
    if not rd.pid:
        return False
    try:
        os.kill(rd.pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists but we cannot signal it -- still running.
        return True
    except OSError:
        return False


def _device_in_use(dev: str) -> Optional[bool]:
    """Best-effort tri-state "is this /dev/unvmeN held by a process?" via ``fuser``.

    Returns ``True`` (held), ``False`` (confirmed free), or ``None`` (cannot tell
    -- no ``fuser`` or the probe errored). A drive bound to unvme carries no record
    of who bound it, so identifiable active use -- not provenance -- gates
    reclaiming it. Reclaim is destructive (it tears down and rebuilds a drive that
    may be serving an unrelated workload), so an indeterminate result fails *closed*:
    the caller refuses the reclaim rather than free a possibly-in-use device. (The
    launch backend already refuses to launch without ``fuser`` for the same reason;
    ``psmisc`` is a runtime dependency.)
    """
    from shutil import which

    if not dev or which("fuser") is None:
        return None
    try:
        import subprocess

        return subprocess.run(
            ["fuser", "-s", dev], capture_output=True, check=False
        ).returncode == 0
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #
def plan_reconciliation(
    config: Config,
    inventory: Inventory,
    existing: Registry,
    desired_domains: List[RegistryDomain],
    *,
    is_alive: Optional[Callable[[RegistryDomain], bool]] = None,
    device_in_use: Optional[Callable[[str], Optional[bool]]] = None,
) -> ReconcilePlan:
    """Classify each desired domain into the action ``apply`` should take.

    ``desired_domains`` is ``build_registry_domains(config, inventory)`` -- the
    shape the config wants, in the same order as ``config.domains``. Returns a
    :class:`ReconcilePlan`; a non-empty ``plan.errors`` means the apply must be
    refused before any host mutation. The ``is_alive``/``device_in_use`` probes
    default to the real host checks and are resolved at call time (so tests can
    override the module-level defaults).
    """
    is_alive = is_alive or _domain_alive
    device_in_use = device_in_use or _device_in_use
    active = set(existing.active_configs)
    is_active = config.name in active
    plan = ReconcilePlan(config=config.name, active=is_active)

    # Registry is the source of truth for what is *managed*: index every resource
    # an ACTIVE config currently owns.
    owner_of_pci: Dict[str, str] = {}   # drive pci -> owning config
    owner_of_gpu: Dict[str, str] = {}   # gpu uuid  -> owning config
    for d in existing.domains:
        if d.config not in active:
            continue
        if d.gpu_uuid:
            owner_of_gpu.setdefault(d.gpu_uuid, d.config)
        for drv in d.drives:
            if drv.pci_bus_id:
                owner_of_pci.setdefault(drv.pci_bus_id, d.config)

    drive_by_pci = {dr.pci_bus_id: dr for dr in inventory.drives if dr.pci_bus_id}

    # 1) Conflict: any target resource owned by a *different* active config.
    for dom, des in zip(config.domains, desired_domains):
        owner = owner_of_gpu.get(des.gpu_uuid)
        if owner and owner != config.name:
            plan.errors.append(
                f"domain {dom.name}: GPU {des.gpu_uuid} is managed by active config "
                f"'{owner}'; stop it (scadactl stop --config {owner}) before applying "
                f"'{config.name}'"
            )
        for pci in (drv.pci_bus_id for drv in des.drives if drv.pci_bus_id):
            owner = owner_of_pci.get(pci)
            if owner and owner != config.name:
                plan.errors.append(
                    f"domain {dom.name}: drive {pci} is managed by active config "
                    f"'{owner}'; stop it (scadactl stop --config {owner}) before applying "
                    f"'{config.name}'"
                )
    if plan.errors:
        return plan  # a conflict blocks the whole apply

    # 2) This config already active: compare desired vs the REGISTRY RECORD (not
    # the possibly-edited file). Identical -> no-op / self-heal; different ->
    # incompatible (stop then apply).
    if is_active:
        entry = next((c for c in existing.configs if c.name == config.name), None)
        if entry and (entry.mps_enabled != config.mps_enabled or entry.mps_multiuser != config.mps_multiuser):
            plan.errors.append(
                f"config '{config.name}' is active with different MPS settings; "
                f"stop it (scadactl stop --config {config.name}) then apply the new version"
            )
            return plan
        recorded = {d.name: d for d in existing.domains if d.config == config.name}
        desired_names = [dom.name for dom in config.domains]
        if set(desired_names) != set(recorded):
            plan.errors.append(
                f"config '{config.name}' is active with a different set of domains "
                f"(running: {sorted(recorded)}; requested: {sorted(desired_names)}); "
                f"stop it (scadactl stop --config {config.name}) then apply the new version"
            )
            return plan
        for dom, des in zip(config.domains, desired_domains):
            if not _domain_matches(des, recorded[dom.name]):
                plan.errors.append(
                    f"config '{config.name}' is active with a different definition for "
                    f"domain {dom.name}; stop it (scadactl stop --config {config.name}) then "
                    f"apply the new version"
                )
                return plan
        # Matches the record: leave healthy domains alone, rebuild dead ones.
        for dom, des in zip(config.domains, desired_domains):
            if is_alive(recorded[dom.name]):
                plan.domains.append(DomainPlan(
                    name=dom.name, action=DomainAction.NOOP,
                    reasons=[f"already running and matches the registry"],
                ))
            else:
                plan.domains.append(DomainPlan(
                    name=dom.name, action=DomainAction.SELF_HEAL,
                    bind_pci=_bind_targets(des, drive_by_pci),
                    reasons=["recorded but not running; rebuilding (self-heal)"],
                ))
        return plan

    # 3) Fresh apply: build each domain, reclaiming any unmanaged orphan drives.
    for dom, des in zip(config.domains, desired_domains):
        bind_pci: List[str] = []
        reclaim_pci: List[str] = []
        reasons: List[str] = []
        held: List[str] = []
        for drv in des.drives:
            pci = drv.pci_bus_id
            if not pci:
                continue
            inv_drv = drive_by_pci.get(pci)
            if des.storage_backend == "kernel" and inv_drv is not None and inv_drv.driver == "unvme":
                # Bound, and (step 1) not owned by another active config, and this
                # config isn't active -> an unmanaged orphan. Free it only when we
                # can CONFIRM it is idle; refuse if it is held, or if usage can't be
                # established (no device-targeted kill primitive; fail closed rather
                # than free a possibly-in-use drive.
                dev = inv_drv.linux_device
                usage = device_in_use(dev) if dev else None
                if usage is None:
                    held.append(
                        f"domain {dom.name}: drive {pci} ({dev or 'no /dev node'}) is bound "
                        f"to unvme but unmanaged, and its usage cannot be determined "
                        f"(fuser unavailable or the probe failed); refusing to reclaim a "
                        f"possibly-in-use drive -- free it manually, then re-apply"
                    )
                    continue
                if usage:
                    held.append(
                        f"domain {dom.name}: drive {pci} ({dev}) is bound to unvme and "
                        f"held by a running process with no registry record; stop that "
                        f"process to free it, then re-apply"
                    )
                    continue
                # Confirmed idle: free the orphan, then rebind it for a clean rebuild.
                reclaim_pci.append(pci)
                reasons.append(f"drive {pci} bound to unvme but unmanaged; free + rebind")
            if des.storage_backend == "kernel":
                bind_pci.append(pci)  # every kernel target drive is (re)bound
        if held:
            plan.errors.extend(held)
            continue
        action = DomainAction.RECLAIM if reclaim_pci else DomainAction.BUILD
        reasons.insert(0, "build" if action is DomainAction.BUILD else "reclaim + build")
        plan.domains.append(DomainPlan(
            name=dom.name, action=action,
            bind_pci=bind_pci, reclaim_pci=reclaim_pci, reasons=reasons,
        ))
    return plan


def _bind_targets(desired: RegistryDomain, drive_by_pci: dict) -> List[str]:
    """Drives of a domain that still need binding (currently on the nvme driver)."""
    if desired.storage_backend != "kernel":
        return []
    out: List[str] = []
    for drv in desired.drives:
        pci = drv.pci_bus_id
        if not pci:
            continue
        inv_drv = drive_by_pci.get(pci)
        if inv_drv is None or inv_drv.driver != "unvme":
            out.append(pci)
    return out


def _domain_matches(desired: RegistryDomain, recorded: RegistryDomain) -> bool:
    """Equality key: GPU, drives, env, and profiler endpoint layout."""
    if (desired.gpu_uuid or "") != (recorded.gpu_uuid or ""):
        return False
    if desired.storage_backend != recorded.storage_backend:
        return False
    if sorted(d.pci_bus_id or "" for d in desired.drives) != \
            sorted(d.pci_bus_id or "" for d in recorded.drives):
        return False
    return (
        desired.socket_group == recorded.socket_group
        and dict(desired.server_env or {}) == dict(recorded.server_env or {})
        and desired.profiler_socket == recorded.profiler_socket
        and desired.profiler_endpoint_mode == recorded.profiler_endpoint_mode
    )
