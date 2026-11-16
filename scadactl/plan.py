# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan: resolve identities, diff desired config vs current registry.

Read-only computation that backs ``scadactl apply --dry-run`` (and the deprecated
``scadactl plan`` alias): it resolves a config's stable identities against
inventory, computes what would change relative to the live registry, and renders
the layout shape (inherited vs per-domain server_env, topology, target socket,
host changes). It never mutates host state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from scadactl import identity, registry as registry_mod
from scadactl.models import Inventory, Config, Registry
from scadactl.models import PROFILER_ENDPOINT_MODE_LEGACY_NSYS


@dataclass
class DomainPlan:
    """Planned state for one domain."""

    name: str
    gpu_alias: Optional[str]
    gpu_uuid: str
    gpu_pci_bus_id: str
    storage_backend: str
    host_cuda_visible_devices: Optional[str]
    drives: List[Dict[str, str]] = field(default_factory=list)
    topology: str = ""
    socket: str = ""
    socket_group: Optional[str] = None
    profiler_socket: str = ""
    profiler_endpoint_mode: str = "managed"
    resolved_server_env: Dict[str, str] = field(default_factory=dict)
    inherited_keys: List[str] = field(default_factory=list)
    overridden_keys: List[str] = field(default_factory=list)
    action: str = "create"  # create | replace | unchanged


@dataclass
class Plan:
    """A computed plan for a config."""

    config_name: str
    domains: List[DomainPlan] = field(default_factory=list)
    host_changes: List[str] = field(default_factory=list)
    removed_domains: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "config": self.config_name,
            "domains": [
                {
                    "name": d.name,
                    "gpu_alias": d.gpu_alias,
                    "gpu_uuid": d.gpu_uuid,
                    "gpu_pci_bus_id": d.gpu_pci_bus_id,
                    "storage_backend": d.storage_backend,
                    "host_cuda_visible_devices": d.host_cuda_visible_devices,
                    "drives": d.drives,
                    "topology": d.topology,
                    "socket": d.socket,
                    "profiler_socket": d.profiler_socket,
                    "socket_group": d.socket_group,
                    "profiler_endpoint_mode": d.profiler_endpoint_mode,
                    "server_env": d.resolved_server_env,
                    "inherited_env_keys": d.inherited_keys,
                    "overridden_env_keys": d.overridden_keys,
                    "action": d.action,
                }
                for d in self.domains
            ],
            "host_changes": self.host_changes,
            "removed_domains": self.removed_domains,
        }


def compute_plan(
    config: Config,
    inventory: Optional[Inventory] = None,
    registry: Optional[Registry] = None,
) -> Plan:
    """Compute the plan for applying ``config``.

    Resolves the host CUDA index from the GPU UUID via inventory (NVML index ==
    CUDA ordinal under ``CUDA_DEVICE_ORDER=PCI_BUS_ID``). When inventory is
    unavailable the host index is left ``None`` (resolved at apply time).
    """
    registry = registry or Registry.empty()
    plan = Plan(config_name=config.name)

    # Existing domain names already owned by this config (reapply -> replace).
    existing_for_config = {
        d.name for d in registry.domains if d.config == config.name
    }

    gpu_alias_by_uuid: Dict[str, str] = {}
    topo_by_pci: Dict[str, str] = {}
    if inventory is not None:
        for g in inventory.gpus:
            if g.uuid:
                gpu_alias_by_uuid[g.uuid] = g.alias
        for d in inventory.drives:
            if d.pci_bus_id:
                topo_by_pci[d.pci_bus_id] = d.topology

    needs_kernel_bind = False
    for dom in config.domains:
        host_cvd: Optional[str] = None
        if inventory is not None:
            idx = identity.gpu_uuid_to_smi_index(inventory, dom.gpu_uuid)
            if idx is not None:
                host_cvd = str(idx)

        resolved_env = dom.resolved_server_env(config.server_env)
        inherited = [k for k in resolved_env if k not in dom.server_env]
        overridden = [k for k in resolved_env if k in dom.server_env]

        gpu_alias = gpu_alias_by_uuid.get(dom.gpu_uuid)
        topo_parts = [topo_by_pci.get(dr.pci_bus_id, "") for dr in dom.drives]
        topology = ",".join(t for t in topo_parts if t)

        socket = registry_mod.socket_for_domain(gpu_alias or _fallback_socket_token(dom))
        profiler_socket = (
            registry_mod.LEGACY_NSYS_PROFILER_SOCKET
            if config.profiler_endpoint_mode == PROFILER_ENDPOINT_MODE_LEGACY_NSYS
            else registry_mod.profiler_socket_for_domain(gpu_alias or _fallback_socket_token(dom))
        )

        dp = DomainPlan(
            name=dom.name,
            gpu_alias=gpu_alias,
            gpu_uuid=dom.gpu_uuid,
            gpu_pci_bus_id=dom.gpu_pci_bus_id,
            storage_backend=dom.storage_backend,
            host_cuda_visible_devices=host_cvd,
            drives=[
                {"serial": dr.serial, "pci_bus_id": dr.pci_bus_id} for dr in dom.drives
            ],
            topology=topology,
            socket=socket,
            profiler_socket=profiler_socket,
            socket_group=dom.socket_group,
            profiler_endpoint_mode=config.profiler_endpoint_mode,
            resolved_server_env=resolved_env,
            inherited_keys=inherited,
            overridden_keys=overridden,
            action="replace" if dom.name in existing_for_config else "create",
        )
        plan.domains.append(dp)
        if dom.drives and dom.storage_backend == "kernel":
            needs_kernel_bind = True

    # Domains this config previously owned but no longer defines -> removed.
    desired_names = {d.name for d in config.domains}
    plan.removed_domains = sorted(existing_for_config - desired_names)

    # Summarize the requested host changes.
    if needs_kernel_bind:
        plan.host_changes.append("bind selected drives to UNVMe if needed")
    if config.mps_enabled:
        plan.host_changes.append("start MPS if needed")
    n = len(config.domains)
    plan.host_changes.append(
        f"start {n} scada-swap-domain process" + ("es" if n != 1 else "")
    )
    if plan.removed_domains:
        plan.host_changes.append(
            f"stop {len(plan.removed_domains)} removed domain(s): "
            + ", ".join(plan.removed_domains)
        )
    plan.host_changes.append(f"publish {registry_mod.registry_path()}")
    return plan


def _fallback_socket_token(dom) -> str:
    """When no GPU alias is known, derive a stable-ish socket token."""
    # Use the last segment of the PCI bus id as a token, else the domain name.
    if dom.gpu_pci_bus_id:
        return dom.gpu_pci_bus_id.replace(":", "").replace(".", "")
    return dom.name


def render_plan(plan: Plan) -> str:
    """Render a human-readable plan."""
    lines: List[str] = []
    lines.append(f"Config: {plan.config_name}")
    lines.append("")
    lines.append("Domains to run:")
    for d in plan.domains:
        gpu_label = "/".join(
            x for x in (d.gpu_alias, d.gpu_uuid, d.gpu_pci_bus_id) if x
        )
        lines.append(f"  {d.name}")
        lines.append(f"    GPU: {gpu_label}")
        lines.append(f"    Storage backend: {d.storage_backend}")
        if d.host_cuda_visible_devices is not None:
            lines.append(f"    Host CUDA_VISIBLE_DEVICES: {d.host_cuda_visible_devices}")
        lines.append("    Drives:")
        for dr in d.drives:
            # VFIO/Lender inventory does not currently expose a drive serial;
            # show its BDF cleanly instead of rendering a leading separator.
            identifiers = [dr.get("serial", ""), dr.get("pci_bus_id", "")]
            lines.append(f"      {' / '.join(value for value in identifiers if value)}")
        if d.topology:
            lines.append(f"    Topology: {d.topology}")
        lines.append(f"    Socket: {d.socket}")
        lines.append(f"    Socket group: {d.socket_group or '(root only)'}")
        lines.append(f"    Profiler ({d.profiler_endpoint_mode}): {d.profiler_socket}")
        lines.append("    Server env:")
        for k, v in d.resolved_server_env.items():
            tag = " (override)" if k in d.overridden_keys else ""
            lines.append(f"      {k}={v}{tag}")
        if d.action != "create":
            lines.append(f"    Action: {d.action}")
    if plan.removed_domains:
        lines.append("")
        lines.append("Domains to remove:")
        for n in plan.removed_domains:
            lines.append(f"  {n}")
    lines.append("")
    lines.append("Host changes:")
    for c in plan.host_changes:
        lines.append(f"  {c}")
    return "\n".join(lines)
