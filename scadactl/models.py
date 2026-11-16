# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data models for the SCADA orchestrator.

Three layers are kept separate:

* **Inventory** -- physical resources detected on the host (GPUs, drives).
* **Config** -- the desired SCADA layout authored by an operator. Persisted
  with stable identifiers only (GPU UUID / GPU PCI bus id / drive serial /
  drive PCI bus id); aliases are accepted on input and resolved on write.
* **Registry** -- the live runtime state published to
  ``/var/run/scada/registry.json`` after apply.

All three are versioned with ``schema_version`` so the on-disk shapes can
evolve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Schema version constants. Bump when the on-disk shape changes incompatibly.
CONFIG_SCHEMA_VERSION = 2
REGISTRY_SCHEMA_VERSION = 2
POLICY_SCHEMA_VERSION = 1

# Profiler endpoint layouts.  ``managed`` is the normal scadactl-owned
# per-GPU socket; ``legacy_nsys`` retains the fixed endpoint required by the
# current Nsight Systems plugin.
PROFILER_ENDPOINT_MODE_MANAGED = "managed"
PROFILER_ENDPOINT_MODE_LEGACY_NSYS = "legacy_nsys"
PROFILER_ENDPOINT_MODES = {
    PROFILER_ENDPOINT_MODE_MANAGED,
    PROFILER_ENDPOINT_MODE_LEGACY_NSYS,
}


# --------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------- #
@dataclass
class GPU:
    """A physical GPU detected on the host."""

    alias: str  # generated, e.g. "gpu0"
    smi_index: Optional[int]  # nvidia-smi / NVML index (PCI-bus ordered)
    uuid: Optional[str]  # GPU-XXXX, the primary stable identity
    pci_bus_id: Optional[str]  # e.g. "0000:81:00.0"
    name: Optional[str] = None  # e.g. "NVIDIA H100 80GB"
    numa: Optional[int] = None
    pcie: Optional[str] = None  # e.g. "gen5 x16"
    notes: str = ""
    # Declared host device policy (allow/block lists); advisory in listings,
    # enforced at apply. Default False = usable, so absent policy is a no-op.
    blocked: bool = False
    block_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alias": self.alias,
            "smi_index": self.smi_index,
            "cuda_visible": self.smi_index,  # equal under CUDA_DEVICE_ORDER=PCI_BUS_ID
            "name": self.name,
            "uuid": self.uuid,
            "pci_bus_id": self.pci_bus_id,
            "numa": self.numa,
            "pcie": self.pcie,
            "notes": self.notes,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
        }

    def policy_label(self) -> str:
        """Policy status for display; '' when the device is usable."""
        return f"blocked - {self.block_reason or 'off-limits'}" if self.blocked else ""


@dataclass
class Drive:
    """A physical NVMe drive detected on the host (pre-bind identity)."""

    alias: str  # generated, e.g. "nvme0"
    linux_device: Optional[str]  # e.g. "/dev/nvme2n1"
    serial: Optional[str]  # primary durable join key
    model: Optional[str] = None
    size: Optional[str] = None  # humanized, e.g. "3.2TB"
    size_bytes: Optional[int] = None  # raw namespace capacity in bytes
    lba_size: Optional[int] = None
    namespace: Optional[int] = None
    pci_bus_id: Optional[str] = None  # immediate join key
    pcie: Optional[str] = None  # e.g. "gen4 x4"
    driver: Optional[str] = None  # e.g. "nvme", "unvme", or "vfio-pci"
    storage_backend: Optional[str] = None  # "kernel" / "lender"
    topology: str = ""  # informational column from nvidia-smi topo -m
    # Advisory mount/usage state, mirroring load-scada-modules' viability
    # checks. None = undetermined. Purely informational: it never blocks
    # selection -- `apply` (which runs the real script) is the gate.
    available: Optional[bool] = None
    unavailable_reason: Optional[str] = None
    # Declared host device policy (allow/block lists); see GPU.blocked.
    blocked: bool = False
    block_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alias": self.alias,
            "linux_device": self.linux_device,
            "serial": self.serial,
            "model": self.model,
            "size": self.size,
            "size_bytes": self.size_bytes,
            "lba_size": self.lba_size,
            "namespace": self.namespace,
            "pci_bus_id": self.pci_bus_id,
            "pcie": self.pcie,
            "driver": self.driver,
            "storage_backend": self.storage_backend,
            "topology": self.topology,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
        }

    def availability_label(self) -> str:
        """One-line mount availability for display; advisory only (apply gates)."""
        if self.available is None:
            return "unknown"
        if self.available:
            return "available"
        return f"unavailable - {self.unavailable_reason or 'in use'}"

    def status_label(self) -> str:
        """Combined usability for display: policy block dominates mount state."""
        if self.blocked:
            return f"blocked - {self.block_reason or 'off-limits'}"
        return self.availability_label()


@dataclass
class Inventory:
    """Host inventory: detected GPUs and drives plus the raw topology matrix."""

    gpus: List[GPU] = field(default_factory=list)
    drives: List[Drive] = field(default_factory=list)
    topology_matrix: str = ""  # raw `nvidia-smi topo -m` text, informational
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gpus": [g.to_dict() for g in self.gpus],
            "drives": [d.to_dict() for d in self.drives],
            "topology_matrix": self.topology_matrix,
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------- #
# Config (desired state, authored by operator)
# --------------------------------------------------------------------------- #
@dataclass
class DriveRef:
    """A reference to a drive inside a config domain (stable identifiers)."""

    serial: str
    pci_bus_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {"serial": self.serial, "pci_bus_id": self.pci_bus_id}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DriveRef":
        return cls(serial=str(d.get("serial", "")), pci_bus_id=str(d.get("pci_bus_id", "")))


@dataclass
class Domain:
    """A SCADA domain in a config: one GPU plus its owned drives."""

    name: str
    gpu_uuid: str
    gpu_pci_bus_id: str
    storage_backend: str = "kernel"
    drives: List[DriveRef] = field(default_factory=list)
    server_env: Dict[str, str] = field(default_factory=dict)  # per-domain overrides
    socket_group: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "name": self.name,
            "gpu_uuid": self.gpu_uuid,
            "gpu_pci_bus_id": self.gpu_pci_bus_id,
            "storage_backend": self.storage_backend,
            "drives": [d.to_dict() for d in self.drives],
        }
        if self.server_env:
            out["server_env"] = dict(self.server_env)
        if self.socket_group is not None:
            out["socket_group"] = self.socket_group
        return out

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Domain":
        return cls(
            name=str(d.get("name", "")),
            gpu_uuid=str(d.get("gpu_uuid", "")),
            gpu_pci_bus_id=str(d.get("gpu_pci_bus_id", "")),
            storage_backend=str(d.get("storage_backend", "kernel")),
            drives=[DriveRef.from_dict(x) for x in (d.get("drives") or [])],
            server_env={k: str(v) for k, v in (d.get("server_env") or {}).items()},
            socket_group=d.get("socket_group"),
        )

    def resolved_server_env(self, global_env: Dict[str, str]) -> Dict[str, str]:
        """Global defaults overlaid with this domain's per-domain overrides."""
        merged = {k: str(v) for k, v in global_env.items()}
        merged.update({k: str(v) for k, v in self.server_env.items()})
        return merged


@dataclass
class Config:
    """A parsed SCADA config (desired layout)."""

    name: str
    mps_enabled: bool = False
    profiler_endpoint_mode: str = PROFILER_ENDPOINT_MODE_MANAGED
    server_env: Dict[str, str] = field(default_factory=dict)  # global defaults
    domains: List[Domain] = field(default_factory=list)
    domain_groups: Dict[str, List[str]] = field(default_factory=dict)
    created: Optional[str] = None   # ISO-ish timestamp; set by init
    modified: Optional[str] = None  # ISO-ish timestamp; bumped by edit
    schema_version: int = CONFIG_SCHEMA_VERSION
    mps_multiuser: bool = False

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "name": self.name,
            "mps": {"enabled": self.mps_enabled, "multiuser": self.mps_multiuser},
            "profiler": {"endpoint_mode": self.profiler_endpoint_mode},
        }
        if self.server_env:
            out["server_env"] = dict(self.server_env)
        out["domains"] = [d.to_dict() for d in self.domains]
        if self.domain_groups:
            out["domain_groups"] = {k: list(v) for k, v in self.domain_groups.items()}
        if self.created:
            out["created"] = self.created
        if self.modified:
            out["modified"] = self.modified
        return out

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        mps = d.get("mps") or {}
        profiler = d.get("profiler") or {}
        return cls(
            name=str(d.get("name", "")),
            mps_enabled=bool(mps.get("enabled", False)),
            mps_multiuser=mps.get("multiuser", False),
            profiler_endpoint_mode=str(profiler.get("endpoint_mode", PROFILER_ENDPOINT_MODE_MANAGED)),
            server_env={k: str(v) for k, v in (d.get("server_env") or {}).items()},
            domains=[Domain.from_dict(x) for x in (d.get("domains") or [])],
            domain_groups={
                k: list(v) for k, v in (d.get("domain_groups") or {}).items()
            },
            created=(str(d["created"]) if d.get("created") else None),
            modified=(str(d["modified"]) if d.get("modified") else None),
            schema_version=int(d.get("schema_version", CONFIG_SCHEMA_VERSION)),
        )


# --------------------------------------------------------------------------- #
# Registry (live state, published after apply)
# --------------------------------------------------------------------------- #
@dataclass
class RegistryDrive:
    """A drive record inside a live registry domain (before-bind + after-bind)."""

    linux_device_before_bind: Optional[str]
    serial: Optional[str]
    model: Optional[str] = None
    namespace: Optional[int] = None
    lba_size: Optional[int] = None
    pci_bus_id: Optional[str] = None
    pcie: Optional[str] = None
    unvme_device: Optional[str] = None  # after-bind /dev/unvmeN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "linux_device_before_bind": self.linux_device_before_bind,
            "serial": self.serial,
            "model": self.model,
            "namespace": self.namespace,
            "lba_size": self.lba_size,
            "pci_bus_id": self.pci_bus_id,
            "pcie": self.pcie,
            "unvme_device": self.unvme_device,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RegistryDrive":
        return cls(
            linux_device_before_bind=d.get("linux_device_before_bind"),
            serial=d.get("serial"),
            model=d.get("model"),
            namespace=d.get("namespace"),
            lba_size=d.get("lba_size"),
            pci_bus_id=d.get("pci_bus_id"),
            pcie=d.get("pcie"),
            unvme_device=d.get("unvme_device"),
        )


@dataclass
class RegistryDomain:
    """A live domain record in the registry."""

    name: str
    config: str  # owning config (the config tag)
    gpu_uuid: str
    gpu_pci_bus_id: str
    host_cuda_visible_devices: str
    socket: str
    profiler_socket: str = ""
    profiler_endpoint_mode: str = PROFILER_ENDPOINT_MODE_MANAGED
    storage_backend: str = "kernel"
    server_env: Dict[str, str] = field(default_factory=dict)  # resolved
    drives: List[RegistryDrive] = field(default_factory=list)
    gpu_alias: Optional[str] = None
    pid: Optional[int] = None
    healthy: bool = False
    socket_group: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "config": self.config,
            "gpu_alias": self.gpu_alias,
            "gpu_uuid": self.gpu_uuid,
            "gpu_pci_bus_id": self.gpu_pci_bus_id,
            "host_cuda_visible_devices": self.host_cuda_visible_devices,
            "socket": self.socket,
            "socket_group": self.socket_group,
            "profiler_socket": self.profiler_socket,
            "profiler_endpoint_mode": self.profiler_endpoint_mode,
            "storage_backend": self.storage_backend,
            "server_env": dict(self.server_env),
            "drives": [d.to_dict() for d in self.drives],
            "pid": self.pid,
            "healthy": self.healthy,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RegistryDomain":
        return cls(
            name=str(d.get("name", "")),
            config=str(d.get("config", "")),
            gpu_alias=d.get("gpu_alias"),
            gpu_uuid=str(d.get("gpu_uuid", "")),
            gpu_pci_bus_id=str(d.get("gpu_pci_bus_id", "")),
            host_cuda_visible_devices=str(d.get("host_cuda_visible_devices", "")),
            socket=str(d.get("socket", "")),
            socket_group=d.get("socket_group"),
            profiler_socket=str(d.get("profiler_socket", "")),
            profiler_endpoint_mode=str(
                d.get("profiler_endpoint_mode", PROFILER_ENDPOINT_MODE_MANAGED)
            ),
            storage_backend=str(d.get("storage_backend", "kernel")),
            server_env={k: str(v) for k, v in (d.get("server_env") or {}).items()},
            drives=[RegistryDrive.from_dict(x) for x in (d.get("drives") or [])],
            pid=d.get("pid"),
            healthy=bool(d.get("healthy", False)),
        )


@dataclass
class ConfigEntry:
    """Lightweight per-config record in the registry's ``configs`` array."""

    name: str
    domains: List[str] = field(default_factory=list)
    domain_groups: Dict[str, List[str]] = field(default_factory=dict)
    mps_enabled: bool = False  # did the owning config request global MPS?
    mps_multiuser: bool = False

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"name": self.name, "domains": list(self.domains)}
        if self.domain_groups:
            out["domain_groups"] = {k: list(v) for k, v in self.domain_groups.items()}
        if self.mps_enabled:
            out["mps"] = True
            out["mps_multiuser"] = self.mps_multiuser
        return out

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConfigEntry":
        return cls(
            name=str(d.get("name", "")),
            domains=list(d.get("domains") or []),
            domain_groups={
                k: list(v) for k, v in (d.get("domain_groups") or {}).items()
            },
            mps_enabled=bool(d.get("mps", False)),
            mps_multiuser=bool(d.get("mps_multiuser", False)),
        )


@dataclass
class Registry:
    """The live runtime registry (``registry.json``)."""

    active_configs: List[str] = field(default_factory=list)
    configs: List[ConfigEntry] = field(default_factory=list)
    socket_dir: str = "/var/run/scada"
    domains: List[RegistryDomain] = field(default_factory=list)
    schema_version: int = REGISTRY_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "active_configs": list(self.active_configs),
            "configs": [p.to_dict() for p in self.configs],
            "socket_dir": self.socket_dir,
            "domains": [d.to_dict() for d in self.domains],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Registry":
        return cls(
            schema_version=int(d.get("schema_version", REGISTRY_SCHEMA_VERSION)),
            active_configs=list(d.get("active_configs") or []),
            configs=[ConfigEntry.from_dict(x) for x in (d.get("configs") or [])],
            socket_dir=str(d.get("socket_dir", "/var/run/scada")),
            domains=[RegistryDomain.from_dict(x) for x in (d.get("domains") or [])],
        )

    @classmethod
    def empty(cls) -> "Registry":
        return cls()


# --------------------------------------------------------------------------- #
# Device policy (host allow/block lists)
# --------------------------------------------------------------------------- #
@dataclass
class PolicyEntry:
    """One device on an allow/block list, keyed by stable identity."""

    key: str  # GPU UUID or drive serial (primary stable id)
    pci_bus_id: Optional[str] = None
    reason: Optional[str] = None
    added: Optional[str] = None  # human timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "pci_bus_id": self.pci_bus_id,
            "reason": self.reason,
            "added": self.added,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PolicyEntry":
        return cls(
            key=str(d.get("key", "")),
            pci_bus_id=d.get("pci_bus_id"),
            reason=d.get("reason"),
            added=d.get("added"),
        )

    def matches(self, primary: Optional[str], pci: Optional[str]) -> bool:
        """True if this entry identifies the given device (by id or PCI)."""
        if primary and self.key and primary == self.key:
            return True
        return bool(pci and self.pci_bus_id and pci == self.pci_bus_id)


@dataclass
class Policy:
    """Host device policy: allow/block lists for GPUs and NVMe drives.

    An empty policy is a complete no-op, so default behavior is unchanged.
    """

    schema_version: int = POLICY_SCHEMA_VERSION
    gpu_block: List[PolicyEntry] = field(default_factory=list)
    gpu_allow: List[PolicyEntry] = field(default_factory=list)
    drive_block: List[PolicyEntry] = field(default_factory=list)
    drive_allow: List[PolicyEntry] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (
            self.gpu_block or self.gpu_allow or self.drive_block or self.drive_allow
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "gpus": {
                "block": [e.to_dict() for e in self.gpu_block],
                "allow": [e.to_dict() for e in self.gpu_allow],
            },
            "drives": {
                "block": [e.to_dict() for e in self.drive_block],
                "allow": [e.to_dict() for e in self.drive_allow],
            },
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Policy":
        d = d or {}
        gpus = d.get("gpus") or {}
        drives = d.get("drives") or {}

        def _entries(x) -> List[PolicyEntry]:
            return [PolicyEntry.from_dict(e) for e in (x or [])]

        return cls(
            schema_version=int(d.get("schema_version", POLICY_SCHEMA_VERSION)),
            gpu_block=_entries(gpus.get("block")),
            gpu_allow=_entries(gpus.get("allow")),
            drive_block=_entries(drives.get("block")),
            drive_allow=_entries(drives.get("allow")),
        )
