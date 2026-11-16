# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client plane (unprivileged, host + container).

Reads the mounted registry, enumerates container-visible GPUs via NVML, and
maps a registry domain's GPU UUID/PCI bus id to a container-local CUDA ordinal.
Implements ``domains``, ``run``, ``env``, and ``check-client``.

NVML is lazy-imported; the registry-reading and env-composition logic that does
not need a GPU is testable without one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from scadactl import registry as registry_mod
from scadactl.backend import mps as mps_mod
from scadactl.models import Registry, RegistryDomain


class GpuNotVisible(Exception):
    """A requested domain's GPU is not visible in the current environment."""

    def __init__(self, domain: RegistryDomain):
        self.domain = domain
        super().__init__(
            f"domain {domain.name} GPU {domain.gpu_uuid} not visible in this environment"
        )


# --------------------------------------------------------------------------- #
# Container GPU enumeration (NVML)
# --------------------------------------------------------------------------- #
def visible_gpus() -> List[Dict[str, Optional[str]]]:
    """Enumerate GPUs visible to the current process via NVML.

    Returns a list ordered by container CUDA ordinal (NVML index), each entry
    ``{"ordinal", "uuid", "pci_bus_id"}``. Raises ``RuntimeError`` with an
    actionable message if NVML is unavailable.
    """
    try:
        import pynvml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "NVML bindings are not installed; container ordinal mapping requires "
            "them. Install with: pip install 'scadactl[gpu]' "
            "(or pip install nvidia-ml-py)."
        ) from exc
    try:
        pynvml.nvmlInit()
    except Exception as exc:
        raise RuntimeError(
            f"NVML init failed ({exc}); is the NVIDIA driver visible in this "
            f"container/host?"
        ) from exc
    out: List[Dict[str, Optional[str]]] = []
    try:
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            uuid = pynvml.nvmlDeviceGetUUID(h)
            pci = pynvml.nvmlDeviceGetPciInfo(h).busId
            out.append(
                {
                    "ordinal": str(i),
                    "uuid": uuid.decode() if isinstance(uuid, bytes) else str(uuid),
                    "pci_bus_id": pci.decode() if isinstance(pci, bytes) else str(pci),
                }
            )
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return out


def map_domain_to_ordinal(
    domain: RegistryDomain, visible: List[Dict[str, Optional[str]]]
) -> Optional[int]:
    """Map a domain to a container CUDA ordinal by UUID (preferred) or PCI.

    Pure function over a ``visible`` list, so it is unit-testable without NVML.
    Returns ``None`` when the domain's GPU is not visible.
    """
    for g in visible:
        if domain.gpu_uuid and g.get("uuid") == domain.gpu_uuid:
            return int(g["ordinal"])
    for g in visible:
        if domain.gpu_pci_bus_id and _norm_pci(g.get("pci_bus_id")) == _norm_pci(
            domain.gpu_pci_bus_id
        ):
            return int(g["ordinal"])
    return None


def _norm_pci(pci: Optional[str]) -> str:
    return (pci or "").lower()


# --------------------------------------------------------------------------- #
# Registry helpers
# --------------------------------------------------------------------------- #
def get_domain(reg: Registry, name: str) -> Optional[RegistryDomain]:
    for d in reg.domains:
        if d.name == name:
            return d
    return None


def group_map(reg: Registry) -> Dict[str, List[str]]:
    """Build the resolvable group -> domain-name map from the registry.

    Groups are carried per-config (``ConfigEntry.domain_groups``) and merged
    here. A built-in ``all`` group (every domain in the registry) is synthesized
    only when no active config already defines one.
    """
    groups: Dict[str, List[str]] = {}
    for p in reg.configs:
        for gname, members in (p.domain_groups or {}).items():
            bucket = groups.setdefault(gname, [])
            for m in members:
                if m not in bucket:
                    bucket.append(m)
    if "all" not in groups:
        groups["all"] = [d.name for d in reg.domains]
    return groups


def resolve_selection(reg: Registry, selection: str) -> List[RegistryDomain]:
    """Resolve a comma-separated domain/group selection, preserving order.

    Each token is either a domain name or a group name (a group expands to its
    member domains in order). Duplicates are dropped while preserving first-seen
    order. Raises ``KeyError`` for an unknown domain or group.
    """
    groups = group_map(reg)
    tokens = [s.strip() for s in selection.split(",") if s.strip()]
    out: List[RegistryDomain] = []
    seen: set = set()
    for tok in tokens:
        members = groups[tok] if tok in groups else [tok]
        for name in members:
            d = get_domain(reg, name)
            if d is None:
                raise KeyError(tok if tok not in groups else name)
            if d.name not in seen:
                seen.add(d.name)
                out.append(d)
    return out


# --------------------------------------------------------------------------- #
# domains
# --------------------------------------------------------------------------- #
@dataclass
class DomainView:
    name: str
    host_gpu: Optional[str]
    gpu_uuid: str
    container_cuda: Optional[int]
    socket: str
    status: str


def list_domains(
    reg: Registry, visible: Optional[List[Dict[str, Optional[str]]]] = None
) -> List[DomainView]:
    """Build the ``scadactl domains`` view for the current environment."""
    if visible is None:
        try:
            visible = visible_gpus()
        except RuntimeError:
            visible = []
    views: List[DomainView] = []
    for d in reg.domains:
        ordinal = map_domain_to_ordinal(d, visible) if visible else None
        views.append(
            DomainView(
                name=d.name,
                host_gpu=d.gpu_alias,
                gpu_uuid=d.gpu_uuid,
                container_cuda=ordinal,
                socket=d.socket,
                status="ok" if ordinal is not None else "gpu-not-visible",
            )
        )
    return views


# --------------------------------------------------------------------------- #
# env / run
# --------------------------------------------------------------------------- #
def compute_env(
    reg: Registry,
    domains: List[RegistryDomain],
    visible: Optional[List[Dict[str, Optional[str]]]] = None,
) -> Tuple[Dict[str, str], List[str]]:
    """Compute the process-local environment for a domain selection.

    Returns ``(env, sockets)``. Preserves the requested domain order. Raises
    :class:`GpuNotVisible` if any selected domain's GPU is not visible.
    """
    if visible is None:
        visible = visible_gpus()
    ordinals: List[int] = []
    sockets: List[str] = []
    for d in domains:
        ordinal = map_domain_to_ordinal(d, visible)
        if ordinal is None:
            raise GpuNotVisible(d)
        ordinals.append(ordinal)
        sockets.append(d.socket)

    env: Dict[str, str] = {
        "CUDA_VISIBLE_DEVICES": ",".join(str(o) for o in ordinals),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "SCADA_DOMAIN": ",".join(d.name for d in domains),
    }
    if len(domains) == 1:
        env["SCADA_SOCKET"] = domains[0].socket
    return env, sockets


def env_json(
    reg: Registry,
    domains: List[RegistryDomain],
    visible: Optional[List[Dict[str, Optional[str]]]] = None,
) -> Dict[str, object]:
    """The ``scadactl env --json`` payload (with ``domain_count``)."""
    env, sockets = compute_env(reg, domains, visible)
    return {
        "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
        "scada_domains": env["SCADA_DOMAIN"],
        "scada_sockets": sockets,
        "domain_count": len(domains),
    }


def render_env_exports(env: Dict[str, str]) -> str:
    """Render ``export K=V`` lines."""
    return "\n".join(f"export {k}={v}" for k, v in env.items())


def run_command(
    reg: Registry,
    selection: str,
    command: List[str],
    visible: Optional[List[Dict[str, Optional[str]]]] = None,
) -> int:
    """Resolve the selection, set the child env, and exec the command.

    Returns the child exit code if the exec returns (it normally does not).
    """
    domains = resolve_selection(reg, selection)
    env, _sockets = compute_env(reg, domains, visible)
    child_env = dict(os.environ)
    child_env.update(env)
    os.execvpe(command[0], command, child_env)
    return 0  # unreachable if exec succeeds


# --------------------------------------------------------------------------- #
# check-client
# --------------------------------------------------------------------------- #
def check_client(
    reg: Registry,
    domain_name: str,
    visible: Optional[List[Dict[str, Optional[str]]]] = None,
    mps_pipe_dir: Optional[str] = None,
    mps_log_dir: Optional[str] = None,
) -> Tuple[bool, str]:
    """Diagnose container visibility for a single domain.

    Returns ``(ok, message)``. On failure, produces the documented
    ``gpu-not-visible`` guidance with a corrected ``docker run`` suggestion.
    """
    d = get_domain(reg, domain_name)
    if d is None:
        return False, f"Requested domain {domain_name} not present in registry."
    if visible is None:
        try:
            visible = visible_gpus()
        except RuntimeError as exc:
            return False, str(exc)

    ordinal = map_domain_to_ordinal(d, visible)
    if ordinal is not None:
        return True, (
            f"Requested domain: {domain_name}\n"
            f"Domain GPU UUID: {d.gpu_uuid}\n"
            f"Container CUDA ordinal: {ordinal}\n"
            f"Socket: {d.socket}\n"
            f"OK: this container can run clients for {domain_name}."
        )

    # Derive the mounts from the *active* paths so the hint stays correct when
    # SCADA_RUNTIME_DIR / CUDA_MPS_PIPE_DIRECTORY / CUDA_MPS_LOG_DIRECTORY are
    # overridden, rather than printing hardcoded defaults. The socket mount comes
    # from the domain's own stored socket path (baked in at apply time), which is
    # authoritative even if it lives outside the current runtime dir.
    socket_dir = os.path.dirname(d.socket) or registry_mod.runtime_dir()
    registry_path = registry_mod.registry_path()
    pipe_dir = mps_pipe_dir or mps_mod.pipe_directory()
    log_dir = mps_log_dir or mps_mod.log_directory()
    suggested = (
        f"  docker run --gpus '\"device={d.gpu_uuid}\"' "
        f"-v {socket_dir}:{socket_dir} "
        f"-v {registry_path}:{registry_path}:ro "
        f"-v {pipe_dir}:{pipe_dir} -v {log_dir}:{log_dir} ..."
    )
    return False, (
        f"Requested domain: {domain_name}\n"
        f"Domain GPU UUID: {d.gpu_uuid}\n"
        f"Container CUDA ordinal: not visible\n\n"
        f"This container cannot run clients for {domain_name} because the "
        f"environment did not expose {d.gpu_uuid}.\n\n"
        f"Suggested container creation:\n{suggested}"
    )
