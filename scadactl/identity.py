# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable-identity resolution, alias generation, and UNVMe correlation.

This module is the single place that translates between volatile names
(``nvidia-smi`` index, ``/dev/unvmeN``, CUDA ordinals) and the stable
identities the configs and registry are keyed on (GPU UUID / GPU PCI bus id /
drive serial / drive PCI bus id).

Pure resolution against an :class:`~scadactl.models.Inventory` needs
no hardware and is fully unit-tested. The UNVMe-correlation helpers touch sysfs
/ subprocess and degrade gracefully.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

from scadactl.models import Drive, GPU, Inventory

UNVME_SYSFS_CLASS = "/sys/class/unvme"


# --------------------------------------------------------------------------- #
# Alias generation
# --------------------------------------------------------------------------- #
def assign_gpu_aliases(gpus: List[GPU]) -> None:
    """Assign ``gpu0..gpuN`` aliases in smi-index (PCI-bus) order, in place."""
    ordered = sorted(
        gpus,
        key=lambda g: (g.smi_index if g.smi_index is not None else 1 << 30, g.pci_bus_id or ""),
    )
    for i, g in enumerate(ordered):
        g.alias = f"gpu{i}"


def assign_drive_aliases(drives: List[Drive]) -> None:
    """Set each drive's alias to its device name, in place.

    ``nvme0`` from ``/dev/nvme0n1``, ``unvme1`` from ``/dev/unvme1``. Those
    names may change, so the alias is derived at every enumeration and never
    stored. Falls back to the PCI bus id for drives with no device node
    (vfio-pci / Lender candidates).
    """
    for d in drives:
        name = os.path.basename(d.linux_device or "")
        d.alias = re.sub(r"n\d+$", "", name) or d.pci_bus_id or ""


def multi_namespace_drives(drives: List[Drive]) -> Dict[str, List[str]]:
    """Map PCI bus id to device names, for controllers with several namespaces."""
    by_pci: Dict[str, List[str]] = {}
    for d in drives:
        if d.pci_bus_id:
            by_pci.setdefault(d.pci_bus_id, []).append(d.linux_device or d.pci_bus_id)
    return {pci: devs for pci, devs in by_pci.items() if len(devs) > 1}


# --------------------------------------------------------------------------- #
# Alias <-> stable id resolution
# --------------------------------------------------------------------------- #
_GPU_ALIAS_RE = re.compile(r"^gpu\d+$")
_NVME_ALIAS_RE = re.compile(r"^nvme\d+$")


def is_gpu_alias(token: str) -> bool:
    return bool(_GPU_ALIAS_RE.match(token))


def is_drive_alias(token: str) -> bool:
    return bool(_NVME_ALIAS_RE.match(token))


def resolve_gpu(inv: Inventory, token: str) -> Optional[GPU]:
    """Resolve a token (alias, UUID, or PCI bus id) to a GPU, or None."""
    for g in inv.gpus:
        if token in (g.alias, g.uuid, g.pci_bus_id):
            return g
    return None


def resolve_drive(inv: Inventory, token: str) -> Optional[Drive]:
    """Resolve a token to a drive, or None.

    Accepts the alias (``nvme3``), serial, PCI BDF, or device path in either
    spelling (``/dev/nvme3n1``, ``nvme3n1``). Device names are never persisted
    as config identity.
    """
    if not token:
        return None
    for d in inv.drives:
        identifiers = (d.alias, d.serial, d.pci_bus_id, d.linux_device)
        if d.linux_device:
            identifiers += (os.path.basename(d.linux_device),)
        if token in identifiers:
            return d
    return None


def gpu_uuid_to_smi_index(inv: Inventory, uuid: str) -> Optional[int]:
    """Map a GPU UUID to its NVML / nvidia-smi (PCI-bus-ordered) index.

    Under ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` this index equals the CUDA ordinal,
    which is why apply launches every server with that env set.
    """
    for g in inv.gpus:
        if g.uuid == uuid:
            return g.smi_index
    return None


def is_alias_token(token: str) -> bool:
    return is_gpu_alias(token) or is_drive_alias(token)


# --------------------------------------------------------------------------- #
# Pre-bind drive identity capture (keyed by PCI bus id)
# --------------------------------------------------------------------------- #
def prebind_identity_by_pci(inv: Inventory) -> Dict[str, Drive]:
    """Index the inventory's drive identities by PCI bus id.

    This is the half of UNVMe correlation that must be captured *before* any
    binding (while drives are still on the kernel ``nvme`` driver). After
    binding, the live ``/dev/unvmeN -> serial/BDF`` mapping is joined back to
    these records (see :func:`correlate_unvme_devices`).
    """
    return {d.pci_bus_id: d for d in inv.drives if d.pci_bus_id}


# --------------------------------------------------------------------------- #
# UNVMe correlation -- PRIMARY path: driver-published sysfs attributes
# --------------------------------------------------------------------------- #
def _read_sysfs_attr(unvme_dir: str, attr: str) -> Optional[str]:
    """Read one plain-text, newline-terminated sysfs attribute."""
    p = os.path.join(unvme_dir, attr)
    try:
        with open(p, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def read_unvme_sysfs_identity(unvme_name: str) -> Optional[Dict[str, str]]:
    """Read the driver-published identity for ``unvme<N>`` from sysfs.

    PRIMARY correlation path. Reads the driver-published attributes under
    ``/sys/class/unvme/unvme<N>/`` (``serial``, ``address``, ``model``; all
    plain text, newline-terminated). ``address`` holds the bound PCI BDF and
    mirrors the nvme driver's ``address`` attribute; we track that value under
    the ``pci_bdf`` key internally. Returns ``None`` if the attributes are absent
    (fall back to the interim bootstrap).
    """
    unvme_dir = os.path.join(UNVME_SYSFS_CLASS, unvme_name)
    serial = _read_sysfs_attr(unvme_dir, "serial")
    # The sysfs attribute is ``address`` (nvme parity); its value is the PCI BDF.
    pci_bdf = _read_sysfs_attr(unvme_dir, "address")
    if serial is None and pci_bdf is None:
        return None
    out: Dict[str, str] = {}
    if serial is not None:
        out["serial"] = serial
    if pci_bdf is not None:
        out["pci_bdf"] = pci_bdf
    model = _read_sysfs_attr(unvme_dir, "model")
    if model is not None:
        out["model"] = model
    for opt in ("nsid", "lba_size", "size_bytes"):
        v = _read_sysfs_attr(unvme_dir, opt)
        if v is not None:
            out[opt] = v
    return out


def list_unvme_devices() -> List[str]:
    """List ``/dev/unvme*`` device paths currently present (best-effort)."""
    try:
        return sorted("/dev/" + n for n in os.listdir("/dev") if n.startswith("unvme"))
    except OSError:
        return []


def list_unvme_sysfs_names() -> List[str]:
    """List ``unvme<N>`` entries under ``/sys/class/unvme`` (best-effort)."""
    try:
        return sorted(n for n in os.listdir(UNVME_SYSFS_CLASS) if n.startswith("unvme"))
    except OSError:
        return []


def correlate_unvme_devices(
    inv: Inventory,
) -> Tuple[Dict[str, Drive], List[str]]:
    """Correlate live ``/dev/unvmeN`` devices to pre-bind drive records.

    PRIMARY path: for each ``unvme<N>`` in sysfs, read the driver-published
    ``serial`` / ``address`` (the PCI BDF) and join to the pre-bind inventory
    record (serial is the restart-proof key; BDF is the immediate one). Returns a mapping of
    ``/dev/unvmeN -> Drive`` plus any warnings about devices that could not be
    attributed. Returns an empty mapping (with a warning) when the sysfs
    attributes are absent -- callers then fall back to the interim bootstrap.
    """
    by_pci = prebind_identity_by_pci(inv)
    by_serial = {d.serial: d for d in inv.drives if d.serial}
    mapping: Dict[str, Drive] = {}
    warnings: List[str] = []

    sysfs_names = list_unvme_sysfs_names()
    if not sysfs_names:
        warnings.append(
            "no /sys/class/unvme entries found; sysfs correlation unavailable "
            "(driver identity attributes may not be present)"
        )
        return mapping, warnings

    any_identity = False
    for name in sysfs_names:
        ident = read_unvme_sysfs_identity(name)
        dev = f"/dev/{name}"
        if not ident:
            warnings.append(f"{name}: no driver-published identity attributes in sysfs")
            continue
        any_identity = True
        drive = None
        if ident.get("serial") in by_serial:
            drive = by_serial[ident["serial"]]
        elif ident.get("pci_bdf") in by_pci:
            drive = by_pci[ident["pci_bdf"]]
        if drive is None:
            warnings.append(
                f"{name}: identity {ident} did not match any pre-bind inventory record"
            )
            continue
        mapping[dev] = drive

    if not any_identity:
        warnings.append(
            "sysfs identity attributes absent on all unvme devices; "
            "use the interim bind-time bootstrap"
        )
    return mapping, warnings


# --------------------------------------------------------------------------- #
# UNVMe correlation -- INTERIM bootstrap (behind a clear seam)
# --------------------------------------------------------------------------- #
class InterimUnvmeBootstrap:
    """Bind-one-at-a-time + before/after ``/dev/unvme*`` diff correlation.

    INTERIM ONLY. Used as a fallback when the driver-published sysfs identity
    attributes are absent (see :func:`read_unvme_sysfs_identity`). It binds a
    single PCI BDF, diffs ``/dev/unvme*`` before and after, and attributes the
    one new node to that BDF.

    LIMITATIONS (documented because they gate ``apply``):

    * It is deterministic only at the *moment of binding*. It cannot attribute
      drives that were already bound before the tool ran, and it cannot
      reconstruct the mapping after a tool/host restart.
    * Therefore ``apply`` using this path must start managed drives from a
      known-unbound state and fail clearly when it finds pre-bound drives it
      cannot attribute.

    This is a bootstrap, not the destination -- the destination is the sysfs
    identity primary path above.
    """

    def __init__(self) -> None:
        self._before: List[str] = []

    def snapshot_before(self) -> None:
        """Record the set of ``/dev/unvme*`` devices before a single bind."""
        self._before = list_unvme_devices()

    def attribute_new_device(self, bound_pci_bdf: str) -> Optional[str]:
        """Return the single new ``/dev/unvmeN`` that appeared since snapshot.

        Returns ``None`` (and is ambiguous) if zero or more than one new node
        appeared -- callers must treat that as a hard failure for ``bound_pci_bdf``.
        """
        after = set(list_unvme_devices())
        new = sorted(after - set(self._before))
        if len(new) == 1:
            return new[0]
        return None
