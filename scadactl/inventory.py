# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host inventory: enumerate GPUs (NVML) and NVMe drives (/sys + nvme).

Best-effort and gracefully degrading: on a machine with no NVML or no
privileges, the enumeration that fails contributes a warning and an empty (or
partial) list rather than raising. ``pynvml`` is lazy-imported so the package
imports without it.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from typing import List, Optional

from scadactl import display, identity, policy
from scadactl.models import Drive, GPU, Inventory


class NvmlUnavailable(RuntimeError):
    """Raised by GPU-dependent paths when NVML/pynvml is not usable."""


def _import_pynvml():
    """Lazy import of pynvml with an actionable error message."""
    try:
        import pynvml  # type: ignore

        return pynvml
    except ImportError as exc:
        raise NvmlUnavailable(
            "NVML bindings are not installed. GPU inventory and GPU-dependent "
            "commands require them. Install with: pip install "
            "'scadactl[gpu]' (or pip install nvidia-ml-py), and ensure "
            "the NVIDIA driver is loaded."
        ) from exc


# --------------------------------------------------------------------------- #
# GPU enumeration via NVML
# --------------------------------------------------------------------------- #
def enumerate_gpus(warnings: Optional[List[str]] = None) -> List[GPU]:
    """Enumerate GPUs via NVML. Raises NvmlUnavailable if NVML cannot be used.

    A GPU NVML cannot open (e.g. lost from the bus) is skipped and reported in
    ``warnings``; the remaining GPUs keep their smi-index aliases.
    """
    pynvml = _import_pynvml()
    try:
        pynvml.nvmlInit()
    except Exception as exc:  # NVMLError when driver absent
        raise NvmlUnavailable(
            f"NVML init failed ({exc}); is the NVIDIA driver loaded?"
        ) from exc

    gpus: List[GPU] = []
    skipped: List[int] = []
    try:
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
            except Exception as exc:  # NVMLError, e.g. GPU lost from the bus
                skipped.append(i)
                if warnings is not None:
                    warnings.append(
                        f"skipping GPU {i}: NVML cannot open it ({exc}); check "
                        f"`nvidia-smi` and dmesg for driver/GPU errors"
                    )
                continue
            gpus.append(
                GPU(
                    alias=f"gpu{i}",
                    smi_index=i,
                    uuid=_nvml_str(pynvml.nvmlDeviceGetUUID(h)),
                    pci_bus_id=_nvml_pci(pynvml, h),
                    name=_nvml_str(pynvml.nvmlDeviceGetName(h)),
                    numa=_nvml_numa(pynvml, h),
                    pcie=_nvml_pcie(pynvml, h),
                    notes="visible",
                )
            )
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    if skipped:
        # Keep aliases tied to smi index so gpuN still means smi index N.
        for g in gpus:
            g.alias = f"gpu{g.smi_index}"
    else:
        identity.assign_gpu_aliases(gpus)
    return gpus


def _nvml_str(v) -> Optional[str]:
    if v is None:
        return None
    return v.decode() if isinstance(v, bytes) else str(v)


def _nvml_pci(pynvml, h) -> Optional[str]:
    try:
        info = pynvml.nvmlDeviceGetPciInfo(h)
        return _nvml_str(info.busId)
    except Exception:
        return None


def _nvml_numa(pynvml, h) -> Optional[int]:
    try:
        return pynvml.nvmlDeviceGetNumaNodeId(h)
    except Exception:
        return None


def _nvml_pcie(pynvml, h) -> Optional[str]:
    try:
        gen = pynvml.nvmlDeviceGetMaxPcieLinkGeneration(h)
        width = pynvml.nvmlDeviceGetMaxPcieLinkWidth(h)
        return f"gen{gen} x{width}"
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Drive enumeration via /sys + nvme
# --------------------------------------------------------------------------- #
def _run(cmd: List[str]) -> Optional[str]:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15, check=False
        )
        if out.returncode != 0:
            return None
        return out.stdout
    except (OSError, subprocess.SubprocessError):
        return None


def enumerate_drives() -> List[Drive]:
    """Enumerate NVMe drives best-effort from ``nvme list -o json`` + sysfs.

    Returns an empty list (not an error) when ``nvme`` is unavailable.
    """
    drives: List[Drive] = []
    out = _run(["nvme", "list", "-o", "json"])
    if out:
        try:
            data = json.loads(out)
            for dev in data.get("Devices", []):
                drives.append(_drive_from_nvme_json(dev))
        except (json.JSONDecodeError, KeyError):
            pass

    if not drives:
        drives = _drives_from_sysfs()

    # Drives bound to UNVMe are gone from `nvme list`; recover them from the
    # driver-published sysfs identity so inventory still shows their binding
    # state (driver=unvme, /dev/unvmeN) after apply.
    _merge_unvme_drives(drives)
    # Storage Lender currently exposes no discovery API.  An NVMe-class PCI
    # function bound to vfio-pci is therefore treated as an optimistic Lender
    # candidate, keyed by its stable BDF rather than a Linux block-device name.
    _merge_vfio_nvme_drives(drives)

    # Advisory mount/usage state (non-gating) so config authoring can flag
    # drives that would fail to bind. `apply` remains the real gate.
    for d in drives:
        d.available, d.unavailable_reason = _drive_availability(
            d.linux_device, d.driver
        )

    identity.assign_drive_aliases(drives)
    return drives


def _drives_from_unvme() -> List[Drive]:
    """Enumerate drives currently bound to UNVMe via ``/sys/class/unvme``."""
    out: List[Drive] = []
    for name in identity.list_unvme_sysfs_names():  # e.g. "unvme1"
        ident = identity.read_unvme_sysfs_identity(name)
        if not ident:
            continue
        pci = ident.get("pci_bdf")
        size_bytes = _parse_int(ident.get("size_bytes"))
        out.append(
            Drive(
                alias="",
                linux_device=f"/dev/{name}",
                serial=ident.get("serial"),
                model=ident.get("model"),
                size=_humanize(size_bytes),
                size_bytes=size_bytes,
                lba_size=_parse_int(ident.get("lba_size")),
                namespace=_parse_int(ident.get("nsid")),
                pci_bus_id=pci,
                pcie=_pcie_for_pci(pci),
                driver="unvme",
                storage_backend="kernel",
            )
        )
    return out


def _merge_unvme_drives(drives: List[Drive]) -> None:
    """Append UNVMe-bound drives not already present (matched by serial/PCI)."""
    seen_serial = {d.serial for d in drives if d.serial}
    seen_pci = {d.pci_bus_id for d in drives if d.pci_bus_id}
    for d in _drives_from_unvme():
        if (d.serial and d.serial in seen_serial) or (
            d.pci_bus_id and d.pci_bus_id in seen_pci
        ):
            continue
        drives.append(d)


def _vfio_nvme_drives() -> List[Drive]:
    """Discover NVMe-class PCI functions supplied through vfio-pci.

    Storage Lender does not yet offer an inventory API.  In this interim mode,
    vfio-pci ownership is the explicit, optimistic signal that a controller may
    be used by a Lender domain.  It is not a confirmation that the Lender service
    has accepted or is currently able to serve the device.
    """
    root = "/sys/bus/pci/devices"
    out: List[Drive] = []
    try:
        entries = os.listdir(root)
    except OSError:
        return out
    for bdf in entries:
        class_code = _read_file(os.path.join(root, bdf, "class"))
        if not class_code or class_code.lower() not in ("0x010802", "010802"):
            continue
        if _driver_for_pci(bdf) != "vfio-pci":
            continue
        out.append(
            Drive(
                alias="",
                linux_device=None,
                serial=None,
                pci_bus_id=bdf,
                pcie=_pcie_for_pci(bdf),
                driver="vfio-pci",
                storage_backend="lender",
                # Explicitly optimistic until Storage Lender exposes a device
                # query/claim API.  Apply remains the final authority.
                available=True,
            )
        )
    return out


def _merge_vfio_nvme_drives(drives: List[Drive]) -> None:
    """Append Lender candidates not already represented by PCI BDF."""
    seen_pci = {d.pci_bus_id for d in drives if d.pci_bus_id}
    for d in _vfio_nvme_drives():
        if d.pci_bus_id not in seen_pci:
            drives.append(d)


def _drive_from_nvme_json(dev: dict) -> Drive:
    devpath = dev.get("DevicePath") or dev.get("Device")
    pci = _pci_for_nvme_device(devpath)
    lba = dev.get("SectorSize") or dev.get("LBASize")
    return Drive(
        alias="",
        linux_device=devpath,
        serial=(dev.get("SerialNumber") or "").strip() or None,
        model=(dev.get("ModelNumber") or "").strip() or None,
        size=_humanize(dev.get("PhysicalSize")),
        size_bytes=_parse_int(dev.get("PhysicalSize")),
        lba_size=int(lba) if lba else None,
        namespace=dev.get("NameSpace"),
        pci_bus_id=pci,
        pcie=_pcie_for_pci(pci),
        driver=_driver_for_pci(pci),
        storage_backend="kernel",
    )


def _drives_from_sysfs() -> List[Drive]:
    """Fallback: enumerate from ``/sys/class/nvme`` when ``nvme`` is missing."""
    drives: List[Drive] = []
    base = "/sys/class/nvme"
    try:
        controllers = sorted(os.listdir(base))
    except OSError:
        return drives
    for ctrl in controllers:
        cdir = os.path.join(base, ctrl)
        serial = _read_file(os.path.join(cdir, "serial"))
        model = _read_file(os.path.join(cdir, "model"))
        pci = _read_link_bdf(os.path.join(cdir, "device"))
        # find a namespace block device
        ns = None
        ns_name = None
        for entry in os.listdir(cdir) if os.path.isdir(cdir) else []:
            if entry.startswith(ctrl + "n"):
                ns_name = entry
                ns = "/dev/" + entry
                break
        size_bytes = _sysfs_block_size_bytes(ns_name) if ns_name else None
        drives.append(
            Drive(
                alias="",
                linux_device=ns,
                serial=serial,
                model=model,
                size=_humanize(size_bytes),
                size_bytes=size_bytes,
                pci_bus_id=pci,
                pcie=_pcie_for_pci(pci),
                driver=_driver_for_pci(pci),
                storage_backend="kernel",
            )
        )
    return drives


def _sysfs_block_size_bytes(block_name: str) -> Optional[int]:
    """Best-effort capacity from ``/sys/block/<name>/size`` (512-byte sectors)."""
    sectors = _parse_int(_read_file(f"/sys/block/{block_name}/size"))
    return sectors * 512 if sectors is not None else None


def _read_file(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


_PCI_BDF_RE = re.compile(r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$")


def _read_link_bdf(path: str) -> Optional[str]:
    try:
        target = os.path.realpath(path)
        # Directly attached controllers normally end in their BDF.  Some
        # topologies have a trailing ``device`` directory instead, so walk the
        # resolved path and return only a real BDF component.
        for component in reversed(target.split(os.sep)):
            if _PCI_BDF_RE.fullmatch(component):
                return component
        return None
    except OSError:
        return None


def _pci_for_nvme_device(devpath: Optional[str]) -> Optional[str]:
    if not devpath:
        return None
    # Map a namespace/controller device name to its controller, e.g.
    # "nvme8n1" -> "nvme8", "nvme10n1p2" -> "nvme10", "nvme8" -> "nvme8".
    name = os.path.basename(devpath)
    m = re.match(r"(nvme\d+)", name)
    if not m:
        return None
    ctrl_name = m.group(1)

    # The NVMe driver's ``address`` attribute is the controller PCI BDF.  It
    # avoids mistaking a trailing sysfs directory named ``device`` for a BDF.
    pci = _read_file(f"/sys/class/nvme/{ctrl_name}/address")
    if pci and _PCI_BDF_RE.fullmatch(pci):
        return pci

    # With native NVMe multipath enabled, a namespace such as /dev/nvme0n2 is
    # virtual and belongs to /sys/devices/virtual/nvme-subsystem/nvme-subsysN.
    # Resolve the sole physical controller in that subsystem instead.  Do not
    # guess when multiple physical paths exist: SCADA binds by PCI BDF and must
    # receive an unambiguous controller identity.
    pci = _pci_for_multipath_namespace(name)
    if pci:
        return pci

    return _read_link_bdf(f"/sys/class/nvme/{ctrl_name}/device")


def _pci_for_multipath_namespace(block_name: str) -> Optional[str]:
    """Return a sole physical controller BDF for a virtual NVMe namespace."""
    block_path = os.path.realpath(f"/sys/class/block/{block_name}")
    subsystem_dir = os.path.dirname(block_path)
    if not os.path.basename(subsystem_dir).startswith("nvme-subsys"):
        return None

    try:
        entries = os.listdir(subsystem_dir)
    except OSError:
        return None

    bdfs = set()
    for entry in entries:
        # Physical controllers are named nvmeN; namespaces are nvmeNnM.
        if not re.fullmatch(r"nvme\d+", entry):
            continue
        pci = _read_file(f"/sys/class/nvme/{entry}/address")
        if pci and _PCI_BDF_RE.fullmatch(pci):
            bdfs.add(pci)

    return next(iter(bdfs)) if len(bdfs) == 1 else None


# PCIe link speed (GT/s, from sysfs max_link_speed) -> generation number.
_PCIE_GEN_BY_GTS = {"2.5": 1, "5.0": 2, "8.0": 3, "16.0": 4, "32.0": 5, "64.0": 6}


def _pcie_for_pci(pci: Optional[str]) -> Optional[str]:
    """Format a drive's PCIe link as ``gen<N> x<width>`` (matching the GPU column).

    Reads sysfs ``max_link_speed`` (e.g. ``16.0 GT/s PCIe``) and ``max_link_width``
    (e.g. ``4``); falls back to the raw speed string if the generation is unknown.
    """
    if not pci:
        return None
    speed = _read_file(f"/sys/bus/pci/devices/{pci}/max_link_speed")
    width = _read_file(f"/sys/bus/pci/devices/{pci}/max_link_width")
    if not speed and not width:
        return None
    gen = _PCIE_GEN_BY_GTS.get(speed.split()[0]) if speed and speed.split() else None
    gen_str = f"gen{gen}" if gen else (speed or "?")
    return f"{gen_str} x{width or '?'}"


def _driver_for_pci(pci: Optional[str]) -> Optional[str]:
    if not pci:
        return None
    drv = os.path.join("/sys/bus/pci/devices", pci, "driver")
    try:
        return os.path.basename(os.path.realpath(drv))
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# Drive availability (advisory, mirrors load-scada-modules is_device_in_use)
# --------------------------------------------------------------------------- #
# Filesystem types that mean a drive is in use and should not be rebound. Kept
# in sync with load-scada-modules; that script (via `apply`) is authoritative.
_IN_USE_FSTYPES = {
    "linux_raid_member": "part of RAID array",
    "LVM2_member": "part of LVM",
    "swap": "swap device",
}


def _drive_availability(linux_device: Optional[str], driver: Optional[str]):
    """Best-effort ``(available, reason)`` for a drive; advisory, never gates.

    Mirrors the viability checks in load-scada-modules' ``is_device_in_use()``:
    a drive is unavailable if it is mounted, a RAID or LVM member, or a swap
    device. Drives already bound to UNVMe are in use by SCADA. Returns
    ``(None, None)`` when availability cannot be determined.
    """
    if driver == "unvme" or (linux_device or "").startswith("/dev/unvme"):
        return False, "bound to unvme (SCADA)"
    if driver == "vfio-pci":
        # Interim Storage Lender policy: vfio-bound NVMe controllers are
        # treated as usable candidates.  Lender-side ownership is not yet
        # queryable, so a later launch can still reject the device.
        return True, None
    if not linux_device:
        return None, None
    reason = _block_device_in_use_reason(linux_device)
    return reason is None, reason


def _block_device_in_use_reason(linux_device: str) -> Optional[str]:
    """Reason an NVMe block device is in use, or None if free."""
    out = _run(["lsblk", "-J", "-o", "NAME,MOUNTPOINT,FSTYPE", linux_device])
    if out:
        try:
            tree = json.loads(out)
            return _scan_lsblk_nodes(tree.get("blockdevices", []))
        except (json.JSONDecodeError, KeyError):
            pass
    # Fallback when lsblk is unavailable: consult /proc directly.
    return _proc_in_use_reason(linux_device)


def _scan_lsblk_nodes(nodes) -> Optional[str]:
    """Walk lsblk device+partition nodes; first in-use reason wins."""
    for node in nodes or []:
        fstype = (node.get("fstype") or "").strip()
        if fstype in _IN_USE_FSTYPES:
            return _IN_USE_FSTYPES[fstype]
        mnt = node.get("mountpoint")
        if mnt and mnt != "[SWAP]":
            return f"mounted at {mnt}"
        child = _scan_lsblk_nodes(node.get("children"))
        if child:
            return child
    return None


def _proc_in_use_reason(linux_device: str) -> Optional[str]:
    name = os.path.basename(linux_device)  # e.g. "nvme2n1"
    if name and name in (_read_file("/proc/mdstat") or ""):
        return "part of RAID array"
    for line in (_read_file("/proc/swaps") or "").splitlines():
        dev = line.split(" ", 1)[0]
        if dev == linux_device or dev.startswith(linux_device + "p"):
            return "swap device"
    for line in (_read_file("/proc/mounts") or "").splitlines():
        parts = line.split(" ")
        dev = parts[0]
        if dev == linux_device or dev.startswith(linux_device + "p"):
            return f"mounted at {parts[1]}" if len(parts) > 1 else "mounted"
    return None


def _parse_int(v) -> Optional[int]:
    """Best-effort int parse; "unknown"/absent/non-numeric -> None, never raises."""
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _humanize(nbytes) -> Optional[str]:
    if not nbytes:
        return None
    try:
        n = float(nbytes)
    except (TypeError, ValueError):
        return None
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if n < 1000:
            return f"{n:.1f}{unit}".replace(".0", "")
        n /= 1000
    return f"{n:.1f}EB"


# --------------------------------------------------------------------------- #
# Topology matrix (informational)
# --------------------------------------------------------------------------- #
def topology_matrix() -> str:
    out = _run(["nvidia-smi", "topo", "-m"])
    return out or ""


# --------------------------------------------------------------------------- #
# Full inventory
# --------------------------------------------------------------------------- #
def collect_inventory() -> Inventory:
    """Collect a best-effort inventory; never raises on missing hardware."""
    inv = Inventory()
    try:
        inv.gpus = enumerate_gpus(inv.warnings)
    except NvmlUnavailable as exc:
        inv.warnings.append(f"GPU inventory unavailable: {exc}")
    try:
        inv.drives = enumerate_drives()
        from shutil import which
        if which("nvme") is None:
            inv.warnings.append(
                "nvme-cli not found; drive inventory came from sysfs (no lba/"
                "namespace detail). Install nvme-cli (a package Recommends) for "
                "richer detail."
            )
    except Exception as exc:  # noqa: BLE001 - best-effort
        inv.warnings.append(f"drive inventory degraded: {exc}")
    for pci, devs in identity.multi_namespace_drives(inv.drives).items():
        inv.warnings.append(
            f"{pci} exposes multiple namespaces ({', '.join(devs)}); scadactl "
            f"supports one namespace per controller and cannot target them apart"
        )
    inv.topology_matrix = topology_matrix()
    _annotate_policy(inv)
    return inv


def _annotate_policy(inv: Inventory) -> None:
    """Tag GPUs/drives with host device policy (advisory here; apply enforces)."""
    try:
        pol = policy.load()
    except Exception:  # noqa: BLE001 - policy is advisory at the inventory layer
        return
    if pol.is_empty():
        return
    for g in inv.gpus:
        g.blocked, g.block_reason = policy.gpu_decision(pol, g.uuid, g.pci_bus_id)
    for d in inv.drives:
        d.blocked, d.block_reason = policy.drive_decision(pol, d.serial, d.pci_bus_id)


def render_inventory(inv: Inventory) -> str:
    """Render GPU and drive inventory tables."""
    lines: List[str] = []
    lines.append("GPUs")
    lines.extend(display.render_table(
        [("alias", 6), ("smi_index", 9), ("cuda_visible", 12), ("name", 18),
         ("uuid", 20), ("pci_bus_id", 14), ("numa", 4), ("pcie", 8), ("notes", 12)],
        [(g.alias, g.smi_index, g.smi_index, g.name, g.uuid, g.pci_bus_id,
          g.numa, g.pcie, g.policy_label() or g.notes) for g in inv.gpus],
    ))
    lines.append("")
    lines.append("Drives")
    lines.extend(display.render_table(
        [("alias", 6), ("linux_device", 14), ("serial", 12), ("model", 12),
         ("size", 8), ("lba", 6), ("pci_bus_id", 14), ("pcie", 8),
         ("driver", 9), ("backend", 7), ("status", 12)],
        [(d.alias, d.linux_device, d.serial, d.model, d.size, d.lba_size,
          d.pci_bus_id, d.pcie, d.driver, d.storage_backend, d.status_label())
         for d in inv.drives],
    ))
    for w in inv.warnings:
        lines.append("")
        lines.append(f"warning: {w}")
    return "\n".join(lines)
