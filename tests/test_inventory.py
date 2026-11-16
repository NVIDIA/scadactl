# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drive enumeration + advisory availability (mirrors load-scada-modules)."""

from __future__ import annotations

import json

import pytest

from scadactl import inventory as inv_mod
from scadactl.models import Drive, GPU, Inventory


def _free_lsblk() -> str:
    return json.dumps({"blockdevices": [{"mountpoint": None, "fstype": None}]})


def _lsblk_only(payload: str):
    """A `_run` stub that answers lsblk with `payload` and everything else None."""
    return lambda cmd: payload if cmd and cmd[0] == "lsblk" else None


def test_pci_for_multipath_namespace_uses_sole_physical_controller(monkeypatch):
    """A virtual /dev/nvmeNnM node resolves through its nvme-subsys directory."""
    monkeypatch.setattr(
        inv_mod.os.path,
        "realpath",
        lambda path: "/sys/devices/virtual/nvme-subsystem/nvme-subsys0/nvme0n2"
        if path == "/sys/class/block/nvme0n2" else path,
    )
    monkeypatch.setattr(inv_mod.os, "listdir", lambda path: ["nvme0n2", "nvme32"])
    monkeypatch.setattr(
        inv_mod,
        "_read_file",
        lambda path: "0000:c7:00.0" if path == "/sys/class/nvme/nvme32/address" else None,
    )

    assert inv_mod._pci_for_nvme_device("/dev/nvme0n2") == "0000:c7:00.0"


def test_pci_for_multipath_namespace_refuses_ambiguous_paths(monkeypatch):
    monkeypatch.setattr(
        inv_mod.os.path,
        "realpath",
        lambda path: "/sys/devices/virtual/nvme-subsystem/nvme-subsys0/nvme0n2"
        if path == "/sys/class/block/nvme0n2" else path,
    )
    monkeypatch.setattr(inv_mod.os, "listdir", lambda path: ["nvme32", "nvme33"])
    monkeypatch.setattr(
        inv_mod,
        "_read_file",
        lambda path: {
            "/sys/class/nvme/nvme32/address": "0000:c7:00.0",
            "/sys/class/nvme/nvme33/address": "0000:c8:00.0",
        }.get(path),
    )

    assert inv_mod._pci_for_nvme_device("/dev/nvme0n2") is None


def test_enumerate_drives_includes_unvme_bound(monkeypatch):
    # `nvme list` only shows the boot drive (still on the nvme driver)...
    nvme_json = json.dumps({"Devices": [{
        "DevicePath": "/dev/nvme0n1", "SerialNumber": "BOOT", "ModelNumber": "OS",
        "SectorSize": 512, "NameSpace": 1,
    }]})

    def fake_run(cmd):
        if cmd[:2] == ["nvme", "list"]:
            return nvme_json
        if cmd and cmd[0] == "lsblk":  # boot drive reports free (mocked)
            return _free_lsblk()
        return None

    monkeypatch.setattr(inv_mod, "_run", fake_run)
    monkeypatch.setattr(inv_mod, "_pci_for_nvme_device", lambda dp: "0015:01:00.0")
    monkeypatch.setattr(inv_mod, "_pcie_for_pci", lambda pci: "gen4 x4")
    monkeypatch.setattr(inv_mod, "_driver_for_pci", lambda pci: "nvme")

    # ...while two data drives are bound to unvme (recovered from sysfs).
    monkeypatch.setattr(inv_mod.identity, "list_unvme_sysfs_names",
                        lambda: ["unvme1", "unvme2"])
    sysfs = {
        "unvme1": {"serial": "S111", "model": "PM", "pci_bdf": "0006:06:00.0"},
        "unvme2": {"serial": "S222", "model": "PM", "pci_bdf": "0006:07:00.0"},
    }
    monkeypatch.setattr(inv_mod.identity, "read_unvme_sysfs_identity",
                        lambda name: sysfs.get(name))

    drives = inv_mod.enumerate_drives()
    by_serial = {d.serial: d for d in drives}
    assert set(by_serial) == {"BOOT", "S111", "S222"}
    assert by_serial["S111"].driver == "unvme"
    assert by_serial["S111"].linux_device == "/dev/unvme1"
    assert by_serial["S111"].pci_bus_id == "0006:06:00.0"
    assert by_serial["BOOT"].driver == "nvme"
    # Advisory availability is attached during enumeration: the boot drive is
    # free (mocked), unvme-bound drives report as in use by SCADA.
    assert by_serial["BOOT"].available is True
    assert by_serial["S111"].available is False
    assert "unvme" in (by_serial["S111"].unavailable_reason or "")


def test_merge_unvme_skips_already_listed(monkeypatch):
    # If a drive somehow appears in both, it's not duplicated.
    monkeypatch.setattr(inv_mod, "_run",
                        lambda cmd: json.dumps({"Devices": [{
                            "DevicePath": "/dev/nvme1n1", "SerialNumber": "S111",
                            "ModelNumber": "PM", "SectorSize": 512, "NameSpace": 1,
                        }]}) if cmd[:2] == ["nvme", "list"] else None)
    monkeypatch.setattr(inv_mod, "_pci_for_nvme_device", lambda dp: "0006:06:00.0")
    monkeypatch.setattr(inv_mod, "_pcie_for_pci", lambda pci: "gen4 x4")
    monkeypatch.setattr(inv_mod, "_driver_for_pci", lambda pci: "nvme")
    monkeypatch.setattr(inv_mod, "_drive_availability", lambda dev, drv: (True, None))
    monkeypatch.setattr(inv_mod.identity, "list_unvme_sysfs_names", lambda: ["unvme1"])
    monkeypatch.setattr(inv_mod.identity, "read_unvme_sysfs_identity",
                        lambda name: {"serial": "S111", "pci_bdf": "0006:06:00.0"})
    drives = inv_mod.enumerate_drives()
    assert [d.serial for d in drives] == ["S111"]


# --------------------------------------------------------------------------- #
# Advisory availability (mirrors load-scada-modules is_device_in_use)
# --------------------------------------------------------------------------- #
def test_availability_unvme_bound_is_in_use():
    avail, reason = inv_mod._drive_availability("/dev/unvme3", "unvme")
    assert avail is False
    assert "unvme" in reason


def test_availability_free_drive(monkeypatch):
    monkeypatch.setattr(inv_mod, "_run", _lsblk_only(json.dumps(
        {"blockdevices": [{"name": "nvme5n1", "mountpoint": None, "fstype": None}]})))
    assert inv_mod._drive_availability("/dev/nvme5n1", "nvme") == (True, None)


def test_availability_mounted_partition(monkeypatch):
    monkeypatch.setattr(inv_mod, "_run", _lsblk_only(json.dumps({"blockdevices": [
        {"name": "nvme5n1", "mountpoint": None, "fstype": None, "children": [
            {"name": "nvme5n1p1", "mountpoint": "/data", "fstype": "ext4"}]}]})))
    assert inv_mod._drive_availability("/dev/nvme5n1", "nvme") == (
        False, "mounted at /data")


@pytest.mark.parametrize("fstype,expected", [
    ("linux_raid_member", "part of RAID array"),
    ("LVM2_member", "part of LVM"),
    ("swap", "swap device"),
])
def test_availability_in_use_fstypes(monkeypatch, fstype, expected):
    monkeypatch.setattr(inv_mod, "_run", _lsblk_only(json.dumps({"blockdevices": [
        {"name": "nvme5n1", "mountpoint": None, "fstype": fstype}]})))
    assert inv_mod._drive_availability("/dev/nvme5n1", "nvme") == (False, expected)


def test_availability_proc_fallback_when_lsblk_missing(monkeypatch):
    # lsblk unavailable -> _run yields None; consult /proc instead.
    monkeypatch.setattr(inv_mod, "_run", lambda cmd: None)
    reads = {
        "/proc/mdstat": "",
        "/proc/swaps": "Filename\tType\tSize\tUsed\tPriority\n",
        "/proc/mounts": "/dev/nvme5n1p2 /mnt/data ext4 rw 0 0\n",
    }
    monkeypatch.setattr(inv_mod, "_read_file", lambda p: reads.get(p))
    assert inv_mod._drive_availability("/dev/nvme5n1", "nvme") == (
        False, "mounted at /mnt/data")


def test_availability_label_formatting():
    d = Drive(alias="nvme0", linux_device="/dev/nvme0n1", serial="S", available=True)
    assert d.availability_label() == "available"
    d.available, d.unavailable_reason = False, "mounted at /data"
    assert d.availability_label() == "unavailable - mounted at /data"
    d.available, d.unavailable_reason = None, None
    assert d.availability_label() == "unknown"


# --------------------------------------------------------------------------- #
# Capacity (size / size_bytes), sourced from `nvme list` or unvme sysfs
# --------------------------------------------------------------------------- #
def test_unvme_capacity_populated(monkeypatch):
    # Driver-published sysfs identity includes the new size_bytes attribute.
    monkeypatch.setattr(inv_mod.identity, "list_unvme_sysfs_names", lambda: ["unvme1"])
    monkeypatch.setattr(inv_mod.identity, "read_unvme_sysfs_identity", lambda name: {
        "serial": "S111", "pci_bdf": "0006:06:00.0", "lba_size": "4096",
        "size_bytes": "3200000000000",
    })
    monkeypatch.setattr(inv_mod, "_pcie_for_pci", lambda pci: "gen4 x4")

    drives = inv_mod._drives_from_unvme()
    assert len(drives) == 1
    d = drives[0]
    assert d.size_bytes == 3200000000000
    assert d.size == "3.2TB"
    assert d.lba_size == 4096


def test_unvme_capacity_unknown_or_absent(monkeypatch):
    # "unknown" and a wholly absent attribute must both degrade to None,
    # never raise.
    monkeypatch.setattr(inv_mod.identity, "list_unvme_sysfs_names",
                        lambda: ["unvme1", "unvme2"])
    sysfs = {
        "unvme1": {"serial": "S1", "pci_bdf": "0006:06:00.0", "size_bytes": "unknown"},
        "unvme2": {"serial": "S2", "pci_bdf": "0006:07:00.0"},
    }
    monkeypatch.setattr(inv_mod.identity, "read_unvme_sysfs_identity",
                        lambda name: sysfs.get(name))
    monkeypatch.setattr(inv_mod, "_pcie_for_pci", lambda pci: None)

    drives = inv_mod._drives_from_unvme()
    assert len(drives) == 2
    for d in drives:
        assert d.size_bytes is None
        assert d.size is None


def test_drive_from_nvme_json_sets_size_bytes():
    dev = {
        "DevicePath": "/dev/nvme0n1", "SerialNumber": "S1", "ModelNumber": "M1",
        "SectorSize": 512, "NameSpace": 1, "PhysicalSize": 3200000000000,
    }
    d = inv_mod._drive_from_nvme_json(dev)
    assert d.size_bytes == 3200000000000
    assert d.size == "3.2TB"


def test_drive_from_nvme_json_missing_size_bytes():
    dev = {"DevicePath": "/dev/nvme0n1", "SerialNumber": "S1"}
    d = inv_mod._drive_from_nvme_json(dev)
    assert d.size_bytes is None
    assert d.size is None


def test_render_inventory_includes_capacity_column():
    inv = Inventory(drives=[
        Drive(alias="nvme0", linux_device="/dev/nvme0n1", serial="S111", model="PM1735",
              size="3.2TB", size_bytes=3200000000000, lba_size=4096, namespace=1,
              pci_bus_id="0000:83:00.0", pcie="gen4 x4", driver="nvme"),
        Drive(alias="nvme1", linux_device="/dev/unvme1", serial="S222", model="PM1735",
              size=None, size_bytes=None, lba_size=4096, namespace=1,
              pci_bus_id="0000:84:00.0", pcie="gen4 x4", driver="unvme"),
    ])
    out = inv_mod.render_inventory(inv)
    lines = out.splitlines()
    drives_idx = lines.index("Drives")
    header = lines[drives_idx + 1]
    assert "size" in header
    row0 = next(l for l in lines if l.startswith("nvme0"))
    assert "3.2TB" in row0
    row1 = next(l for l in lines if l.startswith("nvme1"))
    assert "S222" in row1  # unknown capacity renders blank, not a crash


def test_render_inventory_wraps_long_values_without_shifting_columns():
    gpu_uuid = "GPU-" + "a" * 36
    inv = Inventory(
        gpus=[GPU(alias="gpu0", smi_index=0, uuid=gpu_uuid, pci_bus_id="0007:01:00.0", name="RTX PRO 6000 Blackwell")],
        drives=[Drive(alias="nvme0", linux_device="/dev/nvme3n1", serial="LONG-SERIAL-123456789",
                      model="KIOXIA very long model name", pci_bus_id="0007:02:00.0", driver="nvme")],
    )
    lines = inv_mod.render_inventory(inv).splitlines()
    gpu_header = lines[lines.index("GPUs") + 1]
    gpu_row = lines[lines.index("GPUs") + 2]
    assert gpu_row[gpu_header.index("smi_index") :].startswith("0")
    assert gpu_row[gpu_header.index("pci_bus_id") :].startswith("0007:01:00.0")
    assert gpu_uuid in "".join(line[gpu_header.index("uuid") : gpu_header.index("pci_bus_id")].strip()
                               for line in lines[lines.index("GPUs") + 2 : lines.index("Drives") - 1])
    drive_header = lines[lines.index("Drives") + 1]
    drive_row = lines[lines.index("Drives") + 2]
    assert drive_row[drive_header.index("pci_bus_id") :].startswith("0007:02:00.0")
    assert all(len(line) <= len(drive_header) + 6 for line in lines[lines.index("Drives") + 2 :])


def test_drives_from_unvme_tolerates_unknown_numeric(monkeypatch):
    # The unvme driver publishes the literal "unknown" for nsid/lba_size/
    # size_bytes when the namespace identity is unavailable; parsing must not
    # abort the whole drive scan (regression for raw int()).
    monkeypatch.setattr(inv_mod.identity, "list_unvme_sysfs_names", lambda: ["unvme3"])
    monkeypatch.setattr(
        inv_mod.identity, "read_unvme_sysfs_identity",
        lambda name: {"serial": "S9", "pci_bdf": "0000:81:00.0", "model": "M",
                      "lba_size": "unknown", "nsid": "unknown", "size_bytes": "unknown"},
    )
    monkeypatch.setattr(inv_mod, "_pcie_for_pci", lambda pci: None)
    drives = inv_mod._drives_from_unvme()
    assert len(drives) == 1
    d = drives[0]
    assert d.serial == "S9" and d.driver == "unvme"
    assert d.lba_size is None and d.namespace is None and d.size_bytes is None


def test_collect_inventory_warns_when_nvme_cli_absent(monkeypatch):
    monkeypatch.setattr(inv_mod, "enumerate_gpus", lambda *a, **k: [])
    monkeypatch.setattr(inv_mod, "enumerate_drives", lambda: [])
    monkeypatch.setattr(inv_mod, "topology_matrix", lambda: "")
    monkeypatch.setattr("shutil.which", lambda name: None)
    inv = inv_mod.collect_inventory()
    assert any("nvme-cli not found" in w for w in inv.warnings)


def test_collect_inventory_quiet_when_nvme_cli_present(monkeypatch):
    monkeypatch.setattr(inv_mod, "enumerate_gpus", lambda *a, **k: [])
    monkeypatch.setattr(inv_mod, "enumerate_drives", lambda: [])
    monkeypatch.setattr(inv_mod, "topology_matrix", lambda: "")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/sbin/nvme")
    inv = inv_mod.collect_inventory()
    assert not any("nvme-cli not found" in w for w in inv.warnings)
