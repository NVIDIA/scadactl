# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alias generation, alias<->stable-id resolution, UUID->index, UNVMe sysfs."""

from __future__ import annotations

import os

from scadactl import identity
from scadactl.models import Drive, GPU, Inventory


def test_alias_generation_order():
    gpus = [
        GPU(alias="", smi_index=2, uuid="GPU-C", pci_bus_id="0000:c1:00.0"),
        GPU(alias="", smi_index=0, uuid="GPU-A", pci_bus_id="0000:81:00.0"),
        GPU(alias="", smi_index=1, uuid="GPU-B", pci_bus_id="0000:82:00.0"),
    ]
    identity.assign_gpu_aliases(gpus)
    by_uuid = {g.uuid: g.alias for g in gpus}
    assert by_uuid == {"GPU-A": "gpu0", "GPU-B": "gpu1", "GPU-C": "gpu2"}


def test_resolve_gpu_by_alias_uuid_pci(inventory):
    assert identity.resolve_gpu(inventory, "gpu0").uuid == "GPU-1111"
    assert identity.resolve_gpu(inventory, "GPU-2222").alias == "gpu1"
    assert identity.resolve_gpu(inventory, "0000:c1:00.0").alias == "gpu2"
    assert identity.resolve_gpu(inventory, "nope") is None


def test_resolve_drive(inventory):
    assert identity.resolve_drive(inventory, "nvme3").serial == "S444"
    assert identity.resolve_drive(inventory, "S111").alias == "nvme0"
    assert identity.resolve_drive(inventory, "0000:84:00.0").alias == "nvme1"
    assert identity.resolve_drive(inventory, "/dev/nvme2n1").serial == "S111"
    assert identity.resolve_drive(inventory, "nvme3n1").serial == "S222"


def test_resolve_drive_rejects_empty_token_when_lender_drive_has_no_linux_device():
    inv = Inventory(drives=[Drive(
        alias="nvme0", linux_device=None, serial=None, pci_bus_id="0000:3c:00.0"
    )])

    assert identity.resolve_drive(inv, "") is None


def test_uuid_to_smi_index(inventory):
    assert identity.gpu_uuid_to_smi_index(inventory, "GPU-3333") == 2
    assert identity.gpu_uuid_to_smi_index(inventory, "GPU-NOPE") is None


def test_alias_token_detection():
    assert identity.is_gpu_alias("gpu0")
    assert identity.is_drive_alias("nvme12")
    assert not identity.is_gpu_alias("GPU-1111")
    assert not identity.is_drive_alias("S111")


def test_prebind_identity_by_pci(inventory):
    m = identity.prebind_identity_by_pci(inventory)
    assert m["0000:83:00.0"].serial == "S111"


def test_unvme_sysfs_correlation(tmp_path, monkeypatch, inventory):
    # Build a fake /sys/class/unvme tree with driver-published attributes.
    base = tmp_path / "unvme"
    (base / "unvme0").mkdir(parents=True)
    (base / "unvme0" / "serial").write_text("S444\n")
    (base / "unvme0" / "address").write_text("0000:c4:00.0\n")
    (base / "unvme0" / "model").write_text("PM1735\n")
    monkeypatch.setattr(identity, "UNVME_SYSFS_CLASS", str(base))

    ident = identity.read_unvme_sysfs_identity("unvme0")
    assert ident["serial"] == "S444"
    assert ident["pci_bdf"] == "0000:c4:00.0"

    mapping, warnings = identity.correlate_unvme_devices(inventory)
    assert mapping["/dev/unvme0"].serial == "S444"


def test_unvme_correlation_absent(tmp_path, monkeypatch, inventory):
    base = tmp_path / "empty"
    base.mkdir()
    monkeypatch.setattr(identity, "UNVME_SYSFS_CLASS", str(base))
    mapping, warnings = identity.correlate_unvme_devices(inventory)
    assert mapping == {}
    assert warnings  # warns that sysfs correlation is unavailable


def test_interim_bootstrap_single_new_device(monkeypatch):
    boot = identity.InterimUnvmeBootstrap()
    monkeypatch.setattr(identity, "list_unvme_devices", lambda: ["/dev/unvme0"])
    boot.snapshot_before()
    monkeypatch.setattr(identity, "list_unvme_devices", lambda: ["/dev/unvme0", "/dev/unvme1"])
    assert boot.attribute_new_device("0000:c4:00.0") == "/dev/unvme1"


def test_interim_bootstrap_ambiguous(monkeypatch):
    boot = identity.InterimUnvmeBootstrap()
    monkeypatch.setattr(identity, "list_unvme_devices", lambda: [])
    boot.snapshot_before()
    monkeypatch.setattr(identity, "list_unvme_devices", lambda: ["/dev/unvme0", "/dev/unvme1"])
    assert boot.attribute_new_device("0000:c4:00.0") is None


def test_alias_is_the_controller_name():
    drives = [
        Drive(alias="", linux_device="/dev/nvme0n1", serial="S1", pci_bus_id="0000:01:00.0"),
        Drive(alias="", linux_device="/dev/nvme0n2", serial="S1", pci_bus_id="0000:01:00.0"),
        Drive(alias="", linux_device="/dev/unvme1", serial="S2", pci_bus_id="0000:2b:00.0"),
        Drive(alias="", linux_device=None, serial="S3", pci_bus_id="0000:2c:00.0"),
    ]
    identity.assign_drive_aliases(drives)
    # Namespaces of one controller collide by design: validation rejects them.
    assert [d.alias for d in drives] == ["nvme0", "nvme0", "unvme1", "0000:2c:00.0"]


def test_multi_namespace_controller_is_reported():
    drives = [
        Drive(alias="", linux_device="/dev/nvme0n1", serial="S1", pci_bus_id="0000:01:00.0"),
        Drive(alias="", linux_device="/dev/nvme0n2", serial="S1", pci_bus_id="0000:01:00.0"),
        Drive(alias="", linux_device="/dev/nvme6n1", serial="S2", pci_bus_id="0000:2a:00.0"),
    ]
    # the single-namespace controller is absent from the result
    assert identity.multi_namespace_drives(drives) == {
        "0000:01:00.0": ["/dev/nvme0n1", "/dev/nvme0n2"]
    }
