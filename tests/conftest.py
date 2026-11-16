# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures: a synthetic inventory and a tmp runtime dir."""

from __future__ import annotations

import os

import pytest

from scadactl import apply as apply_mod
from scadactl.models import Drive, GPU, Inventory


@pytest.fixture
def inventory() -> Inventory:
    """A synthetic 4-GPU / 4-drive inventory."""
    gpus = [
        GPU(alias="gpu0", smi_index=0, uuid="GPU-1111", pci_bus_id="0000:81:00.0",
            name="NVIDIA H100 80GB", numa=1, pcie="gen5 x16", notes="visible"),
        GPU(alias="gpu1", smi_index=1, uuid="GPU-2222", pci_bus_id="0000:82:00.0",
            name="NVIDIA H100 80GB", numa=1, pcie="gen5 x16", notes="visible"),
        GPU(alias="gpu2", smi_index=2, uuid="GPU-3333", pci_bus_id="0000:c1:00.0",
            name="NVIDIA H100 80GB", numa=2, pcie="gen5 x16", notes="visible"),
        GPU(alias="gpu3", smi_index=3, uuid="GPU-4444", pci_bus_id="0000:c2:00.0",
            name="NVIDIA H100 80GB", numa=2, pcie="gen5 x16", notes="visible"),
    ]
    drives = [
        Drive(alias="nvme0", linux_device="/dev/nvme2n1", serial="S111", model="PM1735",
              lba_size=4096, namespace=1, pci_bus_id="0000:83:00.0", pcie="gen4 x4", driver="nvme"),
        Drive(alias="nvme1", linux_device="/dev/nvme3n1", serial="S222", model="PM1735",
              lba_size=4096, namespace=1, pci_bus_id="0000:84:00.0", pcie="gen4 x4", driver="nvme"),
        Drive(alias="nvme2", linux_device="/dev/nvme4n1", serial="S333", model="PM1735",
              lba_size=4096, namespace=1, pci_bus_id="0000:c3:00.0", pcie="gen4 x4", driver="nvme"),
        Drive(alias="nvme3", linux_device="/dev/nvme5n1", serial="S444", model="PM1735",
              lba_size=4096, namespace=1, pci_bus_id="0000:c4:00.0", pcie="gen4 x4", driver="nvme"),
    ]
    return Inventory(gpus=gpus, drives=drives)


@pytest.fixture
def runtime_dir(tmp_path, monkeypatch) -> str:
    """Point SCADA_RUNTIME_DIR at a tmp dir (no root needed)."""
    d = tmp_path / "scada-run"
    d.mkdir()
    monkeypatch.setenv("SCADA_RUNTIME_DIR", str(d))
    return str(d)


@pytest.fixture(autouse=True)
def _isolate_config_store(tmp_path_factory, monkeypatch):
    """Point the config store and device policy at per-test tmp paths (no /etc writes)."""
    d = tmp_path_factory.mktemp("scada-configs")
    monkeypatch.setenv("SCADA_CONFIG_DIR", str(d))
    monkeypatch.setenv("SCADA_POLICY_FILE", str(d / "policy.yaml"))
    return str(d)


@pytest.fixture(autouse=True)
def _mock_domain_readiness(monkeypatch):
    """Keep mocked-launch apply tests independent of a real server socket."""
    monkeypatch.setattr(apply_mod, "_wait_for_domain_socket", lambda *args: True)
