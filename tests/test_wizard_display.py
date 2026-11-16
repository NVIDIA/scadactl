# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive wizard display uses host-visible NVMe names."""

from __future__ import annotations

from scadactl import wizard as wizard_mod
from scadactl.wizard import ConfigWizard


def test_drive_table_uses_lsblk_name(inventory, capsys):
    wizard = ConfigWizard(inventory, "demo")

    wizard._print_drive_table([inventory.drives[0]], inventory.gpus[0])

    output = capsys.readouterr().out
    assert "host device name" in output
    assert "nvme2n1" in output
    assert "nvme0" not in output


def test_drive_display_name_uses_namespace_basename(inventory):
    wizard = ConfigWizard(inventory, "demo")

    assert wizard._drive_display_name(inventory.drives[0]) == "nvme2n1"


def test_select_drives_accepts_lsblk_name(inventory, monkeypatch):
    wizard = ConfigWizard(inventory, "demo")
    monkeypatch.setattr(wizard_mod, "_prompt", lambda *args, **kwargs: "nvme2n1")

    selected = wizard._select_drives(inventory.gpus[0])

    assert [d.serial for d in selected] == ["S111"]
