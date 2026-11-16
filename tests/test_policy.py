# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host device policy: allow/block lists, inventory annotation, apply gating."""

from __future__ import annotations

import os

from scadactl import apply as apply_mod
from scadactl import cli
from scadactl import config as config_mod
from scadactl import inventory as inv_mod
from scadactl import policy as policy_mod
from scadactl.models import Config, Domain, DriveRef, Policy, PolicyEntry


def _config_gpu0() -> Config:
    """A minimal valid config selecting gpu0 (GPU-1111) + one drive."""
    return Config(
        name="t",
        mps_enabled=False,
        domains=[
            Domain(name="d0", gpu_uuid="GPU-1111", gpu_pci_bus_id="0000:81:00.0",
                   drives=[DriveRef(serial="S111", pci_bus_id="0000:83:00.0")]),
        ],
    )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def test_load_empty_when_absent():
    assert policy_mod.load().is_empty()


def test_save_load_roundtrip():
    pol = Policy(
        gpu_block=[PolicyEntry(key="GPU-3333", pci_bus_id="0000:c1:00.0",
                               reason="reserved", added="2026-07-18 12:00:00")],
        drive_allow=[PolicyEntry(key="S111", pci_bus_id="0000:83:00.0")],
    )
    policy_mod.save(pol)
    back = policy_mod.load()
    assert [e.key for e in back.gpu_block] == ["GPU-3333"]
    assert back.gpu_block[0].reason == "reserved"
    assert [e.key for e in back.drive_allow] == ["S111"]


# --------------------------------------------------------------------------- #
# Decisions
# --------------------------------------------------------------------------- #
def test_block_by_serial_or_pci():
    pol = Policy(drive_block=[PolicyEntry(key="S444", pci_bus_id="0000:c4:00.0",
                                          reason="boot spare")])
    assert policy_mod.drive_decision(pol, "S444", None) == (True, "boot spare")
    assert policy_mod.drive_decision(pol, None, "0000:c4:00.0")[0] is True   # by PCI
    assert policy_mod.drive_decision(pol, "S111", "0000:83:00.0") == (False, None)


def test_allow_list_is_default_deny():
    pol = Policy(gpu_allow=[PolicyEntry(key="GPU-1111")])
    assert policy_mod.gpu_decision(pol, "GPU-1111", None) == (False, None)
    blocked, reason = policy_mod.gpu_decision(pol, "GPU-2222", None)
    assert blocked and reason == "not in allow-list"


def test_block_wins_over_allow():
    pol = Policy(
        gpu_block=[PolicyEntry(key="GPU-1111", reason="maintenance")],
        gpu_allow=[PolicyEntry(key="GPU-1111")],
    )
    assert policy_mod.gpu_decision(pol, "GPU-1111", None) == (True, "maintenance")


# --------------------------------------------------------------------------- #
# Inventory annotation
# --------------------------------------------------------------------------- #
def test_annotate_marks_blocked_devices(inventory):
    policy_mod.save(Policy(
        gpu_block=[PolicyEntry(key="GPU-3333", reason="reserved")],
        drive_block=[PolicyEntry(key="S444")],
    ))
    inv_mod._annotate_policy(inventory)
    assert inventory.gpus[2].blocked and inventory.gpus[2].block_reason == "reserved"
    assert inventory.drives[3].blocked
    assert inventory.gpus[0].blocked is False
    # Combined status label shows the block, dominating mount availability.
    assert inventory.drives[3].status_label().startswith("blocked -")


def test_empty_policy_is_a_noop(inventory):
    inv_mod._annotate_policy(inventory)  # no file written
    assert all(not g.blocked for g in inventory.gpus)
    assert all(not d.blocked for d in inventory.drives)


# --------------------------------------------------------------------------- #
# Validate (warn) + apply (gate)
# --------------------------------------------------------------------------- #
def test_validate_warns_on_blocked(inventory):
    inventory.gpus[0].blocked = True
    inventory.gpus[0].block_reason = "reserved"
    res = config_mod.validate_config(_config_gpu0(), inventory=inventory)
    assert res.ok  # policy block is a warning at validate time, not a hard error
    assert any("blocked by policy" in w for w in res.warnings)


def test_apply_gate_refuses_blocked(runtime_dir, inventory):
    inventory.gpus[0].blocked = True
    inventory.gpus[0].block_reason = "reserved"
    res = apply_mod.apply_config(_config_gpu0(), inventory, dry_run=True, force=False)
    assert not res.ok
    assert any("blocked by policy" in e for e in res.errors)
    assert any("--force" in e for e in res.errors)


def test_apply_force_overrides_block(runtime_dir, inventory):
    inventory.gpus[0].blocked = True
    inventory.gpus[0].block_reason = "reserved"
    res = apply_mod.apply_config(_config_gpu0(), inventory, dry_run=True, force=True)
    assert res.ok, res.errors
    assert any("overridden by --force" in a for a in res.actions)


# --------------------------------------------------------------------------- #
# CLI (admin-only mutation)
# --------------------------------------------------------------------------- #
def test_policy_block_requires_root(monkeypatch, capsys):
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    assert cli.main(["policy", "block", "--gpus", "gpu0"]) == 2
    assert "root" in capsys.readouterr().err


def test_policy_block_list_unblock_cycle(monkeypatch, capsys, inventory):
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    monkeypatch.setattr(cli.inventory_mod, "collect_inventory", lambda: inventory)

    assert cli.main(["policy", "block", "--gpus", "gpu2", "--drives", "nvme3",
                     "--reason", "reserved for team X"]) == 0
    pol = policy_mod.load()
    assert [e.key for e in pol.gpu_block] == ["GPU-3333"]
    assert [e.key for e in pol.drive_block] == ["S444"]
    assert pol.gpu_block[0].reason == "reserved for team X"

    capsys.readouterr()
    assert cli.main(["policy", "list"]) == 0
    out = capsys.readouterr().out
    assert "GPU-3333" in out and "reserved for team X" in out and "gpu2" in out

    assert cli.main(["policy", "unblock", "--gpus", "gpu2"]) == 0
    assert policy_mod.load().gpu_block == []


def test_policy_block_unknown_device_errors(monkeypatch, capsys, inventory):
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    monkeypatch.setattr(cli.inventory_mod, "collect_inventory", lambda: inventory)
    assert cli.main(["policy", "block", "--gpus", "gpu9"]) == 2
    assert "unknown device" in capsys.readouterr().err


def test_policy_load_rejects_non_mapping():
    # Valid YAML that is not a mapping (a list/scalar) must raise PolicyError,
    # not an unexpected AttributeError inside Policy.from_dict().
    import pytest
    path = policy_mod.policy_file()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("- just\n- a\n- list\n")
    with pytest.raises(policy_mod.PolicyError, match="must be a mapping"):
        policy_mod.load()


def test_policy_list_survives_missing_inventory(monkeypatch, capsys):
    # inventory unavailable (None) must not crash policy list -> degrade to keys.
    pol = Policy(gpu_block=[PolicyEntry(key="GPU-XYZ", pci_bus_id="0000:81:00.0")])
    policy_mod.save(pol)
    monkeypatch.setattr(cli, "_load_inventory_best_effort", lambda: None)
    rc = cli.main(["policy", "list"])
    assert rc == 0
    assert "GPU-XYZ" in capsys.readouterr().out
