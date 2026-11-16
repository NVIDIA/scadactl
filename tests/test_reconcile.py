# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reconciliation planning: classify each desired domain into apply's action."""

from __future__ import annotations

from scadactl import apply as apply_mod
from scadactl import registry as registry_mod
from scadactl import reconcile as rec
from scadactl.reconcile import DomainAction
from scadactl.models import (
    Config, Domain, Drive, DriveRef, GPU, Inventory, Registry,
    RegistryDomain, RegistryDrive,
)

ALIVE = lambda rd: True          # noqa: E731 - test-only liveness stubs
DEAD = lambda rd: False          # noqa: E731
FREE = lambda dev: False         # noqa: E731 - device confirmed not in use
HELD = lambda dev: True          # noqa: E731 - device in active use
UNKNOWN = lambda dev: None       # noqa: E731 - usage indeterminate (no fuser/probe error)


def _inventory(driver="nvme") -> Inventory:
    gpus = [
        GPU(alias="gpu0", smi_index=0, uuid="GPU-1111", pci_bus_id="0000:81:00.0", name="H100"),
        GPU(alias="gpu1", smi_index=1, uuid="GPU-2222", pci_bus_id="0000:82:00.0", name="H100"),
    ]
    drives = [
        Drive(alias="nvme0", linux_device="/dev/unvme0" if driver == "unvme" else "/dev/nvme2n1",
              serial="S111", pci_bus_id="0000:83:00.0", driver=driver),
        Drive(alias="nvme1", linux_device="/dev/unvme1" if driver == "unvme" else "/dev/nvme3n1",
              serial="S222", pci_bus_id="0000:84:00.0", driver=driver),
    ]
    return Inventory(gpus=gpus, drives=drives)


def _config() -> Config:
    return Config(
        name="c", domains=[
            Domain(name="d0", gpu_uuid="GPU-1111", gpu_pci_bus_id="0000:81:00.0",
                   drives=[DriveRef(serial="S111", pci_bus_id="0000:83:00.0")]),
            Domain(name="d1", gpu_uuid="GPU-2222", gpu_pci_bus_id="0000:82:00.0",
                   drives=[DriveRef(serial="S222", pci_bus_id="0000:84:00.0")]),
        ])


def _plan(config, inventory, existing, **kw):
    desired = apply_mod.build_registry_domains(config, inventory)
    return rec.plan_reconciliation(config, inventory, existing, desired, **kw)


def _active_registry(config, inventory) -> Registry:
    """A registry with `config` active and its domains recorded (matching desired)."""
    desired = apply_mod.build_registry_domains(config, inventory)
    return registry_mod.reconcile_add_config(Registry(), config.name, desired)


# --------------------------------------------------------------------------- #
def test_fresh_apply_builds_all_and_binds_all():
    plan = _plan(_config(), _inventory("nvme"), Registry())
    assert plan.ok and not plan.active
    assert [d.action for d in plan.domains] == [DomainAction.BUILD, DomainAction.BUILD]
    assert plan.bind_pci() == ["0000:83:00.0", "0000:84:00.0"]
    assert plan.reclaim_pci() == []
    assert len(plan.actionable()) == 2


def test_unchanged_reapply_is_all_noop():
    cfg, inv = _config(), _inventory("unvme")   # already bound + running
    plan = _plan(cfg, inv, _active_registry(cfg, inv), is_alive=ALIVE)
    assert plan.ok and plan.active
    assert all(d.action == DomainAction.NOOP for d in plan.domains)
    assert plan.actionable() == []
    assert plan.bind_pci() == [] and plan.reclaim_pci() == []


def test_dead_managed_domain_self_heals():
    cfg, inv = _config(), _inventory("unvme")
    reg = _active_registry(cfg, inv)
    # d0 dead, d1 alive.
    alive = lambda rd: rd.name != "d0"          # noqa: E731
    plan = _plan(cfg, inv, reg, is_alive=alive)
    by = {d.name: d for d in plan.domains}
    assert by["d0"].action == DomainAction.SELF_HEAL
    assert by["d1"].action == DomainAction.NOOP
    # Drives already bound to unvme -> self-heal rebinds nothing.
    assert by["d0"].bind_pci == []


def test_self_heal_rebinds_a_drive_that_fell_back_to_nvme():
    cfg = _config()
    inv = _inventory("unvme")
    inv.drives[0].driver = "nvme"               # d0's drive lost its unvme binding
    inv.drives[0].linux_device = "/dev/nvme2n1"
    plan = _plan(cfg, inv, _active_registry(cfg, inv), is_alive=DEAD)
    by = {d.name: d for d in plan.domains}
    assert by["d0"].action == DomainAction.SELF_HEAL
    assert by["d0"].bind_pci == ["0000:83:00.0"]


def test_active_config_with_changed_definition_is_incompatible():
    cfg, inv = _config(), _inventory("unvme")
    reg = _active_registry(cfg, inv)
    reg.domains[0].server_env = {"SCADA_NUM_QUEUE_PAIRS": "16"}   # drifted from file
    plan = _plan(cfg, inv, reg, is_alive=ALIVE)
    assert not plan.ok
    assert any("different definition for domain d0" in e for e in plan.errors)
    assert any("stop it" in e for e in plan.errors)


def test_active_config_with_added_domain_is_incompatible():
    cfg, inv = _config(), _inventory("unvme")
    reg = _active_registry(cfg, inv)
    reg.domains = [d for d in reg.domains if d.name != "d1"]      # registry only has d0
    plan = _plan(cfg, inv, reg, is_alive=ALIVE)
    assert not plan.ok
    assert any("different set of domains" in e for e in plan.errors)


def test_resource_owned_by_other_active_config_conflicts():
    cfg, inv = _config(), _inventory("nvme")
    other = Registry(
        active_configs=["other"],
        domains=[RegistryDomain(
            name="o0", config="other", gpu_uuid="GPU-9", gpu_pci_bus_id="0000:aa:00.0",
            host_cuda_visible_devices="0", socket="/s",
            drives=[RegistryDrive(linux_device_before_bind=None,
                                  serial="S111", pci_bus_id="0000:83:00.0")])],
    )
    plan = _plan(cfg, inv, other)
    assert not plan.ok
    assert any("managed by active config 'other'" in e for e in plan.errors)
    assert any("stop --config other" in e for e in plan.errors)


def test_unmanaged_idle_unvme_drive_is_reclaimed():
    cfg = _config()
    inv = _inventory("unvme")                   # bound, but no registry owns it
    plan = _plan(cfg, inv, Registry(), device_in_use=FREE)
    assert plan.ok and not plan.active
    assert [d.action for d in plan.domains] == [DomainAction.RECLAIM, DomainAction.RECLAIM]
    assert plan.reclaim_pci() == ["0000:83:00.0", "0000:84:00.0"]
    # Reclaimed drives are freed and then rebound, so they are also bind targets.
    assert plan.bind_pci() == ["0000:83:00.0", "0000:84:00.0"]


def test_unmanaged_unvme_drive_in_use_blocks_apply():
    cfg = _config()
    inv = _inventory("unvme")
    plan = _plan(cfg, inv, Registry(), device_in_use=HELD)
    assert not plan.ok
    assert any("held by a running process with no registry record" in e for e in plan.errors)


def test_unmanaged_unvme_drive_with_unknown_use_blocks_apply():
    # Fail closed: if we cannot establish a bound-but-unmanaged drive is idle
    # (no fuser / probe error -> None), refuse to reclaim rather than free a
    # possibly-in-use device.
    cfg = _config()
    inv = _inventory("unvme")
    plan = _plan(cfg, inv, Registry(), device_in_use=UNKNOWN)
    assert not plan.ok
    assert plan.reclaim_pci() == []
    assert any("usage cannot be determined" in e for e in plan.errors)


def test_mixed_reclaim_and_bind():
    cfg = _config()
    inv = _inventory("nvme")
    inv.drives[0].driver = "unvme"              # d0 orphan-bound; d1 fresh
    inv.drives[0].linux_device = "/dev/unvme0"
    plan = _plan(cfg, inv, Registry(), device_in_use=FREE)
    by = {d.name: d for d in plan.domains}
    assert by["d0"].action == DomainAction.RECLAIM
    assert by["d0"].reclaim_pci == ["0000:83:00.0"] and by["d0"].bind_pci == ["0000:83:00.0"]
    assert by["d1"].action == DomainAction.BUILD and by["d1"].bind_pci == ["0000:84:00.0"]
