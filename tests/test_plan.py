# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan diffing: resolve, inherited/overridden env, create/replace actions."""

from __future__ import annotations

from scadactl import plan as plan_mod
from scadactl.models import (
    Domain,
    DriveRef,
    Config,
    Registry,
    RegistryDomain,
    RegistryDrive,
)


def _config():
    return Config(
        name="custom-layout",
        mps_enabled=True,
        server_env={"SCADA_NUM_QUEUE_PAIRS": "64", "SCADA_QUEUE_SIZE_SQ_GPU": "1024"},
        domains=[
            Domain(
                name="gpu0_domain",
                gpu_uuid="GPU-1111",
                gpu_pci_bus_id="0000:81:00.0",
                drives=[DriveRef(serial="S444", pci_bus_id="0000:c4:00.0")],
                server_env={"SCADA_NUM_QUEUE_PAIRS": "128"},
            )
        ],
    )


def test_plan_resolves_host_index_and_socket(inventory):
    p = plan_mod.compute_plan(_config(), inventory=inventory, registry=Registry())
    d = p.domains[0]
    assert d.host_cuda_visible_devices == "0"  # GPU-1111 -> smi index 0
    assert d.gpu_alias == "gpu0"
    assert d.socket.endswith("scadad.swap.gpu0.sock")
    assert d.profiler_endpoint_mode == "managed"
    assert d.profiler_socket.endswith("scadad.profiler.gpu0.sock")
    assert d.action == "create"


def test_plan_env_inherited_vs_override(inventory):
    p = plan_mod.compute_plan(_config(), inventory=inventory, registry=Registry())
    d = p.domains[0]
    assert d.resolved_server_env["SCADA_NUM_QUEUE_PAIRS"] == "128"
    assert "SCADA_NUM_QUEUE_PAIRS" in d.overridden_keys
    assert "SCADA_QUEUE_SIZE_SQ_GPU" in d.inherited_keys


def test_plan_host_changes(inventory):
    p = plan_mod.compute_plan(_config(), inventory=inventory, registry=Registry())
    joined = " ".join(p.host_changes)
    assert "bind selected drives" in joined
    assert "start MPS" in joined
    assert "start 1 scada-swap-domain process" in joined
    assert "publish" in joined


def test_plan_reapply_marks_replace(inventory):
    existing = Registry(
        active_configs=["custom-layout"],
        domains=[
            RegistryDomain(
                name="gpu0_domain",
                config="custom-layout",
                gpu_uuid="GPU-1111",
                gpu_pci_bus_id="0000:81:00.0",
                host_cuda_visible_devices="0",
                socket="/x.sock",
                drives=[RegistryDrive(linux_device_before_bind=None, serial="S444")],
            )
        ],
    )
    p = plan_mod.compute_plan(_config(), inventory=inventory, registry=existing)
    assert p.domains[0].action == "replace"


def test_plan_detects_removed_domains(inventory):
    existing = Registry(
        active_configs=["custom-layout"],
        domains=[
            RegistryDomain(
                name="gone_domain",
                config="custom-layout",
                gpu_uuid="GPU-2222",
                gpu_pci_bus_id="0000:82:00.0",
                host_cuda_visible_devices="1",
                socket="/y.sock",
            )
        ],
    )
    p = plan_mod.compute_plan(_config(), inventory=inventory, registry=existing)
    assert "gone_domain" in p.removed_domains


def test_plan_without_inventory_leaves_index_none():
    p = plan_mod.compute_plan(_config(), inventory=None, registry=Registry())
    assert p.domains[0].host_cuda_visible_devices is None


def test_render_plan_smoke(inventory):
    p = plan_mod.compute_plan(_config(), inventory=inventory, registry=Registry())
    text = plan_mod.render_plan(p)
    assert "Config: custom-layout" in text
    assert "SCADA_NUM_QUEUE_PAIRS=128 (override)" in text


def test_render_plan_displays_pci_only_drive_without_empty_serial_separator():
    plan = plan_mod.Plan(
        config_name="lender",
        domains=[plan_mod.DomainPlan(
            name="d0", gpu_alias="gpu0", gpu_uuid="GPU-1",
            gpu_pci_bus_id="0000:81:00.0", storage_backend="lender",
            host_cuda_visible_devices="0",
            drives=[{"serial": "", "pci_bus_id": "0000:3c:00.0"}],
        )],
    )

    text = plan_mod.render_plan(plan)

    assert "      0000:3c:00.0" in text
    assert "/ 0000:3c:00.0" not in text


def test_build_registry_domains_resolves_pci_only_gpu_and_serial_only_drive(inventory):
    """Execution supports every selector form validation accepts.

    A GPU pinned by PCI **only** (no UUID) and a drive pinned by serial **only**
    (no PCI) both validate, so apply must resolve them through the inventory:
    otherwise the launcher gets an empty GPU column and the drive is never bound.
    """
    from scadactl import apply as apply_mod

    cfg = Config(
        name="selectors",
        domains=[
            Domain(
                name="d0",
                gpu_uuid="",                       # PCI-only GPU selector
                gpu_pci_bus_id="0000:81:00.0",
                drives=[DriveRef(serial="S111", pci_bus_id="")],  # serial-only drive
            )
        ],
    )
    rd = apply_mod.build_registry_domains(cfg, inventory)[0]

    # CUDA column derived from the resolved GPU despite there being no UUID.
    assert rd.host_cuda_visible_devices == "0"
    assert rd.gpu_alias == "gpu0"
    # Identities completed from inventory so the published record is unambiguous.
    assert rd.gpu_uuid == "GPU-1111"
    assert rd.gpu_pci_bus_id == "0000:81:00.0"
    # The drive's PCI (what binding and the launcher need) resolved from serial.
    assert rd.drives[0].pci_bus_id == "0000:83:00.0"
    assert rd.drives[0].serial == "S111"
