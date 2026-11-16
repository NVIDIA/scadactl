# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config validation: positive case + each documented failure/warn case."""

from __future__ import annotations

from scadactl.models import Domain, DriveRef, Config
from scadactl.config import (
    validate_composition,
    validate_config,
)


def _good_config() -> Config:
    return Config(
        name="baseline",
        mps_enabled=True,
        server_env={"SCADA_NUM_QUEUE_PAIRS": "64"},
        domains=[
            Domain(
                name="gpu0_domain",
                gpu_uuid="GPU-1111",
                gpu_pci_bus_id="0000:81:00.0",
                drives=[DriveRef(serial="S111", pci_bus_id="0000:83:00.0")],
            ),
            Domain(
                name="gpu1_domain",
                gpu_uuid="GPU-2222",
                gpu_pci_bus_id="0000:82:00.0",
                drives=[DriveRef(serial="S222", pci_bus_id="0000:84:00.0")],
            ),
        ],
        domain_groups={"all": ["gpu0_domain", "gpu1_domain"]},
    )


def test_valid_config_passes(inventory):
    res = validate_config(_good_config(), inventory=inventory)
    assert res.ok, res.errors
    assert res.errors == []


def test_profiler_endpoint_mode_rejects_unknown_value(inventory):
    p = _good_config()
    p.profiler_endpoint_mode = "anything_goes"
    res = validate_config(p, inventory=inventory)
    assert any("profiler.endpoint_mode" in e for e in res.errors)


def test_legacy_nsys_requires_one_domain(inventory):
    p = _good_config()
    p.profiler_endpoint_mode = "legacy_nsys"
    res = validate_config(p, inventory=inventory)
    assert any("requires exactly one domain" in e for e in res.errors)


def test_legacy_nsys_is_exclusive_across_active_configs(inventory):
    from scadactl.models import Registry, RegistryDomain

    existing = Registry(domains=[
        RegistryDomain(
            name="existing", config="other", gpu_uuid="GPU-1111",
            gpu_pci_bus_id="0000:81:00.0", host_cuda_visible_devices="0",
            socket="/var/run/scada/scadad.swap.gpu0.sock",
            profiler_socket="/tmp/scada_profiler_socket",
            profiler_endpoint_mode="legacy_nsys",
        )
    ])
    p = _good_config()
    p.domains = p.domains[:1]
    p.profiler_endpoint_mode = "legacy_nsys"
    res = validate_config(p, inventory=inventory, registry=existing)
    assert any("already owned" in e for e in res.errors)


def test_unknown_gpu_uuid(inventory):
    p = _good_config()
    p.domains[0].gpu_uuid = "GPU-NOPE"
    p.domains[0].gpu_pci_bus_id = "0000:81:00.0"  # keep pci valid to isolate uuid
    res = validate_config(p, inventory=inventory)
    assert not res.ok
    assert any("unknown GPU UUID" in e for e in res.errors)


def test_unknown_drive_serial(inventory):
    p = _good_config()
    p.domains[0].drives[0].serial = "S-NOPE"
    res = validate_config(p, inventory=inventory)
    assert not res.ok
    assert any("unknown drive serial" in e for e in res.errors)


def test_unknown_gpu_pci(inventory):
    p = _good_config()
    p.domains[0].gpu_pci_bus_id = "0000:ff:00.0"
    res = validate_config(p, inventory=inventory)
    assert any("unknown GPU PCI bus id" in e for e in res.errors)


def test_unresolvable_alias(inventory):
    p = _good_config()
    p.domains[0].gpu_uuid = "gpu9"  # alias that cannot resolve
    p.domains[0].gpu_pci_bus_id = "gpu9"
    res = validate_config(p, inventory=inventory)
    assert not res.ok
    assert any("cannot be resolved" in e for e in res.errors)


def test_duplicate_domain_names(inventory):
    p = _good_config()
    p.domains[1].name = "gpu0_domain"
    res = validate_config(p, inventory=inventory)
    assert any("duplicate domain name" in e for e in res.errors)


def test_duplicate_drive_ownership(inventory):
    p = _good_config()
    p.domains[1].drives[0] = DriveRef(serial="S111", pci_bus_id="0000:83:00.0")
    res = validate_config(p, inventory=inventory)
    assert any("owned by multiple domains" in e for e in res.errors)


def test_duplicate_drive_within_one_domain(inventory):
    p = _good_config()
    p.domains[0].drives = [
        DriveRef(serial="S111", pci_bus_id="0000:83:00.0"),
        DriveRef(serial="S111", pci_bus_id="0000:83:00.0"),
    ]
    res = validate_config(p, inventory=inventory)
    assert not res.ok
    assert any("listed more than once in domain" in e for e in res.errors)
    # Assert EXACTLY ONE such error
    assert len([e for e in res.errors if "listed more than once in domain" in e]) == 1


def test_domain_with_no_drives(inventory):
    p = _good_config()
    p.domains[0].drives = []
    res = validate_config(p, inventory=inventory)
    assert any("has no drives" in e for e in res.errors)


def test_missing_socket_group(inventory):
    p = _good_config()
    p.domains[0].socket_group = "definitely-not-a-real-group-xyz"
    res = validate_config(p, inventory=inventory)
    assert any("socket group" in e for e in res.errors)


def test_non_power_of_two_drive_count_is_allowed(inventory):
    # SCADA's storage layer accepts arbitrary drive counts, so a non-power-of-two
    # count must validate cleanly -- no error and no power-of-two warning.
    p = _good_config()
    # 3 drives, none overlapping domain[1]'s S222 drive, all matching serial/PCI
    # pairs, so nothing else trips.
    p.domains[0].drives = [
        DriveRef(serial="S111", pci_bus_id="0000:83:00.0"),
        DriveRef(serial="S333", pci_bus_id="0000:c3:00.0"),
        DriveRef(serial="S444", pci_bus_id="0000:c4:00.0"),
    ]
    res = validate_config(p, inventory=inventory)
    assert res.ok, res.errors
    assert not any("power" in w.lower() for w in res.warnings)


def test_serial_pci_mismatch(inventory):
    p = _good_config()
    p.domains[0].drives[0] = DriveRef(serial="S111", pci_bus_id="0000:84:00.0")
    res = validate_config(p, inventory=inventory)
    assert not res.ok
    assert any("identify different drives" in e for e in res.errors)


def test_gpu_uuid_pci_mismatch(inventory):
    # GPU-1111 is gpu0's UUID but 0000:c1:00.0 is gpu2's PCI: a pair that points
    # at two different GPUs must be rejected (runtime selection follows the UUID).
    # (gpu2's PCI, not gpu1's, so this doesn't also trip duplicate-GPU with dom[1].)
    p = _good_config()
    p.domains[0].gpu_uuid = "GPU-1111"
    p.domains[0].gpu_pci_bus_id = "0000:c1:00.0"
    res = validate_config(p, inventory=inventory)
    assert not res.ok
    assert any("identify different GPUs" in e for e in res.errors)


def test_duplicate_gpu_ownership(inventory):
    p = _good_config()
    p.domains[1].gpu_uuid = "GPU-1111"
    p.domains[1].gpu_pci_bus_id = "0000:81:00.0"
    res = validate_config(p, inventory=inventory)
    assert any("owned by multiple domains" in e for e in res.errors)


def test_unstable_alias_warning(inventory):
    # Drive referenced by alias in a persisted field -> warning (not error w/o inv? with inv it's also unknown)
    p = _good_config()
    p.domains[0].drives[0] = DriveRef(serial="nvme0", pci_bus_id="0000:83:00.0")
    res = validate_config(p)  # no inventory: only the alias-warning fires
    assert any("alias" in w for w in res.warnings)


def test_resource_in_use_warning(inventory):
    from scadactl.models import (
        Registry,
        RegistryDomain,
        RegistryDrive,
    )

    reg = Registry(
        active_configs=["other"],
        domains=[
            RegistryDomain(
                name="other_dom",
                config="other",
                gpu_uuid="GPU-1111",
                gpu_pci_bus_id="0000:81:00.0",
                host_cuda_visible_devices="0",
                socket="/var/run/scada/x.sock",
                drives=[RegistryDrive(linux_device_before_bind=None, serial="S111")],
            )
        ],
    )
    res = validate_config(_good_config(), inventory=inventory, registry=reg)
    assert any("already in use" in w for w in res.warnings)


def test_registry_domain_name_collision_hard_errors_for_other_config(inventory):
    from scadactl.models import Registry, RegistryDomain, RegistryDrive

    reg = Registry(
        active_configs=["a"],
        domains=[
            RegistryDomain(
                name="gpu0_domain",
                config="a",
                gpu_uuid="GPU-1111",
                gpu_pci_bus_id="0000:81:00.0",
                host_cuda_visible_devices="0",
                socket="/var/run/scada/a.sock",
                drives=[RegistryDrive(linux_device_before_bind=None, serial="S111")],
            )
        ],
    )

    # A different config ("b") reusing the same domain name is a hard error.
    config_b = Config(
        name="b",
        domains=[
            Domain(name="gpu0_domain", gpu_uuid="GPU-2222", gpu_pci_bus_id="0000:82:00.0",
                   drives=[DriveRef(serial="S222", pci_bus_id="0000:84:00.0")]),
        ],
    )
    res = validate_config(config_b, inventory=inventory, registry=reg)
    assert not res.ok
    assert any("already owned by active config" in e for e in res.errors)

    # Reapplying the SAME config ("a") with the same domain name must not error.
    config_a = Config(
        name="a",
        domains=[
            Domain(name="gpu0_domain", gpu_uuid="GPU-1111", gpu_pci_bus_id="0000:81:00.0",
                   drives=[DriveRef(serial="S111", pci_bus_id="0000:83:00.0")]),
        ],
    )
    res_a = validate_config(config_a, inventory=inventory, registry=reg)
    assert res_a.ok, res_a.errors


def test_composition_collision():
    p1 = Config(
        name="a",
        domains=[Domain(name="d1", gpu_uuid="GPU-1111", gpu_pci_bus_id="0000:81:00.0",
                        drives=[DriveRef(serial="S111", pci_bus_id="0000:83:00.0")])],
    )
    p2 = Config(
        name="b",
        domains=[Domain(name="d2", gpu_uuid="GPU-1111", gpu_pci_bus_id="0000:81:00.0",
                        drives=[DriveRef(serial="S999", pci_bus_id="0000:99:00.0")])],
    )
    res = validate_composition([p1, p2])
    assert not res.ok
    assert any("collide on GPU" in e for e in res.errors)


def test_composition_disjoint_ok():
    p1 = Config(name="a", domains=[Domain(name="d1", gpu_uuid="GPU-1111",
                 gpu_pci_bus_id="0000:81:00.0", drives=[DriveRef(serial="S111", pci_bus_id="0000:83:00.0")])])
    p2 = Config(name="b", domains=[Domain(name="d2", gpu_uuid="GPU-2222",
                 gpu_pci_bus_id="0000:82:00.0", drives=[DriveRef(serial="S222", pci_bus_id="0000:84:00.0")])])
    res = validate_composition([p1, p2])
    assert res.ok, res.errors


def test_domain_group_unknown_member(inventory):
    p = _good_config()
    p.domain_groups = {"bad": ["nope"]}
    res = validate_config(p, inventory=inventory)
    assert any("unknown domain" in e for e in res.errors)


def test_composition_collision_by_pci_only():
    # PCI bus id is a stable selector: two configs pinning the same GPU/drive by
    # PCI only (no UUID/serial) must be reported as colliding, not "compatible".
    a = Config(name="a", domains=[Domain(
        name="a0", gpu_uuid="", gpu_pci_bus_id="0000:81:00.0",
        drives=[DriveRef(serial="", pci_bus_id="0000:83:00.0")])])
    b = Config(name="b", domains=[Domain(
        name="b0", gpu_uuid="", gpu_pci_bus_id="0000:81:00.0",
        drives=[DriveRef(serial="", pci_bus_id="0000:84:00.0")])])
    res = validate_composition([a, b])
    assert not res.ok
    assert any("0000:81:00.0" in e for e in res.errors)


def test_composition_single_error_when_uuid_and_pci_both_match():
    def _dom(n):
        return Domain(name=n, gpu_uuid="GPU-1", gpu_pci_bus_id="0000:81:00.0",
                      drives=[DriveRef(serial="S1", pci_bus_id="0000:83:00.0")])
    res = validate_composition([Config(name="a", domains=[_dom("a0")]),
                                Config(name="b", domains=[_dom("b0")])])
    assert len([e for e in res.errors if "on GPU" in e]) == 1
    assert len([e for e in res.errors if "on drive" in e]) == 1


def test_resolve_rejects_device_without_stable_id():
    import pytest
    from scadactl.config import resolve_domain_aliases
    from scadactl.models import Inventory, GPU, Drive

    # GPU resolvable by alias but with no UUID/PCI -> nothing stable to persist.
    inv = Inventory(
        gpus=[GPU(alias="gpu0", smi_index=0, uuid=None, pci_bus_id=None)],
        drives=[Drive(alias="nvme0", linux_device="/dev/nvme0n1", serial="S1",
                      pci_bus_id="0000:83:00.0")],
    )
    dom = Domain(name="d", gpu_uuid="gpu0", gpu_pci_bus_id="gpu0",
                 drives=[DriveRef(serial="nvme0", pci_bus_id="nvme0")])
    with pytest.raises(ValueError, match="no stable identifier"):
        resolve_domain_aliases(dom, inv)

    # GPU is fine, but the drive resolves by alias with no serial/PCI.
    inv2 = Inventory(
        gpus=[GPU(alias="gpu0", smi_index=0, uuid="GPU-1", pci_bus_id="0000:81:00.0")],
        drives=[Drive(alias="nvme0", linux_device="/dev/nvme0n1", serial=None,
                      pci_bus_id=None)],
    )
    dom2 = Domain(name="d", gpu_uuid="gpu0", gpu_pci_bus_id="gpu0",
                  drives=[DriveRef(serial="nvme0", pci_bus_id="nvme0")])
    with pytest.raises(ValueError, match="no stable identifier"):
        resolve_domain_aliases(dom2, inv2)


def test_multi_namespace_drive_is_rejected(inventory):
    # A config stores serial + BDF, which both namespaces share.
    from scadactl.models import Drive
    from scadactl import identity

    inventory.drives.append(
        Drive(alias="", linux_device="/dev/nvme2n2", serial="S111",
              pci_bus_id="0000:83:00.0")
    )
    identity.assign_drive_aliases(inventory.drives)

    res = validate_config(_good_config(), inventory=inventory)
    assert any("multiple namespaces" in e for e in res.errors), res.errors
