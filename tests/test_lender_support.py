# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storage Lender's optimistic vfio inventory and lifecycle behavior."""

from __future__ import annotations

import stat
from types import SimpleNamespace

from scadactl import apply as apply_mod
from scadactl import config as config_mod
from scadactl import inventory as inventory_mod
from scadactl.backend import binding, launcher, lender as lender_backend
from scadactl import diagnostics
from scadactl.models import (
    Config, ConfigEntry, Domain, Drive, DriveRef, GPU, Inventory, Registry,
    RegistryDomain, RegistryDrive,
)


def test_vfio_nvme_is_an_optimistic_lender_candidate(monkeypatch):
    monkeypatch.setattr(inventory_mod.os, "listdir", lambda path: ["0000:3c:00.0"])
    monkeypatch.setattr(
        inventory_mod, "_read_file",
        lambda path: "0x010802" if path.endswith("/class") else None,
    )
    monkeypatch.setattr(inventory_mod, "_driver_for_pci", lambda _pci: "vfio-pci")
    monkeypatch.setattr(inventory_mod, "_pcie_for_pci", lambda _pci: "gen5 x4")

    drives = inventory_mod._vfio_nvme_drives()

    assert len(drives) == 1
    assert drives[0].pci_bus_id == "0000:3c:00.0"
    assert drives[0].linux_device is None
    assert drives[0].driver == "vfio-pci"
    assert drives[0].storage_backend == "lender"
    assert inventory_mod._drive_availability(None, "vfio-pci") == (True, None)


def _lender_inventory() -> Inventory:
    return Inventory(
        gpus=[GPU(alias="gpu0", smi_index=0, uuid="GPU-1", pci_bus_id="0000:81:00.0")],
        drives=[Drive(
            alias="nvme0", linux_device=None, serial=None, pci_bus_id="0000:3c:00.0",
            driver="vfio-pci", storage_backend="lender", available=True,
        )],
    )


def _lender_config() -> Config:
    return Config(name="lender", domains=[Domain(
        name="d0", gpu_uuid="GPU-1", gpu_pci_bus_id="0000:81:00.0",
        storage_backend="lender",
        drives=[DriveRef(serial="", pci_bus_id="0000:3c:00.0")],
    )])


def test_lender_domain_requires_lender_candidate():
    inv = _lender_inventory()
    assert config_mod.validate_config(_lender_config(), inventory=inv).ok

    bad = _lender_config()
    bad.domains[0].storage_backend = "kernel"
    errors = config_mod.validate_config(bad, inventory=inv).errors
    assert any("lender storage candidate, not kernel" in error for error in errors)


def test_config_cannot_mix_kernel_and_lender_domains():
    config = _lender_config()
    config.domains.append(Domain(
        name="d1", gpu_uuid="GPU-2", gpu_pci_bus_id="0000:82:00.0",
        storage_backend="kernel", drives=[DriveRef(serial="S2", pci_bus_id="0000:4c:00.0")],
    ))

    errors = config_mod.validate_config(config).errors

    assert any("config mixes storage backends" in error for error in errors)


def test_add_domain_derives_backend_and_rejects_another_backend():
    inv = _lender_inventory()
    config = Config(name="lender")

    config_mod.add_domain(config, inv, "d0", "gpu0", ["nvme0"])

    assert config.domains[0].storage_backend == "lender"
    kernel_drive = Drive(
        alias="nvme1", linux_device="/dev/nvme1n1", serial="S2",
        pci_bus_id="0000:4c:00.0", driver="nvme", storage_backend="kernel",
    )
    inv.drives.append(kernel_drive)
    try:
        config_mod.add_domain(config, inv, "d1", "gpu0", ["nvme1"])
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("expected backend conflict")


def test_lender_apply_skips_kernel_binding_and_launches_bdf(runtime_dir, monkeypatch):
    inv, cfg = _lender_inventory(), _lender_config()
    monkeypatch.setattr(lender_backend, "is_unix_socket", lambda _path: True)
    bind_calls = []
    launch_calls = []
    monkeypatch.setattr(binding, "bind_drives", lambda *args, **kwargs: bind_calls.append(args))

    def fake_launch(**kwargs):
        launch_calls.append(kwargs)
        return launcher.LaunchResult(returncode=0, config_line="0 0000:3c:00.0", log_dir="/tmp/d0", pid=7)

    monkeypatch.setattr(launcher, "launch_domain", fake_launch)

    result = apply_mod.apply_config(cfg, inv)

    assert result.ok, result.errors
    assert bind_calls == []
    assert launch_calls[0]["unvme_devices"] == ["0000:3c:00.0"]
    assert launch_calls[0]["storage_backend"] == "lender"
    assert result.registry.domains[0].storage_backend == "lender"


def test_lender_apply_requires_service_socket_before_launch(runtime_dir, monkeypatch):
    inv, cfg = _lender_inventory(), _lender_config()
    cfg.domains[0].server_env["STORAGE_LENDER_SOCKET"] = "/tmp/missing-lender.sock"
    launch_calls = []
    monkeypatch.setattr(launcher, "launch_domain", lambda **kwargs: launch_calls.append(kwargs))
    result = apply_mod.apply_config(cfg, inv)

    assert not result.ok
    assert "Storage Lender endpoint unavailable or not a Unix socket" in result.errors[0]
    assert launch_calls == []


def test_lender_apply_uses_the_service_api_socket_by_default(runtime_dir, monkeypatch):
    inv, cfg = _lender_inventory(), _lender_config()
    seen_paths = []
    monkeypatch.setattr(
        lender_backend, "is_unix_socket", lambda path: seen_paths.append(path) or False
    )

    result = apply_mod.apply_config(cfg, inv)

    assert not result.ok
    assert "/run/storage-lender/api.sock" in seen_paths


def test_apply_rolls_back_when_domain_socket_never_becomes_ready(runtime_dir, monkeypatch):
    inv, cfg = _lender_inventory(), _lender_config()
    monkeypatch.setattr(lender_backend, "is_unix_socket", lambda _path: True)
    stop_calls = []
    monkeypatch.setattr(
        launcher, "launch_domain",
        lambda **kwargs: launcher.LaunchResult(returncode=0, pid=17, log_dir="/tmp/d0"),
    )
    monkeypatch.setattr(launcher, "stop_domain", lambda *args, **kwargs: (
        stop_calls.append(args) or launcher.LaunchResult(returncode=0)
    ))
    monkeypatch.setattr(apply_mod, "_wait_for_domain_socket", lambda *args: False)

    result = apply_mod.apply_config(cfg, inv)

    assert not result.ok
    assert any("did not become ready" in error for error in result.errors)
    assert stop_calls == [("lender", "d0", "lender")]


def test_lender_apply_rejects_regular_file_endpoint_before_launch(runtime_dir, monkeypatch):
    inv, cfg = _lender_inventory(), _lender_config()
    lender_endpoint = f"{runtime_dir}/not-a-socket"
    with open(lender_endpoint, "w", encoding="utf-8"):
        pass
    cfg.domains[0].server_env["STORAGE_LENDER_SOCKET"] = lender_endpoint
    launch_calls = []
    monkeypatch.setattr(launcher, "launch_domain", lambda **kwargs: launch_calls.append(kwargs))

    result = apply_mod.apply_config(cfg, inv)

    assert not result.ok
    assert "not a Unix socket" in result.errors[0]
    assert launch_calls == []


def test_lender_stop_leaves_vfio_binding_untouched(runtime_dir, monkeypatch):
    domain = RegistryDomain(
        name="d0", config="lender", gpu_uuid="GPU-1", gpu_pci_bus_id="0000:81:00.0",
        host_cuda_visible_devices="0", socket="/var/run/scada/d0.sock",
        storage_backend="lender",
        drives=[RegistryDrive(linux_device_before_bind=None, serial="",
                              pci_bus_id="0000:3c:00.0")],
    )
    from scadactl import registry as registry_mod
    registry_mod.write_registry(Registry(
        active_configs=["lender"], configs=[ConfigEntry(name="lender", domains=["d0"])],
        domains=[domain],
    ))
    stop_calls, unbind_calls = [], []
    monkeypatch.setattr(launcher, "stop_domain", lambda *args, **kwargs: (
        stop_calls.append(args) or launcher.LaunchResult(returncode=0)
    ))
    monkeypatch.setattr(binding, "unbind_drives", lambda *args, **kwargs: unbind_calls.append(args))

    result = apply_mod.stop_config("lender")

    assert result.ok, result.errors
    assert stop_calls == [("lender", "d0", "lender")]
    assert unbind_calls == []


def test_lender_doctor_checks_lender_socket_not_unvme(monkeypatch):
    domain = RegistryDomain(
        name="d0", config="lender", gpu_uuid="GPU-1", gpu_pci_bus_id="0000:81:00.0",
        host_cuda_visible_devices="0", socket="/var/run/scada/d0.sock",
        storage_backend="lender",
        server_env={"STORAGE_LENDER_SOCKET": "/tmp/test-lender.sock"},
    )
    monkeypatch.setattr(diagnostics, "_check_nvidia_smi", lambda: diagnostics.Check("nvidia", True))
    monkeypatch.setattr(diagnostics, "_check_runtime_dir", lambda: diagnostics.Check("runtime", True))
    monkeypatch.setattr(diagnostics, "_check_registry_present", lambda: diagnostics.Check("registry", True))
    monkeypatch.setattr(lender_backend, "is_unix_socket", lambda path: path == "/tmp/test-lender.sock")

    report = diagnostics.run_doctor(Registry(domains=[domain]))

    assert [check.name for check in report.host_checks] == [
        "nvidia", "Storage Lender socket", "runtime", "registry",
    ]


def test_lender_doctor_rejects_regular_file_endpoint(tmp_path):
    endpoint = tmp_path / "not-a-socket"
    endpoint.write_text("not a socket")

    check = diagnostics._check_lender_socket(str(endpoint))

    assert not check.ok
    assert "not a Unix socket" in check.detail


def test_lender_socket_predicate_accepts_a_unix_socket_mode(monkeypatch):
    monkeypatch.setattr(
        lender_backend.os, "stat", lambda _path: SimpleNamespace(st_mode=stat.S_IFSOCK)
    )

    assert lender_backend.is_unix_socket("/run/storage-lender/api.sock")


def test_lender_status_shows_bdf():
    domain = RegistryDomain(
        name="d0", config="lender", gpu_uuid="GPU-1", gpu_pci_bus_id="0000:81:00.0",
        host_cuda_visible_devices="0", socket="/var/run/scada/d0.sock",
        storage_backend="lender",
        drives=[RegistryDrive(linux_device_before_bind=None, serial="",
                              pci_bus_id="0000:3c:00.0")],
    )

    status = diagnostics.render_status(Registry(domains=[domain]))

    assert "lender" in status
    assert "0000:3c:00.0" in status
