# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Doctor domain checks: unvme verification via driver-published sysfs identity."""

from __future__ import annotations

from scadactl import diagnostics
from scadactl.models import Registry, RegistryDomain, RegistryDrive


def _domain():
    return RegistryDomain(
        name="gpu0_domain", config="scada", gpu_uuid="GPU-1", gpu_pci_bus_id="0008:01:00.0",
        host_cuda_visible_devices="0", socket="/nonexistent.sock", pid=None,
        drives=[
            RegistryDrive(linux_device_before_bind="/dev/nvme0n1",
                          serial="S111", pci_bus_id="0006:06:00.0", unvme_device="/dev/unvme1"),
            RegistryDrive(linux_device_before_bind="/dev/nvme1n1",
                          serial="S222", pci_bus_id="0006:07:00.0", unvme_device="/dev/unvme2"),
        ],
    )


def test_detailed_status_shows_configured_socket_group_and_keeps_columns_aligned():
    domain = _domain()
    domain.socket_group = "scada-users"
    domain.gpu_uuid = "GPU-" + "a" * 36
    domain.socket = "/var/run/scada/a-long-domain-name/scadad.swap.gpu0.sock"
    lines = diagnostics.render_status(Registry(domains=[domain]), details=True).splitlines()
    header = next(line for line in lines if line.startswith("domain"))
    row = lines[lines.index(header) + 1]
    assert row[header.index("profiler") :].startswith(domain.profiler_endpoint_mode)
    assert "    socket_group=scada-users" in lines
    domain.socket_group = None
    assert "    socket_group=(root only)" in diagnostics.render_status(Registry(domains=[domain]), details=True)


def _unvme_check(checks, dev):
    return next(c for c in checks if c.name.endswith(f"unvme {dev}"))


def test_doctor_unvme_verified_by_serial(monkeypatch):
    # sysfs reports matching serials -> verified, no dependence on nvme inventory.
    sysfs = {"unvme1": {"serial": "S111", "pci_bdf": "0006:06:00.0"},
             "unvme2": {"serial": "S222", "pci_bdf": "0006:07:00.0"}}
    monkeypatch.setattr(diagnostics.identity_mod, "read_unvme_sysfs_identity",
                        lambda name: sysfs.get(name))
    checks = diagnostics._check_domain(_domain())
    assert _unvme_check(checks, "/dev/unvme1").ok
    assert _unvme_check(checks, "/dev/unvme2").ok


def test_doctor_unvme_serial_mismatch(monkeypatch):
    sysfs = {"unvme1": {"serial": "WRONG"}, "unvme2": {"serial": "S222"}}
    monkeypatch.setattr(diagnostics.identity_mod, "read_unvme_sysfs_identity",
                        lambda name: sysfs.get(name))
    checks = diagnostics._check_domain(_domain())
    assert not _unvme_check(checks, "/dev/unvme1").ok
    assert "mismatch" in _unvme_check(checks, "/dev/unvme1").detail
    assert _unvme_check(checks, "/dev/unvme2").ok


def test_doctor_unvme_pci_bdf_fallback(monkeypatch):
    # No serial on device -> verify by pci_bdf.
    sysfs = {"unvme1": {"pci_bdf": "0006:06:00.0"}, "unvme2": {"pci_bdf": "0006:07:00.0"}}
    monkeypatch.setattr(diagnostics.identity_mod, "read_unvme_sysfs_identity",
                        lambda name: sysfs.get(name))
    d = _domain()
    for drv in d.drives:
        drv.serial = None  # force the BDF path
    checks = diagnostics._check_domain(d)
    assert _unvme_check(checks, "/dev/unvme1").ok


def test_doctor_unvme_no_sysfs_identity(monkeypatch):
    monkeypatch.setattr(diagnostics.identity_mod, "read_unvme_sysfs_identity",
                        lambda name: None)
    checks = diagnostics._check_domain(_domain())
    assert not _unvme_check(checks, "/dev/unvme1").ok


def test_collect_logs_newest_first(monkeypatch, tmp_path):
    import os
    from scadactl.models import Registry, RegistryDomain

    d = RegistryDomain(name="gpu0_domain", config="scada", gpu_uuid="GPU-1",
                       gpu_pci_bus_id="0", host_cuda_visible_devices="0", socket="s")
    reg = Registry(active_configs=["scada"], domains=[d])

    log_dir = tmp_path / "scada" / "gpu0_domain"
    log_dir.mkdir(parents=True)
    old = log_dir / "scada-swap-domain.gpu0.20260101-000000.log"
    new = log_dir / "scada-swap-domain.gpu0.20260627-120000.log"
    old.write_text("old"); new.write_text("new")
    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))

    monkeypatch.setattr(diagnostics.launcher, "domain_log_dir",
                        lambda config, domain: str(log_dir))
    logs = diagnostics.collect_logs(reg)
    assert logs["gpu0_domain"][0].endswith("20260627-120000.log")  # newest first
    assert logs["gpu0_domain"][1].endswith("20260101-000000.log")
