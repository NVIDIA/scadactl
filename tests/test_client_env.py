# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client env generation and ordinal mapping (no NVML, via injected `visible`)."""

from __future__ import annotations

import pytest

from scadactl import client as client_mod
from scadactl.models import ConfigEntry, Registry, RegistryDomain


def _reg():
    return Registry(
        active_configs=["p"],
        socket_dir="/var/run/scada",
        domains=[
            RegistryDomain(
                name="gpu0_domain", config="p", gpu_alias="gpu0",
                gpu_uuid="GPU-1111", gpu_pci_bus_id="0000:81:00.0",
                host_cuda_visible_devices="0",
                socket="/var/run/scada/scadad.swap.gpu0.sock",
            ),
            RegistryDomain(
                name="gpu2_domain", config="p", gpu_alias="gpu2",
                gpu_uuid="GPU-3333", gpu_pci_bus_id="0000:c1:00.0",
                host_cuda_visible_devices="2",
                socket="/var/run/scada/scadad.swap.gpu2.sock",
            ),
        ],
    )


def _visible():
    # Container exposes GPU-3333 first (ordinal 0), GPU-1111 second (ordinal 1).
    return [
        {"ordinal": "0", "uuid": "GPU-3333", "pci_bus_id": "0000:c1:00.0"},
        {"ordinal": "1", "uuid": "GPU-1111", "pci_bus_id": "0000:81:00.0"},
    ]


def test_map_domain_to_ordinal_by_uuid():
    reg = _reg()
    assert client_mod.map_domain_to_ordinal(reg.domains[0], _visible()) == 1
    assert client_mod.map_domain_to_ordinal(reg.domains[1], _visible()) == 0


def test_map_domain_to_ordinal_by_pci_fallback():
    d = RegistryDomain(name="x", config="p", gpu_uuid="", gpu_pci_bus_id="0000:81:00.0",
                       host_cuda_visible_devices="", socket="")
    assert client_mod.map_domain_to_ordinal(d, _visible()) == 1


def test_map_domain_not_visible():
    d = RegistryDomain(name="x", config="p", gpu_uuid="GPU-9999", gpu_pci_bus_id="0000:ff:00.0",
                       host_cuda_visible_devices="", socket="")
    assert client_mod.map_domain_to_ordinal(d, _visible()) is None


def test_compute_env_single_domain():
    reg = _reg()
    domains = client_mod.resolve_selection(reg, "gpu0_domain")
    env, sockets = client_mod.compute_env(reg, domains, visible=_visible())
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["SCADA_DOMAIN"] == "gpu0_domain"
    assert env["SCADA_SOCKET"].endswith("gpu0.sock")


def test_compute_env_preserves_order():
    reg = _reg()
    # request gpu2 then gpu0 -> ordinals 0,1 in that requested order
    domains = client_mod.resolve_selection(reg, "gpu2_domain,gpu0_domain")
    env, sockets = client_mod.compute_env(reg, domains, visible=_visible())
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert env["SCADA_DOMAIN"] == "gpu2_domain,gpu0_domain"
    assert sockets[0].endswith("gpu2.sock")


def test_env_json_domain_count():
    reg = _reg()
    domains = client_mod.resolve_selection(reg, "gpu0_domain,gpu2_domain")
    payload = client_mod.env_json(reg, domains, visible=_visible())
    assert payload["domain_count"] == 2
    assert payload["cuda_visible_devices"] == "1,0"
    assert len(payload["scada_sockets"]) == 2


def test_compute_env_raises_when_not_visible():
    reg = _reg()
    reg.domains.append(
        RegistryDomain(name="gpu3_domain", config="p", gpu_uuid="GPU-4444",
                       gpu_pci_bus_id="0000:c2:00.0", host_cuda_visible_devices="3",
                       socket="/s")
    )
    domains = client_mod.resolve_selection(reg, "gpu3_domain")
    with pytest.raises(client_mod.GpuNotVisible):
        client_mod.compute_env(reg, domains, visible=_visible())


def test_resolve_selection_unknown():
    with pytest.raises(KeyError):
        client_mod.resolve_selection(_reg(), "nope")


def test_resolve_selection_builtin_all():
    # No config groups defined -> 'all' synthesizes every registry domain.
    reg = _reg()
    names = [d.name for d in client_mod.resolve_selection(reg, "all")]
    assert names == ["gpu0_domain", "gpu2_domain"]


def test_resolve_selection_named_group_and_dedup():
    reg = _reg()
    reg.configs = [ConfigEntry(
        name="p",
        domains=["gpu0_domain", "gpu2_domain"],
        domain_groups={"all": ["gpu0_domain", "gpu2_domain"],
                       "training": ["gpu2_domain"]},
    )]
    # named group expands to its members
    assert [d.name for d in client_mod.resolve_selection(reg, "training")] == ["gpu2_domain"]
    # group + overlapping domain dedups, preserving first-seen order
    assert [d.name for d in client_mod.resolve_selection(reg, "all,gpu0_domain")] == \
        ["gpu0_domain", "gpu2_domain"]


def test_resolve_selection_unknown_group_member_raises():
    reg = _reg()
    reg.configs = [ConfigEntry(name="p", domains=["gpu0_domain"],
                                 domain_groups={"bad": ["ghost_domain"]})]
    with pytest.raises(KeyError):
        client_mod.resolve_selection(reg, "bad")


def test_render_env_exports():
    out = client_mod.render_env_exports({"A": "1", "B": "2"})
    assert "export A=1" in out and "export B=2" in out


def test_list_domains_status():
    reg = _reg()
    views = client_mod.list_domains(reg, visible=_visible())
    by_name = {v.name: v for v in views}
    assert by_name["gpu0_domain"].container_cuda == 1
    assert by_name["gpu0_domain"].status == "ok"


def test_check_client_failure_suggestion():
    reg = _reg()
    reg.domains.append(
        RegistryDomain(name="gpu3_domain", config="p", gpu_uuid="GPU-4444",
                       gpu_pci_bus_id="0000:c2:00.0", host_cuda_visible_devices="3",
                       socket="/s")
    )
    ok, msg = client_mod.check_client(reg, "gpu3_domain", visible=_visible())
    assert not ok
    assert "gpu-not-visible" not in msg  # it's prose, but must include the fix
    assert "device=GPU-4444" in msg
    assert "docker run" in msg


def test_check_client_ok():
    reg = _reg()
    ok, msg = client_mod.check_client(reg, "gpu0_domain", visible=_visible())
    assert ok
    assert "Container CUDA ordinal: 1" in msg


def test_check_client_hint_uses_active_paths(monkeypatch):
    # The docker-run hint must reflect overridden runtime/MPS paths, not literals.
    monkeypatch.setenv("SCADA_RUNTIME_DIR", "/run/custom-scada")
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", "/run/mps-pipe")
    monkeypatch.setenv("CUDA_MPS_LOG_DIRECTORY", "/run/mps-log")
    reg = Registry(domains=[RegistryDomain(
        name="d", config="c", gpu_uuid="GPU-9", gpu_pci_bus_id="0000:81:00.0",
        host_cuda_visible_devices="0", socket="/run/custom-scada/d.sock")])
    ok, msg = client_mod.check_client(reg, "d", visible=[])  # not visible
    assert not ok
    assert "-v /run/custom-scada:/run/custom-scada" in msg
    assert "/run/custom-scada/registry.json:/run/custom-scada/registry.json:ro" in msg
    assert "-v /run/mps-pipe:/run/mps-pipe" in msg
    assert "-v /run/mps-log:/run/mps-log" in msg


def test_check_client_hint_socket_mount_follows_socket_path(monkeypatch):
    # The socket bind-mount is derived from the domain's own stored socket path
    # (baked in at apply time), so it stays correct even when the socket lives
    # outside the current SCADA_RUNTIME_DIR. The registry mount still tracks
    # runtime_dir(), where the registry file actually lives.
    monkeypatch.setenv("SCADA_RUNTIME_DIR", "/run/custom-scada")
    reg = Registry(domains=[RegistryDomain(
        name="d", config="c", gpu_uuid="GPU-9", gpu_pci_bus_id="0000:81:00.0",
        host_cuda_visible_devices="0", socket="/run/sockets-elsewhere/d.sock")])
    ok, msg = client_mod.check_client(reg, "d", visible=[])  # not visible
    assert not ok
    assert "-v /run/sockets-elsewhere:/run/sockets-elsewhere" in msg
    assert "/run/custom-scada/registry.json:/run/custom-scada/registry.json:ro" in msg
