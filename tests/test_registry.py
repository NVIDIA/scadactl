# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry: atomic write, locking, reconcile semantics (tmp runtime dir)."""

from __future__ import annotations

import json
import os
import threading

from scadactl import registry as reg_mod
from scadactl.models import (
    Registry,
    RegistryDomain,
    RegistryDrive,
)


def _domain(name, config, gpu="GPU-1111", serial="S111"):
    return RegistryDomain(
        name=name,
        config=config,
        gpu_uuid=gpu,
        gpu_pci_bus_id="0000:81:00.0",
        host_cuda_visible_devices="0",
        socket=f"/var/run/scada/{name}.sock",
        drives=[RegistryDrive(linux_device_before_bind="/dev/nvme2n1", serial=serial)],
    )


def test_read_absent_returns_empty(runtime_dir):
    reg = reg_mod.read_registry()
    assert reg.active_configs == []
    assert reg.domains == []


def test_atomic_write_and_read_roundtrip(runtime_dir):
    reg = Registry(active_configs=["p"], domains=[_domain("d1", "p")])
    reg_mod.write_registry(reg)
    path = reg_mod.registry_path()
    assert os.path.exists(path)
    # 0644
    assert (os.stat(path).st_mode & 0o777) == 0o644
    # valid complete JSON
    with open(path) as f:
        data = json.load(f)
    assert data["schema_version"] == 2
    back = reg_mod.read_registry()
    assert back.active_configs == ["p"]
    assert back.domains[0].name == "d1"


def test_no_tmp_left_behind(runtime_dir):
    reg_mod.write_registry(Registry(active_configs=["p"], domains=[_domain("d1", "p")]))
    leftovers = [f for f in os.listdir(runtime_dir) if f.endswith(".tmp")]
    assert leftovers == []


def test_reconcile_add_config(runtime_dir):
    base = Registry()
    r1 = reg_mod.reconcile_add_config(base, "p1", [_domain("d1", "p1")])
    assert r1.active_configs == ["p1"]
    assert [d.name for d in r1.domains] == ["d1"]
    assert r1.configs[0].domains == ["d1"]


def test_reconcile_add_second_config_nonconflicting(runtime_dir):
    base = reg_mod.reconcile_add_config(Registry(), "p1", [_domain("d1", "p1")])
    r2 = reg_mod.reconcile_add_config(base, "p2", [_domain("d2", "p2", gpu="GPU-2222")])
    assert set(r2.active_configs) == {"p1", "p2"}
    assert {d.name for d in r2.domains} == {"d1", "d2"}


def test_reapply_replaces_own_entries(runtime_dir):
    base = reg_mod.reconcile_add_config(Registry(), "p1", [_domain("d1", "p1"), _domain("d2", "p1", gpu="GPU-2222")])
    # reapply p1 with only d1
    r = reg_mod.reconcile_add_config(base, "p1", [_domain("d1", "p1")])
    assert [d.name for d in r.domains] == ["d1"]
    assert r.active_configs == ["p1"]


def test_reconcile_name_collision_across_configs(runtime_dir):
    base = reg_mod.reconcile_add_config(Registry(), "p1", [_domain("shared", "p1")])
    try:
        reg_mod.reconcile_add_config(base, "p2", [_domain("shared", "p2")])
        assert False, "expected ValueError on cross-config name collision"
    except ValueError as e:
        assert "already owned by another config" in str(e)


def test_reconcile_remove_config(runtime_dir):
    base = reg_mod.reconcile_add_config(Registry(), "p1", [_domain("d1", "p1")])
    base = reg_mod.reconcile_add_config(base, "p2", [_domain("d2", "p2", gpu="GPU-2222")])
    r = reg_mod.reconcile_remove_config(base, "p1")
    assert r.active_configs == ["p2"]
    assert [d.name for d in r.domains] == ["d2"]


def test_reconcile_remove_all(runtime_dir):
    base = reg_mod.reconcile_add_config(Registry(), "p1", [_domain("d1", "p1")])
    r = reg_mod.reconcile_remove_all(base)
    assert r.active_configs == []
    assert r.domains == []


def test_reconcile_remove_domains_partial(runtime_dir):
    base = reg_mod.reconcile_add_config(
        Registry(),
        "a",
        [_domain("d1", "a"), _domain("d2", "a")],
        domain_groups={"g1": ["d1", "d2"]},
    )
    base = reg_mod.reconcile_add_config(
        base, "b", [_domain("d3", "b", gpu="GPU-2222")]
    )

    r = reg_mod.reconcile_remove_domains(base, [("a", "d1")])

    assert {d.name for d in r.domains} == {"d2", "d3"}
    a_entry = next(c for c in r.configs if c.name == "a")
    assert a_entry.domains == ["d2"]
    assert a_entry.domain_groups == {"g1": ["d2"]}
    b_entry = next(c for c in r.configs if c.name == "b")
    assert b_entry.domains == ["d3"]
    assert set(r.active_configs) == {"a", "b"}


def test_reconcile_remove_domains_preserves_mps_flag(runtime_dir):
    # A partial stop rebuilds each surviving ConfigEntry; it must carry the
    # original mps_enabled forward, else a later `stop --all` sees mps=False on
    # every config and skips stopping the host-global daemon.
    base = reg_mod.reconcile_add_config(
        Registry(),
        "a",
        [_domain("d1", "a"), _domain("d2", "a")],
        mps_enabled=True,
    )

    r = reg_mod.reconcile_remove_domains(base, [("a", "d1")])

    a_entry = next(c for c in r.configs if c.name == "a")
    assert a_entry.domains == ["d2"]
    assert a_entry.mps_enabled is True


def test_reconcile_remove_domains_drops_emptied_config(runtime_dir):
    base = reg_mod.reconcile_add_config(
        Registry(), "a", [_domain("d1", "a"), _domain("d2", "a")]
    )
    base = reg_mod.reconcile_add_config(
        base, "b", [_domain("d3", "b", gpu="GPU-2222")]
    )

    r = reg_mod.reconcile_remove_domains(base, [("a", "d1"), ("a", "d2")])

    assert [c.name for c in r.configs] == ["b"]
    assert r.active_configs == ["b"]
    assert [d.name for d in r.domains] == ["d3"]


def test_concurrent_writers_serialize_and_readers_see_complete(runtime_dir):
    """Hammer the writer lock; the reader must always see complete JSON."""
    errors = []

    def writer(i):
        try:
            for _ in range(20):
                reg = Registry(
                    active_configs=[f"p{i}"],
                    domains=[_domain(f"d{i}", f"p{i}")],
                )
                reg_mod.write_registry(reg)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def reader():
        try:
            for _ in range(50):
                r = reg_mod.read_registry()
                # Must always parse to a complete Registry (no partial JSON).
                assert isinstance(r.active_configs, list)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
    threads += [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    # Final file is valid JSON.
    with open(reg_mod.registry_path()) as f:
        json.load(f)


def test_host_lock_path_is_distinct_from_registry_lock(runtime_dir):
    # Must be a separate file, else an apply holding host_lock and then taking
    # the writer lock to publish would deadlock.
    assert reg_mod.host_lock_path() != reg_mod.lock_path()
    assert reg_mod.host_lock_path().endswith("/host.lock")


def test_host_lock_is_exclusive(runtime_dir):
    """A second host_lock() acquisition blocks until the first releases."""
    acquired = threading.Event()
    release = threading.Event()

    def worker():
        with reg_mod.host_lock():
            acquired.set()
            release.wait(2)

    with reg_mod.host_lock():
        t = threading.Thread(target=worker)
        t.start()
        # While we hold it, the worker cannot acquire the lock.
        assert not acquired.wait(0.3)
    # We released -> the worker acquires promptly.
    assert acquired.wait(2)
    release.set()
    t.join(2)
    assert not t.is_alive()


def test_host_lock_nests_with_writer_lock_without_deadlock(runtime_dir):
    # An apply holds host_lock and still publishes via the writer lock. Distinct
    # lock files must let this nest; a hang here means they collapsed onto one.
    with reg_mod.host_lock():
        reg_mod.write_registry(Registry(active_configs=["p"]))
        reg_mod.update_registry(lambda cur: cur)
    assert reg_mod.read_registry().active_configs == ["p"]


def test_socket_for_domain(runtime_dir):
    s = reg_mod.socket_for_domain("gpu0")
    assert s.endswith("scadad.swap.gpu0.sock")
    assert s.startswith(runtime_dir)


def test_update_registry_merges_against_latest_on_disk(runtime_dir):
    # A writer that reconciled against a stale snapshot must not clobber a
    # concurrently-committed config: update_registry re-reads the latest state
    # inside the lock and merges, so both configs survive.
    from scadactl.models import Registry, RegistryDomain

    def _dom(name, cfg):
        return RegistryDomain(
            name=name, config=cfg, gpu_uuid=f"GPU-{name}",
            gpu_pci_bus_id="0000:81:00.0", host_cuda_visible_devices="0",
            socket=f"/run/{name}",
        )

    # Writer 1 commits config "A".
    reg_mod.write_registry(
        reg_mod.reconcile_add_config(Registry.empty(), "A", [_dom("da", "A")])
    )
    # Writer 2's mutator receives the freshest on-disk state (which has A), not
    # an empty snapshot it might have read earlier -- so it adds B onto {A}.
    result = reg_mod.update_registry(
        lambda cur: reg_mod.reconcile_add_config(cur, "B", [_dom("db", "B")])
    )
    got = {(d.config, d.name) for d in result.domains}
    assert got == {("A", "da"), ("B", "db")}
    disk = {(d.config, d.name) for d in reg_mod.read_registry().domains}
    assert disk == {("A", "da"), ("B", "db")}


def test_config_entry_mps_flag_roundtrips(runtime_dir):
    # The MPS request is recorded per config so stop_all knows whether to quit
    # the host-global daemon; it must survive a write/read roundtrip.
    reg = reg_mod.reconcile_add_config(
        Registry(), "p", [_domain("d1", "p")], mps_enabled=True
    )
    reg_mod.write_registry(reg)
    entry = next(c for c in reg_mod.read_registry().configs if c.name == "p")
    assert entry.mps_enabled is True
