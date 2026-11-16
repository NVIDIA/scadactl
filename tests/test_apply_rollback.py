# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rollback of apply_config's post-mutation side effects on failure.

Covers the bind / MPS / launch failure paths added to ``apply_config``: a
clean rollback should leave the host (and registry) untouched, and an
incomplete rollback should persist a residual registry record rather than
orphan the domains it could not stop.
"""

from __future__ import annotations

from scadactl import apply as apply_mod
from scadactl import reconcile as reconcile_mod
from scadactl import registry as registry_mod
from scadactl.backend import binding, launcher, mps
from scadactl.backend.binding import BindResult
from scadactl.backend.launcher import LaunchResult
from scadactl.backend.mps import MpsResult
from scadactl.models import (
    Config, Domain, Drive, DriveRef, GPU, Inventory, Registry,
    RegistryDomain, RegistryDrive,
)


def _inventory(driver: str = "nvme") -> Inventory:
    """A 2-GPU / 2-drive synthetic inventory (smaller than the shared fixture)."""
    gpus = [
        GPU(alias="gpu0", smi_index=0, uuid="GPU-1111", pci_bus_id="0000:81:00.0",
            name="NVIDIA H100 80GB"),
        GPU(alias="gpu1", smi_index=1, uuid="GPU-2222", pci_bus_id="0000:82:00.0",
            name="NVIDIA H100 80GB"),
    ]
    drives = [
        Drive(alias="nvme0", linux_device="/dev/nvme2n1", serial="S111", model="PM1735",
              lba_size=4096, namespace=1, pci_bus_id="0000:83:00.0", pcie="gen4 x4",
              driver=driver),
        Drive(alias="nvme1", linux_device="/dev/nvme3n1", serial="S222", model="PM1735",
              lba_size=4096, namespace=1, pci_bus_id="0000:84:00.0", pcie="gen4 x4",
              driver=driver),
    ]
    return Inventory(gpus=gpus, drives=drives)


def _config(mps_enabled: bool = False) -> Config:
    """A 2-domain config (1 drive/domain) matching ``_inventory()``'s stable ids."""
    return Config(
        name="rb-cfg",
        mps_enabled=mps_enabled,
        domains=[
            Domain(name="d0", gpu_uuid="GPU-1111", gpu_pci_bus_id="0000:81:00.0",
                   drives=[DriveRef(serial="S111", pci_bus_id="0000:83:00.0")]),
            Domain(name="d1", gpu_uuid="GPU-2222", gpu_pci_bus_id="0000:82:00.0",
                   drives=[DriveRef(serial="S222", pci_bus_id="0000:84:00.0")]),
        ],
    )


def _fake_launch_all_ok(
    config, domain, host_cuda_index, unvme_devices, resolved_server_env,
    socket_group=None, profiler_socket="", dry_run=False
):
    """Both domains launch OK (used to exercise the publish-failure path)."""
    return LaunchResult(
        returncode=0, config_line=f"0 {unvme_devices}", log_dir=f"/tmp/{domain}", pid=222
    )


def _recording_mps_stop(calls, ok=True):
    def _fake(dry_run=False):
        calls.append(dry_run)
        return MpsResult(returncode=0 if ok else 1, stderr="" if ok else "mps stuck")
    return _fake


def _stub_mps(monkeypatch, was_running: bool, stop_calls: list, stop_ok: bool = True):
    """Stub the MPS backend: control whether it was already running, record stops."""
    monkeypatch.setattr(mps, "compatibility_error", lambda multiuser=False: None)
    monkeypatch.setattr(mps, "is_running", lambda: was_running)
    monkeypatch.setattr(mps, "start", lambda dry_run=False, multiuser=False: MpsResult(returncode=0))
    monkeypatch.setattr(mps, "stop", _recording_mps_stop(stop_calls, ok=stop_ok))


# --------------------------------------------------------------------------- #
# Backend stubs
# --------------------------------------------------------------------------- #
def _fake_bind_ok(pci_addresses, dry_run=False, non_interactive=True):
    return BindResult(
        returncode=0,
        bound_pci=list(pci_addresses),
        unvme_by_pci={p: f"/dev/unvme{i}" for i, p in enumerate(pci_addresses)},
    )


def _fake_bind_fail(bound_pci):
    """A bind that fails after having managed to bind ``bound_pci``."""
    def _fake(pci_addresses, dry_run=False, non_interactive=True):
        return BindResult(returncode=1, bound_pci=list(bound_pci), stderr="bind boom")
    return _fake


def _fake_bind_partial(pci_addresses, dry_run=False, non_interactive=True):
    """Exit 0 but only the FIRST drive actually joined unvme (the load script's
    `|| true` swallows the rest), so bound_pci is a strict subset of the request."""
    first = list(pci_addresses)[:1]
    return BindResult(
        returncode=0,
        bound_pci=first,
        unvme_by_pci={p: f"/dev/unvme{i}" for i, p in enumerate(first)},
    )


def _fake_bind_none(pci_addresses, dry_run=False, non_interactive=True):
    """Exit 0 but NOTHING joined unvme (all attaches failed; the load script
    restored them all to nvme). Empty bound set must still fail the completeness
    check rather than proceed to a doomed launch."""
    return BindResult(returncode=0, bound_pci=[], unvme_by_pci={})


def _fake_launch_first_ok_second_fail(
    config, domain, host_cuda_index, unvme_devices, resolved_server_env,
    socket_group=None, profiler_socket="", dry_run=False
):
    """Domain d0 launches OK; d1 fails."""
    if domain == "d0":
        return LaunchResult(
            returncode=0, config_line="0 /dev/unvme0", log_dir="/tmp/d0", pid=111
        )
    return LaunchResult(returncode=1, stderr="boom")


def _recording_unbind(calls, ok=True):
    def _fake(pci_addresses, dry_run=False):
        calls.append(list(pci_addresses))
        if ok:
            return BindResult(returncode=0)
        return BindResult(returncode=1, stderr="unbind boom")
    return _fake


def _recording_stop(calls, ok=True):
    def _fake(config, domain, dry_run=False):
        calls.append((config, domain))
        if ok:
            return LaunchResult(returncode=0)
        return LaunchResult(returncode=1, stderr="stuck")
    return _fake


def _recording_launch(calls):
    def _fake(*args, **kwargs):
        calls.append((args, kwargs))
        return LaunchResult(returncode=0)
    return _fake


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_launch_failure_rolls_back_clean(runtime_dir, monkeypatch):
    inv = _inventory(driver="nvme")
    cfg = _config()

    monkeypatch.setattr(binding, "bind_drives", _fake_bind_ok)
    unbind_calls: list = []
    monkeypatch.setattr(binding, "unbind_drives", _recording_unbind(unbind_calls))
    stop_calls: list = []
    monkeypatch.setattr(launcher, "stop_domain", _recording_stop(stop_calls, ok=True))
    monkeypatch.setattr(launcher, "launch_domain", _fake_launch_first_ok_second_fail)

    res = apply_mod.apply_config(cfg, inv, dry_run=False)

    assert not res.ok
    assert any("launch failed" in e for e in res.errors)
    assert any("rolled back all changes" in a for a in res.actions)
    assert stop_calls == [(cfg.name, "d0")]
    assert unbind_calls == [["0000:83:00.0", "0000:84:00.0"]]

    reg = registry_mod.read_registry()
    assert not [d for d in reg.domains if d.config == cfg.name]


def test_launch_failure_persists_residual_when_rollback_incomplete(runtime_dir, monkeypatch):
    inv = _inventory(driver="nvme")
    cfg = _config()

    monkeypatch.setattr(binding, "bind_drives", _fake_bind_ok)
    unbind_calls: list = []
    monkeypatch.setattr(binding, "unbind_drives", _recording_unbind(unbind_calls))
    stop_calls: list = []
    monkeypatch.setattr(launcher, "stop_domain", _recording_stop(stop_calls, ok=False))
    monkeypatch.setattr(launcher, "launch_domain", _fake_launch_first_ok_second_fail)

    res = apply_mod.apply_config(cfg, inv, dry_run=False)

    assert not res.ok
    assert any("rollback: could not stop domain" in e for e in res.errors)
    assert any("scadactl stop --config" in a for a in res.actions)

    # Rollback must HALT once a domain cannot be stopped: that server is still
    # running and holding its /dev/unvmeN, so its drives must NOT be released.
    assert any("rollback halted" in e for e in res.errors)
    assert unbind_calls == []

    reg = registry_mod.read_registry()
    owned = [d for d in reg.domains if d.config == cfg.name]
    assert [d.name for d in owned] == ["d0"]


def test_conflict_with_other_active_config_does_not_mutate(runtime_dir, monkeypatch):
    # A drive owned by a *different* active config is never seized: apply errors
    # up front and touches nothing (no bind/unbind/launch/stop).
    inv = _inventory(driver="unvme")
    cfg = _config()
    other = registry_mod.reconcile_add_config(
        Registry(), "other",
        [RegistryDomain(name="o0", config="other", gpu_uuid="GPU-1111",
                        gpu_pci_bus_id="0000:81:00.0", host_cuda_visible_devices="0",
                        socket="/s", drives=[RegistryDrive(
                            linux_device_before_bind=None, serial="S111",
                            pci_bus_id="0000:83:00.0")])],
    )
    registry_mod.write_registry(other)

    bind_calls: list = []
    monkeypatch.setattr(binding, "bind_drives",
                        lambda p, dry_run=False: bind_calls.append(list(p)) or _fake_bind_ok(p))
    unbind_calls: list = []
    monkeypatch.setattr(binding, "unbind_drives", _recording_unbind(unbind_calls))
    launch_calls: list = []
    monkeypatch.setattr(launcher, "launch_domain", _recording_launch(launch_calls))

    res = apply_mod.apply_config(cfg, inv, dry_run=False)

    assert not res.ok
    assert bind_calls == [] and unbind_calls == [] and launch_calls == []


def test_unmanaged_orphan_drives_are_reclaimed_and_rebuilt(runtime_dir, monkeypatch):
    # Drives bound to unvme with no registry owner are stale: freed, then rebound
    # and rebuilt. Pin the usage probe to "confirmed free" so reclaim proceeds
    # regardless of whether fuser is installed on the test host (the default now
    # fails closed -- refuses reclaim -- when usage cannot be established).
    inv = _inventory(driver="unvme")
    cfg = _config()

    monkeypatch.setattr(reconcile_mod, "_device_in_use", lambda dev: False)
    monkeypatch.setattr(binding, "bind_drives", _fake_bind_ok)
    unbind_calls: list = []
    monkeypatch.setattr(binding, "unbind_drives", _recording_unbind(unbind_calls))
    monkeypatch.setattr(launcher, "launch_domain", _fake_launch_all_ok)

    res = apply_mod.apply_config(cfg, inv, dry_run=False)

    assert res.ok, res.errors
    assert unbind_calls == [["0000:83:00.0", "0000:84:00.0"]]   # the reclaim
    assert any("reclaim" in a.lower() for a in res.actions)
    assert {d.name for d in res.registry.domains if d.config == cfg.name} == {"d0", "d1"}


def test_unchanged_reapply_is_a_noop(runtime_dir, monkeypatch):
    # Config active, matching the registry, all domains alive -> nothing happens.
    inv = _inventory(driver="unvme")
    cfg = _config()
    desired = apply_mod.build_registry_domains(cfg, inv)
    registry_mod.write_registry(
        registry_mod.reconcile_add_config(Registry(), cfg.name, desired))
    monkeypatch.setattr(reconcile_mod, "_domain_alive", lambda rd: True)

    bind_calls: list = []
    monkeypatch.setattr(binding, "bind_drives",
                        lambda p, dry_run=False: bind_calls.append(list(p)) or _fake_bind_ok(p))
    launch_calls: list = []
    monkeypatch.setattr(launcher, "launch_domain", _recording_launch(launch_calls))

    res = apply_mod.apply_config(cfg, inv, dry_run=False)

    assert res.ok
    assert bind_calls == [] and launch_calls == []
    assert any("nothing to do" in a for a in res.actions)


def test_dead_managed_domain_is_self_healed(runtime_dir, monkeypatch):
    # Config active + matches the registry, but one domain's process died: only
    # that domain is relaunched; the healthy one is left alone.
    inv = _inventory(driver="unvme")
    cfg = _config()
    desired = apply_mod.build_registry_domains(cfg, inv)
    registry_mod.write_registry(
        registry_mod.reconcile_add_config(Registry(), cfg.name, desired))
    monkeypatch.setattr(reconcile_mod, "_domain_alive", lambda rd: rd.name == "d1")

    monkeypatch.setattr(binding, "bind_drives", _fake_bind_ok)
    monkeypatch.setattr(binding, "unbind_drives", _recording_unbind([]))
    launch_calls: list = []
    monkeypatch.setattr(launcher, "launch_domain", _recording_launch(launch_calls))

    res = apply_mod.apply_config(cfg, inv, dry_run=False)

    assert res.ok, res.errors
    # Only d0 (dead) is relaunched.
    assert [kw["domain"] for _a, kw in launch_calls] == ["d0"]
    assert {d.name for d in res.registry.domains if d.config == cfg.name} == {"d0", "d1"}


def test_orphan_rollback_when_relaunch_fails(runtime_dir, monkeypatch):
    inv = _inventory(driver="unvme")  # both drives orphan-bound before this apply
    cfg = _config()

    monkeypatch.setattr(reconcile_mod, "_device_in_use", lambda dev: False)
    monkeypatch.setattr(binding, "bind_drives", _fake_bind_ok)
    unbind_calls: list = []
    monkeypatch.setattr(binding, "unbind_drives", _recording_unbind(unbind_calls))
    stop_calls: list = []
    monkeypatch.setattr(launcher, "stop_domain", _recording_stop(stop_calls, ok=True))
    monkeypatch.setattr(launcher, "launch_domain", _fake_launch_first_ok_second_fail)

    res = apply_mod.apply_config(cfg, inv, dry_run=False)

    assert not res.ok
    assert stop_calls == [(cfg.name, "d0")]
    # unbind called twice: once to reclaim the orphans, once to roll back our bind.
    assert unbind_calls == [
        ["0000:83:00.0", "0000:84:00.0"],   # reclaim
        ["0000:83:00.0", "0000:84:00.0"],   # rollback of the rebind
    ]


def test_partial_bind_is_rejected_before_launch(runtime_dir, monkeypatch):
    # bind exits 0 but only 1 of 2 requested drives joined unvme: apply must
    # refuse (not launch a silently under-provisioned domain) and roll back the
    # one drive that did bind.
    inv = _inventory(driver="nvme")
    cfg = _config()

    monkeypatch.setattr(binding, "bind_drives", _fake_bind_partial)
    unbind_calls: list = []
    monkeypatch.setattr(binding, "unbind_drives", _recording_unbind(unbind_calls))
    monkeypatch.setattr(launcher, "stop_domain", _recording_stop([], ok=True))
    launch_calls: list = []
    monkeypatch.setattr(launcher, "launch_domain", _recording_launch(launch_calls))

    res = apply_mod.apply_config(cfg, inv, dry_run=False)

    assert not res.ok
    assert any("binding incomplete" in e and "0000:84:00.0" in e for e in res.errors)
    assert launch_calls == []                       # never launched a partial domain
    assert unbind_calls == [["0000:83:00.0"]]       # rolled back the one that bound


def test_all_binds_failed_is_rejected(runtime_dir, monkeypatch):
    # bind exits 0 but the BOUND set is empty (every attach failed; the load
    # script restored them to nvme). Apply must fail the completeness check rather
    # than bypass it, and roll back nothing (nothing is on unvme).
    inv = _inventory(driver="nvme")
    cfg = _config()

    monkeypatch.setattr(binding, "bind_drives", _fake_bind_none)
    unbind_calls: list = []
    monkeypatch.setattr(binding, "unbind_drives", _recording_unbind(unbind_calls))
    launch_calls: list = []
    monkeypatch.setattr(launcher, "launch_domain", _recording_launch(launch_calls))

    res = apply_mod.apply_config(cfg, inv, dry_run=False)

    assert not res.ok
    assert any("binding incomplete" in e for e in res.errors)
    assert launch_calls == []        # never launched
    assert unbind_calls == []        # nothing bound -> nothing to roll back


def test_apply_holds_host_lock_but_dry_run_does_not(runtime_dir, monkeypatch):
    # A real apply runs under the operation-wide host lock (validate -> mutate ->
    # publish); a dry-run mutates nothing and takes no lock.
    import contextlib

    calls: list = []

    @contextlib.contextmanager
    def _recording_lock():
        calls.append("enter")
        try:
            yield
        finally:
            calls.append("exit")

    monkeypatch.setattr(registry_mod, "host_lock", _recording_lock)
    monkeypatch.setattr(binding, "bind_drives", _fake_bind_ok)
    monkeypatch.setattr(binding, "unbind_drives", _recording_unbind([]))
    monkeypatch.setattr(launcher, "launch_domain", _fake_launch_all_ok)

    inv = _inventory("nvme")
    cfg = _config()

    apply_mod.apply_config(cfg, inv, dry_run=False)
    assert calls == ["enter", "exit"]

    calls.clear()
    apply_mod.apply_config(cfg, inv, dry_run=True)
    assert calls == []


def test_bind_failure_rolls_back_and_releases(runtime_dir, monkeypatch):
    inv = _inventory(driver="nvme")
    cfg = _config()

    monkeypatch.setattr(binding, "bind_drives", _fake_bind_fail(["0000:83:00.0"]))
    unbind_calls: list = []
    monkeypatch.setattr(binding, "unbind_drives", _recording_unbind(unbind_calls))
    stop_calls: list = []
    monkeypatch.setattr(launcher, "stop_domain", _recording_stop(stop_calls, ok=True))
    launch_calls: list = []
    monkeypatch.setattr(launcher, "launch_domain", _recording_launch(launch_calls))

    res = apply_mod.apply_config(cfg, inv, dry_run=False)

    assert not res.ok
    assert any("drive binding failed" in e for e in res.errors)
    assert unbind_calls == [["0000:83:00.0"]]
    assert launch_calls == []
    assert stop_calls == []


# --------------------------------------------------------------------------- #
# MPS is part of the rollback -- but only when THIS apply started it
# --------------------------------------------------------------------------- #
def test_rollback_stops_mps_when_this_apply_started_it(runtime_dir, monkeypatch):
    inv = _inventory(driver="nvme")
    cfg = _config(mps_enabled=True)

    monkeypatch.setattr(binding, "bind_drives", _fake_bind_ok)
    monkeypatch.setattr(binding, "unbind_drives", _recording_unbind([]))
    monkeypatch.setattr(launcher, "stop_domain", _recording_stop([], ok=True))
    monkeypatch.setattr(launcher, "launch_domain", _fake_launch_first_ok_second_fail)
    mps_stops: list = []
    _stub_mps(monkeypatch, was_running=False, stop_calls=mps_stops)

    res = apply_mod.apply_config(cfg, inv, dry_run=False)

    assert not res.ok
    assert any("rolled back all changes" in a for a in res.actions)
    # MPS was not running before this apply started it, so rollback must stop it.
    assert mps_stops == [False]


def test_rollback_leaves_mps_alone_when_a_domain_could_not_be_stopped(runtime_dir, monkeypatch):
    # Even though THIS apply started MPS, a domain that failed to stop may still be
    # using it (and still holds its drives), so rollback must halt before the
    # MPS/unbind steps rather than pull resources out from under a live process.
    inv = _inventory(driver="nvme")
    cfg = _config(mps_enabled=True)

    monkeypatch.setattr(binding, "bind_drives", _fake_bind_ok)
    unbind_calls: list = []
    monkeypatch.setattr(binding, "unbind_drives", _recording_unbind(unbind_calls))
    monkeypatch.setattr(launcher, "stop_domain", _recording_stop([], ok=False))
    monkeypatch.setattr(launcher, "launch_domain", _fake_launch_first_ok_second_fail)
    mps_stops: list = []
    _stub_mps(monkeypatch, was_running=False, stop_calls=mps_stops)

    res = apply_mod.apply_config(cfg, inv, dry_run=False)

    assert not res.ok
    assert any("rollback halted" in e for e in res.errors)
    assert mps_stops == []      # daemon left up for the still-running domain
    assert unbind_calls == []   # drives left bound under the live process


def test_rollback_leaves_mps_running_when_already_up(runtime_dir, monkeypatch):
    inv = _inventory(driver="nvme")
    cfg = _config(mps_enabled=True)

    monkeypatch.setattr(binding, "bind_drives", _fake_bind_ok)
    monkeypatch.setattr(binding, "unbind_drives", _recording_unbind([]))
    monkeypatch.setattr(launcher, "stop_domain", _recording_stop([], ok=True))
    monkeypatch.setattr(launcher, "launch_domain", _fake_launch_first_ok_second_fail)
    mps_stops: list = []
    _stub_mps(monkeypatch, was_running=True, stop_calls=mps_stops)

    res = apply_mod.apply_config(cfg, inv, dry_run=False)

    assert not res.ok
    # MPS was already running (another workload may rely on it) -> never stopped.
    assert mps_stops == []


# --------------------------------------------------------------------------- #
# A publish failure after all domains launched must also roll back
# --------------------------------------------------------------------------- #
def test_publish_failure_rolls_back(runtime_dir, monkeypatch):
    inv = _inventory(driver="nvme")
    cfg = _config()

    monkeypatch.setattr(binding, "bind_drives", _fake_bind_ok)
    unbind_calls: list = []
    monkeypatch.setattr(binding, "unbind_drives", _recording_unbind(unbind_calls))
    stop_calls: list = []
    monkeypatch.setattr(launcher, "stop_domain", _recording_stop(stop_calls, ok=True))
    monkeypatch.setattr(launcher, "launch_domain", _fake_launch_all_ok)

    def _boom_publish(_mutate):
        raise OSError("registry write boom")

    monkeypatch.setattr(registry_mod, "update_registry", _boom_publish)

    res = apply_mod.apply_config(cfg, inv, dry_run=False)

    assert not res.ok
    # A non-ValueError publish failure is caught and routed through rollback,
    # rather than escaping and orphaning the launched domains.
    assert any("registry write boom" in e for e in res.errors)
    assert any("rolled back all changes" in a for a in res.actions)
    # Both launched domains stopped (reverse order), drives released.
    assert stop_calls == [(cfg.name, "d1"), (cfg.name, "d0")]
    assert unbind_calls == [["0000:83:00.0", "0000:84:00.0"]]


def test_residual_publish_preserves_untouched_sibling_domain(runtime_dir, monkeypatch):
    """A partial-failure residual publish must not delete a LIVE sibling domain.

    ``reconcile_add_config`` replaces a config's registry entries wholesale, so
    publishing only the domains this apply rebuilt would erase the healthy no-op
    siblings it never touched -- orphaning running servers that 'stop --config'
    could then no longer find (while 'stop --all' would still unbind their drives).
    """
    inv = _inventory(driver="unvme")
    cfg = _config()

    # Config already active with BOTH domains recorded, and a group map.
    desired = apply_mod.build_registry_domains(cfg, inv)
    seeded = registry_mod.reconcile_add_config(
        Registry(), cfg.name, desired,
        domain_groups={"all": ["d0", "d1"]}, mps_enabled=False,
    )
    registry_mod.write_registry(seeded)
    cfg.domain_groups = {"all": ["d0", "d1"]}

    # d1 is healthy (NOOP, untouched); d0's process died -> SELF_HEAL.
    monkeypatch.setattr(reconcile_mod, "_domain_alive", lambda rd: rd.name == "d1")
    monkeypatch.setattr(binding, "bind_drives", _fake_bind_ok)
    monkeypatch.setattr(binding, "unbind_drives", _recording_unbind([]))
    monkeypatch.setattr(launcher, "launch_domain", _fake_launch_all_ok)
    # Rollback cannot stop the relaunched d0 -> teardown halts, residual persisted.
    monkeypatch.setattr(launcher, "stop_domain", _recording_stop([], ok=False))

    real_update = registry_mod.update_registry
    calls = {"n": 0}

    def _fail_first_publish(mutate):
        # Fail the apply's own publish, but let the residual publish through.
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("registry write boom")
        return real_update(mutate)

    monkeypatch.setattr(registry_mod, "update_registry", _fail_first_publish)

    res = apply_mod.apply_config(cfg, inv, dry_run=False)

    assert not res.ok
    # Prove the residual-publish path actually ran (otherwise the assertions below
    # would pass vacuously against the untouched seeded registry).
    assert any("registry write boom" in e for e in res.errors)
    assert any("rollback halted" in e for e in res.errors)
    assert any("rollback incomplete: recorded" in a for a in res.actions), res.actions

    reg = registry_mod.read_registry()
    names = {d.name for d in reg.domains if d.config == cfg.name}
    # d1 was never touched by this apply and is still running: it MUST survive.
    assert "d1" in names, f"live sibling erased from registry: {names}"
    # d0 could not be stopped, so it stays tracked too.
    assert "d0" in names
    # The config's group map must not be wiped by the residual publish.
    entry = next(c for c in reg.configs if c.name == cfg.name)
    assert entry.domain_groups == {"all": ["d0", "d1"]}
