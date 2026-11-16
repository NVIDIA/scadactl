# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stop paths: partial-teardown accounting for stop_config / stop_all.

Covers the fix for silently purging a still-running domain from the registry
when its stop (or drive release) fails: a domain is dropped from the registry
only once it is fully torn down (process stopped AND drives released), any
failure is recorded in ``ApplyResult.errors`` so ``cmd_stop``'s exit code goes
non-zero, and the host-global teardown in ``stop_all`` only runs once every
domain has stopped cleanly.
"""

from __future__ import annotations

import stat
from types import SimpleNamespace

from scadactl import apply as apply_mod
from scadactl import registry as registry_mod
from scadactl.backend import binding, launcher, mps
from scadactl.backend.binding import BindResult
from scadactl.backend.launcher import LaunchResult
from scadactl.backend.mps import MpsResult
from scadactl.models import ConfigEntry, Registry, RegistryDomain, RegistryDrive


def _domain(name, config, pci="0000:83:00.0", gpu="GPU-1111"):
    return RegistryDomain(
        name=name,
        config=config,
        gpu_uuid=gpu,
        gpu_pci_bus_id="0000:81:00.0",
        host_cuda_visible_devices="0",
        socket=f"/var/run/scada/{name}.sock",
        drives=[
            RegistryDrive(
                linux_device_before_bind="/dev/nvme2n1",
                serial="S1",
                pci_bus_id=pci,
            )
        ],
    )


def _seed(domains, active_configs, configs):
    """Publish a starting registry so stop_config/stop_all have state to read."""
    reg = Registry(active_configs=active_configs, configs=configs, domains=domains)
    registry_mod.write_registry(reg)
    return reg


def _recording_stop(results):
    """``results``: ``{(config, domain): LaunchResult}`` for the failing cases."""
    calls = []

    def _fake(config, domain, dry_run=False):
        calls.append((config, domain))
        return results.get((config, domain), LaunchResult(returncode=0))

    return _fake, calls


def _recording_unbind_drives(ok=True):
    calls = []

    def _fake(pci_addresses, dry_run=False):
        calls.append(list(pci_addresses))
        return BindResult(returncode=0 if ok else 1, stderr="" if ok else "unbind boom")

    return _fake, calls


def _recording_unbind_all(ok=True):
    calls = []

    def _fake(dry_run=False):
        calls.append(True)
        return BindResult(returncode=0 if ok else 1, stderr="" if ok else "module unload boom")

    return _fake, calls


def _recording_mps_stop(ok=True):
    calls = []

    def _fake(dry_run=False):
        calls.append(True)
        return MpsResult(returncode=0 if ok else 1, stderr="" if ok else "mps boom")

    return _fake, calls


# --------------------------------------------------------------------------- #
# stop_config
# --------------------------------------------------------------------------- #
def test_stop_config_all_succeed(runtime_dir, monkeypatch):
    d1 = _domain("d1", "cfg1")
    _seed([d1], ["cfg1"], [ConfigEntry(name="cfg1", domains=["d1"])])

    stop_fake, stop_calls = _recording_stop({})
    monkeypatch.setattr(launcher, "stop_domain", stop_fake)
    unbind_fake, unbind_calls = _recording_unbind_drives(ok=True)
    monkeypatch.setattr(binding, "unbind_drives", unbind_fake)

    res = apply_mod.stop_config("cfg1", dry_run=False)

    assert res.ok
    assert any("stopped 1 of 1" in a for a in res.actions)
    reg = registry_mod.read_registry()
    assert reg.domains == []
    assert reg.configs == []
    assert reg.active_configs == []


def test_stop_config_removes_stale_managed_profiler_socket(runtime_dir, monkeypatch):
    d1 = _domain("d1", "cfg1")
    d1.profiler_socket = "/var/run/scada/d1.profiler.sock"
    _seed([d1], ["cfg1"], [ConfigEntry(name="cfg1", domains=["d1"])])
    monkeypatch.setattr(launcher, "stop_domain", _recording_stop({})[0])
    monkeypatch.setattr(binding, "unbind_drives", _recording_unbind_drives()[0])
    monkeypatch.setattr(
        apply_mod.os, "lstat", lambda path: SimpleNamespace(st_mode=stat.S_IFSOCK)
    )
    removed = []
    monkeypatch.setattr(apply_mod.os, "unlink", lambda path: removed.append(path))

    res = apply_mod.stop_config("cfg1", dry_run=False)

    assert res.ok
    assert removed == [d1.profiler_socket]
    assert any("removed stale profiler socket" in action for action in res.actions)


def test_stop_config_stops_mps_when_it_was_the_last_mps_config(runtime_dir, monkeypatch):
    # cfg1 (MPS) + cfg2 (no MPS): stopping cfg1 individually must stop the daemon
    # -- no remaining config needs it, and a later `stop --all` never would.
    d1 = _domain("d1", "cfg1")
    d2 = _domain("d2", "cfg2", pci="0000:84:00.0")
    _seed([d1, d2], ["cfg1", "cfg2"],
          [ConfigEntry(name="cfg1", domains=["d1"], mps_enabled=True),
           ConfigEntry(name="cfg2", domains=["d2"])])
    monkeypatch.setattr(launcher, "stop_domain", _recording_stop({})[0])
    monkeypatch.setattr(binding, "unbind_drives", _recording_unbind_drives(ok=True)[0])
    mps_stop_fake, mps_stop_calls = _recording_mps_stop(ok=True)
    monkeypatch.setattr(mps, "stop", mps_stop_fake)

    res = apply_mod.stop_config("cfg1", dry_run=False)

    assert res.ok
    assert mps_stop_calls == [True]
    assert any("stop MPS" in a for a in res.actions)


def test_stop_config_keeps_mps_when_another_mps_config_remains(runtime_dir, monkeypatch):
    # cfg1 (MPS) + cfg2 (MPS): stopping cfg1 must NOT stop the daemon; cfg2 relies on it.
    d1 = _domain("d1", "cfg1")
    d2 = _domain("d2", "cfg2", pci="0000:84:00.0")
    _seed([d1, d2], ["cfg1", "cfg2"],
          [ConfigEntry(name="cfg1", domains=["d1"], mps_enabled=True),
           ConfigEntry(name="cfg2", domains=["d2"], mps_enabled=True)])
    monkeypatch.setattr(launcher, "stop_domain", _recording_stop({})[0])
    monkeypatch.setattr(binding, "unbind_drives", _recording_unbind_drives(ok=True)[0])
    mps_stop_fake, mps_stop_calls = _recording_mps_stop(ok=True)
    monkeypatch.setattr(mps, "stop", mps_stop_fake)

    res = apply_mod.stop_config("cfg1", dry_run=False)

    assert res.ok
    assert mps_stop_calls == []   # cfg2 still needs MPS


def test_stop_config_stop_domain_fails_keeps_it_tracked(runtime_dir, monkeypatch):
    d1 = _domain("d1", "cfg1", pci="0000:83:00.0")
    d2 = _domain("d2", "cfg1", pci="0000:84:00.0")
    _seed([d1, d2], ["cfg1"], [ConfigEntry(name="cfg1", domains=["d1", "d2"])])

    stop_fake, stop_calls = _recording_stop(
        {("cfg1", "d2"): LaunchResult(returncode=1, stderr="stuck")}
    )
    monkeypatch.setattr(launcher, "stop_domain", stop_fake)
    unbind_fake, unbind_calls = _recording_unbind_drives(ok=True)
    monkeypatch.setattr(binding, "unbind_drives", unbind_fake)

    res = apply_mod.stop_config("cfg1", dry_run=False)

    assert not res.ok
    assert any("stop failed" in e for e in res.errors)
    assert any("still tracked" in a for a in res.actions)

    reg = registry_mod.read_registry()
    names = {d.name for d in reg.domains}
    assert names == {"d2"}
    assert "d1" not in names


def test_stop_config_unbind_fails_keeps_it_tracked(runtime_dir, monkeypatch):
    d1 = _domain("d1", "cfg1")
    _seed([d1], ["cfg1"], [ConfigEntry(name="cfg1", domains=["d1"])])

    stop_fake, stop_calls = _recording_stop({})
    monkeypatch.setattr(launcher, "stop_domain", stop_fake)
    unbind_fake, unbind_calls = _recording_unbind_drives(ok=False)
    monkeypatch.setattr(binding, "unbind_drives", unbind_fake)

    res = apply_mod.stop_config("cfg1", dry_run=False)

    assert not res.ok
    assert any("drive release failed" in e for e in res.errors)

    reg = registry_mod.read_registry()
    assert {d.name for d in reg.domains} == {"d1"}


def test_stop_config_mps_failure_retains_config_then_retries(runtime_dir, monkeypatch):
    # A failed MPS teardown must NOT drop the owning config: the daemon would leak
    # and a rerun would see nothing to retry. Keep it tracked; a rerun (domain now
    # dead) skips the non-idempotent stop, re-reaches MPS, and retries.
    from scadactl import reconcile as reconcile_mod

    d1 = _domain("d1", "cfg1")
    d1.pid = 4242
    _seed([d1], ["cfg1"], [ConfigEntry(name="cfg1", domains=["d1"], mps_enabled=True)])

    monkeypatch.setattr(launcher, "stop_domain", _recording_stop({})[0])
    monkeypatch.setattr(binding, "unbind_drives", _recording_unbind_drives(ok=True)[0])
    monkeypatch.setattr(reconcile_mod, "_domain_alive", lambda d: True)  # first run: alive
    mps_fail, mps_fail_calls = _recording_mps_stop(ok=False)
    monkeypatch.setattr(mps, "stop", mps_fail)

    res = apply_mod.stop_config("cfg1", dry_run=False)

    assert not res.ok
    assert any("MPS stop failed" in e for e in res.errors)
    assert mps_fail_calls == [True]
    reg = registry_mod.read_registry()
    assert {d.name for d in reg.domains} == {"d1"}        # config retained for retry
    assert reg.active_configs == ["cfg1"]

    # Rerun: the domain is now dead, so its stop is skipped (idempotent) and MPS
    # cleanup is retried -- this time successfully -- clearing the registry.
    monkeypatch.setattr(reconcile_mod, "_domain_alive", lambda d: False)
    stop_fake2, stop_calls2 = _recording_stop({})
    monkeypatch.setattr(launcher, "stop_domain", stop_fake2)
    mps_ok2, mps_ok2_calls = _recording_mps_stop(ok=True)
    monkeypatch.setattr(mps, "stop", mps_ok2)

    res2 = apply_mod.stop_config("cfg1", dry_run=False)

    assert res2.ok, res2.errors
    assert mps_ok2_calls == [True]      # MPS retried
    assert stop_calls2 == []            # already-dead domain not re-stopped
    reg2 = registry_mod.read_registry()
    assert reg2.domains == [] and reg2.configs == []


# --------------------------------------------------------------------------- #
# stop_all
# --------------------------------------------------------------------------- #
def test_stop_all_succeeds(runtime_dir, monkeypatch):
    d1 = _domain("d1", "cfg1")
    d2 = _domain("d2", "cfg2", pci="0000:84:00.0", gpu="GPU-2222")
    _seed(
        [d1, d2],
        ["cfg1", "cfg2"],
        [ConfigEntry(name="cfg1", domains=["d1"], mps_enabled=True),
         ConfigEntry(name="cfg2", domains=["d2"])],
    )

    stop_fake, stop_calls = _recording_stop({})
    monkeypatch.setattr(launcher, "stop_domain", stop_fake)
    unbind_all_fake, unbind_all_calls = _recording_unbind_all(ok=True)
    monkeypatch.setattr(binding, "unbind_all", unbind_all_fake)
    mps_stop_fake, mps_stop_calls = _recording_mps_stop(ok=True)
    monkeypatch.setattr(mps, "stop", mps_stop_fake)

    res = apply_mod.stop_all(dry_run=False)

    assert res.ok
    assert unbind_all_calls == [True]
    assert mps_stop_calls == [True]
    assert any("full unbind + module unload" in a for a in res.actions)
    assert any("stop MPS" in a for a in res.actions)

    reg = registry_mod.read_registry()
    assert reg.domains == []
    assert reg.active_configs == []


def test_stop_all_skips_mps_when_no_config_enabled_it(runtime_dir, monkeypatch):
    # A config that never requested MPS must not stop the host-global MPS daemon.
    d1 = _domain("d1", "cfg1")
    _seed([d1], ["cfg1"], [ConfigEntry(name="cfg1", domains=["d1"])])  # mps off

    stop_fake, _ = _recording_stop({})
    monkeypatch.setattr(launcher, "stop_domain", stop_fake)
    unbind_all_fake, unbind_all_calls = _recording_unbind_all(ok=True)
    monkeypatch.setattr(binding, "unbind_all", unbind_all_fake)
    mps_stop_fake, mps_stop_calls = _recording_mps_stop(ok=True)
    monkeypatch.setattr(mps, "stop", mps_stop_fake)

    res = apply_mod.stop_all(dry_run=False)

    assert res.ok
    assert unbind_all_calls == [True]        # module still unloaded
    assert mps_stop_calls == []              # but MPS is left alone
    assert not any("stop MPS" in a for a in res.actions)


def test_stop_all_domain_failure_skips_unload(runtime_dir, monkeypatch):
    d1 = _domain("d1", "cfg1")
    d2 = _domain("d2", "cfg2", pci="0000:84:00.0", gpu="GPU-2222")
    _seed(
        [d1, d2],
        ["cfg1", "cfg2"],
        [ConfigEntry(name="cfg1", domains=["d1"]), ConfigEntry(name="cfg2", domains=["d2"])],
    )

    stop_fake, stop_calls = _recording_stop(
        {("cfg2", "d2"): LaunchResult(returncode=1, stderr="stuck")}
    )
    monkeypatch.setattr(launcher, "stop_domain", stop_fake)
    unbind_all_fake, unbind_all_calls = _recording_unbind_all(ok=True)
    monkeypatch.setattr(binding, "unbind_all", unbind_all_fake)
    mps_stop_fake, mps_stop_calls = _recording_mps_stop(ok=True)
    monkeypatch.setattr(mps, "stop", mps_stop_fake)

    res = apply_mod.stop_all(dry_run=False)

    assert not res.ok
    assert any("still running" in e for e in res.errors)
    assert unbind_all_calls == []

    reg = registry_mod.read_registry()
    assert {d.name for d in reg.domains} == {"d1", "d2"}
    assert set(reg.active_configs) == {"cfg1", "cfg2"}


def test_stop_all_unload_fails_retains_registry(runtime_dir, monkeypatch):
    d1 = _domain("d1", "cfg1")
    _seed([d1], ["cfg1"], [ConfigEntry(name="cfg1", domains=["d1"])])

    stop_fake, stop_calls = _recording_stop({})
    monkeypatch.setattr(launcher, "stop_domain", stop_fake)
    unbind_all_fake, unbind_all_calls = _recording_unbind_all(ok=False)
    monkeypatch.setattr(binding, "unbind_all", unbind_all_fake)
    mps_stop_fake, mps_stop_calls = _recording_mps_stop(ok=True)
    monkeypatch.setattr(mps, "stop", mps_stop_fake)

    res = apply_mod.stop_all(dry_run=False)

    assert not res.ok
    assert any("module unload failed" in e for e in res.errors)
    assert any("registry retained" in a for a in res.actions)

    reg = registry_mod.read_registry()
    assert {d.name for d in reg.domains} == {"d1"}


def test_stop_all_mps_failure_retains_registry_then_retries(runtime_dir, monkeypatch):
    # stop_all must also keep the registry when MPS stop fails (not just on module
    # unload failure), so a rerun retries the MPS cleanup.
    from scadactl import reconcile as reconcile_mod

    d1 = _domain("d1", "cfg1")
    d1.pid = 4242
    _seed([d1], ["cfg1"], [ConfigEntry(name="cfg1", domains=["d1"], mps_enabled=True)])

    monkeypatch.setattr(launcher, "stop_domain", _recording_stop({})[0])
    monkeypatch.setattr(binding, "unbind_all", _recording_unbind_all(ok=True)[0])
    monkeypatch.setattr(reconcile_mod, "_domain_alive", lambda d: True)
    mps_fail, mps_fail_calls = _recording_mps_stop(ok=False)
    monkeypatch.setattr(mps, "stop", mps_fail)

    res = apply_mod.stop_all(dry_run=False)

    assert not res.ok
    assert any("MPS stop failed" in e for e in res.errors)
    assert any("registry retained (MPS stop failed)" in a for a in res.actions)
    reg = registry_mod.read_registry()
    assert {d.name for d in reg.domains} == {"d1"}        # retained for retry

    # Rerun: domain now dead -> re-reaches host-global teardown, MPS retried OK.
    monkeypatch.setattr(reconcile_mod, "_domain_alive", lambda d: False)
    stop_fake2, stop_calls2 = _recording_stop({})
    monkeypatch.setattr(launcher, "stop_domain", stop_fake2)
    monkeypatch.setattr(binding, "unbind_all", _recording_unbind_all(ok=True)[0])
    mps_ok2, mps_ok2_calls = _recording_mps_stop(ok=True)
    monkeypatch.setattr(mps, "stop", mps_ok2)

    res2 = apply_mod.stop_all(dry_run=False)

    assert res2.ok, res2.errors
    assert mps_ok2_calls == [True]
    assert stop_calls2 == []
    reg2 = registry_mod.read_registry()
    assert reg2.domains == []


def test_stop_holds_host_lock_but_dry_run_does_not(runtime_dir, monkeypatch):
    # stop runs under the same operation-wide host lock as apply, so a teardown
    # cannot interleave with a concurrent apply's mutations. Dry-run takes none.
    import contextlib

    registry_mod.write_registry(Registry(
        active_configs=["cfg"],
        configs=[ConfigEntry(name="cfg", domains=["d1"])],
        domains=[_domain("d1", "cfg")],
    ))
    calls: list = []

    @contextlib.contextmanager
    def _recording_lock():
        calls.append("enter")
        try:
            yield
        finally:
            calls.append("exit")

    monkeypatch.setattr(registry_mod, "host_lock", _recording_lock)
    monkeypatch.setattr(launcher, "stop_domain",
                        lambda c, d, dry_run=False: LaunchResult(returncode=0))
    monkeypatch.setattr(binding, "unbind_drives",
                        lambda p, dry_run=False: BindResult(returncode=0))

    apply_mod.stop_config("cfg", dry_run=False)
    assert calls == ["enter", "exit"]

    calls.clear()
    apply_mod.stop_config("cfg", dry_run=True)
    assert calls == []
