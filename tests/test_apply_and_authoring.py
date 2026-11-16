# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply dry-run reconcile, config authoring, and CLI dispatch smoke tests."""

from __future__ import annotations

import json

from scadactl import apply as apply_mod
from scadactl import cli, config as config_mod, registry as reg_mod
from scadactl.backend import binding, launcher
from scadactl.backend.binding import BindResult
from scadactl.models import Domain, DriveRef, Config


def _config():
    return Config(
        name="custom-layout",
        mps_enabled=True,
        server_env={"SCADA_NUM_QUEUE_PAIRS": "64"},
        domains=[
            Domain(name="gpu0_domain", gpu_uuid="GPU-1111", gpu_pci_bus_id="0000:81:00.0",
                   drives=[DriveRef(serial="S111", pci_bus_id="0000:83:00.0")],
                   server_env={"SCADA_NUM_QUEUE_PAIRS": "128"}),
        ],
    )


def test_apply_dry_run_no_host_changes(runtime_dir, inventory):
    res = apply_mod.apply_config(_config(), inventory, dry_run=True)
    assert res.ok, res.errors
    # dry-run does NOT write the registry
    import os
    assert not os.path.exists(reg_mod.registry_path())
    # the in-memory registry is built and tagged with the config
    assert res.registry.active_configs == ["custom-layout"]
    d = res.registry.domains[0]
    assert d.host_cuda_visible_devices == "0"
    assert d.server_env["SCADA_NUM_QUEUE_PAIRS"] == "128"
    assert any("publish" in a for a in res.actions)


def test_kernel_dry_run_shows_bdf_when_unvme_mapping_is_unavailable(runtime_dir, inventory):
    res = apply_mod.apply_config(_config(), inventory, dry_run=True)

    assert res.ok, res.errors
    assert any("config '0 0000:83:00.0'" in action for action in res.actions)


def test_kernel_apply_waits_for_socket_after_launch(runtime_dir, inventory, monkeypatch):
    cfg = _config()
    cfg.mps_enabled = False
    readiness_calls = []
    monkeypatch.setattr(
        binding, "bind_drives",
        lambda pci_addresses, **kwargs: BindResult(
            returncode=0, bound_pci=list(pci_addresses),
            unvme_by_pci={pci: "/dev/unvme0" for pci in pci_addresses},
        ),
    )
    monkeypatch.setattr(
        launcher, "launch_domain",
        lambda **kwargs: launcher.LaunchResult(returncode=0, pid=17, log_dir="/tmp/gpu0_domain"),
    )
    monkeypatch.setattr(
        apply_mod, "_wait_for_domain_socket",
        lambda *args: readiness_calls.append(args) or True,
    )

    res = apply_mod.apply_config(cfg, inventory)

    assert res.ok, res.errors
    assert len(readiness_calls) == 1
    assert readiness_calls[0][0] == res.registry.domains[0].socket


def test_kernel_apply_rolls_back_when_socket_never_becomes_ready(runtime_dir, inventory, monkeypatch):
    cfg = _config()
    cfg.mps_enabled = False
    stop_calls, unbind_calls = [], []
    monkeypatch.setattr(
        binding, "bind_drives",
        lambda pci_addresses, **kwargs: BindResult(
            returncode=0, bound_pci=list(pci_addresses),
            unvme_by_pci={pci: "/dev/unvme0" for pci in pci_addresses},
        ),
    )
    monkeypatch.setattr(
        binding, "unbind_drives",
        lambda pci_addresses, **kwargs: unbind_calls.append(list(pci_addresses))
        or BindResult(returncode=0),
    )
    monkeypatch.setattr(
        launcher, "launch_domain",
        lambda **kwargs: launcher.LaunchResult(returncode=0, pid=17, log_dir="/tmp/gpu0_domain"),
    )
    monkeypatch.setattr(
        launcher, "stop_domain",
        lambda *args, **kwargs: stop_calls.append(args) or launcher.LaunchResult(returncode=0),
    )
    monkeypatch.setattr(apply_mod, "_wait_for_domain_socket", lambda *args: False)

    res = apply_mod.apply_config(cfg, inventory)

    assert not res.ok
    assert any("did not become ready" in error for error in res.errors)
    assert stop_calls == [("custom-layout", "gpu0_domain")]
    assert unbind_calls == [["0000:83:00.0"]]


def test_apply_carries_domain_groups_into_registry(runtime_dir, inventory):
    from scadactl import client as client_mod
    p = Config(
        name="multi",
        domains=[
            Domain(name="gpu0_domain", gpu_uuid="GPU-1111", gpu_pci_bus_id="0000:81:00.0",
                   drives=[DriveRef(serial="S111", pci_bus_id="0000:83:00.0")]),
            Domain(name="gpu1_domain", gpu_uuid="GPU-2222", gpu_pci_bus_id="0000:82:00.0",
                   drives=[DriveRef(serial="S222", pci_bus_id="0000:84:00.0")]),
        ],
        domain_groups={"all": ["gpu0_domain", "gpu1_domain"]},
    )
    res = apply_mod.apply_config(p, inventory, dry_run=True)
    assert res.ok, res.errors
    reg = res.registry
    assert reg.configs[0].domain_groups["all"] == ["gpu0_domain", "gpu1_domain"]
    # the client can now resolve the group name
    assert [d.name for d in client_mod.resolve_selection(reg, "all")] == \
        ["gpu0_domain", "gpu1_domain"]


def test_apply_dry_run_blocks_on_validation_error(runtime_dir, inventory):
    p = _config()
    p.domains[0].gpu_uuid = "GPU-NOPE"
    res = apply_mod.apply_config(p, inventory, dry_run=True)
    assert not res.ok
    assert any("unknown GPU UUID" in e for e in res.errors)


def test_build_registry_domains_drive_identity(inventory):
    domains = apply_mod.build_registry_domains(_config(), inventory)
    drv = domains[0].drives[0]
    assert drv.serial == "S111"
    assert drv.linux_device_before_bind == "/dev/nvme2n1"
    assert drv.model == "PM1735"


def test_authoring_add_domain_writes_stable_ids(inventory):
    prof = config_mod.empty_config("p")
    assert prof.server_env["SCADA_NUM_QUEUE_PAIRS"] == "16"
    config_mod.add_domain(prof, inventory, name="d1", gpu_token="gpu1", drive_tokens=["nvme1"])
    d = prof.domains[0]
    # aliases resolved to stable identifiers
    assert d.gpu_uuid == "GPU-2222"
    assert d.gpu_pci_bus_id == "0000:82:00.0"
    assert d.drives[0].serial == "S222"
    assert d.drives[0].pci_bus_id == "0000:84:00.0"


def test_authoring_add_domain_rejects_empty_drive_selection(inventory):
    config = config_mod.empty_config("p")

    try:
        config_mod.add_domain(config, inventory, name="d1", gpu_token="gpu1", drive_tokens=[])
    except ValueError as exc:
        assert str(exc) == "a domain requires at least one drive"
    else:
        raise AssertionError("expected empty drive selection to fail")


def test_cli_add_domain_creates_new_file(runtime_dir, tmp_path, capsys, monkeypatch, inventory):
    # add-domain with -f pointing at a nonexistent file should CREATE it,
    # not raise FileNotFoundError.
    monkeypatch.setattr("scadactl.cli.inventory_mod.collect_inventory", lambda: inventory)
    path = tmp_path / "new.yaml"
    assert not path.exists()
    rc = cli.main(["config", "add-domain", "--name", "d1", "--gpu", "gpu0",
                   "--drives", "nvme0", "-f", str(path)])
    assert rc == 0
    assert path.exists()
    loaded = config_mod.load_config(str(path))
    assert loaded.domains[0].gpu_uuid == "GPU-1111"


def test_seed_from_resources(inventory):
    prof = config_mod.seed_config_from_resources(inventory, "p", ["gpu0", "gpu1"], ["nvme0", "nvme1"])
    assert len(prof.domains) == 2
    assert prof.domains[0].gpu_uuid == "GPU-1111"
    assert "all" in prof.domain_groups


def test_config_yaml_roundtrip(inventory, tmp_path):
    prof = _config()
    path = tmp_path / "p.yaml"
    config_mod.save_config(prof, str(path))
    loaded = config_mod.load_config(str(path))
    assert loaded.name == "custom-layout"
    assert loaded.mps_enabled is True
    assert loaded.domains[0].server_env["SCADA_NUM_QUEUE_PAIRS"] == "128"
    assert loaded.schema_version == 2


def test_legacy_nsys_config_roundtrip_and_registry_endpoint(inventory, tmp_path):
    prof = _config()
    prof.profiler_endpoint_mode = "legacy_nsys"
    # Legacy mode is intentionally limited to a single domain.
    prof.domains = prof.domains[:1]
    path = tmp_path / "legacy.yaml"
    config_mod.save_config(prof, str(path))
    loaded = config_mod.load_config(str(path))
    assert loaded.profiler_endpoint_mode == "legacy_nsys"
    rd = apply_mod.build_registry_domains(loaded, inventory)[0]
    assert rd.profiler_socket == "/tmp/scada_profiler_socket"
    assert rd.profiler_endpoint_mode == "legacy_nsys"


# ---- CLI dispatch smoke tests ---- #
def test_cli_help_runs(capsys):
    assert cli.main([]) == 0  # prints help, exits 0
    out = capsys.readouterr().out
    assert "inventory" in out and "apply" in out


def test_normalize_command_strips_leading_separator():
    # argparse REMAINDER keeps the '--'; cmd_run must drop a single leading one.
    assert cli._normalize_command(["--", "python", "train.py"]) == ["python", "train.py"]
    assert cli._normalize_command(["python", "train.py"]) == ["python", "train.py"]
    assert cli._normalize_command(["--", "torchrun", "--nproc_per_node=2"]) == \
        ["torchrun", "--nproc_per_node=2"]
    assert cli._normalize_command([]) == []
    assert cli._normalize_command(None) == []


def test_cli_run_missing_command(runtime_dir, capsys):
    # selection ok but only the separator after `--` -> no command
    rc = cli.main(["run", "--domain", "gpu0_domain", "--"])
    assert rc == 2
    assert "no command" in capsys.readouterr().err


def test_cli_status_empty(runtime_dir, capsys):
    rc = cli.main(["status"])
    assert rc == 0
    assert "No active domains" in capsys.readouterr().out


def test_cli_validate(runtime_dir, tmp_path, capsys, monkeypatch, inventory):
    # avoid touching real NVML/nvme by stubbing inventory collection
    monkeypatch.setattr("scadactl.cli.inventory_mod.collect_inventory", lambda: inventory)
    path = tmp_path / "p.yaml"
    config_mod.save_config(_config(), str(path))
    rc = cli.main(["config", "validate", "-f", str(path)])
    assert rc == 0
    assert "OK" in capsys.readouterr().out


def test_cli_apply_dry_run_json(runtime_dir, tmp_path, capsys, monkeypatch, inventory):
    monkeypatch.setattr("scadactl.cli.inventory_mod.collect_inventory", lambda: inventory)
    path = tmp_path / "p.yaml"
    config_mod.save_config(_config(), str(path))
    rc = cli.main(["apply", "-f", str(path), "--dry-run", "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["config"] == "custom-layout"
    assert data["domains"][0]["host_cuda_visible_devices"] == "0"


def test_cli_apply_dry_run_commands_tolerates_missing_inventory(
    runtime_dir, tmp_path, capsys, monkeypatch
):
    # Inventory collection failing -> _load_inventory_best_effort() returns None;
    # apply --dry-run --commands must still render, not crash (regression).
    def _boom():
        raise RuntimeError("inventory exploded")

    monkeypatch.setattr(
        "scadactl.cli.inventory_mod.collect_inventory", _boom
    )
    path = tmp_path / "p.yaml"
    config_mod.save_config(_config(), str(path))
    rc = cli.main(["apply", "-f", str(path), "--dry-run", "--commands"])
    assert "Commands:" in capsys.readouterr().out


def test_cli_apply_dry_run(runtime_dir, tmp_path, capsys, monkeypatch, inventory):
    monkeypatch.setattr("scadactl.cli.inventory_mod.collect_inventory", lambda: inventory)
    path = tmp_path / "p.yaml"
    config_mod.save_config(_config(), str(path))
    rc = cli.main(["apply", "-f", str(path), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "gpu0_domain" in out


def _drive_interactive(monkeypatch, inventory, answers):
    """Wire up a fake TTY + scripted ``input()`` for interactive-init tests."""
    import io

    class _Tty(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr("scadactl.cli.inventory_mod.collect_inventory",
                        lambda: inventory)
    monkeypatch.setattr("sys.stdin", _Tty())
    it = iter(answers)
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(it))


def test_cli_interactive_init_build(runtime_dir, tmp_path, monkeypatch, inventory):
    # 2 domains, MPS kept on (new default), NUM_QUEUE_PAIRS changed to 8.
    _drive_interactive(
        monkeypatch,
        inventory,
        [
            "multi",  # config name
            "gpu0",  # domain 1 GPU
            "nvme0,nvme1",  # its drives
            "",  # root-only socket
            "gpu1",  # domain 2 GPU
            "nvme2,nvme3",  # its drives
            "",  # root-only socket
            "",  # blank GPU -> finish
            "",  # done defining domains? -> yes
            "",  # 'all' group? -> yes
            "",  # enable MPS? -> default yes
            "",  # multiuser MPS? -> default yes
            "n",  # keep default server env? -> no, edit
            "8",  # SCADA_NUM_QUEUE_PAIRS
            "",  # SCADA_QUEUE_SIZE_SQ_GPU -> keep
            "",  # SCADA_QUEUE_SIZE_CQ_GPU -> keep
            "",  # write this config? -> yes
        ],
    )
    out = tmp_path / "multi.yaml"
    rc = cli.main(["config", "init", "-o", str(out)])
    assert rc == 0
    prof = config_mod.load_config(str(out))
    assert prof.name == "multi"
    assert [d.name for d in prof.domains] == ["gpu0_domain", "gpu1_domain"]
    assert prof.domains[0].gpu_uuid == "GPU-1111"
    assert prof.domains[0].drives[0].serial == "S111"
    assert prof.domains[1].drives[1].serial == "S444"
    assert prof.mps_enabled is True
    assert prof.mps_multiuser is True
    assert prof.server_env["SCADA_NUM_QUEUE_PAIRS"] == "8"
    assert prof.server_env["SCADA_QUEUE_SIZE_SQ_GPU"] == "1024"
    assert prof.domain_groups["all"] == ["gpu0_domain", "gpu1_domain"]
    assert prof.profiler_endpoint_mode == "managed"


def test_cli_interactive_init_single_gpu_can_select_legacy_nsys(runtime_dir, tmp_path, monkeypatch, inventory):
    _drive_interactive(
        monkeypatch,
        inventory,
        [
            "single",  # config name
            "gpu0",
            "nvme0",  # one domain
            "",  # root-only socket
            "",
            "",  # finish domains
            "",  # MPS -> default yes
            "",  # multiuser MPS -> default yes
            "legacy_nsys",  # profiler endpoint mode
            "",  # keep server env
            "",  # write config
        ],
    )
    out = tmp_path / "single.yaml"
    assert cli.main(["config", "init", "-o", str(out)]) == 0
    prof = config_mod.load_config(str(out))
    assert prof.profiler_endpoint_mode == "legacy_nsys"


def test_cli_interactive_init_no_gpu_surfaces_warning(runtime_dir, monkeypatch, capsys):
    # With NVML unavailable the inventory is empty but carries a warning; `config
    # init` must surface that reason, not just a bare "no GPUs detected" (the
    # symptom that looks wrong when nvidia-smi clearly shows GPUs).
    import io
    from scadactl.models import Inventory

    class _Tty(io.StringIO):
        def isatty(self):
            return True

    empty = Inventory(
        gpus=[], drives=[],
        warnings=["GPU inventory unavailable: NVML bindings are not installed."],
    )
    monkeypatch.setattr(
        "scadactl.cli.inventory_mod.collect_inventory", lambda: empty)
    monkeypatch.setattr("sys.stdin", _Tty())

    rc = cli.main(["config", "init", "--name", "x"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "NVML bindings are not installed" in err
    assert "no GPUs detected" in err


def test_cli_interactive_init_edit_mps(runtime_dir, tmp_path, monkeypatch, inventory):
    # Reject the summary, edit MPS off, then confirm.
    _drive_interactive(
        monkeypatch,
        inventory,
        [
            "multi",
            "gpu0",
            "nvme0,nvme1",
            "",  # root-only socket
            "gpu1",
            "nvme2,nvme3",
            "",  # root-only socket
            "",
            "",  # finish domains
            "",  # 'all' group? -> yes
            "",  # MPS? -> yes
            "",  # multiuser MPS? -> yes
            "",  # keep server env? -> yes
            "n",  # write? -> no -> edit menu
            "3",  # edit MPS
            "n",  # enable MPS? -> no
            "",  # write? -> yes
        ],
    )
    out = tmp_path / "m.yaml"
    assert cli.main(["config", "init", "-o", str(out)]) == 0
    prof = config_mod.load_config(str(out))
    assert prof.mps_enabled is False
    assert len(prof.domains) == 2


def test_cli_interactive_init_remove_domain(runtime_dir, tmp_path, monkeypatch, inventory):
    # Build 2, reject, remove one; 'all' group drops since <2 remain.
    _drive_interactive(
        monkeypatch,
        inventory,
        [
            "multi",
            "gpu0",
            "nvme0,nvme1",
            "",  # root-only socket
            "gpu1",
            "nvme2,nvme3",
            "",  # root-only socket
            "",
            "",  # finish domains
            "",  # 'all' group? -> yes
            "",  # MPS? -> yes
            "",  # multiuser MPS? -> yes
            "",  # keep server env? -> yes
            "n",  # write? -> no
            "1",  # edit domains
            "r",  # remove
            "gpu1",  # remove gpu1_domain
            "",  # finish removing
            "",  # write? -> yes
        ],
    )
    out = tmp_path / "m.yaml"
    assert cli.main(["config", "init", "-o", str(out)]) == 0
    prof = config_mod.load_config(str(out))
    assert [d.name for d in prof.domains] == ["gpu0_domain"]
    assert "all" not in prof.domain_groups


def test_cli_interactive_init_modify_pool_filtering(runtime_dir, tmp_path, monkeypatch, inventory):
    # Modify gpu0_domain: a drive held by gpu1_domain is rejected; its own
    # freed drives are selectable.
    _drive_interactive(
        monkeypatch,
        inventory,
        [
            "multi",
            "gpu0",
            "nvme0,nvme1",
            "",  # root-only socket
            "gpu1",
            "nvme2,nvme3",
            "",  # root-only socket
            "",
            "",  # finish domains
            "",  # 'all' group? -> yes
            "",  # MPS? -> yes
            "",  # multiuser MPS? -> yes
            "",  # keep server env? -> yes
            "n",  # write? -> no
            "1",  # edit domains
            "m",  # modify
            "gpu0",  # modify gpu0_domain
            "nvme2",  # held by gpu1_domain -> rejected
            "nvme0",  # accepted (1 drive)
            "",  # keep root-only socket
            "",  # write? -> yes
        ],
    )
    out = tmp_path / "m.yaml"
    assert cli.main(["config", "init", "-o", str(out)]) == 0
    prof = config_mod.load_config(str(out))
    by_name = {d.name: d for d in prof.domains}
    assert [dr.serial for dr in by_name["gpu0_domain"].drives] == ["S111"]
    assert [dr.serial for dr in by_name["gpu1_domain"].drives] == ["S333", "S444"]


def test_per_domain_log_dirs_are_distinct(runtime_dir, inventory):
    from scadactl.backend import launcher
    # two domains in one config must not share a log dir / pidfile
    assert launcher.domain_log_dir("p", "d1") != launcher.domain_log_dir("p", "d2")
    assert launcher.domain_log_dir("p", "d1").endswith("/p/d1")

    p = Config(
        name="multi",
        domains=[
            Domain(name="d0", gpu_uuid="GPU-1111", gpu_pci_bus_id="0000:81:00.0",
                   drives=[DriveRef(serial="S111", pci_bus_id="0000:83:00.0")]),
            Domain(name="d1", gpu_uuid="GPU-2222", gpu_pci_bus_id="0000:82:00.0",
                   drives=[DriveRef(serial="S222", pci_bus_id="0000:84:00.0")]),
        ],
    )
    res = apply_mod.apply_config(p, inventory, dry_run=True)
    assert res.ok, res.errors
    launch_actions = [a for a in res.actions if a.startswith("launch domain")]
    assert "log-dir /var/log/scada/multi/d0" in launch_actions[0]
    assert "log-dir /var/log/scada/multi/d1" in launch_actions[1]


def test_read_launched_pid_parses_pidfile(tmp_path):
    from scadactl.backend import launcher
    pf = tmp_path / launcher.PIDFILE_NAME
    pf.write_text(
        "# scada-swap-domains pidfile (managed by manage-scada-domains)\n"
        "0\t12345\t/usr/bin/scada-swap-domain\t99887766\n"
        "2\t12346\t/usr/bin/scada-swap-domain\t99887767\n"
    )
    assert launcher.read_launched_pid(str(tmp_path), "0") == 12345
    assert launcher.read_launched_pid(str(tmp_path), "2") == 12346
    assert launcher.read_launched_pid(str(tmp_path), "1") is None
    assert launcher.read_launched_pid(str(tmp_path), "9") is None


# ---- CLI dry-run reconciliation tests ---- #
def test_cli_apply_dry_run_shows_noop(runtime_dir, tmp_path, capsys, monkeypatch, inventory):
    """An unchanged re-apply of an already-active config is a NO-OP."""
    from scadactl import reconcile as reconcile_mod
    from scadactl import registry as registry_mod

    prof = _config()
    path = tmp_path / "p.yaml"
    config_mod.save_config(prof, str(path))

    # Seed registry with the config active and its domains matching the desired shape.
    desired = apply_mod.build_registry_domains(prof, inventory)
    registry_mod.write_registry(
        registry_mod.reconcile_add_config(registry_mod.Registry(), prof.name, desired, mps_enabled=prof.mps_enabled)
    )

    # All domains alive.
    monkeypatch.setattr(reconcile_mod, "_domain_alive", lambda rd: True)
    monkeypatch.setattr("scadactl.cli.inventory_mod.collect_inventory", lambda: inventory)

    rc = cli.main(["apply", "-f", str(path), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "no changes" in out or "no-op" in out


def test_cli_apply_dry_run_warns_on_reclaim(runtime_dir, tmp_path, capsys, monkeypatch):
    """An unmanaged orphan drive triggers a WARNING in dry-run."""
    from scadactl.models import Inventory, Drive, GPU

    # Build a minimal inventory: 1 GPU + 1 unvme drive (unmanaged orphan).
    orphan_inv = Inventory(
        gpus=[
            GPU(alias="gpu0", smi_index=0, uuid="GPU-1111", pci_bus_id="0000:81:00.0",
                name="NVIDIA H100 80GB"),
        ],
        drives=[
            Drive(alias="nvme0", linux_device="/dev/unvme0", serial="S111", model="PM1735",
                  lba_size=4096, namespace=1, pci_bus_id="0000:83:00.0", pcie="gen4 x4",
                  driver="unvme"),
        ],
    )

    # Config targeting that GPU and drive.
    prof = Config(
        name="test-reclaim",
        domains=[
            Domain(name="gpu0_domain", gpu_uuid="GPU-1111", gpu_pci_bus_id="0000:81:00.0",
                   drives=[DriveRef(serial="S111", pci_bus_id="0000:83:00.0")]),
        ],
    )
    path = tmp_path / "p.yaml"
    config_mod.save_config(prof, str(path))

    # Empty registry (drive is unmanaged).
    monkeypatch.setattr("scadactl.cli.inventory_mod.collect_inventory", lambda: orphan_inv)
    # Pin the usage probe to "confirmed free" so the reclaim path is exercised
    # deterministically regardless of whether fuser/psmisc is present. The default
    # fails closed (refuses reclaim) when it cannot tell -- e.g. the CI lint image
    # has no fuser -- which is a different (also-tested) path than this WARNING one.
    from scadactl import reconcile as reconcile_mod
    monkeypatch.setattr(reconcile_mod, "_device_in_use", lambda dev: False)

    rc = cli.main(["apply", "-f", str(path), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "reclaim" in out.lower()
    assert "0000:83:00.0" in out


def test_cli_apply_dry_run_refuses_incompatible(runtime_dir, tmp_path, capsys, monkeypatch, inventory):
    """An active config with a different definition is REFUSED."""
    from scadactl import reconcile as reconcile_mod
    from scadactl import registry as registry_mod
    from scadactl.models import Registry, RegistryDomain, RegistryDrive

    prof = _config()
    path = tmp_path / "p.yaml"
    config_mod.save_config(prof, str(path))

    # Seed registry with the config active, but mutate one domain's definition
    # (change server_env to differ from what the config has).
    desired = apply_mod.build_registry_domains(prof, inventory)
    reg = registry_mod.reconcile_add_config(registry_mod.Registry(), prof.name, desired)

    # Mutate the first domain to differ from the config's definition.
    if reg.domains:
        reg.domains[0].server_env = {"X": "1"}  # A different server_env.

    registry_mod.write_registry(reg)

    # All domains alive.
    monkeypatch.setattr(reconcile_mod, "_domain_alive", lambda rd: True)
    monkeypatch.setattr("scadactl.cli.inventory_mod.collect_inventory", lambda: inventory)

    rc = cli.main(["apply", "-f", str(path), "--dry-run"])
    assert rc == 1
    out = capsys.readouterr().out
    # Should report incompatibility and refuse.
    assert "different definition" in out.lower() or "incompatible" in out.lower() or "refuse" in out.lower()


def test_cli_apply_requires_root_for_real_run(tmp_path, capsys, monkeypatch):
    # A real (non-dry-run) apply must fail fast for a non-root user with a clear
    # "run as root" message -- BEFORE reading the config or touching the host --
    # rather than a deep, incidental PermissionError (config store or host lock).
    import os
    path = tmp_path / "p.yaml"
    config_mod.save_config(_config(), str(path))
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    called = {"n": 0}
    monkeypatch.setattr(apply_mod, "apply_config",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1))

    rc = cli.main(["apply", "-f", str(path)])

    assert rc == 2
    err = capsys.readouterr().err
    assert "root" in err and "scadactl apply" in err
    assert called["n"] == 0            # never reached the host-mutating step


def test_cli_apply_dry_run_stays_unprivileged(tmp_path, capsys, monkeypatch, inventory):
    # Dry-run is a read-only preview: it must NOT require root.
    import os
    path = tmp_path / "p.yaml"
    config_mod.save_config(_config(), str(path))
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    monkeypatch.setattr(cli, "_load_inventory_best_effort", lambda: inventory)

    rc = cli.main(["apply", "-f", str(path), "--dry-run"])

    out = capsys.readouterr()
    assert rc == 0
    assert "must be run as root" not in out.err


def test_cli_stop_requires_root_for_real_run(capsys, monkeypatch):
    import os
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    stop_calls = {"all": 0, "config": 0}
    monkeypatch.setattr(apply_mod, "stop_all",
                        lambda *a, **k: stop_calls.__setitem__("all", stop_calls["all"] + 1))
    monkeypatch.setattr(apply_mod, "stop_config",
                        lambda *a, **k: stop_calls.__setitem__("config", stop_calls["config"] + 1))

    assert cli.main(["stop", "--all"]) == 2
    assert "root" in capsys.readouterr().err
    assert cli.main(["stop", "--config", "demo"]) == 2
    assert "root" in capsys.readouterr().err
    assert stop_calls == {"all": 0, "config": 0}
