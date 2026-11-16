# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Non-root socket access and explicit multiuser MPS setup."""

import subprocess

import pytest

from scadactl import apply as apply_mod, config as config_mod, reconcile, registry, wizard
from scadactl.backend import mps
from scadactl.models import Config, ConfigEntry, Domain, DriveRef, Registry, RegistryDomain


def _config():
    return Config(
        name="qa",
        mps_enabled=True,
        domains=[
            Domain(
                name="d",
                gpu_uuid="GPU-1111",
                gpu_pci_bus_id="0000:81:00.0",
                drives=[DriveRef(serial="S111", pci_bus_id="0000:83:00.0")],
            )
        ],
    )


def test_old_config_defaults_to_single_user():
    assert not Config.from_dict({"name": "old", "mps": {"enabled": True}}).mps_multiuser
    assert not ConfigEntry.from_dict({"name": "old", "mps": True}).mps_multiuser


def test_config_and_registry_roundtrip(inventory):
    cfg = _config()
    cfg.mps_multiuser = True
    cfg.domains[0].socket_group = "users"
    assert Config.from_dict(cfg.to_dict()) == cfg
    rd = apply_mod.build_registry_domains(cfg, inventory)[0]
    assert RegistryDomain.from_dict(rd.to_dict()).socket_group == "users"
    reg = registry.reconcile_add_config(Registry(), cfg.name, [rd], mps_enabled=True, mps_multiuser=True)
    assert Registry.from_dict(reg.to_dict()).configs[0].mps_multiuser


@pytest.mark.parametrize("value", ["false", 1, None])
def test_multiuser_requires_boolean(value):
    cfg = Config.from_dict({"name": "qa", "mps": {"enabled": True, "multiuser": value}})
    assert "mps.multiuser must be a boolean" in config_mod.validate_config(cfg).errors


def test_multiuser_requires_enabled():
    cfg = Config(name="qa", mps_multiuser=True)
    assert "mps.multiuser requires mps.enabled: true" in config_mod.validate_config(cfg).errors


@pytest.mark.parametrize("change", ["group", "mode", "enabled"])
def test_active_changes_require_stop(inventory, change):
    cfg = _config()
    rd = apply_mod.build_registry_domains(cfg, inventory)
    reg = registry.reconcile_add_config(Registry(), cfg.name, rd, mps_enabled=True)
    if change == "group":
        cfg.domains[0].socket_group = "users"
    elif change == "mode":
        cfg.mps_multiuser = True
    else:
        cfg.mps_enabled = False
    plan = reconcile.plan_reconciliation(cfg, inventory, reg, apply_mod.build_registry_domains(cfg, inventory))
    assert not plan.ok
    assert "stop it" in plan.errors[0]


def _daemon(monkeypatch, args):
    monkeypatch.setattr(mps.os, "geteuid", lambda: 0)
    monkeypatch.setattr(mps.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "101", ""))
    monkeypatch.setattr(mps, "_proc_pipe_dir", lambda pid: mps.pipe_directory())
    monkeypatch.setattr(mps, "_proc_args", lambda pid: args)


@pytest.mark.parametrize("flag", ["-M", "--multiuser-server", "-multiuser-server"])
def test_compatible_multiuser_daemon_is_reused(monkeypatch, flag):
    _daemon(monkeypatch, ["/usr/bin/nvidia-cuda-mps-control", "-d", flag])
    assert mps.start(multiuser=True).ok


@pytest.mark.parametrize("requested", [False, True])
def test_mode_mismatch_is_rejected(monkeypatch, requested):
    args = ["nvidia-cuda-mps-control", "-d"] + ([] if requested else ["-M"])
    _daemon(monkeypatch, args)
    result = mps.start(multiuser=requested)
    assert not result.ok and "requested" in result.stderr


def test_unreadable_daemon_mode_is_rejected(monkeypatch):
    _daemon(monkeypatch, None)
    assert "cannot determine" in mps.compatibility_error(True)


def test_shell_mentioning_mps_is_not_a_daemon(monkeypatch):
    _daemon(monkeypatch, ["bash", "-c", "nvidia-cuda-mps-control -d"])
    assert not mps.is_running()
    assert mps.compatibility_error(True) is None


def test_start_multiuser_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(mps.os, "geteuid", lambda: 0)
    monkeypatch.setattr(mps, "compatibility_error", lambda mode: None)
    monkeypatch.setattr(mps, "is_running", lambda: False)
    monkeypatch.setattr(mps.os, "makedirs", lambda *a, **kw: None)
    monkeypatch.setattr(
        mps.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", "")
    )
    assert mps.start(multiuser=True).ok
    assert calls == [[mps.control_binary(), "-d", "-multiuser-server"]]
    assert "-multiuser-server" in mps.start(dry_run=True, multiuser=True).stdout


def test_nonroot_multiuser_is_rejected(monkeypatch):
    monkeypatch.setattr(mps.os, "geteuid", lambda: 2524)
    assert "as root" in mps.compatibility_error(True)


def test_missing_pgrep_blocks_start_with_setup_error(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("pgrep")

    monkeypatch.setattr(mps.subprocess, "run", missing)
    result = mps.start()
    assert not result.ok
    assert "install procps" in result.stderr


def test_pgrep_failure_blocks_start(monkeypatch):
    monkeypatch.setattr(
        mps.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 2, "", "process inspection failed")
    )
    result = mps.start()
    assert not result.ok and "process inspection failed" in result.stderr


def test_composition_rejects_different_mps_modes():
    result = config_mod.validate_composition(
        [
            Config(name="a", mps_enabled=True),
            Config(name="b", mps_enabled=True, mps_multiuser=True),
        ]
    )
    assert not result.ok and "incompatible MPS modes" in result.errors[0]


def test_active_other_config_mode_conflict():
    reg = Registry(active_configs=["other"], configs=[ConfigEntry(name="other", mps_enabled=True)])
    result = config_mod.validate_config(Config(name="qa", mps_enabled=True, mps_multiuser=True), registry=reg)
    assert any("different MPS mode" in error for error in result.errors)


def test_incompatible_daemon_blocks_before_binding(runtime_dir, inventory, monkeypatch):
    cfg = _config()
    cfg.mps_multiuser = True
    monkeypatch.setattr(mps, "compatibility_error", lambda mode: "incompatible daemon")
    monkeypatch.setattr(apply_mod.binding, "bind_drives", lambda *a, **kw: pytest.fail("must not bind"))
    result = apply_mod.apply_config(cfg, inventory)
    assert result.errors == ["incompatible daemon"]


def test_wizard_group_create_edit_and_summary(inventory, monkeypatch, capsys):
    w = wizard.ConfigWizard(inventory, "qa")
    monkeypatch.setattr(w, "_select_drives", lambda *a, **kw: [inventory.drives[0]])
    monkeypatch.setattr(config_mod, "_group_exists", lambda group: group == "users")
    answers = iter(["missing", "users"])
    monkeypatch.setattr(wizard, "_prompt", lambda *a, **kw: next(answers))
    w._add_domain_for_gpu(inventory.gpus[0])
    assert w.config.domains[0].socket_group == "users"
    w.config.domains[0].server_env = {"SCADA_NUM_QUEUE_PAIRS": "16"}
    answers = iter(["gpu0", "users"])
    w._modify_domain()
    assert w.config.domains[0].socket_group == "users"
    assert w.config.domains[0].server_env == {"SCADA_NUM_QUEUE_PAIRS": "16"}
    w._print_summary()
    assert "Socket group: users" in capsys.readouterr().out


def test_wizard_multiuser_prompt_and_disable(inventory, monkeypatch):
    w = wizard.ConfigWizard(inventory, "qa")
    answers = iter([True, True, False])
    monkeypatch.setattr(wizard, "_prompt_yes_no", lambda *a, **kw: next(answers))
    w._mps()
    assert w.config.mps_enabled and w.config.mps_multiuser
    w._mps()
    assert not w.config.mps_enabled and not w.config.mps_multiuser


def test_wizard_can_remove_socket_group(inventory, monkeypatch):
    w = wizard.ConfigWizard(inventory, "qa")
    monkeypatch.setattr(wizard, "_prompt", lambda *a, **kw: "-")
    assert w._socket_group("users") is None


@pytest.mark.parametrize("socket_group", [None, "users"])
def test_new_wizard_defaults_to_multiuser(inventory, monkeypatch, socket_group):
    w = wizard.ConfigWizard(inventory, "qa")
    w.config.domains = _config().domains
    w.config.domains[0].socket_group = socket_group
    defaults = []
    monkeypatch.setattr(wizard, "_prompt_yes_no", lambda text, default: defaults.append(default) or default)
    w._mps()
    assert defaults == [True, True]
    assert w.config.mps_enabled and w.config.mps_multiuser


def test_edit_existing_single_user_preserves_mode_default(inventory, monkeypatch):
    w = wizard.ConfigWizard(inventory, "qa", config=_config())
    monkeypatch.setattr(wizard, "_prompt_yes_no", lambda text, default: default)
    w._mps()
    assert w.config.mps_enabled and not w.config.mps_multiuser


def test_enabling_mps_defaults_to_multiuser(inventory, monkeypatch):
    w = wizard.ConfigWizard(inventory, "qa", config=Config(name="qa"))
    monkeypatch.setattr(wizard, "_prompt_yes_no", lambda text, default: True if text == "Enable MPS?" else default)
    w._mps()
    assert w.config.mps_enabled and w.config.mps_multiuser
