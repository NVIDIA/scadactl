# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Managed config store + config management commands (list/show/rm/import/compat),
apply-by-name, and the kubectl-style -f convention."""

from __future__ import annotations

import json
import os
import stat

import pytest

from scadactl import cli, config as config_mod, store as store_mod
from scadactl.models import Domain, DriveRef, Config


def _config(name="multi"):
    return Config(
        name=name,
        mps_enabled=True,
        server_env={"SCADA_NUM_QUEUE_PAIRS": "8"},
        domains=[
            Domain(name="gpu0_domain", gpu_uuid="GPU-1111", gpu_pci_bus_id="0000:81:00.0",
                   drives=[DriveRef(serial="S111", pci_bus_id="0000:83:00.0")]),
            Domain(name="gpu1_domain", gpu_uuid="GPU-2222", gpu_pci_bus_id="0000:82:00.0",
                   drives=[DriveRef(serial="S222", pci_bus_id="0000:84:00.0")]),
        ],
        domain_groups={"all": ["gpu0_domain", "gpu1_domain"]},
    )


# ---- store module ---- #
def test_store_save_load_list_delete(_isolate_config_store):
    assert store_mod.list_names() == []
    store_mod.save(_config("alpha"))
    store_mod.save(_config("beta"))
    assert store_mod.list_names() == ["alpha", "beta"]
    assert store_mod.exists("alpha")
    loaded = store_mod.load("alpha")
    assert loaded.name == "alpha"
    assert loaded.domains[0].drives[0].serial == "S111"
    store_mod.delete("alpha")
    assert store_mod.list_names() == ["beta"]


def test_store_overwrite_guard(_isolate_config_store):
    store_mod.save(_config("alpha"))
    with pytest.raises(store_mod.StoreError):
        store_mod.save(_config("alpha"))            # exists, no overwrite
    store_mod.save(_config("alpha"), overwrite=True)  # ok


def test_store_load_missing_raises(_isolate_config_store):
    with pytest.raises(store_mod.StoreError):
        store_mod.load("ghost")


def test_save_config_roundtrip(tmp_path):
    path = tmp_path / "cfg.yaml"
    config_mod.save_config(_config("alpha"), str(path))
    loaded = config_mod.load_config(str(path))
    assert loaded.name == "alpha"
    assert loaded.domains[0].drives[0].serial == "S111"


def test_save_config_is_world_readable(tmp_path):
    cfg = _config("test_cfg")
    path = tmp_path / "p.yaml"
    config_mod.save_config(cfg, str(path))
    mode = stat.S_IMODE(os.stat(str(path)).st_mode)
    assert mode == 0o644, oct(mode)
    reloaded = config_mod.load_config(str(path))
    assert reloaded.name == cfg.name


def test_save_config_is_atomic_on_failure(monkeypatch, tmp_path):
    # A crash mid-write must leave the original file intact (never truncated) and
    # no temp file behind: save_config writes to a temp file + os.replace, so a
    # dump error means os.replace never runs and the original survives.
    path = tmp_path / "cfg.yaml"
    config_mod.save_config(_config("orig"), str(path))
    original = path.read_text()

    def _boom(_config):
        raise RuntimeError("dump boom")

    monkeypatch.setattr(config_mod, "dump_config", _boom)
    with pytest.raises(RuntimeError, match="dump boom"):
        config_mod.save_config(_config("replacement"), str(path))

    assert path.read_text() == original                       # original untouched
    assert [p.name for p in tmp_path.iterdir()] == ["cfg.yaml"]  # no .config.*.tmp left


def test_config_created_modified_roundtrip():
    p = _config()
    p.created = "2026-06-27 10:00:00"
    p.modified = "2026-06-27 11:00:00"
    p2 = Config.from_dict(p.to_dict())
    assert p2.created == "2026-06-27 10:00:00"
    assert p2.modified == "2026-06-27 11:00:00"


# ---- CLI: list / show / rm / import ---- #
def _stub_inv(monkeypatch, inventory):
    monkeypatch.setattr("scadactl.cli.inventory_mod.collect_inventory", lambda: inventory)
    monkeypatch.setattr("scadactl.cli._load_inventory_best_effort", lambda: inventory)


def test_cli_config_list_json(monkeypatch, capsys, inventory):
    store_mod.save(_config("alpha"))
    rc = cli.main(["config", "list", "--json"])
    assert rc == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["name"] == "alpha"
    assert rows[0]["domains"] == 2 and rows[0]["drives"] == 2 and rows[0]["mps"] is True


def test_cli_config_show_by_name_and_file(monkeypatch, capsys, inventory, tmp_path):
    _stub_inv(monkeypatch, inventory)
    store_mod.save(_config("alpha"))
    assert cli.main(["config", "show", "alpha"]) == 0
    assert "Config: alpha" in capsys.readouterr().out
    f = tmp_path / "x.yaml"
    config_mod.save_config(_config("fromfile"), str(f))
    assert cli.main(["config", "show", "-f", str(f)]) == 0
    assert "Config: fromfile" in capsys.readouterr().out


def test_cli_config_show_name_and_file_mutually_exclusive(monkeypatch, capsys, inventory, tmp_path):
    _stub_inv(monkeypatch, inventory)
    store_mod.save(_config("alpha"))
    f = tmp_path / "x.yaml"
    config_mod.save_config(_config("alpha"), str(f))
    rc = cli.main(["config", "show", "alpha", "-f", str(f)])
    assert rc == 2
    assert "not both" in capsys.readouterr().err


def test_cli_config_rm(capsys):
    store_mod.save(_config("alpha"))
    assert cli.main(["config", "rm", "alpha"]) == 0
    assert not store_mod.exists("alpha")
    assert cli.main(["config", "rm", "ghost"]) == 2  # missing


def test_cli_config_import(tmp_path, capsys):
    f = tmp_path / "x.yaml"
    config_mod.save_config(_config("imported"), str(f))
    assert cli.main(["config", "import", str(f)]) == 0
    assert store_mod.exists("imported")
    stored = store_mod.load("imported")
    assert stored.modified  # import stamps modified
    # re-import without --force fails, with --force succeeds
    assert cli.main(["config", "import", str(f)]) == 2
    assert cli.main(["config", "import", str(f), "--force"]) == 0


# ---- CLI: apply by name / -f, dry-run ---- #
def test_cli_apply_by_name_dry_run(monkeypatch, capsys, inventory, runtime_dir):
    _stub_inv(monkeypatch, inventory)
    store_mod.save(_config("alpha"))
    rc = cli.main(["apply", "alpha", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "gpu0_domain" in out


def test_cli_apply_dry_run_commands(monkeypatch, capsys, inventory, runtime_dir):
    _stub_inv(monkeypatch, inventory)
    store_mod.save(_config("alpha"))
    rc = cli.main(["apply", "alpha", "--dry-run", "--commands"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Commands:" in out
    assert "scada-load-modules --non-interactive" in out
    assert "manage-scada-domains launch" in out
    assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in out


def test_cli_apply_unknown_name(monkeypatch, capsys, inventory, runtime_dir):
    _stub_inv(monkeypatch, inventory)
    assert cli.main(["apply", "ghost", "--dry-run"]) == 2


# ---- CLI: compat ---- #
def test_cli_compat_compatible(monkeypatch, capsys, inventory):
    _stub_inv(monkeypatch, inventory)
    store_mod.save(Config(name="a", domains=[
        Domain(name="a0", gpu_uuid="GPU-1111", gpu_pci_bus_id="0000:81:00.0",
               drives=[DriveRef(serial="S111", pci_bus_id="0000:83:00.0")])]))
    store_mod.save(Config(name="b", domains=[
        Domain(name="b0", gpu_uuid="GPU-2222", gpu_pci_bus_id="0000:82:00.0",
               drives=[DriveRef(serial="S222", pci_bus_id="0000:84:00.0")])]))
    rc = cli.main(["config", "compat", "a", "b"])
    assert rc == 0
    assert "compatible" in capsys.readouterr().out


def test_cli_compat_conflict(monkeypatch, capsys, inventory):
    _stub_inv(monkeypatch, inventory)
    dom = Domain(name="x0", gpu_uuid="GPU-1111", gpu_pci_bus_id="0000:81:00.0",
                 drives=[DriveRef(serial="S111", pci_bus_id="0000:83:00.0")])
    store_mod.save(Config(name="a", domains=[dom]))
    store_mod.save(Config(name="b", domains=[dom]))
    rc = cli.main(["config", "compat", "a", "b"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "conflict" in out and "GPU-1111" in out


def test_cli_compat_needs_two(capsys):
    store_mod.save(_config("only"))
    assert cli.main(["config", "compat", "only"]) == 2


def test_store_rejects_path_traversal_names():
    # A config name is a bare filename; traversal/separators are refused so
    # `config show '../policy'` can't read (nor `config rm` delete) files
    # outside the store.
    for bad in ("../policy", "a/b", "..", ".", "", "foo/../bar", "x\x00y", "sub/dir/x"):
        with pytest.raises(store_mod.StoreError):
            store_mod._path(bad)
    # An ordinary name is accepted and stays inside the store.
    assert store_mod._path("scada").endswith("/scada.yaml")


def test_cli_config_rm_rejects_traversal(capsys):
    rc = cli.main(["config", "rm", "../policy"])
    assert rc == 2
    assert "invalid config name" in capsys.readouterr().err


def test_stop_all_and_config_mutually_exclusive():
    # argparse rejects both at parse time (exit 2).
    with pytest.raises(SystemExit) as ei:
        cli.main(["stop", "--all", "--config", "x"])
    assert ei.value.code == 2


def test_stop_handler_rejects_all_and_config(capsys):
    import types
    args = types.SimpleNamespace(all=True, config="x", dry_run=False)
    assert cli.cmd_stop(args) == 2
    assert "mutually exclusive" in capsys.readouterr().err


def test_add_domain_invalid_file_errors(monkeypatch, capsys, tmp_path, inventory):
    monkeypatch.setattr(
        "scadactl.cli.inventory_mod.collect_inventory", lambda: inventory
    )
    bad = tmp_path / "bad.yaml"
    bad.write_text("key: [unclosed\n")  # malformed YAML
    rc = cli.main(["config", "add-domain", "--name", "d", "--gpu", "gpu0",
                   "--drives", "nvme0", "-f", str(bad)])
    assert rc == 2
    assert "error:" in capsys.readouterr().err




def test_list_names_surfaces_permission_error(_isolate_config_store, monkeypatch):
    # A store dir that exists but cannot be listed (wrong mode) must NOT be masked
    # as "no configs" -- that contradicts a `config init` that still sees a config.
    # (Tested via monkeypatch because CI runs as root, which bypasses file perms.)
    def _boom(_d):
        raise PermissionError(13, "Permission denied")
    monkeypatch.setattr(store_mod.os, "listdir", _boom)
    with pytest.raises(store_mod.StoreError) as ei:
        store_mod.list_names()
    assert "world-readable" in str(ei.value)


def test_list_names_absent_dir_is_empty(_isolate_config_store, monkeypatch):
    # A genuinely-absent store directory is normal -> [] (not an error).
    def _gone(_d):
        raise FileNotFoundError(2, "No such file or directory")
    monkeypatch.setattr(store_mod.os, "listdir", _gone)
    assert store_mod.list_names() == []


def test_cli_config_list_surfaces_unreadable_config(_isolate_config_store, monkeypatch, capsys):
    # A config present in the store but unreadable (e.g. a root-only 0600 file
    # written by an older build) must be surfaced, not silently dropped from the
    # listing while `config init` still reports it exists.
    store_mod.save(_config("readable"))
    store_mod.save(_config("broken"))
    real_load = store_mod.load

    def _load(name):
        if name == "broken":
            raise store_mod.StoreError(
                "cannot read stored config 'broken': [Errno 13] Permission denied"
            )
        return real_load(name)

    monkeypatch.setattr(store_mod, "load", _load)
    rc = cli.main(["config", "list"])
    out = capsys.readouterr()
    assert rc == 0
    assert "readable" in out.out                        # the good one still lists
    assert "broken" in out.err and "could not be read" in out.err  # bad one surfaced


def test_save_makes_store_dir_world_readable(_isolate_config_store):
    # Even if the store dir was created with a restrictive mode (umask 077),
    # save() must leave it world-readable so unprivileged `config list` works.
    d = _isolate_config_store
    os.chmod(d, 0o700)
    store_mod.save(_config("x"))
    mode = stat.S_IMODE(os.stat(d).st_mode)
    assert mode == 0o755, oct(mode)
