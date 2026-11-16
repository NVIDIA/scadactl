# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""backend/binding.py: bind command construction and BOUND-line parsing."""

from __future__ import annotations

import pytest

from scadactl.backend import binding


def test_bind_drives_dry_run_includes_print_result():
    res = binding.bind_drives(["0000:01:00.0"], dry_run=True)
    assert res.ok
    assert "--non-interactive" in res.stdout
    assert "--print-result" in res.stdout


def test_parse_bound_set_maps_pci_to_dev_node():
    stdout = "BOUND 0000:01:00.0 unvme3\nBOUND 0000:02:00.0 unvme5\n"
    bound, unvme_by_pci = binding._parse_bound_set(stdout)
    assert bound == ["0000:01:00.0", "0000:02:00.0"]
    assert unvme_by_pci == {
        "0000:01:00.0": "/dev/unvme3",
        "0000:02:00.0": "/dev/unvme5",
    }


def test_parse_bound_set_empty_falls_back():
    bound, unvme_by_pci = binding._parse_bound_set("")
    assert bound == []
    assert unvme_by_pci == {}


def test_bind_drives_populates_result_from_bound_lines(monkeypatch):
    import subprocess

    def fake_run(cmd, dry_run):
        return subprocess.CompletedProcess(
            cmd, 0, stdout="BOUND 0000:01:00.0 unvme3\n", stderr=""
        )

    monkeypatch.setattr(binding, "_run", fake_run)
    res = binding.bind_drives(["0000:01:00.0"], dry_run=False)
    assert res.ok
    assert res.bound_pci == ["0000:01:00.0"]
    assert res.unvme_by_pci == {"0000:01:00.0": "/dev/unvme3"}


def test_bind_drives_no_bound_lines_yields_empty_set(monkeypatch):
    import subprocess

    # Script exits 0 but emits no BOUND lines (all binds failed via `|| true`).
    # bound_pci must stay EMPTY -- not fall back to the requested set -- so apply's
    # completeness check can tell "nothing confirmed" from "all confirmed".
    def fake_run(cmd, dry_run):
        return subprocess.CompletedProcess(cmd, 0, stdout="no results here\n", stderr="")

    monkeypatch.setattr(binding, "_run", fake_run)
    res = binding.bind_drives(["0000:01:00.0", "0000:02:00.0"], dry_run=False)
    assert res.ok
    assert res.bound_pci == []
    assert res.unvme_by_pci == {}


def test_relative_script_override_is_rejected(monkeypatch):
    monkeypatch.setenv("SCADA_LOAD_SCRIPT", "scripts/scada-load-modules")
    with pytest.raises(ValueError, match="SCADA_LOAD_SCRIPT must be an absolute path"):
        binding.load_script_path()


def test_absolute_script_override_is_accepted(monkeypatch):
    monkeypatch.setenv("SCADA_LOAD_SCRIPT", "/opt/scada/bin/scada-load-modules")
    assert binding.load_script_path() == "/opt/scada/bin/scada-load-modules"
