# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""backend/launcher.py: per-domain log-path normalization."""

from __future__ import annotations

import os

import pytest

from scadactl.backend import launcher


def test_domain_log_dir_normal():
    assert launcher.domain_log_dir("cfg", "gpu0_domain") == (
        "/var/log/scada/cfg/gpu0_domain"
    )


def _contained(d):
    # A proper boundary check: startswith("/var/log/scada") would also accept a
    # sibling like /var/log/scada2, so compare on path components instead.
    root = os.path.realpath("/var/log/scada")
    return os.path.commonpath([os.path.realpath(d), root]) == root


def test_domain_log_dir_rejects_traversal():
    # config/domain are unconstrained strings; separators or .. must not escape
    # LOG_ROOT when makedirs runs as root at launch.
    d = launcher.domain_log_dir("../../etc", "..")
    assert d.startswith("/var/log/scada/")
    assert _contained(d)


def test_domain_log_dir_is_injective():
    # A component with a separator is percent-encoded to a single segment rather
    # than collapsed to its basename, so distinct names never share one log dir
    # (and thus never share a pidfile / overwrite each other's logs).
    assert launcher.domain_log_dir("a/b", "c/d") == "/var/log/scada/a%2Fb/c%2Fd"
    assert launcher.domain_log_dir("foo/bar", "d") != launcher.domain_log_dir("bar", "d")
    assert launcher.domain_log_dir("a", "x/y") != launcher.domain_log_dir("a", "y")
    assert _contained(launcher.domain_log_dir("foo/bar", "x/y"))


def _launch_cmd(**kw):
    return launcher.launch_domain(
        config="c", domain="d0", host_cuda_index="3",
        unvme_devices=["/dev/unvme0"], resolved_server_env={},
        dry_run=True, **kw,
    ).cmd


def test_launch_forwards_socket_group_as_group_flag():
    # socket_group must reach manage-scada-domains as `--group NAME`, else the
    # server creates a root-owned socket and unprivileged clients cannot connect.
    cmd = _launch_cmd(socket_group="scada-users")
    assert "--group" in cmd
    assert cmd[cmd.index("--group") + 1] == "scada-users"


def test_launch_omits_group_flag_when_unset():
    # No socket_group -> no --group (server default: root-owned socket).
    assert "--group" not in _launch_cmd()
    assert "--group" not in _launch_cmd(socket_group=None)


def test_launch_forwards_profiler_socket():
    cmd = _launch_cmd(profiler_socket="/var/run/scada/scadad.profiler.gpu3.sock")
    assert "--profiler-socket" in cmd
    assert cmd[cmd.index("--profiler-socket") + 1] == "/var/run/scada/scadad.profiler.gpu3.sock"


def test_lender_launch_uses_the_packaged_binary_and_backend_flag():
    cmd = launcher.launch_domain(
        config="c", domain="d0", host_cuda_index="3",
        unvme_devices=["0000:3c:00.0"], resolved_server_env={},
        storage_backend="lender", dry_run=True,
    ).cmd
    assert cmd[cmd.index("--storage-backend") + 1] == "lender"
    assert cmd[cmd.index("--domain-binary") + 1] == "scada-swap-domain"


def test_manage_script_requires_absolute_override(monkeypatch):
    monkeypatch.setenv("SCADA_MANAGE_SCRIPT", "scripts/manage-scada-domains")
    with pytest.raises(ValueError, match="SCADA_MANAGE_SCRIPT must be an absolute path"):
        launcher.manage_script_path()


def test_manage_script_accepts_absolute_override(monkeypatch):
    monkeypatch.setenv("SCADA_MANAGE_SCRIPT", "/opt/scada/manage-scada-domains")
    assert launcher.manage_script_path() == "/opt/scada/manage-scada-domains"
