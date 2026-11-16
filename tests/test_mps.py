# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""backend/mps.py: is_running() scoped to the resolved CUDA_MPS_PIPE_DIRECTORY."""

from __future__ import annotations

import subprocess

from scadactl.backend import mps


def _fake_pgrep(pids):
    def _run(cmd, **kw):
        return subprocess.CompletedProcess(
            cmd, 0 if pids else 1, stdout="\n".join(pids), stderr=""
        )
    return _run


def test_is_running_true_only_for_our_pipe_dir(monkeypatch):
    monkeypatch.setattr(mps, "_proc_args", lambda pid: ["nvidia-cuda-mps-control", "-d"])
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", "/run/mps-a")
    monkeypatch.setattr(subprocess, "run", _fake_pgrep(["101", "102"]))
    # 101 serves a different pipe dir, 102 serves ours.
    dirs = {"101": "/run/mps-other", "102": "/run/mps-a"}
    monkeypatch.setattr(mps, "_proc_pipe_dir", lambda pid: dirs.get(pid))
    assert mps.is_running() is True


def test_is_running_false_when_only_other_pipe_dirs(monkeypatch):
    monkeypatch.setattr(mps, "_proc_args", lambda pid: ["nvidia-cuda-mps-control", "-d"])
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", "/run/mps-a")
    monkeypatch.setattr(subprocess, "run", _fake_pgrep(["101"]))
    monkeypatch.setattr(mps, "_proc_pipe_dir", lambda pid: "/run/mps-other")
    assert mps.is_running() is False


def test_is_running_false_when_no_process(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_pgrep([]))
    assert mps.is_running() is False
