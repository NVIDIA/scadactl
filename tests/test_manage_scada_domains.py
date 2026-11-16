# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interface checks for the packaged manage-scada-domains command."""

from __future__ import annotations

import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "manage-scada-domains"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SCRIPT), *args], capture_output=True, text=True, check=False
    )


def test_manager_is_executable() -> None:
    assert SCRIPT.is_file()
    assert SCRIPT.stat().st_mode & 0o111


def test_manager_help_describes_public_interface() -> None:
    result = _run("--help")
    assert result.returncode == 0
    assert "launch --config FILE --log-dir DIR" in result.stdout
    assert "stop --log-dir DIR" in result.stdout
    assert "--storage-backend" in result.stdout


def test_subcommand_help_does_not_require_root() -> None:
    for subcommand in ("launch", "stop"):
        result = _run(subcommand, "--help")
        assert result.returncode == 0
        assert "Usage" in result.stdout


def test_manager_rejects_unknown_subcommand() -> None:
    result = _run("unknown")
    assert result.returncode != 0
    assert "unknown subcommand" in result.stderr
