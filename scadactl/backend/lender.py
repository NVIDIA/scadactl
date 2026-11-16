# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storage Lender endpoint helpers."""

from __future__ import annotations

import os
import stat


def is_unix_socket(path: str) -> bool:
    """Return whether ``path`` exists and is a Unix-domain socket."""
    try:
        return stat.S_ISSOCK(os.stat(path).st_mode)
    except OSError:
        return False
