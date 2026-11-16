# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small, width-bounded tables for human-readable command output."""

from __future__ import annotations

import textwrap
from typing import Sequence


def render_table(columns: Sequence[tuple[str, int]], rows: Sequence[Sequence[object]]) -> list[str]:
    """Wrap long cells instead of letting them shift later columns."""
    widths = [max(len(heading), width) for heading, width in columns]
    lines = [" ".join(heading.ljust(width) for (heading, _), width in zip(columns, widths)).rstrip()]
    for row in rows:
        cells = [
            textwrap.wrap(str(value) if value is not None else "", width, break_on_hyphens=False) or [""]
            for value, width in zip(row, widths)
        ]
        for i in range(max(map(len, cells))):
            lines.append(" ".join((cell[i] if i < len(cell) else "").ljust(width)
                                  for cell, width in zip(cells, widths)).rstrip())
    return lines
