# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SCADA orchestration CLI package.

This package provides the ``scadactl`` command-line control plane for configuring,
launching, validating, and using SCADA on hosts with GPUs and NVMe storage.

The package is importable on machines with no GPU and no root: hardware-,
driver-, and root-dependent functionality (NVML inventory, drive binding,
domain launch, container ordinal mapping) is implemented against real
interfaces but lazy-imported and guarded so the hardware-free commands and the
unit tests run anywhere.
"""

from scadactl.models import (
    CONFIG_SCHEMA_VERSION,
    REGISTRY_SCHEMA_VERSION,
)

__all__ = ["CONFIG_SCHEMA_VERSION", "REGISTRY_SCHEMA_VERSION", "__version__"]

__version__ = "0.6.0"
