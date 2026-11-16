# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Privileged backend wrappers around installed system capabilities.

Each wrapper isolates a host-changing operation behind a narrow Python API so
the rest of the orchestrator stays read-only/unprivileged. The wrappers
invoke capabilities by their installed paths.
"""
