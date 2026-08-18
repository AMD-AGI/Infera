# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""oneline installer: a single-line run command, optionally gated by check_cmd."""

from __future__ import annotations

from .base import ShellInstaller


class OnelineInstaller(ShellInstaller):
    name = "oneline"
