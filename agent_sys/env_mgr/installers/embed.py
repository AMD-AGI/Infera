# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""embed installer: a multi-line script body, optionally gated by check_cmd."""

from __future__ import annotations

from ..recipe import Item
from .base import ShellInstaller


class EmbedInstaller(ShellInstaller):
    name = "embed"

    def _plan_message(self, body: str) -> str:
        return f"would run script:\n{body}"

    def _install_message(self, item: Item, rc: int, body: str) -> str:
        return f"{item.name} script rc={rc}"
