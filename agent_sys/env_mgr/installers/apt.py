# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""apt installer: detect packages, PRINT the apt-get line. Never sudo."""

from __future__ import annotations

from ..outcome import Outcome
from ..recipe import Item, Target
from .base import _run_bootstrap, level_for_missing, run_cmd


class AptInstaller:
    name = "apt"

    def _missing(self, item: Item) -> list[str]:
        provides = item.spec.get("provides", {})
        missing = []
        for pkg in item.spec.get("packages", []):
            cmd = provides.get(pkg) if isinstance(provides, dict) else None
            if cmd:
                rc, _ = run_cmd(f"command -v {cmd}")
            else:
                rc, _ = run_cmd(f"dpkg -s {pkg}")
            if rc != 0:
                missing.append(pkg)
        return missing

    def check(self, item: Item, target: Target) -> list[Outcome]:
        missing = self._missing(item)
        if not missing:
            return [Outcome("ok", "all apt packages present")]
        return [
            Outcome(
                level_for_missing(item.importance),
                f"missing apt packages: {', '.join(missing)}",
                {"missing": missing, "apt_get": f"sudo apt-get install -y {' '.join(missing)}"},
            )
        ]

    def plan(self, item: Item, target: Target) -> list[Outcome]:
        missing = self._missing(item)
        if not missing:
            return [Outcome("ok", "all apt packages present")]
        line = f"sudo apt-get install -y {' '.join(missing)}"
        return [Outcome("info", f"apt-get install needed: {line}", {"apt_get": line})]

    def install(self, item: Item, target: Target) -> list[Outcome]:
        # v1: never sudo. Same behavior as plan — print, don't run.
        return self.plan(item, target)

    def bootstrap(self, item: Item, target: Target) -> list[Outcome]:
        return _run_bootstrap(item, target)
