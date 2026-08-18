# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""bin installer: install one executable via a command; probe via check_cmd."""

from __future__ import annotations

from ..outcome import Outcome
from ..recipe import Item, Target
from ..versions import satisfies
from .base import _run_bootstrap, level_for_missing, probe_version, run_cmd


class BinInstaller:
    name = "bin"

    def check(self, item: Item, target: Target) -> list[Outcome]:
        check_cmd = item.spec.get("check_cmd", "")
        actual = probe_version(check_cmd, cwd=target.path) if check_cmd else None
        present = actual is not None
        ok_version = satisfies(actual, item.version)
        if present and ok_version:
            return [Outcome("ok", f"{item.name} present", {"version": actual})]
        level = level_for_missing(item.importance)
        msg = (
            f"{item.name} missing"
            if not present
            else (f"{item.name} {actual} does not satisfy {item.version}")
        )
        return [Outcome(level, msg, {"version": actual, "want": item.version})]

    def plan(self, item: Item, target: Target) -> list[Outcome]:
        if self._satisfied(item, target):
            return [Outcome("ok", f"{item.name} already present")]
        return [Outcome("info", f"would run: {item.spec.get('install', '')}")]

    def install(self, item: Item, target: Target) -> list[Outcome]:
        if self._satisfied(item, target):
            return [Outcome("ok", f"{item.name} already present (skip)")]
        cmd = item.spec.get("install", "")
        rc, out = run_cmd(cmd, cwd=target.path)
        if rc == 0:
            return [Outcome("ok", f"installed {item.name}", {"cmd": cmd})]
        return [
            Outcome(
                level_for_missing(item.importance),
                f"install failed for {item.name}",
                {"cmd": cmd, "output": out},
            )
        ]

    def bootstrap(self, item: Item, target: Target) -> list[Outcome]:
        return _run_bootstrap(item, target)

    def _satisfied(self, item: Item, target: Target) -> bool:
        check_cmd = item.spec.get("check_cmd", "")
        actual = probe_version(check_cmd, cwd=target.path) if check_cmd else None
        return actual is not None and satisfies(actual, item.version)
