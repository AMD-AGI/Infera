# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""claude installer: install Claude Code plugins."""

from __future__ import annotations

from ..outcome import Outcome
from ..recipe import Item, Target
from .base import _run_bootstrap, level_for_missing, run_cmd


class ClaudeInstaller:
    name = "claude"

    def _installed(self) -> tuple[bool, str]:
        rc, out = run_cmd("claude plugin list")
        return rc == 0, out

    @staticmethod
    def _present_names(out: str) -> set[str]:
        # `claude plugin list` prints one plugin per line, name first. Match the
        # bare name of each non-empty line exactly, so a prefix name (e.g.
        # "super") is not falsely matched against "superpowers".
        names = set()
        for line in out.splitlines():
            line = line.strip()
            if line:
                names.add(line.split()[0])
        return names

    def check(self, item: Item, target: Target) -> list[Outcome]:
        ok, out = self._installed()
        if not ok:
            return [Outcome(level_for_missing(item.importance), "claude CLI not available")]
        present = self._present_names(out)
        missing = [p for p in item.spec.get("plugins", []) if p.split("@")[0] not in present]
        if missing:
            return [
                Outcome(
                    level_for_missing(item.importance),
                    f"missing plugins: {', '.join(missing)}",
                    {"missing": missing},
                )
            ]
        return [Outcome("ok", "all claude plugins present")]

    def plan(self, item: Item, target: Target) -> list[Outcome]:
        return [Outcome("info", f"would install plugins: {item.spec.get('plugins', [])}")]

    def install(self, item: Item, target: Target) -> list[Outcome]:
        ok, out = self._installed()
        if not ok:
            return [Outcome(level_for_missing(item.importance), "claude CLI not available")]
        present = self._present_names(out)
        outs: list[Outcome] = []
        for spec in item.spec.get("plugins", []):
            if spec.split("@")[0] in present:
                outs.append(Outcome("ok", f"{spec} already installed"))
                continue
            rc, o = run_cmd(f"claude plugin install {spec}")
            outs.append(
                Outcome("ok" if rc == 0 else "warn", f"plugin {spec} rc={rc}", {"output": o})
            )
        return outs

    def bootstrap(self, item: Item, target: Target) -> list[Outcome]:
        return _run_bootstrap(item, target)
