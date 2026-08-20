# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""uv installer: ref form (uv pip install -e) and tool form (uv tool install)."""

from __future__ import annotations

from pathlib import Path

from ..outcome import Outcome
from ..recipe import Item, Target
from .base import _run_bootstrap, level_for_missing, probe_version, run_cmd


class UvInstaller:
    name = "uv"

    def _is_tool(self, item: Item) -> bool:
        return "tool" in item.spec

    def check(self, item: Item, target: Target) -> list[Outcome]:
        if probe_version("uv --version") is None:
            return [Outcome(level_for_missing(item.importance), "uv not on PATH")]
        if self._is_tool(item):
            prov = item.spec.get("provides", "")
            rc, _ = run_cmd(f"command -v {prov}") if prov else (1, "")
            if rc == 0:
                return [Outcome("ok", f"{prov} present (uv tool)")]
            return [Outcome(level_for_missing(item.importance), f"{prov} missing (uv tool)")]
        ref = item.spec.get("ref", "")
        if not ref:
            return [Outcome("warn", f"{item.name}: uv item declares neither 'ref' nor 'tool'")]
        if not (Path(target.path) / ref).exists():
            return [
                Outcome(
                    level_for_missing(item.importance), f"ref {ref} not found under {target.path}"
                )
            ]
        return [Outcome("ok", f"uv ref {ref} present")]

    def plan(self, item: Item, target: Target) -> list[Outcome]:
        # ref form: preview via uv's own --dry-run instead of a hand-written
        # string. This is read-only (no mutation) and shows uv's own "already
        # satisfied / would install" view, so plan never needs to duplicate
        # uv's logic. `uv pip install --dry-run` is supported.
        #
        # tool form: `uv tool install` has no --dry-run flag (it errors on
        # unknown argument), so plan is a static, non-executing preview here.
        if self._is_tool(item):
            cmd = self._cmd(item)
            return [
                Outcome(
                    "info",
                    f"would run: {cmd}",
                    {
                        "cmd": cmd,
                        "dry_run": False,
                        "note": "uv tool install has no dry-run flag; static preview only",
                    },
                )
            ]
        cmd = f"{self._cmd(item)} --dry-run"
        rc, out = run_cmd(cmd, cwd=target.path)
        return [Outcome("info", f"dry-run: {cmd}", {"rc": rc, "output": out})]

    def install(self, item: Item, target: Target) -> list[Outcome]:
        # No hand-rolled "already satisfied" gate here: uv itself is idempotent
        # (an already-installed package/tool is left as-is unless
        # --reinstall/--force is passed), so re-running the install command is
        # already a safe no-op.
        cmd = self._cmd(item)
        rc, out = run_cmd(cmd, cwd=target.path)
        level = "ok" if rc == 0 else level_for_missing(item.importance)
        return [Outcome(level, f"ran: {cmd}", {"rc": rc, "output": out})]

    def bootstrap(self, item: Item, target: Target) -> list[Outcome]:
        return _run_bootstrap(item, target)

    def _cmd(self, item: Item) -> str:
        if self._is_tool(item):
            return f"uv tool install {item.spec['tool']}"
        extras = item.spec.get("extras", [])
        spec = ".[" + ",".join(extras) + "]" if extras else "."
        return f"uv pip install -e {spec}"
