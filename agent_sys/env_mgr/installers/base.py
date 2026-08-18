# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Installer protocol + shared command/version helpers."""

from __future__ import annotations

import re
import subprocess
from typing import Protocol, runtime_checkable

from ..outcome import Outcome
from ..recipe import (
    Item,
    Target,  # noqa: F401  (re-export for installer type hints)
)

_VERSION_RE = re.compile(r"\d+\.\d+(?:\.\d+)?")

_MISSING_LEVEL = {
    "required": "fail",
    "strongly-suggested": "warn",
    "suggested": "info",
}


def run_cmd(cmd: str, cwd: str | None = None) -> tuple[int, str]:
    """Run a shell string; return (returncode, combined stdout+stderr stripped).

    Never raises on non-zero exit.
    """
    proc = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def probe_version(check_cmd: str, cwd: str | None = None) -> str | None:
    rc, out = run_cmd(check_cmd, cwd=cwd)
    if rc != 0:
        return None
    m = _VERSION_RE.search(out)
    return m.group(0) if m else None


def level_for_missing(importance: str) -> str:
    return _MISSING_LEVEL.get(importance, "warn")


@runtime_checkable
class Installer(Protocol):
    name: str

    def check(self, item: Item, target: Target) -> list[Outcome]: ...
    def plan(self, item: Item, target: Target) -> list[Outcome]: ...
    def install(self, item: Item, target: Target) -> list[Outcome]: ...
    def bootstrap(self, item: Item, target: Target) -> list[Outcome]: ...


class ShellInstaller:
    """Shared base for installers that run a shell string from item.spec['run'],
    gated by an optional check_cmd. Subclasses set `name` and may override the
    message hooks to phrase plan/install output differently.
    """

    name = "shell"

    def _satisfied(self, item: Item, target: Target) -> bool:
        check_cmd = item.spec.get("check_cmd")
        if not check_cmd:
            return False
        rc, _ = run_cmd(check_cmd, cwd=target.path)
        return rc == 0

    def _plan_message(self, body: str) -> str:
        return f"would run: {body}"

    def _install_message(self, item: Item, rc: int, body: str) -> str:
        return f"ran: {body}"

    def check(self, item: Item, target: Target) -> list[Outcome]:
        if self._satisfied(item, target):
            return [Outcome("ok", f"{item.name} satisfied")]
        return [Outcome(level_for_missing(item.importance), f"{item.name} not satisfied")]

    def plan(self, item: Item, target: Target) -> list[Outcome]:
        if self._satisfied(item, target):
            return [Outcome("ok", f"{item.name} already satisfied")]
        return [Outcome("info", self._plan_message(item.spec.get("run", "")))]

    def install(self, item: Item, target: Target) -> list[Outcome]:
        if self._satisfied(item, target):
            return [Outcome("ok", f"{item.name} already satisfied (skip)")]
        body = item.spec.get("run", "")
        rc, out = run_cmd(body, cwd=target.path)
        level = "ok" if rc == 0 else level_for_missing(item.importance)
        return [Outcome(level, self._install_message(item, rc, body), {"rc": rc, "output": out})]

    def bootstrap(self, item: Item, target: Target) -> list[Outcome]:
        return _run_bootstrap(item, target)


def _run_bootstrap(item: Item, target: Target) -> list[Outcome]:
    """Run item.spec['bootstrap']: a nested {run:...} or a list of them."""
    boot = item.spec.get("bootstrap")
    if not boot:
        return []
    steps = boot if isinstance(boot, list) else [boot]
    outs: list[Outcome] = []
    for step in steps:
        cmd = step.get("run", "") if isinstance(step, dict) else str(step)
        if not cmd:
            continue
        rc, out = run_cmd(cmd, cwd=target.path)
        level = "ok" if rc == 0 else "warn"
        outs.append(Outcome(level, f"bootstrap: {cmd}", {"rc": rc, "output": out}))
    return outs
