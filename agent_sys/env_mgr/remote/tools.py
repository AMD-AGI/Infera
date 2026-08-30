# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The tool-call surface. Design §10.3, criterion 18.

Spec §5.5: the whole remote↔local surface is exposed to agents as **tool calls**,
not as a procedure described in prose, because *"an agent given a
natural-language description of how to sync a directory will improvise, and the
improvisation will be wrong in a way nobody notices"*. A tool call has a schema,
a name, and a result.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, NamedTuple

from env_mgr.fs.path import contained
from env_mgr.fs.zone import Zone
from env_mgr.remote.connection import Connection

__all__ = ["ToolDef", "tools"]


class ToolDef(NamedTuple):
    name: str
    description: str
    schema: dict[str, Any]  # JSON Schema for the arguments object
    call: Callable[..., Any]


def _inside(zone: Zone, rel: str) -> str:
    """Every path argument is resolved **relative to the zone**, and checked.

    Closing over the zone is what makes criterion 10 true on this surface too:
    the zone root is never taken from agent-supplied input, because the tool does
    not accept one.
    """
    if os.path.isabs(rel):
        raise PermissionError(f"{rel!r} is absolute; paths are relative to the zone")
    path = os.path.join(zone.root, rel)
    if not contained(path, zone.root):
        raise PermissionError(f"{rel!r} resolves outside zone {zone.root!r}")
    return path


def tools(conn: Connection, zone: Zone) -> tuple[ToolDef, ...]:
    """Three tools, each closed over this attempt's zone."""

    def env_remote_run(command: list[str], cwd: str = "") -> dict[str, Any]:
        proc = conn.run(command, cwd=_inside(zone, cwd) if cwd else zone.root)
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}

    def env_remote_push(path: str, remote: str) -> dict[str, Any]:
        report = conn.push(_inside(zone, path), remote)
        return report._asdict()

    def env_remote_pull(remote: str, path: str) -> dict[str, Any]:
        report = conn.pull(remote, _inside(zone, path))
        return report._asdict()

    return (
        ToolDef(
            name="env_remote_run",
            description="Run a command on the remote side of this task's mapping.",
            schema={
                "type": "object",
                "properties": {
                    "command": {"type": "array", "items": {"type": "string"}},
                    "cwd": {
                        "type": "string",
                        "description": "relative to the zone; absolute paths are refused",
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
            call=env_remote_run,
        ),
        ToolDef(
            name="env_remote_push",
            description="Copy a path from this zone to the remote side.",
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "relative to the zone"},
                    "remote": {"type": "string"},
                },
                "required": ["path", "remote"],
                "additionalProperties": False,
            },
            call=env_remote_push,
        ),
        ToolDef(
            name="env_remote_pull",
            description="Copy a path from the remote side into this zone.",
            schema={
                "type": "object",
                "properties": {
                    "remote": {"type": "string"},
                    "path": {"type": "string", "description": "relative to the zone"},
                },
                "required": ["remote", "path"],
                "additionalProperties": False,
            },
            call=env_remote_pull,
        ),
    )
