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

from env_mgr.fs.path import contained, contained_syntactically
from env_mgr.fs.zone import Zone
from env_mgr.remote.connection import Connection

__all__ = ["ToolDef", "tools"]


class ToolDef(NamedTuple):
    name: str
    description: str
    schema: dict[str, Any]  # JSON Schema for the arguments object
    call: Callable[..., Any]


def _inside(zone: Zone, rel: str) -> str:
    """A path argument on **this** machine, resolved under the zone and checked.

    Closing over the zone is what makes criterion 10 true on this surface too:
    the zone root is never taken from agent-supplied input, because the tool does
    not accept one.

    **The message is read by a model.** It is `str(e)` on an `isError` tool
    result — measured, `scratch/single-real-task-2026-08/c_probe_tool_refusal_visible.py`
    — so it has to say what the agent should do differently, and it must name a
    path that exists on the side the argument was about. This one is about the
    local zone, which is the agent's own `cwd`, so naming it is actionable.
    """
    if os.path.isabs(rel):
        raise PermissionError(
            f"{rel!r} is an absolute path. This argument is a path in your own "
            f"zone and must be relative to it."
        )
    path = os.path.join(zone.root, rel)
    if not contained(path, zone.root):
        raise PermissionError(f"{rel!r} resolves outside your zone {zone.root!r}.")
    return path


def _inside_remote(remote_root: str, rel: str) -> str:
    """The same question about the **far** side, where there is no filesystem here.

    `contained` resolves both sides and requires the root to exist locally, so
    against a remote root it denies everything. `contained_syntactically` is the
    weaker check that works without a filesystem, and its weakness — a symlink
    on the far side defeats it — is stated where it is defined.
    """
    path = contained_syntactically(rel, remote_root)
    if path is None:
        raise PermissionError(
            f"{rel!r} is not a path inside your remote zone. This argument is "
            f"relative to {remote_root!r} on the remote side; absolute paths and "
            f"paths that climb out with '..' are refused."
        )
    return path


def tools(conn: Connection, zone: Zone, remote_root: str) -> tuple[ToolDef, ...]:
    """Three tools, closed over this attempt's zone **and its far-side twin**.

    **`remote_root` is not optional and was the defect.** This took `(conn,
    zone)` and passed `zone.root` — a *local* absolute path — as the `cwd` for a
    command run on another machine. `cd /var/tmp/yihou/…` on the far side finds
    nothing, because the mirror lives at `/data/yihou/…`. The one configuration
    where it appeared to work is a **strong** mapping, where the two paths are
    the same by definition — which is the worst way for a defect to hide, since
    it would have shipped looking correct.

    `sync.remote_root(zone, mapping)` is where the value comes from; `prepare`
    has the mapping and this module does not, which is why it is a parameter and
    not something computed here.

    `conn` is a `Connection` and deliberately not a `SyncTransport`: **a
    container is a valid tool target and an invalid sync transport**, because
    `docker exec` runs commands fine while `docker cp` cannot express
    `--delete`. Nothing constructs a `DockerExec` yet; the door is left open,
    not walked through.
    """

    def env_remote_run(command: list[str], cwd: str = "") -> dict[str, Any]:
        proc = conn.run(command, cwd=_inside_remote(remote_root, cwd) if cwd else remote_root)
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}

    def env_remote_push(path: str, remote: str) -> dict[str, Any]:
        report = conn.push(_inside(zone, path), remote)
        return report._asdict()

    def env_remote_pull(remote: str, path: str) -> dict[str, Any]:
        report = conn.pull(remote, _inside(zone, path))
        return report._asdict()

    # **The far side, named.** These three descriptions used to say "the remote
    # side of this task's mapping" and nothing else — true, and unusable: it
    # names no machine, so a reader of the tool cannot tell *which* one it is
    # talking to. Measured, run `20260901T080901-50ecb9`: the agent's first act
    # was `env_remote_run(["hostname","-f"])`, because the tool surface withheld
    # something it knew.
    #
    # That is the defect this closes, and it is a design one rather than a bug.
    # The knowledge existed and the only route to it was a package remembering
    # to write it down — the same shape as the mechanisms in this repository
    # that were wired, correct, and reached by no production caller.
    #
    # **What is deliberately not here: whether the work belongs over there.**
    # That is the task's call and not this module's — `env_mgr` cannot know
    # whether a package wants a remote *resource* while working locally, and a
    # claim made here is one the package has no way to contradict. Identity is
    # ours; intent is the package's, keyed on whether these tools exist at all.
    # `Ssh.describe`'s docstring records the revision in which this line was
    # crossed and why it was walked back.
    where = conn.describe()
    return (
        ToolDef(
            name="env_remote_run",
            # **The locative sentence belongs to `describe()`, not here**, and
            # that was measured rather than reasoned. This first read "…{where}.
            # This is the far side of this task's mapping and it is where this
            # task's work is meant to happen." — a sentence that is true over
            # `Ssh` and self-contradictory over `LocalConnection`, whose
            # `describe()` says the two ends are one host. Asked to read it, a
            # model said so unprompted: *"这与「远端」的措辞本身就相互矛盾…既拿不到
            # 机器标识，也无法确定它是否与本地是同一台"*
            # (`p3_description_reaches_the_model.py`, control arm).
            #
            # A fixed clause here can only be right for one kind of connection.
            # Each class now says its own whole locative sentence and this
            # supplies only what is uniform: the zone root and the `cwd` rule.
            # (The *"where the work is meant to happen"* half of that sentence
            # was dropped altogether, for a second and unrelated reason — see
            # the note above on identity versus intent.)
            description=(
                f"Run a command on {where} Your own zone there is "
                f"{remote_root!r}, which is the working directory unless you "
                f"pass `cwd`."
            ),
            schema={
                "type": "object",
                "properties": {
                    "command": {"type": "array", "items": {"type": "string"}},
                    "cwd": {
                        "type": "string",
                        "description": (
                            "relative to your zone ON THE REMOTE SIDE; absolute "
                            "paths and '..' are refused. Omit it to start at the "
                            "remote zone root."
                        ),
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
            call=env_remote_run,
        ),
        ToolDef(
            name="env_remote_push",
            description=(
                f"Copy a path from your zone on this machine to {where} "
                f"Its zone root there is {remote_root!r}."
            ),
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
            description=(
                f"Copy a path into your zone on this machine from {where} "
                f"Its zone root there is {remote_root!r}."
            ),
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
