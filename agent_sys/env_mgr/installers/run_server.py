# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""run_server installer: a long-lived server, declared in a recipe like anything else.

    - installer: run_server
      name: serena                 # keys the once-per-run guard
      command: "serena start-mcp-server --port 24282"
      port: 24282
      ready_timeout: 30            # seconds to wait for the port; default below

What it produces outlives the process that produced it. The server leads its own
process group so `servers.stop_all` can `killpg` it; its output goes to a file,
never inherited pipes, since a detached server holding them open hangs the
capturing parent. `bootstrap` starts nothing, to avoid doubling the server. This
installer is for a server reached over a **port**; a stdio MCP server is spawned
by its own client and declared in `.mcp.json` instead.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time

from ..outcome import Outcome
from ..recipe import Item, Target
from ..servers import (
    ServerRecord,
    already_running,
    cmdline_of,
    inspect_port,
    port_conflict,
    record_spawn,
    registry_path,
    starttime_of,
)
from .base import _run_bootstrap, level_for_missing

#: How long `install` waits for the port to accept a connection. Overridable per
#: item; a server that needs longer says so rather than everything paying for it.
DEFAULT_READY_TIMEOUT = 20.0

_POLL_SECONDS = 0.1


def _port_of(item: Item) -> int | None:
    raw = item.spec.get("port")
    try:
        return int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _accepts(port: int) -> bool:
    """Is something accepting connections on `port`?

    A real connect, not `ss`: a socket in `LISTEN` the process has not begun
    serving would answer the wrong question.
    """
    with socket.socket() as sock:
        sock.settimeout(0.5)
        try:
            return sock.connect_ex(("127.0.0.1", port)) == 0
        except OSError:
            return False


class RunServerInstaller:
    name = "run_server"

    # ----------------------------------------------------------------- read-only

    def check(self, item: Item, target: Target) -> list[Outcome]:
        port = _port_of(item)
        if port is None:
            return [Outcome(level_for_missing(item.importance), f"{item.name}: no port declared")]
        holder = inspect_port(port)
        if not holder.occupied:
            return [Outcome("info", f"{item.name}: port {port} is free", {"port": port})]
        conflict = port_conflict(port, item.spec.get("command", ""), item.name)
        assert conflict is not None  # occupied, so port_conflict returns a verdict
        return [conflict]

    def plan(self, item: Item, target: Target) -> list[Outcome]:
        port = _port_of(item)
        command = item.spec.get("command", "")
        if port is not None:
            conflict = port_conflict(port, command, item.name)
            if conflict is not None:
                return [conflict]
        return [Outcome("info", f"would start: {command}", {"port": port})]

    # -------------------------------------------------------------------- acting

    def install(self, item: Item, target: Target) -> list[Outcome]:
        command = item.spec.get("command", "")
        if not command:
            return [
                Outcome(level_for_missing(item.importance), f"{item.name}: no command declared")
            ]
        port = _port_of(item)
        if port is None:
            return [Outcome(level_for_missing(item.importance), f"{item.name}: no port declared")]

        path = registry_path()
        if path is None:
            return [
                Outcome(
                    level_for_missing(item.importance),
                    f"{item.name}: no server registry was named, so a server started here "
                    f"could never be stopped; refusing to start one",
                    {"variable": "AGENT_SYS_SERVER_REGISTRY"},
                )
            ]

        # Started once per run; a second declaration warns. Checked before the
        # port check, since a duplicate declaration's own server would
        # otherwise be reported as a stranger (`fail` instead of `warn`).
        running = already_running(path, item.name)
        if running is not None:
            return [
                Outcome(
                    "warn",
                    f"{item.name}: already started by this run (pid {running.pid}, "
                    f"port {running.port}); not started again",
                    {"pid": running.pid, "port": running.port},
                )
            ]

        # The port policy is the whole of the *someone else* case: an
        # already-served port is never started onto, whether the verdict is the
        # warn (the same program, started by something that is not this run) or
        # the fail.
        conflict = port_conflict(port, command, item.name)
        if conflict is not None:
            return [conflict]

        log = os.path.join(target.path, f"{item.name}.server.log")
        try:
            handle = open(log, "w", encoding="utf-8")
        except OSError as error:
            return [Outcome(level_for_missing(item.importance), f"{item.name}: {error}")]
        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=target.path,
                stdout=handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as error:
            return [Outcome(level_for_missing(item.importance), f"{item.name}: {error}")]
        finally:
            # Ours is closed either way: the child holds its own descriptor, and
            # leaving this one open is what would keep the recipe's pipes alive.
            handle.close()

        record_spawn(
            path,
            ServerRecord(
                name=item.name,
                pid=proc.pid,
                port=port,
                command=command,
                # Read before the readiness wait, because that is when the pid
                # is certainly still ours. `cmdline` is reporting only and is
                # legitimately empty here: the exec has not happened yet.
                starttime=starttime_of(proc.pid),
                cmdline=cmdline_of(proc.pid),
                started_at=time.time(),
            ),
        )
        return [self._await_ready(item, proc, port, log)]

    def bootstrap(self, item: Item, target: Target) -> list[Outcome]:
        return _run_bootstrap(item, target)

    # ------------------------------------------------------------------ readiness

    def _await_ready(
        self, item: Item, proc: subprocess.Popen[bytes], port: int, log: str
    ) -> Outcome:
        """Alive and bound, or a named failure saying which half went wrong.

        An exited server is a broken command; one running but unbound is slow
        or misconfigured. Reported separately rather than as "did not start".
        """
        timeout = _timeout_of(item)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return Outcome(
                    level_for_missing(item.importance),
                    f"{item.name}: exited immediately with rc {proc.returncode}",
                    {"rc": proc.returncode, "log": log, "pid": proc.pid},
                )
            if _accepts(port):
                return Outcome(
                    "ok",
                    f"started {item.name} on port {port} (pid {proc.pid})",
                    {"pid": proc.pid, "port": port, "log": log},
                )
            time.sleep(_POLL_SECONDS)
        return Outcome(
            level_for_missing(item.importance),
            f"{item.name}: still running after {timeout:g}s but nothing is listening on "
            f"port {port}",
            {"pid": proc.pid, "port": port, "log": log, "timeout": timeout},
        )


def _timeout_of(item: Item) -> float:
    try:
        return float(item.spec["ready_timeout"])
    except (KeyError, TypeError, ValueError):
        return DEFAULT_READY_TIMEOUT
