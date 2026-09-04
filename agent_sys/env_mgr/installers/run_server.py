# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""run_server installer: a long-lived server, declared in a recipe like anything else.

    - installer: run_server
      importance: required
      name: serena-mcp
      command: "serena start-mcp-server --port 24282"
      port: 24282
      ready_timeout: 20        # optional

**The one thing that makes this different from every other installer**: what it
produces outlives the process that produced it. `check`/`plan` are read-only as
usual; `install` starts a process that must still be running when this installer,
its recipe, and the `python -m env_mgr` child are all gone. Three consequences,
each of which is a defect if forgotten:

1. **The server leads its own process group** (``start_new_session=True``), so
   that `servers.stop_all` can `killpg` it and take its workers with it.
   **Not** so that it survives its parent -- it survives either way, by ordinary
   orphaning, and an earlier revision of this line claimed otherwise. Removing
   the flag was measured: `test_the_server_outlives_the_process_that_started_it`
   stayed **green**, and the three tests that *stop* a server went red, because
   `killpg` was then aimed at a group id that is not the server's.
2. **Its output goes to a file, never to the inherited pipes.** Measured
   2026-09-04: `agent_assets.py::_run_recipe` uses
   ``subprocess.run(capture_output=True)``, which waits for the *pipes* to
   close rather than for the child to exit. A detached server holding them open
   made that call return after **25 s** -- the server's whole lifetime --
   against **0 s** with the output redirected. Inheriting them would hang the
   recipe until `RECIPE_TIMEOUT_SECONDS`, 20 minutes later, and then report a
   perfectly healthy server as a timeout.
3. **It is recorded before it is known to work.** See
   `env_mgr.servers.record_spawn` for why that ordering is the only one that
   survives a kill in the gap.

`bootstrap` deliberately does **not** start anything: `runner.run` calls
``install`` and then ``bootstrap`` for the bootstrap stage, and a start in both
would be two servers for one declaration.

**This is for a server reached over a port -- HTTP or SSE -- and no shipped
recipe uses one yet. That is not an omission.** A **stdio** MCP server is
spawned by its client, by definition: the transport *is* the child's stdin and
stdout, so there is nothing to start separately and nothing to connect to
afterwards. Both addons this repository ships are ``"type": "stdio"``, measured,
and serena is one of them -- so declaring serena here would be wrong, not merely
unnecessary. The split is: **stdio, the harness spawns it; port-based, this
does.** The second case has not arisen yet.
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

    A real connect, not `ss`. This is the readiness question -- *can a client
    reach it* -- and a socket in `LISTEN` that the process has not yet begun
    serving answers the wrong one.
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

        # **Started once per run, and a second declaration warns.** The registry
        # is the whole answer: it already knows what is up, so nothing has to
        # say *whether this run wants it* and no second declaration route is
        # needed.
        #
        # **Before the port check, and that ordering is load-bearing.** A
        # duplicate declaration finds its own first server holding the port, and
        # the port check would report it as a stranger -- `fail` for what is
        # meant to be a `warn`. The first revision had these the other way round
        # and `test_a_second_declaration_of_the_same_server_warns_and_starts_
        # nothing` caught it. "This run started it" is also simply the truer
        # sentence when it is true.
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
        """Alive **and** bound, or a named failure saying which half went wrong.

        The two failures are different things and are reported as different
        things: a server that exited is a broken command, and a server that is
        running but has not bound is a slow or misconfigured one. Collapsing
        them into "did not start" would send the reader to the wrong file.
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
