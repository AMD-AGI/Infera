# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The servers this run started, and the stopping of them.

Recorded to a file because the process that starts a server (a short-lived
recipe child) exits long before the run ends, so an in-memory registry would
not survive. Both sides agree on one registry path, carried via
`REGISTRY_ENV_VAR`; the path is a parameter and never defaulted.

Servers are stopped on normal and handled-error exit, when `owned_servers`
unwinds. They leak on `SIGTERM` to the supervisor, because no handler is
installed, and on `SIGKILL`, which cannot be handled at all. A sweep to reap
entries left by a crashed run is not built here; see `docs/TODO.md` 4j.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import subprocess
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from .outcome import Outcome

__all__ = [
    "REGISTRY_ENV_VAR",
    "PortHolder",
    "already_running",
    "cmdline_of",
    "starttime_of",
    "ServerRecord",
    "inspect_port",
    "owned_servers",
    "port_conflict",
    "record_spawn",
    "records",
    "registry_path",
    "stop_all",
]

#: The one variable both sides read. Exported by whoever owns the run's state
#: directory; never defaulted here -- see the module docstring.
REGISTRY_ENV_VAR = "AGENT_SYS_SERVER_REGISTRY"

#: Between ``SIGTERM`` and ``SIGKILL`` when stopping. A server that has not gone
#: in this long is not going to be talked round.
STOP_GRACE_SECONDS = 5.0

#: `ss -ltnp` renders the holder as ``users:(("python3",pid=728122,fd=3))``.
#: Parsed rather than reformatted by a second tool: this is the output of a
#: program this repository does not author, and the shape was read off this host
#: on 2026-09-04 rather than remembered.
_SS_PID = re.compile(r"pid=(\d+)")


@dataclass(frozen=True)
class ServerRecord:
    """One server this run started.

    `starttime` (field 22 of `/proc/<pid>/stat`), not `cmdline`, is the
    identity guard for the stop -- a pid can be reused.
    """

    name: str
    pid: int
    port: int | None
    command: str
    starttime: str
    cmdline: str
    started_at: float


@dataclass(frozen=True)
class PortHolder:
    """What could be learned about whoever holds a port.

    `occupied=False` means nothing is listening. Otherwise `pid` is ``None``
    when the holder belongs to another uid and is unreadable without privilege.
    """

    #: False when nothing is listening on the port at all.
    occupied: bool
    pid: int | None = None
    cmdline: str = ""


def registry_path(environ: Mapping[str, str] | None = None) -> Path | None:
    """The registry file, or ``None`` when the caller named none."""
    env = os.environ if environ is None else environ
    raw = env.get(REGISTRY_ENV_VAR, "").strip()
    return Path(raw) if raw else None


# --------------------------------------------------------------------------- #
# writing: the recipe child's side


def record_spawn(path: Path, record: ServerRecord) -> None:
    """Append one server to the registry, durably, before it is known good.

    Recorded at spawn, before the readiness check, so a kill during that wait
    still leaves an entry.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # A torn last line has no terminating newline; appending straight onto it
    # would splice the new record into the wreckage and lose both.
    prefix = "\n" if _ends_mid_line(path) else ""
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(prefix + json.dumps(asdict(record), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _ends_mid_line(path: Path) -> bool:
    try:
        with open(path, "rb") as handle:
            if handle.seek(0, os.SEEK_END) == 0:
                return False
            handle.seek(-1, os.SEEK_END)
            return handle.read(1) != b"\n"
    except OSError:
        return False


def records(path: Path) -> list[ServerRecord]:
    """Every server in the registry, oldest first.

    A malformed line is skipped rather than fatal, so one corrupt entry does
    not strand the well-formed entries after it.
    """
    if not path.exists():
        return []
    out: list[ServerRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(ServerRecord(**json.loads(line)))
        except (json.JSONDecodeError, TypeError):
            continue
    return out


# --------------------------------------------------------------------------- #
# ports


def already_running(path: Path, name: str) -> ServerRecord | None:
    """The live entry this run already has for `name`, if there is one.

    Live, not merely present: an entry whose process has since died does not
    count, using the same `starttime` guard a stop uses.
    """
    found = None
    for record in records(path):
        if record.name == name and _still_ours(record):
            found = record
    return found


def cmdline_of(pid: int) -> str:
    """`/proc/<pid>/cmdline` as a space-joined string, empty if unreadable.

    Readable across uids, unlike `/proc/<pid>/exe`, which is one reason the
    port check keys on cmdline.
    """
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.decode("utf-8", "replace").replace("\0", " ").strip()


def inspect_port(port: int) -> PortHolder:
    """Who is listening on `port`, as far as an unprivileged process can tell.

    Runs `ss` directly rather than through `installers/base.py::run_cmd`,
    since installers use the registry, not the other way round.
    """
    try:
        # `-p` is load-bearing: without it `ss` prints no Process column at
        # all, and every holder -- same uid or not -- reads as unknowable.
        proc = subprocess.run(
            ["ss", "-ltnpH", f"sport = :{int(port)}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        # No `ss`, or it hung. "Occupied, holder unknowable" is the honest
        # answer, which `port_conflict` renders as a fail.
        return PortHolder(occupied=True)
    out = proc.stdout + proc.stderr
    if proc.returncode != 0 or not out.strip():
        return PortHolder(occupied=False)
    match = _SS_PID.search(out)
    if match is None:
        # Occupied, holder unknowable: see `PortHolder`.
        return PortHolder(occupied=True)
    pid = int(match.group(1))
    return PortHolder(occupied=True, pid=pid, cmdline=cmdline_of(pid))


#: Words that may stand in front of the real program in a declared command, and
#: are skipped when picking the token. A short closed list, not a parser. A
#: command this list does not cover degrades to a `fail`, never to a false
#: `warn`.
_COMMAND_PREFIXES = ("exec", "env", "nohup", "setsid", "stdbuf", "time")


def _program_token(command: str) -> str:
    """The basename of the program a `run_server` item declares.

    Leading wrappers are skipped, as is anything of the form ``VAR=value``.
    """
    for part in command.split():
        if part in _COMMAND_PREFIXES or "=" in part.split("/")[0]:
            continue
        return os.path.basename(part)
    return ""


def _is_basically_the_same(declared: str, holder_cmdline: str) -> bool:
    """Is the process holding the port *the thing we were about to start*?

    Keys on our own declaration, not the holder's identity: does the holder's
    command line mention the program this item says it runs?
    """
    token = _program_token(declared)
    if not token:
        return False
    return any(os.path.basename(part) == token for part in holder_cmdline.split())


def port_conflict(
    port: int, declared_command: str, name: str, holder: PortHolder | None = None
) -> Outcome | None:
    """The verdict on an occupied port, or ``None`` when it is free.

    ``warn`` when the holder looks like the same program, ``fail`` otherwise.
    `holder` is an injectable seam for testing.
    """
    if holder is None:
        holder = inspect_port(port)
    if not holder.occupied:
        return None
    if holder.pid is None:
        return Outcome(
            "fail",
            f"{name}: port {port} is held by another user; its command line cannot be "
            f"read, so it cannot be shown to be the same server",
            {"port": port, "holder": "unreadable"},
        )
    if _is_basically_the_same(declared_command, holder.cmdline):
        return Outcome(
            "warn",
            f"{name}: port {port} already served by pid {holder.pid}, which looks like "
            f"the same program; not started again",
            {"port": port, "pid": holder.pid, "cmdline": holder.cmdline},
        )
    return Outcome(
        "fail",
        f"{name}: port {port} is held by pid {holder.pid}, which is a different program",
        {"port": port, "pid": holder.pid, "cmdline": holder.cmdline},
    )


# --------------------------------------------------------------------------- #
# stopping: the supervisor's side


def _stat_fields(pid: int) -> list[str]:
    """`/proc/<pid>/stat` from field 3 on, or empty if the process is gone.

    Sliced from the last ``)`` rather than split on spaces, since field 2 (the
    executable name) is unescaped and may contain spaces or `)`.
    """
    try:
        data = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return []
    try:
        return data[data.rindex(")") + 2 :].split()
    except ValueError:
        return []


def starttime_of(pid: int) -> str:
    """Field 22 of `/proc/<pid>/stat`, or ``""`` if the process is gone.

    A zombie reports as gone deliberately, since an un-`wait`ed process keeps
    its `starttime` after it dies.
    """
    fields = _stat_fields(pid)
    if not fields or fields[0] == "Z":
        return ""
    try:
        return fields[19]
    except IndexError:
        return ""


def _still_ours(record: ServerRecord) -> bool:
    """Is `record.pid` still the process we started?

    Signal a thing that can still be identified, or signal nothing.
    """
    return bool(record.starttime) and starttime_of(record.pid) == record.starttime


def _stop_one(record: ServerRecord) -> Outcome:
    if not _still_ours(record):
        return Outcome(
            "info",
            f"{record.name}: pid {record.pid} is gone or is no longer ours; not signalled",
            {"pid": record.pid},
        )
    # The process group, not the pid: `run_server` starts each server with
    # `start_new_session=True`, so it leads its own group and a server that
    # forked workers takes them with it.
    for sig, wait in ((signal.SIGTERM, STOP_GRACE_SECONDS), (signal.SIGKILL, 1.0)):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(record.pid, sig)
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            if not _still_ours(record):
                verb = "stopped" if sig == signal.SIGTERM else "killed"
                return Outcome(
                    "ok", f"{record.name}: {verb} (pid {record.pid})", {"pid": record.pid}
                )
            time.sleep(0.05)
    return Outcome(
        "warn",
        f"{record.name}: pid {record.pid} survived SIGKILL",
        {"pid": record.pid},
    )


def stop_all(path: Path) -> list[Outcome]:
    """Stop every server in the registry, newest first, and empty it.

    Truncated only after the stops, so a crash during shutdown leaves entries
    for whatever is still up rather than losing them.
    """
    entries = records(path)
    outs = [_stop_one(record) for record in reversed(entries)]
    if entries:
        with contextlib.suppress(OSError):
            path.write_text("", encoding="utf-8")
    return outs


@contextmanager
def owned_servers(path: Path | None) -> Iterator[Path | None]:
    """The servers this run starts are this block's to stop.

    `path` may be ``None``, meaning no registry and nothing to stop. Unwinds
    on normal and handled-error exit; see the module docstring for the rest.
    """
    try:
        yield path
    finally:
        if path is not None:
            stop_all(path)
