# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The servers this run started, and the stopping of them.

Two halves of one fact, kept in one module because they are written and read by
**different processes**. A recipe runs as a short-lived child --
``agent_assets.py::_run_recipe`` shells ``python -m env_mgr bootstrap`` and waits
-- so the process that starts a server is gone seconds later, and cannot be the
process that stops it. An in-memory registry would therefore record nothing that
survives to the end of the run, which is why this one is a file.

The two sides agree on **one path, carried in one environment variable**
(`REGISTRY_ENV_VAR`). ``_run_recipe`` passes ``env=dict(environ)`` and
``installers/base.py::run_cmd`` adds no ``env=`` of its own, so a variable
exported by the caller reaches the installer unchanged -- the same route
``CLAUDE_CONFIG_DIR`` takes. The path is a **parameter and never a default**:
there is no root to default to. `AGENT_SYS_HOME` is named in `docs/TODO.md` and
`docs/spec.md` and **in no code in this tree** (measured 2026-09-04), and
inventing a provisional root is exactly what the owner's standing ruling
forbids. A caller with nowhere to put the file exports nothing, and then
`run_server` records nothing and says so.

## What is guaranteed, and what is not

**Servers are stopped on normal and handled-error exit** -- whenever
`owned_servers` unwinds. That is the whole of it. It is *not* "servers are
always stopped":

- ``SIGTERM`` to the supervisor: no handler is installed, so the manager does
  not unwind and the servers **leak**.
- ``SIGKILL``: nothing can unwind. They **leak**.

**The closing mechanism is known and is named here so the limitation reads as a
decision rather than an oversight.** Measured on this host 2026-09-04:
``prctl(PR_SET_PDEATHSIG, SIGTERM)`` on the child does close the ``SIGKILL``
case -- with it the child died when its parent was ``kill -9``'d, without it the
child survived. **It is nonetheless the wrong tool at the site that spawns
these**, because the spawning process is the recipe child, which exits within
seconds of the spawn: the server would be killed the moment its own install
finished. It would work only if the supervisor were the direct parent, which is
a different architecture, not a flag.

The cheaper half of the same problem -- a sweep that reaps entries left by a
crashed run -- is `docs/TODO.md` and not built here.
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

    `starttime` is the **identity guard** for the stop, and the choice of field
    is measured rather than aesthetic. A pid is reused, so a registry entry that
    outlives its process would otherwise aim a signal at whatever inherited the
    number -- someone else's process, on a shared host.

    **It is field 22 of `/proc/<pid>/stat`, not the command line.** `cmdline`
    was the obvious key and is the wrong one, twice over. Measured 2026-09-04:

    - immediately after `Popen`, `/proc/<pid>/cmdline` is **empty** -- the exec
      has not happened yet. A guard keyed on it would have compared ``""``
      against a later real value and **refused every stop**, silently, because a
      refusal is an `info` line and not a failure.
    - `shell=True` means the recorded string would be the shell's, and a shell
      running one simple command may `exec` into it, changing the value under
      the guard afterwards.

    `starttime` has neither problem: it is set at fork, never changes, is
    readable across uids (checked against pid 1), and a reused pid gets a
    different one. `cmdline` is kept for the human reading the report and is
    read on a best-effort basis.
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

    Three states, and the third is the one that is easy to miss: `pid` is
    ``None`` **with the port occupied** when the holder belongs to another uid.
    Measured 2026-09-04 against a root-owned port on this host: `ss -ltnp`
    prints the socket with an empty Process column, `lsof -nP -iTCP:<port>`
    prints nothing at all, and `/proc/<pid>/fd` is `Permission denied` so the
    inode in `/proc/net/tcp` cannot be walked back to a pid either. There is no
    second link in the chain, so the holder's binary is not merely unknown --
    it is **unknowable** without privilege we do not have and must not take.
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
    """Append one server to the registry, durably, **before it is known good**.

    One JSON object per line, opened in append mode, flushed and `fsync`ed.

    **Recorded at spawn rather than after the readiness check, and that is not
    redundant** -- it is the only ordering that survives the gap. A server is
    already a live detached process the instant `Popen` returns, and the bind
    wait that follows takes seconds. Anything that kills the recipe child during
    those seconds -- `_run_recipe`'s own 20-minute `subprocess.run(timeout=)`,
    an operator, an OOM -- leaves that process running. If the entry were
    written after the check, or carried home in the report JSON, it would die
    with the child and the server would leak **with nothing recording that it
    exists**. Written first, the worst case is an entry for a server that never
    came up, and stopping something already dead is a no-op with a guard on it.

    Append-only, one line per record, for the same reason: a rewritten JSON
    array has a window in which the file is truncated, and that window is
    exactly when the kill lands.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # A torn last line -- the very thing this file is written to survive -- has
    # no terminating newline, and appending straight onto it would splice the
    # new record into the wreckage and lose **both**. Measured by
    # `test_a_corrupt_line_does_not_strand_the_entries_after_it`, which failed
    # on exactly that before this line existed.
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

    A malformed line is skipped rather than fatal. The reader is the shutdown
    path: refusing to parse would strand every *well-formed* entry after it, so
    a corrupt tail -- the signature of a kill mid-write, which is the case this
    file exists for -- would cost more processes than it saved.
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


def cmdline_of(pid: int) -> str:
    """`/proc/<pid>/cmdline` as a space-joined string, empty if unreadable.

    Readable across uids on this host (measured against pid 1); `/proc/<pid>/exe`
    is **not**, which is one reason the port check keys on cmdline.
    """
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.decode("utf-8", "replace").replace("\0", " ").strip()


def inspect_port(port: int) -> PortHolder:
    """Who is listening on `port`, as far as an unprivileged process can tell.

    An argv list rather than `installers/base.py::run_cmd`'s shell string, and
    not only for quoting: this module sits **below the decoupling wall** and
    `installers/` is a sibling below it, so importing the installers' helper
    here would point the dependency backwards -- installers use the registry,
    not the other way round.
    """
    try:
        # `-p` is load-bearing and was once missing. Without it `ss` prints the
        # socket with **no Process column at all**, which this module reads as
        # "holder unknowable" -- so every same-uid holder looked foreign, and
        # the `warn` branch the owner's rule turns on could never be reached.
        # Measured both ways on this host: `-ltnH` gives
        # ``LISTEN 0 5 127.0.0.1:38941 0.0.0.0:*`` and `-ltnpH` appends
        # ``users:(("python3",pid=881479,fd=3))``.
        proc = subprocess.run(
            ["ss", "-ltnpH", f"sport = :{int(port)}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        # No `ss`, or it hung. Reporting "free" would invite a start onto a port
        # that may well be taken, so the honest answer is "occupied, holder
        # unknowable" -- which `port_conflict` renders as a fail.
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


def _program_token(command: str) -> str:
    """The basename of the program a `run_server` item declares."""
    parts = command.split()
    return os.path.basename(parts[0]) if parts else ""


def _is_basically_the_same(declared: str, holder_cmdline: str) -> bool:
    """Is the process holding the port *the thing we were about to start*?

    **The question is deliberately not "what is this process".** That one has no
    answer: measured on this host, a serena MCP server, an `agentsview` server
    and a bare `python3 -m http.server` all report `comm=python3`, and
    `/proc/<pid>/exe` is an interpreter path for all three. Keying on either
    would call every Python server "basically the same" as every other, and by
    the owner's rule that case is an error rather than a warning.

    So the key comes from **our own declaration**, not from the holder: does the
    holder's command line mention the program this item says it runs? The
    holder's ``argv[0]`` -- the interpreter -- never enters the comparison.
    """
    token = _program_token(declared)
    if not token:
        return False
    return any(os.path.basename(part) == token for part in holder_cmdline.split())


def port_conflict(
    port: int, declared_command: str, name: str, holder: PortHolder | None = None
) -> Outcome | None:
    """The verdict on an occupied port, or ``None`` when it is free.

    Levels are `env_mgr.outcome.LEVELS` and nothing new: ``warn`` when the
    holder is basically the same program, ``fail`` otherwise. ``fail`` is what
    `status_from` rolls up to ``FAIL`` and the CLI turns into exit 2.

    `holder` exists so the **foreign-uid** verdict can be tested. Reaching it
    for real needs a port held by another user, which is not something a test
    may arrange on a shared host, and leaving it untested is not an option: that
    branch is where an unreadable holder lands, and a missing `-p` flag once put
    *every* holder there, making the `warn` case unreachable. It is a seam, not
    a mock — the default is the real measurement and both live cases use it.
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

    Sliced from the last ``)`` rather than split on spaces: field 2 is the
    executable name **in parentheses and not escaped**, so a process named
    ``foo bar)baz`` shifts every later field. Reading a format this repository
    does not author, so it was checked against a live process rather than
    assumed. Index 0 of the result is field 3, the state.
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

    **A zombie reports as gone**, and that is a correction rather than a nicety.
    A process this one spawned and has not `wait`ed for keeps its `/proc` entry
    and its `starttime` after it dies, so an identity check that looked only at
    `starttime` called a corpse "still ours" — and `stop_all` then reported
    ``pid N survived SIGKILL``, which is alarming, wrong, and was caught by
    `test_a_started_server_is_running_recorded_and_stoppable`. In production the
    server is orphaned and reaped by init, so this state is invisible there; it
    appears whenever the starter is still around, which is exactly what a test
    is.
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

    The guard is the same rule as *never `docker rm` a container you did not
    start*, in process form: **signal a thing you can still identify, or signal
    nothing.** See `ServerRecord` for why the identity is `starttime` and not
    the command line.

    An empty recorded `starttime` means the record was written for a process we
    could not identify even then, so it is never signalled.
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

    Reverse order because a later server may depend on an earlier one.

    The file is truncated only after the stops, so a crash *during* shutdown
    leaves the entries for whatever is still up rather than losing them -- the
    same reasoning as recording the spawn first, at the other end of the life.
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

    Named for what it owns rather than for when it runs. `env_mgr` starts these
    processes, so `env_mgr` stops them: a caller entering this block must not
    also have to remember to clean up, and the one line that enters it is a call
    site rather than a transfer of responsibility.

    ``path`` may be ``None`` -- a caller with no run state names no registry,
    `run_server` then starts nothing, and there is correspondingly nothing to
    stop. That is a caller with the feature switched off, not an error.

    Read the module docstring for what this does **not** guarantee: it unwinds
    on normal and handled-error exit, and on neither ``SIGTERM`` nor ``SIGKILL``.
    """
    try:
        yield path
    finally:
        if path is not None:
            stop_all(path)
