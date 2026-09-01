# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ssh and docker exec behind one Protocol. Design §10.2.

Three methods, because that is what sync and prepare call. `fabric` and
`paramiko` were considered and rejected: a library would add a dependency to
hide a ``subprocess`` call, and neither expresses ``docker exec``.

The transport-specific options that always leak through such an abstraction are
held in the `RemoteMapping` that `meta` persists, not passed per call — so the
leak is in configuration, where it is inspectable, rather than in a signature.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from collections.abc import Sequence
from typing import Protocol

from env_mgr.protocols import SyncReport

__all__ = ["Connection", "DockerExec", "LocalConnection", "Ssh"]


class Connection(Protocol):
    def run(
        self, argv: Sequence[str], *, cwd: str | None = None, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]: ...

    def push(self, local: str, remote: str) -> SyncReport: ...

    def pull(self, remote: str, local: str) -> SyncReport: ...


def _rsync(src: str, dst: str, *, rsh: Sequence[str] = ()) -> SyncReport:
    rsync = shutil.which("rsync")
    if rsync is None:
        raise RuntimeError("rsync is not installed")
    cmd = [rsync, "-a", "--stats"]
    if rsh:
        cmd += ["-e", " ".join(rsh)]
    cmd += [src.rstrip(os.sep) + os.sep, dst.rstrip(os.sep) + os.sep]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    sent = sum(
        1
        for line in proc.stdout.splitlines()
        if line.startswith("Number of regular files transferred")
    )
    return SyncReport(sent=sent, received=0, conflicts=())


class Ssh:
    def __init__(self, host: str, *, options: Sequence[str] = ()) -> None:
        self.host = host
        self.options = tuple(options)

    def run(
        self, argv: Sequence[str], *, cwd: str | None = None, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        """**`shlex.join`, not `" ".join`.**

        `ssh host <string>` hands the string to the far side's *shell*, so an
        argument is re-split over there by whitespace and re-interpreted for
        globs, `$`, quotes and `;`. Joined with a plain space, `["echo", "a b"]`
        and `["echo", "a", "b"]` become the same command — the argv boundary the
        caller expressed is destroyed in transit, silently, and a path with a
        space in it becomes two paths.

        `LocalConnection` and `DockerExec` pass a real argv and have no such
        problem, so this class was the only one of the three that could not keep
        the Protocol's promise. It shipped that way and was never called; this is
        the commit that first constructs an `Ssh`, so it is where it is fixed.
        """
        remote = shlex.join(argv)
        if cwd:
            remote = f"cd {shlex.quote(cwd)} && {remote}"
        return subprocess.run(
            ["ssh", *self.options, self.host, remote],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def push(self, local: str, remote: str) -> SyncReport:
        return _rsync(local, f"{self.host}:{remote}", rsh=("ssh", *self.options))

    def pull(self, remote: str, local: str) -> SyncReport:
        return _rsync(f"{self.host}:{remote}", local, rsh=("ssh", *self.options))


class DockerExec:
    def __init__(self, container: str) -> None:
        self.container = container

    def run(
        self, argv: Sequence[str], *, cwd: str | None = None, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        head = ["docker", "exec"]
        if cwd:
            head += ["-w", cwd]
        return subprocess.run(
            [*head, self.container, *argv], capture_output=True, text=True, timeout=timeout
        )

    def push(self, local: str, remote: str) -> SyncReport:
        subprocess.run(
            ["docker", "cp", local, f"{self.container}:{remote}"],
            capture_output=True,
            text=True,
            check=True,
        )
        return SyncReport(sent=1, received=0, conflicts=())

    def pull(self, remote: str, local: str) -> SyncReport:
        subprocess.run(
            ["docker", "cp", f"{self.container}:{remote}", local],
            capture_output=True,
            text=True,
            check=True,
        )
        return SyncReport(sent=0, received=1, conflicts=())


class LocalConnection:
    """The same three methods against this machine.

    Not a test double: spec §5.2's *strong* mapping is one mount, where the two
    paths are the same bytes and "remote" is a name for the far end of a mapping
    rather than a different host.
    """

    def run(
        self, argv: Sequence[str], *, cwd: str | None = None, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(list(argv), cwd=cwd, capture_output=True, text=True, timeout=timeout)

    def push(self, local: str, remote: str) -> SyncReport:
        return _rsync(local, remote)

    def pull(self, remote: str, local: str) -> SyncReport:
        return _rsync(remote, local)
