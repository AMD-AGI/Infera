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

__all__ = [
    "Connection",
    "DockerExec",
    "LocalConnection",
    "Ssh",
    "SyncTransport",
    "sync_transport",
]


class Connection(Protocol):
    def run(
        self, argv: Sequence[str], *, cwd: str | None = None, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]: ...

    def push(self, local: str, remote: str) -> SyncReport: ...

    def pull(self, remote: str, local: str) -> SyncReport: ...

    def describe(self) -> str:
        """**Where this connection lands, as a phrase a model will read.**

        `remote.tools` puts this into the tool descriptions, which is how the
        agent learns *where the far side is* from `env_mgr` rather than from a
        task package's prose. The user's framing of the whole remote stage is
        "只要告诉他 hostname 就完了" — telling the agent where the work happens
        is this module's job, and a package that had to re-say it would be one
        package away from a package that forgets.

        **A method on the Protocol rather than a `getattr(conn, "host", …)` in
        `tools`.** The duck-typed version works for the three classes here and
        has one silent-wrong case that matters: a fourth transport — a `Kubectl`,
        a `Slurm` — would land in the fallback and be described to the agent as
        *this machine* while executing somewhere else. That is
        `interfaces.md` §4.11's family, a plausible value consumed as if it were
        the right one, and it is exactly the failure this stage exists to
        prevent. Declared here, a transport that cannot say where it goes does
        not satisfy `Connection`.

        Each class answers for itself, so there is no type switch and no table
        for a new transport to be missing from. Phrase, not bare identifier: the
        *"is this a different machine"* half is what the agent acts on, and only
        the class knows it — `LocalConnection`'s far side really is this host.
        """
        ...


class SyncTransport(Connection, Protocol):
    """A `Connection` that `sync` can also drive an ``rsync`` across.

    **Two Protocols rather than one with an optional capability**, because
    `docker cp` is not ``rsync`` and no amount of arranging makes it one. A
    single Protocol whose third implementation accepted ``delete=True`` and
    raised would be capability negotiation with the branch hidden one level
    down — the matrix the main spec's structural decisions rule out, because
    *"a matrix makes every caller branch on what a backend can do, and those
    branches are untested in the configuration a site actually runs"*. Split in
    two, *"this transport cannot sync"* is a type-level fact and no call site
    branches on it.

    **The name says `rsync` on purpose.** This seam is rsync-shaped: it hands
    back the two things an ``rsync`` command line needs to address the far side
    and nothing else. A neutral name would invite an implementation over
    something that is not rsync, and whoever tried would rediscover the
    distinction this split exists to record.

    **What is deliberately *not* here: the copy's semantics.** ``--delete`` is
    the whole of spec §5.3's *"local and remote are made identical"* and
    ``--exclude=playground/**`` is criterion 16, and both belong to `sync`. A
    transport that could drop either silently is exactly the defect this split
    was written to avoid — `Connection.push` has neither flag, so routing
    `sync` through it would have lost both with every existing test still green.
    """

    def rsync_spec(self) -> tuple[Sequence[str], str]:
        """``(rsh_argv, path_prefix)``.

        `rsh_argv` is what goes after ``rsync -e``, empty for a local copy.
        `path_prefix` is prepended to a path on the far side — ``"host:"`` for
        ssh, empty locally. `sync` builds the command; this only says how to
        reach the other end.
        """
        ...


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

    def rsync_spec(self) -> tuple[Sequence[str], str]:
        return ("ssh", *self.options), f"{self.host}:"

    def describe(self) -> str:
        """The host, and the fact that it is not the agent's own machine.

        The second clause is the load-bearing one. A hostname alone leaves the
        agent to infer whether it is somewhere else, and the inference is
        unreliable in the configuration that matters most: the weights and the
        image may be visible from both sides — measured, `/apps` is one NFS
        export — so everything the agent can *see* looks the same on either
        machine.

        **And it stops there, deliberately.** This sentence ended *"…and it is
        where this task's work is meant to happen"* for one revision, and that
        clause was wrong in a way worth recording: **where a task's work belongs
        is the task's decision, not this module's.** `env_mgr` knows where the
        far side *is* — nobody else can — and it does not know whether a given
        package wants its work there. A package may legitimately want a remote
        *resource* while working locally, and this sentence would have told its
        agent otherwise, in a voice the package cannot contradict.

        The split is worth keeping in mind for anything else that gets added
        here: **identity is ours, intent is the package's.** The task-side half
        lives in `examples/single_real_task/assets/serve_qwen.task/readme.md`,
        keyed on whether these tools are present at all.
        """
        return (
            f"{self.host}, a different machine from the one your own shell runs "
            f"on. It is the far side of this task's mapping."
        )


class DockerExec:
    """**A `Connection` and deliberately not a `SyncTransport`.**

    It exists for `remote.tools`, where the three methods are what an agent
    needs to reach into a container. `docker cp` cannot express identity — no
    ``--delete``, no exclude — so there is no honest `rsync_spec` for it, and
    declaring one that raised would be the capability matrix `SyncTransport`'s
    docstring rules out.
    """

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

    def describe(self) -> str:
        """A container, and **on this host** — the one case where "remote" does
        not mean "another machine". Saying so is the point: an agent told only
        the container name could reasonably read it as a second host and reason
        about the network between them.
        """
        return (
            f"the container {self.container}, running on the machine your own "
            f"shell runs on rather than on a second computer."
        )


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

    def rsync_spec(self) -> tuple[Sequence[str], str]:
        """No `-e`, no prefix. Both ends are ordinary paths on this machine."""
        return (), ""

    def describe(self) -> str:
        """**"This machine", said out loud rather than left to be assumed.**

        Spec §5.2's strong mapping is one mount seen from two places, and here
        both ends are this host. An agent that read the remote tools as implying
        a second machine would go looking for a network that is not there.
        """
        return (
            "this machine. The two ends of this mapping are paths on one host, "
            "so 'remote' here names the far end of a mapping rather than a "
            "second computer, and there is no network between them."
        )


def sync_transport(transport: str, target: str) -> SyncTransport:
    """The declared `RemoteMapping.transport` / `.target`, as an object.

    `meta.RemoteMapping` has carried both since it was written and **nothing
    anywhere read either of them** — `mapping_roots()` returns local→remote
    strings and drops the rest. This is the reader.

    An unknown transport raises **here, at composition**, rather than at the
    first copy. `docker` is the case that matters: it is a legitimate
    `Connection` and cannot be a sync transport, so a mapping that asks for one
    is a configuration fault and is named as one, at the point where a human is
    reading configuration.
    """
    if transport == "ssh" and target:
        return Ssh(target)
    if not target:
        # No far-side host: the two ends are paths on this machine. Spec §5.2's
        # strong mapping, and what the R0 rung ran against.
        return LocalConnection()
    raise ValueError(
        f"no sync transport for transport={transport!r} target={target!r}: "
        f"`docker` can be a Connection but not a SyncTransport (docker cp "
        f"cannot express --delete or an exclude); use ssh, or a shared mount "
        f"with an empty target"
    )
