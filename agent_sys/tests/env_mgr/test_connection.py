# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""`Ssh` — the argv boundary, which `ssh` does not preserve for free.

`Ssh.run` hands one string to the far side's shell, so the argv the caller built
has to survive being re-split over there. Joined with a plain space it does not:
`["echo", "a b"]` and `["echo", "a", "b"]` produce the same remote command, and
a path with a space in it becomes two paths.

**No test here connects to anything.** `subprocess.run` is captured, so what is
asserted is the command that *would* be sent — which is the thing the defect was
in. A round-trip against the real host is a probe in `scratch/`, and it is
evidence of a particular machine rather than a regression test.
"""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from env_mgr.remote.connection import DockerExec, LocalConnection, Ssh


@pytest.fixture()
def sent(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Every argv handed to `subprocess.run`, and nothing executed."""
    calls: list[list[str]] = []

    def fake(argv: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake)
    return calls


def test_one_argument_with_a_space_is_not_two_arguments(sent: list[list[str]]) -> None:
    """The defect, stated as the two cases it could not tell apart.

    This is the whole of it: if these two produce the same remote command, the
    caller's argv boundary was destroyed in transit and no error is raised
    anywhere. `assertNotEqual` is the load-bearing line — the individual
    spellings below are `shlex`'s business, not this module's.
    """
    Ssh("h").run(["echo", "a b"])
    Ssh("h").run(["echo", "a", "b"])
    one, two = sent[0][-1], sent[1][-1]
    assert one != two, f"argv boundary lost: both became {one!r}"
    assert "'a b'" in one


def test_shell_metacharacters_do_not_reach_the_far_side_shell(sent: list[list[str]]) -> None:
    """An argument is data. `;`, `$`, `*` and backticks in it are not commands.

    Its control is the line below: the *command* is still an ordinary bare word,
    so this is quoting arguments rather than quoting everything and hoping.
    """
    Ssh("h").run(["echo", "; rm -rf /tmp/x", "$HOME", "*"])
    remote = sent[0][-1]
    assert "; rm -rf /tmp/x" not in remote.replace("'; rm -rf /tmp/x'", "")
    assert "'$HOME'" in remote and "'*'" in remote
    assert remote.startswith("echo ")  # control: the verb is not quoted into oblivion


def test_cwd_is_quoted_too(sent: list[list[str]]) -> None:
    """`cd {cwd} && …` has exactly the same hole, and a zone path is generated
    rather than typed — so it is the one most likely to contain a surprise."""
    Ssh("h").run(["true"], cwd="/data/yihou/a dir")
    remote = sent[0][-1]
    assert remote.startswith("cd '/data/yihou/a dir' && ")


def test_the_host_and_options_are_argv_not_string(sent: list[list[str]]) -> None:
    """The near side is a real argv and stays one: only the far side is a shell.

    Without this, a fix that wrapped the whole thing in quotes would pass the
    tests above and break `ssh` itself.
    """
    Ssh("h", options=("-o", "BatchMode=yes")).run(["true"])
    assert sent[0][:4] == ["ssh", "-o", "BatchMode=yes", "h"]


def test_the_other_two_transports_never_had_the_problem(sent: list[list[str]]) -> None:
    """`LocalConnection` and `DockerExec` pass a real argv to `subprocess`, so
    there is no shell to re-split it. Asserted rather than assumed, because a
    later "consistency" edit routing them through a shell would reintroduce the
    defect in two more places.
    """
    LocalConnection().run(["echo", "a b"])
    DockerExec("c").run(["echo", "a b"])
    for argv in sent:
        assert "a b" in argv, argv  # intact as one element, not re-split


# ------------------------------------------------------------------ SyncTransport


def test_docker_exec_is_a_connection_and_not_a_sync_transport() -> None:
    """**The split, asserted as a type-level fact.**

    `docker cp` cannot express `--delete` or an exclude, so `DockerExec` has no
    honest `rsync_spec`. Declaring one that raised would be capability
    negotiation with the branch hidden one level down — the matrix the main
    spec's structural decisions rule out. The assertion is that the method is
    *absent*, because that is what makes a call site unable to branch on it.
    """
    assert not hasattr(DockerExec("c"), "rsync_spec")
    # Control: it is still a full `Connection`, which is what `remote.tools`
    # needs and the only reason it exists.
    for name in ("run", "push", "pull"):
        assert callable(getattr(DockerExec("c"), name))


def test_the_two_sync_transports_say_how_to_reach_their_far_side() -> None:
    assert Ssh("h").rsync_spec() == (("ssh",), "h:")
    assert Ssh("h", options=("-o", "BatchMode=yes")).rsync_spec() == (
        ("ssh", "-o", "BatchMode=yes"),
        "h:",
    )
    # Local is not a degenerate remote: no `-e`, and no prefix to prepend.
    assert LocalConnection().rsync_spec() == ((), "")


def test_a_mapping_becomes_the_transport_it_declared() -> None:
    """`RemoteMapping.transport` and `.target` have existed since the file was
    written and nothing read either. This is the reader."""
    from env_mgr.remote.connection import sync_transport

    ssh = sync_transport("ssh", "somehost")
    assert isinstance(ssh, Ssh) and ssh.host == "somehost"
    # No target: both ends are on this machine. Spec §5.2's strong mapping, and
    # what every configuration before R1 was.
    assert isinstance(sync_transport("ssh", ""), LocalConnection)


def test_a_transport_that_cannot_sync_is_refused_at_configuration_time() -> None:
    """Named while a human is reading configuration, not at the first copy.

    Its control is the line below: the *same* value is a perfectly good
    `Connection`, so this is refusing a role rather than refusing a transport.
    """
    from env_mgr.remote.connection import sync_transport

    with pytest.raises(ValueError, match="not a SyncTransport"):
        sync_transport("docker", "some-container")
    assert DockerExec("some-container").run(["true"]) is not None


# --------------------------------------------------------------------------- #
# `Connection` conformance — the gap that `tests/agent/test_doubles_conform.py`
# left open, closed here because `Connection` is `env_mgr`'s type.


def test_every_connection_a_double_or_a_real_one_provides_what_tools_reads() -> None:
    """**A Protocol is not enforced at runtime, so a missing member is silent.**

    `Connection` gained `describe()` so `remote.tools` could name the far side
    to the agent. Nothing checks that an implementation has it: a `Protocol`
    member is a type-checker's business, and a double that omits it stays green
    until production happens to call it — which is the *"wired but reached by
    nobody"* shape, at six occurrences in this repository and every one of them
    green unit tests over an unexercised seam. `StubStore.seal` is the nearest
    relative: absent for months while a broad `except` turned every
    `AttributeError` into a "refusal", 174 tests green and the seal never once
    running.

    So the members are read out of `tools.py` **by AST** rather than listed
    here. A list would be a second declaration of the contract and would go
    stale the next time a member is added — the failure this test exists to
    catch, reintroduced in the test itself.

    The checkers are imported from `tests/agent/test_doubles_conform.py` rather
    than copied: they are themselves tested there, in both directions, and a
    transcription would be an untested copy of a tested thing.
    """
    from pathlib import Path

    import env_mgr.remote.tools as tools_module
    from tests.agent.test_doubles_conform import attributes_read_on, missing, public
    from tests.env_mgr.test_sync import FakeTransport
    from tests.env_mgr.test_tools import RecordingConnection

    reads = attributes_read_on("conn", Path(tools_module.__file__))
    # Guard the instrument: if the AST reader silently found nothing, every
    # assertion below would pass vacuously.
    assert {"run", "push", "pull", "describe"} <= reads, reads

    for impl in (Ssh("somehost"), DockerExec("c"), LocalConnection(), RecordingConnection(), FakeTransport(far_exists=True)):
        assert not missing(reads, public(impl)), f"{type(impl).__name__} is missing members"


def test_the_conformance_rule_catches_a_connection_missing_describe() -> None:
    """The control. Without it the test above passes if `missing` stopped working.

    Deliberately shaped like the doubles as they were **before** `describe()`
    existed, because that is the omission that would actually happen: someone
    writes a new transport against the three methods they can see being called.
    """
    from pathlib import Path

    import env_mgr.remote.tools as tools_module
    from tests.agent.test_doubles_conform import attributes_read_on, missing, public

    class ConnectionWithoutDescribe:
        def run(self, argv, *, cwd=None, timeout=None):  # noqa: ANN001, ANN201
            raise AssertionError("never called")

        def push(self, local: str, remote: str):  # noqa: ANN201
            raise AssertionError("never called")

        def pull(self, remote: str, local: str):  # noqa: ANN201
            raise AssertionError("never called")

    reads = attributes_read_on("conn", Path(tools_module.__file__))
    assert missing(reads, public(ConnectionWithoutDescribe())) == {"describe"}
