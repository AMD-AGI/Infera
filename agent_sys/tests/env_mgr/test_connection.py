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
