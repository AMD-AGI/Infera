# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Criterion 18, and criterion 10 on the remote surface.

Spec §5.5: an agent given a natural-language description of how to sync a
directory will improvise, and the improvisation will be wrong in a way nobody
notices. A tool call has a schema, a name, and a result.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from env_mgr.fs.zone import Zone
from env_mgr.remote.connection import LocalConnection
from env_mgr.remote.tools import REMOTE_CALL_SECONDS, tools
from task_graph.ids import TaskId


@pytest.fixture
def far_root(tmp_path: Path) -> str:
    """The zone's twin on the far side, at a **different path**.

    Different on purpose: when the two coincide -- which is what a *strong*
    mapping is -- a tool that confuses them looks correct. That is exactly how
    `tools()` passing the local `zone.root` as a remote `cwd` survived.

    Created, because `sync` creates it before any tool runs — and because a
    `LocalConnection` really does `cd` there, so a fixture that only named the
    path would make every `env_remote_run` fail for a reason unrelated to what
    is being tested.
    """
    far = tmp_path / "far" / "zone"
    far.mkdir(parents=True)
    return str(far)


@pytest.fixture
def zone(tmp_path: Path) -> Zone:
    root = tmp_path / "zone"
    (root / "out").mkdir(parents=True)
    (root / "out" / "result.txt").write_text("result")
    (tmp_path / "outside").mkdir()
    return Zone(TaskId.new(), 0, str(root.resolve()))


def _by_name(zone: Zone, far_root: str) -> dict[str, object]:
    return {t.name: t for t in tools(LocalConnection(), zone, far_root)}


# ---------------------------------------------------------- criterion 18


def test_the_descriptions_name_the_far_side_so_a_package_need_not(
    far_root: str, zone: Zone
) -> None:
    """**Criterion 18's other half: a tool call has a name, a schema — and an
    address.**

    The descriptions said *"the remote side of this task's mapping"*, which is
    true and tells an agent nothing about *where its work goes*. Measured in run
    `20260901T080901-50ecb9`: the agent's first act was to ask
    `env_remote_run(["hostname","-f"])`, because only its task readme said the
    work was remote and the tool surface did not.

    That made the fact a package's to remember, and a package that must remember
    is one package away from a package that forgets — the shape of the five
    mechanisms here that were wired, correct and reached by nobody. This asserts
    `env_mgr` says it itself.

    **Asserted against `Ssh` rather than `LocalConnection`**, because
    `LocalConnection`'s far side *is* this machine: a test that used it would
    pass while saying nothing, which is the "working instrument pointed at the
    safe case" failure. The control below is that the two answers differ.
    """
    from env_mgr.remote.connection import Ssh

    defs = {t.name: t for t in tools(Ssh("gpu-01"), zone, far_root)}

    run = defs["env_remote_run"].description
    assert "gpu-01" in run
    assert far_root in run
    # The half an agent acts on: a hostname alone leaves "is this elsewhere?" to
    # inference, and inference is unreliable exactly where it matters -- weights
    # and images visible from both sides make the two machines look alike.
    assert "different machine" in run

    for name in ("env_remote_push", "env_remote_pull"):
        assert "gpu-01" in defs[name].description, name
        assert far_root in defs[name].description, name

    # CONTROL: the same call over a connection whose far side really is this
    # host must NOT claim otherwise. Without this, `describe()` returning a
    # constant would satisfy every assertion above.
    local = {t.name: t for t in tools(LocalConnection(), zone, far_root)}
    assert "different machine" not in local["env_remote_run"].description
    assert "this machine" in local["env_remote_run"].description


def test_remote_tools_have_schemas(zone: Zone, far_root: str) -> None:
    defs = tools(LocalConnection(), zone, far_root)
    assert {t.name for t in defs} == {"env_remote_run", "env_remote_push", "env_remote_pull"}
    for tool in defs:
        assert tool.description
        assert tool.schema["type"] == "object"
        assert tool.schema["additionalProperties"] is False
        assert tool.schema["required"]
        for name in tool.schema["required"]:
            assert name in tool.schema["properties"], f"{tool.name} requires an undeclared {name}"


def test_tool_call_round_trip(zone: Zone, far_root: str) -> None:
    result = _by_name(zone, far_root)["env_remote_run"].call(command=["echo", "hello"])  # type: ignore[attr-defined]
    assert result["returncode"] == 0
    assert result["stdout"].strip() == "hello"


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync is not installed")
def test_push_and_pull_round_trip(zone: Zone, far_root: str) -> None:
    """`remote` is relative to the far-side zone, on the same terms as `cwd`.

    **This asserted the opposite and said so**: the destination was an absolute
    `tmp_path` outside the far root, under a comment reading *"`remote` is not
    confined to the remote zone"*. That was an accurate record of the code and
    of a surface where one of three far-side path arguments was checked and two
    were not — a rule a reader of the schemas could not see stopping.

    `design.md` §10.4 is unchanged and still governs: the far side **is** less
    isolated than the local one, deliberately, and `_inside_remote` is syntactic,
    so this is a guardrail against a malformed argument rather than a boundary.
    """
    far = Path(far_root) / "push-target"
    far.mkdir(parents=True)
    report = _by_name(zone, far_root)["env_remote_push"].call(path="out", remote="push-target")  # type: ignore[attr-defined]
    assert (far / "result.txt").read_text() == "result"
    assert report["conflicts"] == ()


def test_a_remote_command_carries_a_bound(zone: Zone, far_root: str) -> None:
    """`conn.run` accepts a `timeout` and `tools` passed none.

    Under `asyncio.to_thread` a `subprocess.run(timeout=None)` is not
    cancellable, so a wedged session pins a worker thread for the process
    lifetime and the SDK's own turn timeouts cannot reclaim it.

    The control is the second half: the bound is an *hour*, because a
    first-start CUDA-graph capture legitimately runs for tens of minutes and a
    tight default would turn a working bring-up into a failure. A test that only
    asserted "not None" would be satisfied by one second.
    """
    seen: dict[str, Any] = {}

    class Recording:
        def run(self, argv, *, cwd=None, timeout=None):  # noqa: ANN001, ANN202
            seen["timeout"] = timeout
            return subprocess.CompletedProcess(list(argv), 0, "", "")

        def push(self, local: str, remote: str):  # noqa: ANN202
            raise AssertionError

        def pull(self, remote: str, local: str):  # noqa: ANN202
            raise AssertionError

        def describe(self) -> str:
            return "somewhere"

    defs = {t.name: t for t in tools(Recording(), zone, far_root)}  # type: ignore[arg-type]
    defs["env_remote_run"].call(command=["true"])  # type: ignore[attr-defined]
    assert seen["timeout"] == REMOTE_CALL_SECONDS
    assert seen["timeout"] >= 1800, "a graph capture takes tens of minutes; do not cut one short"


@pytest.mark.parametrize("proposal", ["/etc/cron.d", "../outside", "x/../../outside"])
def test_the_remote_argument_cannot_leave_the_far_zone(
    zone: Zone, far_root: str, proposal: str
) -> None:
    """CONTROL for the test above, and the reason this pair is not vacuous.

    Without it, dropping `_inside_remote` from `env_remote_push`/`pull` would
    leave every other assertion in this file green — a relative `remote` reaches
    the same place either way, so only a *refusal* discriminates.

    `/etc/cron.d` and `/home/other/.ssh/id_rsa` are the two the review named.

    **Note for anyone re-running this against the unfixed code**: it does not
    merely fail, it writes. A relative `remote` handed straight to `rsync`
    resolves against the *process's* working directory, so `"../outside"` put a
    real directory beside the repository root — observed 2026-09-01 while
    checking that this control is not vacuous. That is the finding demonstrating
    itself, and it is a better argument for the guardrail than the docstring is.
    """
    by_name = _by_name(zone, far_root)
    with pytest.raises(PermissionError):
        by_name["env_remote_push"].call(path="out", remote=proposal)  # type: ignore[attr-defined]
    with pytest.raises(PermissionError):
        by_name["env_remote_pull"].call(remote=proposal, path="out")  # type: ignore[attr-defined]


# ------------------------------- criterion 10, on this surface too


def test_tool_takes_no_zone_argument(zone: Zone, far_root: str) -> None:
    """Closing over the zone is what makes criterion 10 true here: the zone root
    is never taken from agent-supplied input, because the tool does not accept
    one."""
    for tool in tools(LocalConnection(), zone, far_root):
        properties = tool.schema["properties"]
        assert "zone" not in properties
        assert "working_directory" not in properties
        assert (
            "cwd" not in properties or "relative to your zone" in properties["cwd"]["description"]
        )


@pytest.mark.parametrize("proposal", ["/etc", "../outside", "out/../../outside"])
def test_a_path_argument_cannot_leave_the_zone(zone: Zone, far_root: str, proposal: str) -> None:
    with pytest.raises(PermissionError):
        _by_name(zone, far_root)["env_remote_push"].call(path=proposal, remote="/tmp/x")  # type: ignore[attr-defined]


# ------------------------------------------- the far side is not the near side


class RecordingConnection:
    """A `Connection` that records the `cwd` it was asked for and runs nothing.

    Structural, not a subclass: `Connection` is a `Protocol`, and a stub that had
    to inherit would not be testing what the production classes satisfy.
    """

    def __init__(self) -> None:
        self.cwds: list[str | None] = []

    def run(self, argv, *, cwd=None, timeout=None):  # noqa: ANN001, ANN201
        import subprocess

        self.cwds.append(cwd)
        return subprocess.CompletedProcess(list(argv), 0, "", "")

    def push(self, local: str, remote: str):  # noqa: ANN201
        raise AssertionError("not used here")

    def pull(self, remote: str, local: str):  # noqa: ANN201
        raise AssertionError("not used here")

    def describe(self) -> str:
        """`Connection` gained this so `tools` could name the far side to the
        agent. A structural stub has to grow it too — which is the point of
        declaring it on the Protocol rather than reaching for it with `getattr`:
        the omission is a failure here, loudly, instead of a fourth transport
        being described to a model as *this machine* while executing elsewhere.
        """
        return "a recording stub"


def test_a_remote_command_runs_in_the_remote_zone_not_the_local_one(
    zone: Zone, far_root: str
) -> None:
    """**The defect this signature exists to fix.**

    `tools()` took `(conn, zone)` and passed `zone.root` — a *local* absolute
    path — as the `cwd` of a command executed on another machine. Over ssh that
    is `cd /var/tmp/…` on a host where the mirror lives at `/data/…`: it finds
    nothing, always.

    The reason it survived a read is in the fixture: **the only configuration
    where it appears to work is a strong mapping, where the two paths are equal
    by definition.** So this test insists they differ, and asserts on the value
    the connection was asked for rather than on whether the call succeeded.
    """
    conn = RecordingConnection()
    by_name = {t.name: t for t in tools(conn, zone, far_root)}

    by_name["env_remote_run"].call(command=["true"])
    by_name["env_remote_run"].call(command=["true"], cwd="out")

    assert conn.cwds == [far_root, f"{far_root}/out"]
    assert zone.root not in conn.cwds, "a local path was used as a remote working directory"


@pytest.mark.parametrize("proposal", ["/etc", "../outside", "out/../../outside"])
def test_a_remote_cwd_cannot_leave_the_remote_zone(
    zone: Zone, far_root: str, proposal: str
) -> None:
    """Criterion 10 on the far side, where `contained` cannot help.

    `contained` resolves both sides and needs the root to exist *here*; against a
    remote root it would deny everything. The syntactic check is what runs, and
    its control is the test above — a legitimate relative `cwd` must still be
    accepted, or this rule would be refusing everything and proving nothing.
    """
    conn = RecordingConnection()
    by_name = {t.name: t for t in tools(conn, zone, far_root)}
    with pytest.raises(PermissionError):
        by_name["env_remote_run"].call(command=["true"], cwd=proposal)
    assert conn.cwds == [], "the command must be refused before it is run"


def test_the_refusal_names_the_side_the_argument_was_about(zone: Zone, far_root: str) -> None:
    """These messages are read by a model — measured: they arrive as `str(e)` on
    an `isError` result. A refusal about a remote path that named the *local*
    zone would send the agent to inspect a directory on the wrong machine."""
    conn = RecordingConnection()
    by_name = {t.name: t for t in tools(conn, zone, far_root)}

    with pytest.raises(PermissionError, match=r"remote"):
        by_name["env_remote_run"].call(command=["true"], cwd="/etc")
    # Its control: a refusal about a *local* argument still names the local zone,
    # because that one is about the agent's own working directory.
    with pytest.raises(PermissionError, match=r"your zone"):
        by_name["env_remote_push"].call(path="../outside", remote="/tmp/x")
