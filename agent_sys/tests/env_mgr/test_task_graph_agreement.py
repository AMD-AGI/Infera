# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The seam, against the **real** `task_graph` types rather than the stubs.

`stubs.py` exists because this package was written while `task_graph` rev. 12
was being written, and the plan says to code against the documented shape. Now
that the shape is shipped, a stub that agrees with a document and disagrees with
the code would be the worst of the three outcomes — so this file imports the
real `Access`, `Grant`, `Permissions`, `Task`, `Execution` and `Handoff` and
drives `grants.resolve_all` with them.

Tests are not under `interfaces.md` §4's import rule, and `env_mgr` may import
`task_graph` in any case.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from env_mgr.fs.domain import DomainKind, DomainRegistry
from env_mgr.fs.layout import handoff_version_dir
from env_mgr.grants import mode_for, resolve_all
from env_mgr.protocols import Context, Mode, Tier, UnresolvedGrant
from task_graph import Access, Grant, Permissions
from task_graph.ids import AgentId, HandoffId
from task_graph.models import Execution, Handoff, HandoffVersion, Task

from . import stubs


@pytest.fixture
def store(tmp_path: Path) -> str:
    root = tmp_path / "store"
    root.mkdir()
    return str(root)


def _ctx(store: str, handoffs: dict, tmp_path: Path) -> Context:
    reg = DomainRegistry()
    reg.register("store", str(tmp_path / "root"), DomainKind.HANDOFF_STORAGE)
    return Context(
        domains=reg,
        handoffs=handoffs,
        store_root=store,
        main_repo="",
        mapping={},
        interpreter_grants=(),
        tier=Tier.PRODUCTION,
    )


def _handoff(hid: HandoffId, kind: str) -> Handoff:
    return Handoff(id=hid, type=kind, versions=[HandoffVersion(version=0)])


def _shape(annotation: object) -> str:
    """A type, normalised so a dataclass string and a pydantic object compare.

    `stubs.py` has ``from __future__ import annotations``, so its field types
    are already strings; pydantic's are objects. Module qualifiers are stripped
    from the real side because the stub cannot carry them.
    """
    import re

    if isinstance(annotation, type):
        return annotation.__name__
    return re.sub(r"\b[a-z_][A-Za-z0-9_]*\.", "", str(annotation))


def test_the_stub_matches_the_shipped_shape() -> None:
    """Every field the stub declares must exist on the real type **and have the
    same type**, because a stub that has drifted asserts the old contract with
    full confidence.

    **Names are not enough, and that is not hypothetical.** `validator` had a
    stub of this package's `ValidationZone` whose *field set* was still correct
    after `materials` changed from `tuple[str, ...]` to a mapping — so
    `tuple(placed.materials)` silently yielded **keys where paths were
    expected**, and their whole suite passed. Comparing names would not have
    caught it; comparing types does.

    Checking it here found the same defect one layer over: this stub declared
    ``Task.repos``, **which the real `Task` has never had**. `prepare` read it
    with `getattr` and got `()` against every real task — dead code that looked
    live because the stub had invented the field it needed.

    Subset, not equality: the stub deliberately carries only the fields this
    package reads. What it must not do is carry one the real type lacks, or
    carry one at the wrong type.
    """
    import dataclasses

    assert [m.value for m in stubs.Access] == [m.value for m in Access]

    pairs = (
        (stubs.Grant, Grant),
        (stubs.Permissions, Permissions),
        (stubs.Task, Task),
        (stubs.Execution, Execution),
        (stubs.Handoff, Handoff),
    )
    problems: list[str] = []
    for stub, real in pairs:
        for f in dataclasses.fields(stub):
            declared = real.model_fields.get(f.name)
            if declared is None:
                problems.append(f"{stub.__name__}.{f.name} is absent from the real type")
            elif f.type != _shape(declared.annotation):
                problems.append(
                    f"{stub.__name__}.{f.name}: stub {f.type!r}, real "
                    f"{_shape(declared.annotation)!r}"
                )
    assert not problems, "the stub has drifted from the shipped type:\n  " + "\n  ".join(problems)


def test_the_stub_check_can_tell_two_shapes_apart() -> None:
    """The negative control for the test above.

    A guard that has never been shown capable of failing is indistinguishable
    from one that cannot — `validator`'s lesson, applied to the guard written
    an hour after learning it. Both directions: a field the real type lacks,
    and a field at the wrong type.
    """
    import dataclasses

    @dataclasses.dataclass
    class Drifted:
        path: str
        access: Access
        kind: int  # the real one is `str | None`
        invented: str = ""

    seen = {
        f.name: (
            f.name in Grant.model_fields and f.type == _shape(Grant.model_fields[f.name].annotation)
        )
        for f in dataclasses.fields(Drifted)
    }
    assert seen == {"path": True, "access": True, "kind": False, "invented": False}


def test_resolve_all_against_real_permissions(store: str, tmp_path: Path) -> None:
    """A real frozen `Grant` on a real `Task`, resolved to a real store path."""
    hid = HandoffId.new()
    task = Task(
        agent_spec="writer",
        inputs=[hid],
        kinds={hid: "trace"},
        permissions=Permissions(grants=(Grant(kind="trace"),)),
    )
    execution = task.push_execution(AgentId.new(), {hid: 2})

    granted = resolve_all(task, execution, _ctx(store, {hid: _handoff(hid, "trace")}, tmp_path))
    assert [g.path for g in granted] == [os.path.join(handoff_version_dir(store, hid, 2), "content")]
    assert granted[0].mode is Mode.READ_EXEC


def test_a_write_grant_reaches_the_kernel_as_read_write(store: str, tmp_path: Path) -> None:
    """The two vocabularies, across the real seam. `Access` is what the author
    declared; `Mode` is what the kernel gets, and `READ_EXEC` has no
    declaration-side member to correspond to."""
    hid = HandoffId.new()
    task = Task(
        agent_spec="writer",
        outputs=[hid],
        permissions=Permissions(grants=(Grant(kind="report", access=Access.WRITE),)),
    )
    execution = task.push_execution(AgentId.new(), output_versions={hid: 0})

    granted = resolve_all(task, execution, _ctx(store, {hid: _handoff(hid, "report")}, tmp_path))
    assert granted[0].mode is Mode.READ_WRITE
    assert mode_for(Access.WRITE) is Mode.READ_WRITE
    assert mode_for(Access.READ) is Mode.READ_EXEC


def test_an_unlabelled_output_still_raises(store: str, tmp_path: Path) -> None:
    """**Keep the raise.** `Task.kinds` now reaches `Handoff.type`, which closes
    the common case — but a task submitted without `kinds` still produces
    ``type == ""``, and `task_graph` asserts that it does. The omission is still
    expressible, so being loud still costs one raise and still catches it."""
    hid = HandoffId.new()
    task = Task(
        agent_spec="writer",
        inputs=[hid],
        permissions=Permissions(grants=(Grant(kind="trace"),)),
    )
    execution = task.push_execution(AgentId.new(), {hid: 0})

    with pytest.raises(UnresolvedGrant, match="no slot has that kind"):
        resolve_all(task, execution, _ctx(store, {hid: _handoff(hid, "")}, tmp_path))


def test_we_do_not_disagree_with_covers(store: str, tmp_path: Path) -> None:
    """`closure` design §6.3 forbids a second interpreter of a declared name.

    `Permissions.covers` is `task_graph`'s lookup and this module's resolution is
    a different function over the same field, so the two must agree about which
    kinds are named. Asserted rather than assumed, in both directions and for
    the WRITE-implies-READ rule that only `covers` knows.
    """
    a, b = HandoffId.new(), HandoffId.new()
    permissions = Permissions(
        grants=(Grant(kind="trace"), Grant(kind="report", access=Access.WRITE))
    )
    task = Task(agent_spec="w", inputs=[a, b], permissions=permissions)
    execution = task.push_execution(AgentId.new(), {a: 0, b: 0})
    handoffs = {a: _handoff(a, "trace"), b: _handoff(b, "report")}

    granted = resolve_all(task, execution, _ctx(store, handoffs, tmp_path))
    # Three paths for two grants: the WRITE one resolves to `content/` **and** a
    # `claim/` sibling, because the manifest is the seal and an agent granted
    # `v<N>/` could publish its own unsealed version. `covers` is a question
    # about *kinds* and is unaffected -- which is the agreement being asserted.
    assert len(granted) == 3

    assert permissions.covers("trace", Access.READ) is True
    assert permissions.covers("report", Access.READ) is True  # WRITE implies READ
    assert permissions.covers("trace", Access.WRITE) is False
    # And a kind neither names resolves to nothing, loudly, on both sides.
    assert permissions.covers("absent", Access.READ) is False
    with pytest.raises(UnresolvedGrant):
        resolve_all(
            Task(
                agent_spec="w",
                inputs=[a],
                permissions=Permissions(grants=(Grant(kind="absent"),)),
            ),
            execution,
            _ctx(store, handoffs, tmp_path),
        )


def test_grant_path_is_stored_verbatim_and_we_are_the_ones_who_refuse(
    store: str, tmp_path: Path
) -> None:
    """`task_graph` does not normalise `Grant.path`, by design — it interprets
    none of it. So the canonical-form rule is entirely ours, and a
    non-canonical path must be refused **here** or nowhere."""
    assert Grant(path="/usr/../etc").path == "/usr/../etc"
    task = Task(agent_spec="w", permissions=Permissions(grants=(Grant(path="/usr/../etc"),)))
    execution = task.push_execution(AgentId.new())
    with pytest.raises(UnresolvedGrant, match="canonical"):
        resolve_all(task, execution, _ctx(store, {}, tmp_path))


def test_prepare_takes_a_real_execution(tmp_path: Path) -> None:
    """A retry gets a different granted set, on the real `Execution`.

    `Task.push_execution` sets `attempt = len(history)`, so the zone name and
    the resolved version both move with the attempt without this module
    computing either.
    """
    from env_mgr.fs import layout

    reg = DomainRegistry()
    reg.register("store", str(tmp_path / "root"), DomainKind.HANDOFF_STORAGE)
    task = Task(agent_spec="w")

    first = task.push_execution(AgentId.new())
    task.close_execution(task.status)
    zone_a = layout.create(task, first, reg)
    second = task.push_execution(AgentId.new())
    zone_b = layout.create(task, second, reg)

    assert (first.attempt, second.attempt) == (0, 1)
    assert zone_a.root != zone_b.root
