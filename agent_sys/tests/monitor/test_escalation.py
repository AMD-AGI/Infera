"""Escalation — criteria 12 and 17."""

from __future__ import annotations

from monitor import (
    ESCALATION_TARGET,
    TARGET_USER,
    EventKind,
    EventRecord,
    NullUserSink,
    PusherMonitor,
    Unit,
    event,
    reached_the_user,
)
from monitor.record import EVENT_KIND
from task_graph.registry import Registry

from .conftest import StubTask, StubTaskMgr


def kinds_for(registry: Registry, task: StubTask) -> list[str]:
    return [
        r["kind"]
        for r in registry.get("store_mgr").read_all(EVENT_KIND)
        if r["task_id"] == str(task.id)
    ]


def test_walks_parent_chain_to_root(
    registry: Registry, task_mgr: StubTaskMgr, monitor: PusherMonitor
) -> None:
    """Criterion 17: **a monitor that cannot resolve an exception escalates to
    the monitor of the parent task**, up to the root, and each escalation is
    recorded.

    **Up the task tree, never the monitor topology.** Global monitors are a flat
    pool; the tree that matters is `task_graph`'s, and `Task.parent` is the edge
    `unfold` sets. So the target is always *the monitor of my task's parent*,
    whichever kind either one is — and the scope always moves in the right
    direction, because a parent's zone contains its children's.
    """
    root = task_mgr.add(StubTask(parent=None))
    middle = task_mgr.add(StubTask(parent=root.id))
    leaf = task_mgr.add(StubTask(parent=middle.id))
    for task in (root, middle, leaf):
        monitor.set_task(task.id)

    sink = NullUserSink()
    monitor._sink = sink

    # Escalate from the leaf; then hand each forwarded report on, as the loop
    # would, so the whole chain is walked.
    monitor._escalate(Unit(leaf.id, (event(EventKind.VALIDATION_FAILED, leaf.id),)), "cannot fix")
    for _ in range(3):
        unit = monitor._buffer.get(0.05)
        if unit is None:
            break
        monitor._escalate(unit, "cannot fix")
        monitor._buffer.done(unit.task_id)

    for task in (leaf, middle, root):
        assert EventKind.ESCALATED.value in kinds_for(registry, task), (
            f"no escalation recorded at {task.id}"
        )
    assert sink.delivered, "the root's escalation target is the user"
    assert sink.delivered[-1][0].attributes["target"] == "user"


def test_the_escalated_record_is_rekeyed_to_the_parent(
    registry: Registry, task_mgr: StubTaskMgr, monitor: PusherMonitor
) -> None:
    """It must be: the parent's monitor runs it through `_transition`, which
    refuses a task that is not the one it is handling. A record keeping the
    child's `task_id` would be a scope violation on arrival."""
    parent = task_mgr.add(StubTask())
    child = task_mgr.add(StubTask(parent=parent.id))
    for task in (parent, child):
        monitor.set_task(task.id)

    original = event(EventKind.BUDGET_EXCEEDED, child.id, attempt=2)
    monitor._escalate(Unit(child.id, (original,)), "over budget")

    unit = monitor._buffer.get(0.05)
    assert unit is not None
    assert unit.task_id == parent.id
    forwarded = unit.newest
    assert forwarded.attributes["from_task"] == str(child.id)
    assert forwarded.kind is EventKind.BUDGET_EXCEEDED  # the payload is unchanged
    assert forwarded.id != original.id
    assert forwarded.fingerprint != original.fingerprint  # it names another task now


def test_unpushable_still_records(
    registry: Registry, task_mgr: StubTaskMgr, monitor: PusherMonitor
) -> None:
    """Criterion 12: **the ceiling is on the response, never on the reporting.**

    A task that failed its output validation is terminal with no agent running,
    so none of push / resume / restart applies. The alpha's monitor cannot fix
    such a branch and **must not leave it unremarked** — a dead branch nobody is
    told about is how a graph stops without anyone noticing. This is the defect
    main spec §10 recorded and principle 1 withdrew.
    """
    task = task_mgr.add(StubTask(parent=None))
    monitor.set_task(task.id)
    sink = NullUserSink()
    monitor._sink = sink

    monitor._run_guarded(
        task.id, monitor._handle, Unit(task.id, (event(EventKind.VALIDATION_FAILED, task.id),))
    )

    recorded = kinds_for(registry, task)
    assert EventKind.ESCALATED.value in recorded
    assert EventKind.PUSH_ATTEMPTED.value not in recorded, "there was nothing to push"
    assert sink.delivered
    assert "terminal" in sink.delivered[-1][1]


def test_escalate_makes_exactly_one_hop(task_mgr: StubTaskMgr, monitor: PusherMonitor) -> None:
    """The walk is one hop per monitor, not a loop inside one.

    That is why it **needs no visited set**: each monitor reports to its parent's
    and returns, and `unfold` sets `parent` on tasks it has just created and
    therefore cannot close a loop. A second guard for one invariant is the
    two-writers failure, so the absence is a decision rather than an oversight.
    """
    root = task_mgr.add(StubTask(parent=None))
    middle = task_mgr.add(StubTask(parent=root.id))
    leaf = task_mgr.add(StubTask(parent=middle.id))
    for task in (root, middle, leaf):
        monitor.set_task(task.id)

    monitor._escalate(Unit(leaf.id, (event(EventKind.VALIDATION_FAILED, leaf.id),)), "cannot fix")

    unit = monitor._buffer.get(0.05)
    assert unit is not None and unit.task_id == middle.id
    assert len(monitor._buffer) == 0, "one call walked more than one edge"


def test_reached_the_user_separates_a_resting_state_from_a_stall(
    registry: Registry, task_mgr: StubTaskMgr, monitor: PusherMonitor
) -> None:
    """`demo`'s requirement, as a question rather than two magic strings.

    > *A state the system is specified to rest in must be distinguishable, from
    > the outside, from one it is stuck in.*

    An escalation that reached the root and found no sink is **not** a stall —
    the system is doing exactly what spec §11 says. Both present as a task in
    `running` that stopped changing, so a stall detector that cannot tell them
    apart reports a specified resting state as a hang.

    The distinction is `target: "user"` and **not merely that an escalation
    exists** — an escalation still walking the tree is an ordinary failure. Both
    arms are asserted here for that reason.
    """
    root = task_mgr.add(StubTask(parent=None))
    child = task_mgr.add(StubTask(parent=root.id))
    for task in (root, child):
        monitor.set_task(task.id)

    # one hop: still walking the tree, not resting
    monitor._escalate(Unit(child.id, (event(EventKind.VALIDATION_FAILED, child.id),)), "cannot fix")
    mid = [
        EventRecord.model_validate(r)
        for r in registry.get("store_mgr").read_all(EVENT_KIND)
        if r["task_id"] == str(child.id)
    ]
    assert mid and not any(reached_the_user(r) for r in mid), "a hop up the tree is not resting"

    # the top: no parent left
    monitor._escalate(Unit(root.id, (event(EventKind.VALIDATION_FAILED, root.id),)), "cannot fix")
    top = [
        EventRecord.model_validate(r)
        for r in registry.get("store_mgr").read_all(EVENT_KIND)
        if r["task_id"] == str(root.id)
    ]
    assert any(reached_the_user(r) for r in top), "an escalation that ran out of tree"


def test_reached_the_user_is_false_for_every_other_kind(task_mgr: StubTaskMgr) -> None:
    """It asks one narrow thing. A `PUSH_ATTEMPTED` carrying the same attribute
    by accident is not an escalation that ran out of tree."""
    tid = task_mgr.add(StubTask()).id
    for kind in EventKind:
        r = event(kind, tid, attributes={ESCALATION_TARGET: TARGET_USER})
        assert reached_the_user(r) is (kind is EventKind.ESCALATED)
