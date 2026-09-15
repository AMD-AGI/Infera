"""The index invariant — criterion 12.

`_move` is the only thing that assigns `task.status` or mutates a pool, so the
index cannot disagree with the TaskMgr. This drives a long sequence of
operations and re-checks after every one.
"""

import logging
import random

from task_graph.models import TaskStatus
from task_graph.resource import RenewableMgr

from .conftest import DISPATCHED, LIVE, make_task, new_handoffs, rebuild


def check(scheduler, task_mgr, registry=None) -> None:
    tasks = task_mgr.all()
    indexed = set().union(*scheduler.pools.values()) if scheduler.pools else set()

    assert indexed == {t.id for t in tasks}, "the index and the TaskMgr disagree"
    for task in tasks:
        assert task.id in scheduler.pools[task.status], (
            f"{task.id} is {task.status.value} but not in that pool"
        )
    seen = [tid for pool in scheduler.pools.values() for tid in pool]
    assert len(seen) == len(set(seen)), "a task is in two pools at once"

    if registry is None:
        return
    # Resource conservation. Free capacity plus every live reservation — plus,
    # for a consumable, everything already settled — must equal the total. This
    # is what catches a leaked or double-booked lease, which the pool/status
    # correspondence alone cannot see.
    for pool_mgr in registry.resolve("resource:*"):
        reserved = sum(t.resources.get(pool_mgr.name, 0.0) for t in tasks if t.status in LIVE)
        # A consumable shrinks as spend accrues, so `spent` is part of the sum.
        # It is the half where the subtle accounting lives: D12's bug was a
        # reservation leaking into the durable record, and a renewable-only
        # check could never have seen it.
        spent = 0.0 if isinstance(pool_mgr, RenewableMgr) else pool_mgr.spent
        assert pool_mgr.available + reserved + spent == pool_mgr.capacity, (
            f"{pool_mgr.name}: {pool_mgr.available} free + {reserved} reserved "
            f"+ {spent} spent != {pool_mgr.capacity}"
        )


def test_the_invariant_holds_through_a_long_mixed_sequence(
    scheduler, task_mgr, runner, registry, caplog
):
    """A fixed sequence — deterministic, so a failure is reproducible."""
    # A spec whose agent cannot be built: the launch fails after the lease is
    # taken, which is the path that exercises resource conservation.
    registry.get("agent_mgr").register("doomed")
    real_instantiate = registry.get("agent_mgr").instantiate

    def instantiate(spec, task_id):
        if spec == "doomed":
            raise RuntimeError("agent factory down")
        return real_instantiate(spec, task_id)

    registry.get("agent_mgr").instantiate = instantiate
    caplog.set_level(logging.CRITICAL, logger="task_graph.scheduler")  # the failures are expected

    rng = random.Random(20260820)
    (shared,) = new_handoffs(1)
    live: list = []
    performed: list[str] = []

    check(scheduler, task_mgr, registry)

    for _ in range(200):
        choice = rng.randrange(8)

        if choice == 0:
            task = make_task(
                inputs=[shared] if rng.random() < 0.4 else [],
                outputs=[shared] if rng.random() < 0.3 else new_handoffs(1),
                resources={"gpu": rng.choice([0, 1, 4]), "token": 100},
            )
            scheduler.submit(task)
            live.append(task.id)
            performed.append("submit")

        elif choice == 7:
            # A launch that fails after the lease is taken. Without this the
            # conservation half of `check` is never exercised, and a leaked
            # lease would pass the whole sequence.
            doomed = make_task(spec="doomed", resources={"gpu": rng.choice([1, 4])})
            scheduler.submit(doomed)
            live.append(doomed.id)
            performed.append("failed-launch")

        elif choice == 1 and live:
            tid = rng.choice(live)
            if task_mgr.get(tid).status in (
                TaskStatus.WAITING_HANDOFF,
                TaskStatus.WAITING_RESOURCE,
            ):
                scheduler.remove_queued(tid)
                performed.append("remove")

        elif choice in (2, 3):
            # `runner.running` still holds a task between stop() and its
            # acknowledgement, so pick on status rather than on membership.
            actually_running = [
                tid for tid in runner.running if task_mgr.get(tid).status is DISPATCHED
            ]
            if not actually_running:
                pass
            elif choice == 2:
                tid = rng.choice(actually_running)
                if rng.random() < 0.5:
                    runner.produce(registry, tid)
                runner.finish(
                    tid, TaskStatus.SUCCEEDED if rng.random() < 0.7 else TaskStatus.FAILED
                )
                performed.append("finish")
            else:
                scheduler.stop(rng.choice(actually_running))
                performed.append("stop")

        elif choice == 4 and runner.stop_requested:
            tid = runner.stop_requested.pop(0)
            if task_mgr.get(tid).status is TaskStatus.STOPPING:
                runner.ack_stop(tid)
                performed.append("ack")

        elif choice == 5 and live:
            tid = rng.choice(live)
            if task_mgr.get(tid).status in (TaskStatus.FAILED, TaskStatus.SUSPENDED):
                scheduler.resume_task(tid)
                performed.append("resume")

        elif choice == 6 and live:
            tid = rng.choice(live)
            if task_mgr.get(tid).status in (
                TaskStatus.WAITING_HANDOFF,
                TaskStatus.WAITING_RESOURCE,
            ):
                scheduler.update_task(tid, resources={"gpu": rng.choice([0, 2])})
                performed.append("update")

        check(scheduler, task_mgr, registry)

    # The sequence is only meaningful if it actually reached every operation.
    assert set(performed) == {
        "submit",
        "remove",
        "finish",
        "stop",
        "ack",
        "resume",
        "update",
        "failed-launch",
    }, sorted(set(performed))


def test_no_resource_leaks_after_everything_settles(scheduler, task_mgr, runner, registry):
    """A renewable pool must return to full once nothing is running."""
    tasks = [make_task(resources={"gpu": 2}) for _ in range(6)]
    for task in tasks:
        scheduler.submit(task)

    while runner.running:
        tid = next(iter(runner.running))
        runner.finish(tid)

    assert registry.get("resource:gpu").available == 8
    assert not scheduler.pools[DISPATCHED]
    check(scheduler, task_mgr, registry)


def test_a_queued_task_never_holds_a_resource(scheduler, task_mgr, registry):
    """The all-or-nothing consequence, stated as an invariant."""
    scheduler.submit(make_task(resources={"gpu": 5}))
    for _ in range(4):
        scheduler.submit(make_task(resources={"gpu": 4, "token": 100}))

    queued = task_mgr.by_status(TaskStatus.WAITING_RESOURCE)
    assert queued
    assert registry.get("resource:gpu").available == 3  # only the running task's 5
    assert registry.get("resource:token").available == 1_000_000


def test_the_invariant_survives_a_restart(scheduler, task_mgr, runner, registry, store):
    for _ in range(4):
        scheduler.submit(make_task(resources={"gpu": 3}, outputs=new_handoffs(1)))
    scheduler.submit(make_task(inputs=new_handoffs(1)))
    runner.finish(next(iter(runner.running)))

    fresh = rebuild(store)
    from task_graph.registry import resume_all

    resume_all(fresh)

    check(fresh.get("scheduler"), fresh.get("task_mgr"), fresh)
    assert len(fresh.get("task_mgr").all()) == 5


def test_move_is_idempotent_and_self_healing(scheduler, task_mgr):
    """Discarding from every pool rather than from the recorded status means a
    stale entry cannot survive, even if the stored status was already wrong."""
    task = make_task(inputs=new_handoffs(1))
    scheduler.submit(task)

    scheduler.pools[TaskStatus.RUNNING].add(task.id)  # corrupt the index by hand
    scheduler._move(task.id, TaskStatus.WAITING_HANDOFF)

    check(scheduler, task_mgr)
    assert scheduler.pools[TaskStatus.RUNNING] == set()
