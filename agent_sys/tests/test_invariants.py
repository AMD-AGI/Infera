"""The index invariant — criterion 12.

`_move` is the only thing that assigns `task.status` or mutates a pool, so the
index cannot disagree with the TaskMgr. This drives a long sequence of
operations and re-checks after every one.
"""

import random

from agent_sys.models import TaskStatus

from .conftest import make_task, new_handoffs, rebuild


def check(scheduler, task_mgr) -> None:
    tasks = task_mgr.all()
    indexed = set().union(*scheduler.pools.values()) if scheduler.pools else set()

    assert indexed == {t.id for t in tasks}, "the index and the TaskMgr disagree"
    for task in tasks:
        assert task.id in scheduler.pools[task.status], (
            f"{task.id} is {task.status.value} but not in that pool"
        )
    seen = [tid for pool in scheduler.pools.values() for tid in pool]
    assert len(seen) == len(set(seen)), "a task is in two pools at once"


def test_the_invariant_holds_through_a_long_mixed_sequence(scheduler, task_mgr, runner, registry):
    """A fixed sequence — deterministic, so a failure is reproducible."""
    rng = random.Random(20260820)
    (shared,) = new_handoffs(1)
    live: list = []
    performed: list[str] = []

    check(scheduler, task_mgr)

    for _ in range(200):
        choice = rng.randrange(7)

        if choice == 0:
            task = make_task(
                inputs=[shared] if rng.random() < 0.4 else [],
                outputs=[shared] if rng.random() < 0.3 else new_handoffs(1),
                resources={"gpu": rng.choice([0, 1, 4]), "token": 100},
            )
            scheduler.submit(task)
            live.append(task.id)
            performed.append("submit")

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
                tid for tid in runner.running if task_mgr.get(tid).status is TaskStatus.RUNNING
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

        check(scheduler, task_mgr)

    # The sequence is only meaningful if it actually reached every operation.
    assert set(performed) == {"submit", "remove", "finish", "stop", "ack", "resume", "update"}, (
        sorted(set(performed))
    )


def test_no_resource_leaks_after_everything_settles(scheduler, task_mgr, runner, registry):
    """A renewable pool must return to full once nothing is running."""
    tasks = [make_task(resources={"gpu": 2}) for _ in range(6)]
    for task in tasks:
        scheduler.submit(task)

    while runner.running:
        tid = next(iter(runner.running))
        runner.finish(tid)

    assert registry.get("resource:gpu").available == 8
    assert not scheduler.pools[TaskStatus.RUNNING]
    check(scheduler, task_mgr)


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
    from agent_sys.registry import resume_all

    resume_all(fresh)

    check(fresh.get("scheduler"), fresh.get("task_mgr"))
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
