"""The three phases — criteria 39, 40, 41, and criterion 8 extended.

A task runs input validations, then its main work, then output validations. Only
the middle one is a graph. The two validation phases are not tasks the scheduler
dispatches, they take no pool slot, and no policy orders them — which is what
makes the phase states a property of the *task* rather than of a queue.
"""

import pytest

from task_graph.models import PHASES, TaskStateError, TaskStatus
from task_graph.registry import resume_all

from .conftest import DISPATCHED, gpu, make_task, new_handoffs, rebuild


class SpyPolicy:
    """Records everything it was ever asked to order."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.seen: list = []

    def select(self, eligible, snapshot):
        self.seen.extend(eligible)
        return self.inner.select(eligible, snapshot)


# ------------------------------------------------------------- criterion 39


def test_a_validation_phase_is_invisible_to_the_scheduler(store):
    """One dispatch for the whole run; no pool holds a validator; the policy is
    never asked to order one. Asserted over a spy, not inferred."""
    from task_graph.policy import DepthFirstPolicy

    policy = SpyPolicy(DepthFirstPolicy())
    registry = rebuild(store, policy=policy)
    scheduler, runner, task_mgr = (
        registry.get("scheduler"),
        registry.get("runner"),
        registry.get("task_mgr"),
    )

    task = make_task()
    scheduler.submit(task)
    runner.advance(registry, task.id)  # -> RUNNING
    runner.advance(registry, task.id)  # -> OUTPUT_VALIDATING
    runner.finish(task.id)

    assert runner.started == [task.id]  # dispatched exactly once
    assert len(task_mgr.all()) == 1  # no validator became a task
    assert {t.id for t in policy.seen} == {task.id}
    assert sum(len(pool) for pool in scheduler.pools.values()) == 1


def test_the_phase_sequence_is_the_observable_states(scheduler, runner, registry, task_mgr):
    seen = [task_mgr.all()]
    task = make_task()
    scheduler.submit(task)
    observed = [task_mgr.get(task.id).status]
    for _ in range(2):
        runner.advance(registry, task.id)
        observed.append(task_mgr.get(task.id).status)
    assert observed == list(PHASES)
    assert seen  # the fixture built a registry


def test_a_runner_cannot_skip_a_phase_by_advancing_twice(scheduler, task_mgr):
    """The guard is what makes the sequence enforceable rather than advisory."""
    task = make_task()
    scheduler.submit(task)
    with pytest.raises(TaskStateError, match="phase sequence"):
        task.enter_phase(TaskStatus.OUTPUT_VALIDATING)


def test_advancing_past_the_last_phase_is_rejected(scheduler, runner, registry):
    task = make_task()
    scheduler.submit(task)
    runner.advance(registry, task.id)
    runner.advance(registry, task.id)
    with pytest.raises(TaskStateError, match="phase sequence"):
        task.enter_phase(TaskStatus.SUCCEEDED)


def test_a_queued_task_cannot_enter_a_phase(scheduler):
    task = make_task(inputs=new_handoffs(1))
    scheduler.submit(task)
    with pytest.raises(TaskStateError, match="expected a phase state"):
        task.enter_phase(TaskStatus.RUNNING)


# ------------------------------------------------------------- criterion 40


def test_a_leaf_holds_one_lease_across_all_three_phases(scheduler, runner, registry):
    """Acquired once at the WAITING_RESOURCE transition, released once at the
    terminal state. Every pool unchanged between the phase transitions."""
    task = make_task(resources={"gpu": 3})
    scheduler.submit(task)

    after_dispatch = gpu(registry).available
    assert after_dispatch == 5

    runner.advance(registry, task.id)
    assert gpu(registry).available == after_dispatch
    runner.advance(registry, task.id)
    assert gpu(registry).available == after_dispatch

    runner.finish(task.id)
    assert gpu(registry).available == 8


def test_output_validation_needs_no_second_admission(scheduler, runner, registry, task_mgr):
    """The pool is fully occupied by the task itself; if the last phase had to
    re-acquire, it could never start."""
    task = make_task(resources={"gpu": 8})
    scheduler.submit(task)
    runner.advance(registry, task.id)
    runner.advance(registry, task.id)
    assert task_mgr.get(task.id).status is TaskStatus.OUTPUT_VALIDATING
    assert gpu(registry).available == 0


# ------------------------------------------------------------- criterion 41


def test_a_skipped_phase_advances_and_is_reported(scheduler, runner, registry, task_mgr):
    task = make_task()
    scheduler.submit(task)
    runner.skip_phase(registry, task.id, "already validated by an earlier run")

    assert task_mgr.get(task.id).status is TaskStatus.RUNNING
    assert runner.skipped == [
        (task.id, TaskStatus.INPUT_VALIDATING, "already validated by an earlier run")
    ]
    assert "already validated" in task_mgr.get(task.id).current.detail


def test_a_skip_is_not_silent_in_the_persisted_record(scheduler, runner, registry, store):
    task = make_task()
    scheduler.submit(task)
    runner.skip_phase(registry, task.id, "config")
    record = store.read("task", str(task.id))
    assert "skipped input_validating: config" in record["history"][0]["detail"]


# ---------------------------------------------------------- stop, all three


@pytest.mark.parametrize("advances", [0, 1, 2])
def test_stop_is_accepted_in_all_three_phase_states(
    scheduler, runner, registry, task_mgr, advances
):
    """All three are a running task from the outside."""
    task = make_task()
    scheduler.submit(task)
    for _ in range(advances):
        runner.advance(registry, task.id)
    assert task_mgr.get(task.id).status is PHASES[advances]

    scheduler.stop(task.id)
    assert task_mgr.get(task.id).status is TaskStatus.STOPPING
    runner.ack_stop(task.id)
    assert task_mgr.get(task.id).status is TaskStatus.SUSPENDED


@pytest.mark.parametrize("advances", [0, 1, 2])
def test_completion_is_accepted_in_all_three_phase_states(
    scheduler, runner, registry, task_mgr, advances
):
    task = make_task(resources={"gpu": 2})
    scheduler.submit(task)
    for _ in range(advances):
        runner.advance(registry, task.id)
    runner.finish(task.id)
    assert task_mgr.get(task.id).status is TaskStatus.SUCCEEDED
    assert gpu(registry).available == 8


# ------------------------------------------------- criterion 8, extended


@pytest.mark.parametrize("advances", [0, 1, 2])
def test_recovery_demotes_all_three_phase_states_identically(store, advances):
    """The lease is gone in each, so there is nothing to distinguish them."""
    registry = rebuild(store)
    scheduler, runner = registry.get("scheduler"), registry.get("runner")
    task = make_task(resources={"gpu": 4})
    scheduler.submit(task)
    for _ in range(advances):
        runner.advance(registry, task.id)

    fresh = rebuild(store)
    resume_all(fresh)

    restored = fresh.get("task_mgr").get(task.id)
    # It is demoted to WAITING_RESOURCE and then dispatched again in the same
    # pass, so what recovery is asserted on is that it holds a fresh lease and
    # a second execution record.
    assert restored.status is DISPATCHED
    assert len(restored.history) == 2
    assert fresh.get("resource:gpu").available == 4


def test_a_demoted_phase_state_that_cannot_be_redispatched_waits(store):
    """With the pool too small to re-admit it, the demotion itself is visible."""
    from task_graph.resource import GpuMgr

    registry = rebuild(store)
    scheduler = registry.get("scheduler")
    task = make_task(resources={"gpu": 8})
    scheduler.submit(task)
    registry.get("runner").advance(registry, task.id)

    fresh = rebuild(store)
    fresh.register("resource:gpu", GpuMgr(fresh, capacity=2))
    resume_all(fresh)
    assert fresh.get("task_mgr").get(task.id).status is TaskStatus.WAITING_RESOURCE
