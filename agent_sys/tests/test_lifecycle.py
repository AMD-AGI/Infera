"""Lifecycle — criteria 5, 6, 11 and 23.

Stop is a request, not a state change: the task is SUSPENDED only when the
runner acknowledges.
"""

import pytest

from agent_sys.models import TaskStatus

from .conftest import gpu, make_task, new_handoffs, token

# --------------------------------------------------------------- remove_queued


def test_removing_a_queued_task_cancels_it(scheduler, task_mgr, runner):
    """Criterion 5."""
    task = make_task(inputs=new_handoffs(1))
    scheduler.submit(task)
    scheduler.remove_queued(task.id)

    assert task_mgr.get(task.id).status is TaskStatus.CANCELLED
    assert scheduler.pools[TaskStatus.CANCELLED] == {task.id}
    assert runner.started == []


def test_removing_an_eligible_task_cancels_it(scheduler, task_mgr):
    scheduler.submit(make_task(resources={"gpu": 8}))
    queued = make_task(resources={"gpu": 1})
    scheduler.submit(queued)

    scheduler.remove_queued(queued.id)
    assert task_mgr.get(queued.id).status is TaskStatus.CANCELLED


def test_removing_a_running_task_is_rejected(scheduler, task_mgr):
    """Criterion 11: stop it instead."""
    task = make_task()
    scheduler.submit(task)
    with pytest.raises(ValueError, match="running"):
        scheduler.remove_queued(task.id)
    assert task_mgr.get(task.id).status is TaskStatus.RUNNING


def test_a_cancelled_task_is_not_dispatched_when_room_appears(scheduler, runner):
    hog = make_task(resources={"gpu": 8})
    scheduler.submit(hog)
    queued = make_task(resources={"gpu": 1})
    scheduler.submit(queued)
    scheduler.remove_queued(queued.id)

    runner.finish(hog.id)
    assert queued.id not in runner.started


# ------------------------------------------------------------------ stop


def test_stop_asks_the_runner_and_waits_for_the_acknowledgement(scheduler, task_mgr, runner):
    """Criterion 6."""
    task = make_task(resources={"gpu": 2})
    scheduler.submit(task)

    scheduler.stop(task.id)
    assert task_mgr.get(task.id).status is TaskStatus.STOPPING
    assert runner.stop_requested == [task.id]

    runner.ack_stop(task.id)
    assert task_mgr.get(task.id).status is TaskStatus.SUSPENDED


def test_the_resources_are_held_until_the_acknowledgement(scheduler, runner, registry):
    task = make_task(resources={"gpu": 2})
    scheduler.submit(task)

    scheduler.stop(task.id)
    assert gpu(registry).available == 6

    runner.ack_stop(task.id)
    assert gpu(registry).available == 8


def test_a_stop_settles_a_consumable_at_the_full_reservation(scheduler, runner, registry):
    """No usage figures arrive with an acknowledgement, so assuming the agent
    spent what it reserved is the safe direction for a budget."""
    task = make_task(resources={"token": 500})
    scheduler.submit(task)
    scheduler.stop(task.id)
    runner.ack_stop(task.id)

    assert token(registry).available == 1_000_000 - 500


def test_stopping_closes_the_execution_record(scheduler, task_mgr, runner):
    task = make_task()
    scheduler.submit(task)
    scheduler.stop(task.id)
    runner.ack_stop(task.id)

    top = task_mgr.get(task.id).current
    assert not top.is_open
    assert top.outcome is TaskStatus.SUSPENDED


def test_stopping_a_queued_task_is_rejected(scheduler):
    task = make_task(inputs=new_handoffs(1))
    scheduler.submit(task)
    with pytest.raises(ValueError, match="waiting_handoff"):
        scheduler.stop(task.id)


def test_stopping_a_finished_task_is_rejected(scheduler, runner):
    task = make_task()
    scheduler.submit(task)
    runner.finish(task.id)
    with pytest.raises(ValueError, match="succeeded"):
        scheduler.stop(task.id)


def test_a_stop_frees_room_for_a_queued_task(scheduler, task_mgr, runner):
    hog = make_task(resources={"gpu": 8})
    scheduler.submit(hog)
    queued = make_task(resources={"gpu": 8})
    scheduler.submit(queued)

    scheduler.stop(hog.id)
    runner.ack_stop(hog.id)
    assert task_mgr.get(queued.id).status is TaskStatus.RUNNING


# ----------------------------------------------------------------- resume


def test_a_suspended_task_resumes_with_a_new_attempt(scheduler, task_mgr, agent_mgr, runner):
    task = make_task()
    scheduler.submit(task)
    first_agent = task_mgr.get(task.id).current.agent_id
    scheduler.stop(task.id)
    runner.ack_stop(task.id)

    scheduler.resume_task(task.id)

    restored = task_mgr.get(task.id)
    assert restored.status is TaskStatus.RUNNING
    assert [e.attempt for e in restored.history] == [0, 1]
    assert restored.current.agent_id != first_agent
    assert agent_mgr.get(restored.current.agent_id).task_id == task.id


def test_a_failed_task_resumes(scheduler, task_mgr, runner):
    task = make_task()
    scheduler.submit(task)
    runner.finish(task.id, TaskStatus.FAILED)

    scheduler.resume_task(task.id)
    assert task_mgr.get(task.id).status is TaskStatus.RUNNING


def test_a_resumed_task_whose_inputs_went_stale_waits_again(scheduler, task_mgr, runner, registry):
    """Eligibility is recomputed on resume, not remembered."""
    producer = make_task(outputs=new_handoffs(1))
    scheduler.submit(producer)
    runner.produce(registry, producer.id)
    runner.finish(producer.id)

    consumer = make_task(inputs=producer.outputs)
    scheduler.submit(consumer)
    runner.finish(consumer.id, TaskStatus.FAILED)

    # the slot is reopened by another producer and not yet sealed
    refresher = make_task(outputs=producer.outputs)
    scheduler.submit(refresher)
    registry.get("handoff_mgr").get(producer.outputs[0]).open_next(
        refresher.id, task_mgr.get(refresher.id).current.agent_id
    )

    scheduler.resume_task(consumer.id)
    assert task_mgr.get(consumer.id).status is TaskStatus.WAITING_HANDOFF


@pytest.mark.parametrize("status", ["waiting_handoff", "running", "succeeded", "cancelled"])
def test_resuming_a_task_that_is_not_stopped_or_failed_is_rejected(scheduler, runner, status):
    queued = status in ("waiting_handoff", "cancelled")
    task = make_task(inputs=new_handoffs(1) if queued else [])
    scheduler.submit(task)
    if status == "succeeded":
        runner.finish(task.id)
    elif status == "cancelled":
        scheduler.remove_queued(task.id)

    with pytest.raises(ValueError, match=status):
        scheduler.resume_task(task.id)


# -------------------------------------------------------------- update_task


def test_update_replaces_the_definition_and_keeps_the_id(scheduler, task_mgr):
    task = make_task(inputs=new_handoffs(1), resources={"gpu": 1})
    scheduler.submit(task)

    scheduler.update_task(task.id, agent_spec="tuner", resources={"gpu": 4})

    updated = task_mgr.get(task.id)
    assert updated.id == task.id
    assert updated.agent_spec == "tuner"
    assert updated.resources == {"gpu": 4}
    assert updated.status is TaskStatus.WAITING_HANDOFF


def test_update_keeps_the_place_in_fifo_order(scheduler, runner, task_mgr):
    """`model_copy` preserves created_at: an update does not cost a task its
    place in the queue."""
    scheduler.submit(make_task(resources={"gpu": 8}))
    first = make_task(resources={"gpu": 8})
    second = make_task(resources={"gpu": 8})
    scheduler.submit(first)
    scheduler.submit(second)

    scheduler.update_task(first.id, agent_spec="tuner")
    assert task_mgr.get(first.id).created_at == first.created_at

    runner.finish(runner.started[0])
    assert runner.started[-1] == first.id


def test_update_clears_the_history(scheduler, task_mgr, runner):
    task = make_task()
    scheduler.submit(task)
    runner.finish(task.id, TaskStatus.FAILED)
    scheduler.resume_task(task.id)
    runner.finish(task.id, TaskStatus.FAILED)
    assert len(task_mgr.get(task.id).history) == 2

    scheduler.resume_task(task.id)  # back to running; stop it to make it queued
    scheduler.stop(task.id)
    runner.ack_stop(task.id)
    with pytest.raises(ValueError):
        scheduler.update_task(task.id)  # SUSPENDED is not queued


def test_update_behaves_exactly_like_a_resubmission(scheduler, task_mgr, registry, store):
    """Criterion 23. `update_task` is literally remove_queued + submit, so this
    holds by construction; the test is what keeps it that way."""
    from .conftest import rebuild

    (hid,) = new_handoffs(1)
    original = make_task(inputs=[hid], resources={"gpu": 1})
    scheduler.submit(original)
    updated = scheduler.update_task(original.id, agent_spec="tuner", resources={"gpu": 4})

    # the manual arm: a fresh system, cancel then submit by hand
    other = rebuild(store=type(store)())
    twin = original.model_copy(deep=True)
    other.get("scheduler").submit(twin)
    other.get("scheduler").remove_queued(twin.id)
    manual = twin.model_copy(
        update={
            "agent_spec": "tuner",
            "resources": {"gpu": 4},
            "status": TaskStatus.WAITING_HANDOFF,
            "history": [],
        },
        deep=True,
    )
    other.get("scheduler").submit(manual)

    exclude = {"created_at"}  # the manual arm is constructed later
    assert updated.model_dump(exclude=exclude) == other.get("task_mgr").get(manual.id).model_dump(
        exclude=exclude
    )


def test_update_of_a_running_task_is_rejected(scheduler, task_mgr):
    task = make_task()
    scheduler.submit(task)
    with pytest.raises(ValueError, match="running"):
        scheduler.update_task(task.id, agent_spec="tuner")
    assert task_mgr.get(task.id).agent_spec == "profiler"


def test_update_redeclares_outputs_without_destroying_written_versions(
    scheduler, registry, runner, task_mgr
):
    """`declare` is idempotent, which is what makes this safe."""
    (hid,) = new_handoffs(1)
    producer = make_task(outputs=[hid])
    scheduler.submit(producer)
    runner.produce(registry, producer.id, content="written")
    runner.finish(producer.id)

    consumer = make_task(inputs=new_handoffs(1), outputs=[hid])
    scheduler.submit(consumer)
    scheduler.update_task(consumer.id, agent_spec="tuner")

    handoff = registry.get("handoff_mgr").get(hid)
    assert handoff.latest.content == "written"
    assert len(handoff.versions) == 1


def test_a_rejected_update_leaves_the_task_where_it_was(scheduler, task_mgr):
    """`update_task` is cancel-then-submit, so a submit that rejects would
    otherwise silently destroy the very task it was asked to change."""
    task = make_task(inputs=new_handoffs(1), resources={"gpu": 2})
    scheduler.submit(task)

    with pytest.raises(ValueError, match="tpu"):
        scheduler.update_task(task.id, resources={"tpu": 1})

    survivor = task_mgr.get(task.id)
    assert survivor.status is TaskStatus.WAITING_HANDOFF
    assert survivor.resources == {"gpu": 2}  # the old definition, unchanged
    assert survivor.created_at == task.created_at  # and its place in the queue
    assert task.id in scheduler.pools[TaskStatus.WAITING_HANDOFF]


def test_a_rejected_update_does_not_mutate_the_stored_task(scheduler, store):
    task = make_task(inputs=new_handoffs(1), resources={"gpu": 2})
    scheduler.submit(task)

    with pytest.raises(KeyError):
        scheduler.update_task(task.id, agent_spec="nope")

    assert store.read("task", str(task.id))["agent_spec"] == "profiler"
    assert store.read("task", str(task.id))["status"] == "waiting_handoff"
