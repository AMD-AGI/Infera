"""Completion — criteria 4, 7 and 13.

There is no result object. The runner reports a status and what it spent; the
scheduler reads output versions from the HandoffMgr for itself, exactly as it
read the input versions at dispatch.
"""

import pytest

from agent_sys.models import TaskStatus

from .conftest import gpu, make_task, new_handoffs, token


def test_completion_releases_a_renewable_in_full(scheduler, runner, registry):
    """Criterion 4."""
    task = make_task(resources={"gpu": 3})
    scheduler.submit(task)
    runner.finish(task.id)
    assert gpu(registry).available == 8


def test_completion_settles_a_consumable_at_reported_usage(scheduler, runner, registry):
    task = make_task(resources={"token": 500})
    scheduler.submit(task)
    runner.finish(task.id, usage={"token": 120})
    assert token(registry).available == 1_000_000 - 120


def test_a_completion_reporting_no_usage_charges_the_whole_reservation(scheduler, runner, registry):
    task = make_task(resources={"token": 500})
    scheduler.submit(task)
    runner.finish(task.id, usage={})
    assert token(registry).available == 1_000_000 - 500


def test_usage_for_a_renewable_is_ignored(scheduler, runner, registry):
    task = make_task(resources={"gpu": 3})
    scheduler.submit(task)
    runner.finish(task.id, usage={"gpu": 1})
    assert gpu(registry).available == 8


def test_a_failure_releases_just_as_a_success_does(scheduler, runner, registry):
    task = make_task(resources={"gpu": 3, "token": 500})
    scheduler.submit(task)
    runner.finish(task.id, TaskStatus.FAILED, usage={"token": 200})

    assert gpu(registry).available == 8
    assert token(registry).available == 1_000_000 - 200


# -------------------------------------------------------------- the record


def test_completion_records_the_output_versions_the_run_wrote(scheduler, runner, registry):
    task = make_task(outputs=new_handoffs(2))
    scheduler.submit(task)
    runner.produce(registry, task.id)
    runner.finish(task.id)

    execution = registry.get("task_mgr").get(task.id).current
    assert execution.output_versions == dict.fromkeys(task.outputs, 0)
    assert execution.outcome is TaskStatus.SUCCEEDED
    assert not execution.is_open


def test_an_output_the_agent_never_wrote_is_still_recorded_at_its_declared_version(
    scheduler, runner, registry
):
    task = make_task(outputs=new_handoffs(1))
    scheduler.submit(task)
    runner.finish(task.id)  # no produce()

    assert registry.get("task_mgr").get(task.id).current.output_versions == {task.outputs[0]: 0}
    assert not registry.get("handoff_mgr").check_if_latest_valid(task.outputs[0])


def test_completion_persists(scheduler, runner, store):
    task = make_task()
    scheduler.submit(task)
    runner.finish(task.id)

    record = store.read("task", str(task.id))
    assert record["status"] == "succeeded"
    assert record["history"][0]["outcome"] == "succeeded"


# ----------------------------------- status and handoff validity are separate


def test_a_succeeded_task_may_leave_an_invalid_output(scheduler, runner, registry):
    """Criterion 13: the two say different things and the scheduler conflates
    neither. Whether the work is usable is the handoff's answer."""
    task = make_task(outputs=new_handoffs(1))
    scheduler.submit(task)
    runner.produce(registry, task.id, valid=False)
    runner.finish(task.id, TaskStatus.SUCCEEDED)

    assert registry.get("task_mgr").get(task.id).status is TaskStatus.SUCCEEDED
    assert not registry.get("handoff_mgr").check_if_latest_valid(task.outputs[0])


def test_a_failed_task_may_leave_a_valid_output(scheduler, runner, registry):
    task = make_task(outputs=new_handoffs(1))
    scheduler.submit(task)
    runner.produce(registry, task.id, valid=True)
    runner.finish(task.id, TaskStatus.FAILED)

    assert registry.get("task_mgr").get(task.id).status is TaskStatus.FAILED
    assert registry.get("handoff_mgr").check_if_latest_valid(task.outputs[0])


def test_a_consumer_unblocks_on_handoff_validity_not_on_task_success(scheduler, runner, registry):
    producer = make_task(outputs=new_handoffs(1))
    consumer = make_task(inputs=producer.outputs)
    scheduler.submit(producer)
    scheduler.submit(consumer)

    runner.produce(registry, producer.id, valid=True)
    runner.finish(producer.id, TaskStatus.FAILED)

    assert registry.get("task_mgr").get(consumer.id).status is TaskStatus.RUNNING


def test_a_consumer_stays_blocked_when_a_succeeded_producer_wrote_nothing_valid(
    scheduler, runner, registry
):
    producer = make_task(outputs=new_handoffs(1))
    consumer = make_task(inputs=producer.outputs)
    scheduler.submit(producer)
    scheduler.submit(consumer)

    runner.produce(registry, producer.id, valid=False)
    runner.finish(producer.id, TaskStatus.SUCCEEDED)

    assert registry.get("task_mgr").get(consumer.id).status is TaskStatus.WAITING_HANDOFF


# --------------------------------------------------------------- rejections


def test_completing_a_task_that_is_not_running_is_rejected(scheduler, runner):
    task = make_task()
    scheduler.submit(task)
    runner.finish(task.id)

    with pytest.raises(ValueError, match="succeeded"):
        scheduler.on_task_done(task.id, TaskStatus.SUCCEEDED, {})


def test_a_completion_arriving_after_a_stop_raises_into_the_runner(scheduler, runner):
    """Open question O2, asserted so the behaviour is deliberate rather than
    discovered. Nothing leaks: on_stopped still releases the resources."""
    task = make_task(resources={"gpu": 2})
    scheduler.submit(task)
    scheduler.stop(task.id)

    with pytest.raises(ValueError, match="stopping"):
        runner.finish(task.id)


def test_completion_triggers_the_next_dispatch(scheduler, runner, task_mgr):
    hog = make_task(resources={"gpu": 8})
    scheduler.submit(hog)
    queued = make_task(resources={"gpu": 8})
    scheduler.submit(queued)

    runner.finish(hog.id)
    assert task_mgr.get(queued.id).status is TaskStatus.RUNNING


def test_a_failed_run_leaves_its_dependents_waiting(scheduler, runner, registry):
    """Criterion 7: released, and nothing downstream moved."""
    producer = make_task(outputs=new_handoffs(1), resources={"gpu": 3})
    consumer = make_task(inputs=producer.outputs)
    scheduler.submit(producer)
    scheduler.submit(consumer)

    runner.finish(producer.id, TaskStatus.FAILED)

    assert gpu(registry).available == 8
    assert registry.get("task_mgr").get(consumer.id).status is TaskStatus.WAITING_HANDOFF


def test_an_output_left_generating_does_not_satisfy_a_consumer(
    scheduler, runner, registry, handoff_mgr, task_mgr
):
    """Criterion 7's second half."""
    producer = make_task(outputs=new_handoffs(1))
    consumer = make_task(inputs=producer.outputs)
    scheduler.submit(producer)
    scheduler.submit(consumer)

    handoff_mgr.get(producer.outputs[0]).open_next(
        producer.id, task_mgr.get(producer.id).current.agent_id
    )
    handoff_mgr.persist(producer.outputs[0])
    runner.finish(producer.id)

    assert not handoff_mgr.check_if_latest_valid(producer.outputs[0])
    assert task_mgr.get(consumer.id).status is TaskStatus.WAITING_HANDOFF
