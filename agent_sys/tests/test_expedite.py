"""Expedite — criterion 9.

A handoff-complete closure goes to the front. The rejection is what makes the
word mean something: an expedited task that still had to wait would be a
priority flag, not an expedite.
"""

import pytest

from agent_sys.models import TaskStatus

from .conftest import make_task, new_handoffs


def test_an_expedited_task_runs_before_earlier_ones(scheduler, runner):
    scheduler.submit(make_task(resources={"gpu": 8}))  # occupy the pool
    early = make_task(resources={"gpu": 8})
    scheduler.submit(early)

    urgent = make_task(resources={"gpu": 8})
    scheduler.expedite(urgent)

    runner.finish(runner.started[0])
    assert runner.started[-1] == urgent.id


def test_expedite_marks_the_task(scheduler, task_mgr):
    task = make_task(resources={"gpu": 1})
    scheduler.expedite(task)
    assert task_mgr.get(task.id).expedited


def test_an_ordinary_submission_is_not_expedited(scheduler, task_mgr):
    task = make_task()
    scheduler.submit(task)
    assert not task_mgr.get(task.id).expedited


def test_expediting_a_task_with_an_unmet_input_is_rejected(scheduler, task_mgr):
    task = make_task(inputs=new_handoffs(1))
    with pytest.raises(ValueError, match="not valid"):
        scheduler.expedite(task)
    assert task_mgr.all() == []


def test_expediting_a_task_with_a_declared_but_unwritten_input_is_rejected(scheduler, handoff_mgr):
    producer = make_task(outputs=new_handoffs(1))
    scheduler.submit(producer)  # declared, still CREATED

    with pytest.raises(ValueError, match="not valid"):
        scheduler.expedite(make_task(inputs=producer.outputs))


def test_expediting_a_task_whose_inputs_are_valid_is_accepted(
    scheduler, task_mgr, runner, registry
):
    producer = make_task(outputs=new_handoffs(1))
    scheduler.submit(producer)
    runner.produce(registry, producer.id)
    runner.finish(producer.id)

    urgent = make_task(inputs=producer.outputs)
    scheduler.expedite(urgent)
    assert task_mgr.get(urgent.id).status is TaskStatus.RUNNING


def test_an_expedited_task_with_no_inputs_is_accepted(scheduler, task_mgr):
    """Vacuously handoff-complete."""
    task = make_task()
    scheduler.expedite(task)
    assert task_mgr.get(task.id).status is TaskStatus.RUNNING


def test_expedited_tasks_keep_fifo_order_among_themselves(scheduler, runner):
    scheduler.submit(make_task(resources={"gpu": 8}))
    first, second = make_task(resources={"gpu": 8}), make_task(resources={"gpu": 8})
    scheduler.expedite(first)
    scheduler.expedite(second)

    runner.finish(runner.started[0])
    assert runner.started[-1] == first.id
    runner.finish(first.id)
    assert runner.started[-1] == second.id


def test_expedite_still_waits_for_resources(scheduler, task_mgr):
    """It reorders the queue; it does not preempt a running task."""
    scheduler.submit(make_task(resources={"gpu": 8}))
    urgent = make_task(resources={"gpu": 1})
    scheduler.expedite(urgent)
    assert task_mgr.get(urgent.id).status is TaskStatus.WAITING_RESOURCE


def test_a_rejected_expedite_leaves_no_trace(scheduler, task_mgr, handoff_mgr):
    task = make_task(inputs=new_handoffs(1), outputs=new_handoffs(1))
    with pytest.raises(ValueError):
        scheduler.expedite(task)

    assert task_mgr.all() == []
    assert handoff_mgr.all_ids() == []
    assert not task.expedited
