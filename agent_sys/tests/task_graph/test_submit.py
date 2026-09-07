"""Submission — criteria 1, 2 and 35.

Where a task lands is decided by asking the handoffs, not by counting
dependencies: a task with no inputs is immediately eligible.
"""

import logging

import pytest

from task_graph.models import HandoffStatus, Task, TaskStatus
from task_graph.registry import Registry
from task_graph.resource import GpuMgr

from .conftest import DISPATCHED, make_task, new_handoffs


def test_a_task_with_no_inputs_is_immediately_eligible(scheduler, task_mgr):
    """Criterion 1. The pool is saturated first so the landing state is
    observable — otherwise submit dispatches it straight through to RUNNING."""
    scheduler.submit(make_task(resources={"gpu": 8}))

    task = make_task(resources={"gpu": 1})
    scheduler.submit(task)
    assert task_mgr.get(task.id).status is TaskStatus.WAITING_RESOURCE


def test_an_eligible_task_with_nothing_blocking_runs_at_once(scheduler, task_mgr):
    task = make_task()
    scheduler.submit(task)
    assert task_mgr.get(task.id).status is DISPATCHED


def test_a_task_with_an_unmet_input_waits_for_handoffs(scheduler, task_mgr):
    """Criterion 2."""
    task = make_task(inputs=new_handoffs(1))
    scheduler.submit(task)
    assert task_mgr.get(task.id).status is TaskStatus.WAITING_HANDOFF


def test_a_task_whose_inputs_are_already_valid_is_eligible(
    scheduler, handoff_mgr, task_mgr, runner, registry
):
    producer = make_task(outputs=new_handoffs(1))
    scheduler.submit(producer)
    runner.produce(registry, producer.id)
    runner.finish(producer.id)

    consumer = make_task(inputs=producer.outputs)
    scheduler.submit(consumer)
    assert task_mgr.get(consumer.id).status is DISPATCHED


def test_all_inputs_must_be_valid_not_just_one(scheduler, task_mgr, runner, registry):
    first, second = new_handoffs(2)
    producer = make_task(outputs=[first])
    scheduler.submit(producer)
    runner.produce(registry, producer.id)
    runner.finish(producer.id)

    consumer = make_task(inputs=[first, second])
    scheduler.submit(consumer)
    assert task_mgr.get(consumer.id).status is TaskStatus.WAITING_HANDOFF


def test_an_invalid_input_does_not_count(scheduler, task_mgr, runner, registry):
    producer = make_task(outputs=new_handoffs(1))
    scheduler.submit(producer)
    runner.produce(registry, producer.id, valid=False)
    runner.finish(producer.id, TaskStatus.FAILED)

    consumer = make_task(inputs=producer.outputs)
    scheduler.submit(consumer)
    assert task_mgr.get(consumer.id).status is TaskStatus.WAITING_HANDOFF


# ------------------------------------------------------------------ outputs


def test_submit_declares_the_outputs(scheduler, handoff_mgr):
    task = make_task(outputs=new_handoffs(2))
    scheduler.submit(task)

    for hid in task.outputs:
        handoff = handoff_mgr.get(hid)
        assert handoff.latest.status is HandoffStatus.CREATED
        assert handoff.latest.producer_task_id == task.id


def test_declaring_an_output_does_not_make_it_valid(scheduler, handoff_mgr):
    task = make_task(outputs=new_handoffs(1))
    scheduler.submit(task)
    assert not handoff_mgr.check_if_latest_valid(task.outputs[0])


def test_a_task_that_consumes_its_own_declared_output_waits(scheduler, task_mgr):
    """Declaration is not production."""
    (hid,) = new_handoffs(1)
    task = make_task(inputs=[hid], outputs=[hid])
    scheduler.submit(task)
    assert task_mgr.get(task.id).status is TaskStatus.WAITING_HANDOFF


# --------------------------------------------------------------- rejections


def test_a_duplicate_id_is_rejected(scheduler):
    task = make_task()
    scheduler.submit(task)
    with pytest.raises(KeyError):
        scheduler.submit(Task(id=task.id, agent_spec="tuner"))


def test_an_unregistered_resource_is_rejected(scheduler, task_mgr):
    task = make_task(resources={"tpu": 1})
    with pytest.raises(ValueError, match="tpu"):
        scheduler.submit(task)
    assert task_mgr.all() == []


@pytest.mark.parametrize("amount", [-1, float("nan"), float("inf")])
def test_a_malformed_resource_amount_is_rejected(scheduler, task_mgr, registry, amount):
    """It would pass `can_afford` for other pools and then raise inside `take`,
    leaving the partial reservation all-or-nothing exists to prevent."""
    task = make_task(resources={"gpu": 1, "token": amount})
    with pytest.raises(ValueError, match="token"):
        scheduler.submit(task)

    assert task_mgr.all() == []
    assert registry.get("resource:gpu").available == 8  # nothing was taken


def test_zero_is_a_legitimate_amount(scheduler, task_mgr):
    task = make_task(resources={"gpu": 0})
    scheduler.submit(task)
    assert task_mgr.get(task.id).status is DISPATCHED


def test_a_registered_resource_is_accepted(scheduler, task_mgr):
    task = make_task(resources={"gpu": 2, "token": 500})
    scheduler.submit(task)
    assert task_mgr.get(task.id).resources == {"gpu": 2, "token": 500}


def test_an_unregistered_agent_spec_is_rejected(scheduler, task_mgr):
    with pytest.raises(KeyError, match="nope"):
        scheduler.submit(make_task(spec="nope"))
    assert task_mgr.all() == []


def test_a_rejected_task_leaves_no_trace(scheduler, task_mgr, handoff_mgr):
    """Rejection happens before anything is written."""
    task = make_task(resources={"tpu": 1}, outputs=new_handoffs(1))
    with pytest.raises(ValueError):
        scheduler.submit(task)

    assert task_mgr.all() == []
    assert handoff_mgr.all_ids() == []
    assert all(not pool for pool in scheduler.pools.values())


# ----------------------------------------------------- the depends_on check


def test_a_missing_depends_on_edge_warns_and_does_not_reject(
    scheduler, task_mgr, runner, registry, caplog
):
    """Criterion 35. Rejecting would make declaration order matter; repairing
    would make depends_on derived and unable to express an edge that shares no
    handoff."""
    producer = make_task(outputs=new_handoffs(1))
    scheduler.submit(producer)
    runner.produce(registry, producer.id)
    runner.finish(producer.id)

    consumer = make_task(inputs=producer.outputs)  # depends_on left empty
    with caplog.at_level(logging.WARNING):
        scheduler.submit(consumer)

    assert str(producer.id) in caplog.text
    assert "depends_on" in caplog.text
    assert task_mgr.get(consumer.id).status is DISPATCHED  # accepted


def test_a_correct_depends_on_edge_is_silent(scheduler, runner, registry, caplog):
    producer = make_task(outputs=new_handoffs(1))
    scheduler.submit(producer)
    runner.produce(registry, producer.id)
    runner.finish(producer.id)

    consumer = make_task(inputs=producer.outputs, depends_on=[producer.id])
    # The producer's own submit warns that nothing was pinned for its outputs —
    # `tests/task_graph` runs storeless by design. That is setup noise here, not
    # the thing under test, so it is cleared rather than tolerated by loosening
    # the assertion below: "this submit is silent" stays exact.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        scheduler.submit(consumer)

    assert caplog.text == ""


def test_the_check_passes_vacuously_when_the_producer_is_not_yet_known(scheduler, caplog):
    """Open question O8, asserted so the limitation is visible rather than
    discovered later."""
    consumer = make_task(inputs=new_handoffs(1))
    with caplog.at_level(logging.WARNING):
        scheduler.submit(consumer)
    assert caplog.text == ""


# ---------------------------------------------------------------- the index


def test_the_pool_index_agrees_with_the_stored_status(scheduler, task_mgr):
    hog = make_task(resources={"gpu": 8})
    eligible = make_task(resources={"gpu": 1})
    waiting = make_task(inputs=new_handoffs(1))
    for task in (hog, eligible, waiting):
        scheduler.submit(task)

    assert scheduler.pools[DISPATCHED] == {hog.id}
    assert scheduler.pools[TaskStatus.WAITING_RESOURCE] == {eligible.id}
    assert scheduler.pools[TaskStatus.WAITING_HANDOFF] == {waiting.id}
    for task in (hog, eligible, waiting):
        assert task.id in scheduler.pools[task_mgr.get(task.id).status]


def test_submit_persists_the_task(scheduler, store):
    task = make_task(inputs=new_handoffs(1))
    scheduler.submit(task)
    assert store.read("task", str(task.id))["status"] == "waiting_handoff"


def test_a_pool_with_no_registered_resources_still_works(store):
    """A system may declare no pools at all; a task requesting none still runs."""
    from task_graph.bootstrap import build_registry

    registry = build_registry(store=store, resources=[])
    registry.get("agent_mgr").register("profiler")
    scheduler = registry.get("scheduler")

    task = make_task()
    scheduler.submit(task)
    assert registry.get("task_mgr").get(task.id).status is DISPATCHED


def test_resource_names_come_from_the_registry_not_a_fixed_list(store):
    from task_graph.bootstrap import build_registry

    registry = Registry()
    pools = [GpuMgr(registry, capacity=4, name="mi300")]
    registry = build_registry(store=store, resources=pools)
    registry.get("agent_mgr").register("profiler")

    task = make_task(resources={"mi300": 2})
    registry.get("scheduler").submit(task)
    assert registry.get("task_mgr").get(task.id).status is DISPATCHED


def test_a_storeless_dispatch_of_a_task_with_outputs_says_so(scheduler, registry, caplog):
    """`bootstrap.py:216` deliberately leaves `handoff_store` unregistered when
    no root was supplied, so the first resolution is a loud `KeyError` rather
    than a store rooted at a default nobody chose.

    **Two tolerant readers turned that loudness back into nothing**: this early
    return, and `agent._seal_outputs` skipping an output with no pinned version.
    Between them a task that declares outputs produces none and says why
    nowhere. It cannot raise — the storeless mode is real and this suite runs
    entirely in it — so it says so instead.
    """
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="task_graph.scheduler"):
        scheduler.submit(make_task(outputs=new_handoffs(2)))

    assert "no handoff_store is registered" in caplog.text
    assert "declares 2 output(s)" in caplog.text
    # The consequence, stated correctly. The first version of this message said
    # the gate would report `OUTPUT_ABSENT` naming the wrong cause. `monitor`
    # measured the real methods: `_gate` returns `[]` for want of the same
    # store, so there are no failures and `_main` reports the task planned.
    # **It succeeds, having published nothing it declared** — worse than a
    # misattributed absence, and asserted here so the message cannot drift back.
    assert "will nevertheless succeed" in caplog.text


def test_a_storeless_dispatch_of_a_task_with_no_outputs_is_silent(scheduler, caplog):
    """Nothing to pin and nothing lost, so there is nothing to say. A warning
    that fires for a whole class is one a reader learns to discount — `agent`'s
    gate made that argument about `SELF_CHECK_UNSET` and it applies here."""
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="task_graph.scheduler"):
        scheduler.submit(make_task())

    assert caplog.text == ""
