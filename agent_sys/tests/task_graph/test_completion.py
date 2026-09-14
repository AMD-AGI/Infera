"""Completion — criteria 4, 7 and 13.

There is no result object: the runner reports a status and what it spent.

**Output versions are no longer read here.** `interfaces.md` §4.14 moved them to
dispatch, where the store allocates the directory the write grant resolves to,
so completion records nothing about them and the two tests that used to assert
it now assert the pin at the moment it happens.
"""

import pytest

from task_graph.bootstrap import build_registry
from task_graph.models import TaskStatus
from task_graph.store import MemoryStoreMgr

from .conftest import DISPATCHED, gpu, make_task, new_handoffs, token


@pytest.fixture
def stored_registry(tmp_path):
    """A registry **with** a `handoff_store`, which the shared one has not.

    `bootstrap.py:214` registers the store only when a root is supplied, and the
    rest of this suite deliberately runs without one — it exercises the
    scheduler against `FakeRunner`, and an artefact store rooted at a default
    nobody chose is worse than a loud `KeyError`. Pinning an output version
    means allocating a directory, so the two tests that assert it need the mode
    where a store exists.
    """
    r = build_registry(store=MemoryStoreMgr(), handoff_root=str(tmp_path / "store"))
    r.get("agent_mgr").register("profiler")
    return r


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


def test_an_output_version_is_pinned_at_dispatch_and_survives_completion(stored_registry):
    """§4.14, and the half `env_mgr` is blocked on.

    The number has to be on the `Execution` **while the attempt is open**: that
    is the whole point of moving it. `env_mgr/grants.py:85` reads it there to
    build `<store>/<hid>/v<N>/`, and before this it was empty until close, so
    every output write grant raised `UnresolvedGrant` for the duration of the
    attempt that was supposed to fill it.
    """
    scheduler, runner = stored_registry.get("scheduler"), stored_registry.get("runner")
    task = make_task(outputs=new_handoffs(2))
    scheduler.submit(task)

    open_attempt = stored_registry.get("task_mgr").get(task.id).current
    assert open_attempt.is_open, "the assertion below is only interesting mid-attempt"
    assert open_attempt.output_versions == dict.fromkeys(task.outputs, 0)

    runner.produce(stored_registry, task.id)
    runner.finish(task.id)

    execution = stored_registry.get("task_mgr").get(task.id).current
    assert execution.output_versions == dict.fromkeys(task.outputs, 0)
    assert execution.outcome is TaskStatus.SUCCEEDED
    assert not execution.is_open


def test_a_retry_is_pinned_to_a_fresh_version_rather_than_the_previous_one(stored_registry):
    """Criterion 16 is what this protects, and it is why the number cannot come
    from `HandoffMgr.latest`.

    At the second dispatch that slot's latest is the version the first attempt
    left behind, so deriving from it would grant the retry **the previous
    version's directory** and let it overwrite bytes criterion 16 promises are
    byte-identical forever. Allocating instead hands out a directory nobody
    holds.
    """
    scheduler, runner = stored_registry.get("scheduler"), stored_registry.get("runner")
    task = make_task(outputs=new_handoffs(1))
    (hid,) = task.outputs
    scheduler.submit(task)
    runner.produce(stored_registry, task.id)
    runner.finish(task.id, TaskStatus.FAILED)

    scheduler.resume_task(task.id)

    attempts = stored_registry.get("task_mgr").get(task.id).history
    assert [e.output_versions[hid] for e in attempts] == [0, 1]


def test_an_output_the_agent_never_wrote_still_carries_the_version_it_was_granted(
    stored_registry,
):
    """The pin records **what this attempt was given**, not what it delivered.

    Whether anything was written is the handoff's answer and stays separate —
    which is criterion 13's real content, and the half of it §4.14 did not move.
    """
    scheduler, runner = stored_registry.get("scheduler"), stored_registry.get("runner")
    task = make_task(outputs=new_handoffs(1))
    scheduler.submit(task)
    runner.finish(task.id)  # no produce()

    current = stored_registry.get("task_mgr").get(task.id).current
    assert current.output_versions == {task.outputs[0]: 0}
    assert not stored_registry.get("handoff_mgr").check_if_latest_valid(task.outputs[0])


def test_without_a_handoff_store_no_output_version_is_pinned(scheduler, runner, registry):
    """The store-less mode, asserted rather than left as an accident.

    Most of this suite runs in it, so if pinning silently required a store the
    absence would show up as 358 errors rather than as a decision. There is no
    directory to grant, so there is no number to pin.
    """
    assert "handoff_store" not in registry
    task = make_task(outputs=new_handoffs(1))
    scheduler.submit(task)
    runner.finish(task.id)

    assert registry.get("task_mgr").get(task.id).current.output_versions == {}


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

    assert registry.get("task_mgr").get(consumer.id).status is DISPATCHED


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
    assert task_mgr.get(queued.id).status is DISPATCHED


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


def test_one_pool_failing_to_release_does_not_strand_the_task(scheduler, runner, registry, caplog):
    """The run is over and cannot be un-finished. An exception escaping the
    release would leave the task RUNNING with an open execution record and its
    other leases held — recoverable only by a restart."""

    class WedgedPool:
        name = "gpu"
        available = 0.0

        def give_back(self, amount, actual=None):
            raise RuntimeError("pool wedged")

    task = make_task(resources={"gpu": 2, "token": 100})
    scheduler.submit(task)
    registry.register("resource:gpu", WedgedPool())

    with caplog.at_level("ERROR"):
        runner.finish(task.id, usage={"token": 40})

    finished = registry.get("task_mgr").get(task.id)
    assert finished.status is TaskStatus.SUCCEEDED
    assert not finished.is_running
    assert token(registry).available == 1_000_000 - 40  # the healthy pool settled
    assert "could not release" in caplog.text


def test_the_runners_detail_reaches_the_execution_record(scheduler, runner, task_mgr):
    """`Execution.detail` is "from the runner; for a human" and was empty on
    every failed task, because `OnDone` was a `Callable` alias that could not
    express a keyword argument. The field, the scheduler's parameter and the
    plumbing were all in place; the declared type was the whole gap.
    """
    task = make_task()
    scheduler.submit(task)
    runner.finish(task.id, TaskStatus.FAILED, detail="KeyError: 'agent'")

    top = task_mgr.get(task.id).history[-1]
    assert top.outcome is TaskStatus.FAILED
    assert top.detail == "KeyError: 'agent'"


def test_a_runner_with_nothing_to_say_passes_nothing(scheduler, runner, task_mgr):
    """`detail` defaults, so the common path is unchanged and no runner is
    obliged to invent a string."""
    task = make_task()
    scheduler.submit(task)
    runner.finish(task.id)
    assert task_mgr.get(task.id).history[-1].detail == ""


def test_on_done_is_a_protocol_that_admits_the_keyword():
    """A `Callable[[...], None]` alias cannot express `detail`, and
    `Callable[..., None]` would give up the first three as well. The first three
    are positional-only, so an implementation may name them what it likes —
    `Scheduler.on_task_done` calls its first parameter `tid`."""
    import inspect

    from task_graph.runner import OnDone

    params = inspect.signature(OnDone.__call__).parameters
    positional = [p.name for p in params.values() if p.kind is p.POSITIONAL_ONLY]
    assert positional == ["self", "task_id", "status", "usage"]
    assert params["detail"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["detail"].default == ""


def test_a_non_leaf_charges_spend_it_never_reserved(scheduler, runner, registry, task_mgr):
    """Design §6.3.1, and the path had no test.

    A non-leaf acquires nothing (criterion 53), yet its validation phases may
    run an AI agent and spend tokens. `give_back` cannot record that — it clamps
    the settlement to the reservation and the reservation is zero — so
    `on_task_done` charges any pool named in `usage` that the task did not
    declare. Found by auditing rather than by a failure: the code was correct
    and unexercised, which is the state in which it stops being correct.
    """
    task = make_task()  # declares no resources at all
    scheduler.submit(task)
    before = token(registry).available

    runner.finish(task.id, TaskStatus.SUCCEEDED, {"token": 500})

    assert token(registry).available == before - 500
    assert token(registry).spent == 500
    assert task_mgr.get(task.id).status is TaskStatus.SUCCEEDED


def test_an_undeclared_renewable_in_usage_is_logged_and_not_charged(
    scheduler, runner, registry, caplog
):
    """A GPU that was never taken cannot be given back, and "spend" is not a
    concept a renewable has — spec §3.4's two release semantics are exactly this
    distinction. So it is a caller error, logged rather than silently absorbed.
    """
    import logging

    task = make_task()
    scheduler.submit(task)
    caplog.set_level(logging.WARNING, logger="task_graph.scheduler")

    runner.finish(task.id, TaskStatus.SUCCEEDED, {"gpu": 3})

    assert gpu(registry).available == 8, "nothing was taken, so nothing moves"
    assert "did not declare" in caplog.text
