"""Recovery — criteria 8, 19, 24, 25 and 34.

A restart is not a fresh start and it is not a replay: state is reloaded,
verdicts are never re-derived, and eligibility is recomputed by asking the
handoffs again.
"""

import pytest

from agent_sys.bootstrap import build_registry
from agent_sys.models import HandoffStateError, HandoffStatus, TaskStatus
from agent_sys.registry import RESUME_ORDER, Registry, resume_all
from agent_sys.store import JsonFileStoreMgr

from .conftest import make_task, new_handoffs, rebuild

# ------------------------------------------------------------- resume_all


def test_resume_all_visits_every_manager_in_order(registry):
    log = []

    class Spy:
        def __init__(self, name):
            self.name = name

        def resume_system(self):
            log.append(self.name)

    for name in ["scheduler", "task_mgr", "handoff_mgr", "agent_mgr", "resource:gpu"]:
        registry.register(name, Spy(name))
    resume_all(registry)

    assert log == RESUME_ORDER[:3] + ["resource:gpu", "scheduler"]


def test_a_system_with_no_state_resumes_to_nothing(store):
    fresh = rebuild(store)
    resume_all(fresh)
    assert fresh.get("task_mgr").all() == []
    assert all(not pool for pool in fresh.get("scheduler").pools.values())


# -------------------------------------------------------------- demotion


def test_a_running_task_is_demoted_and_its_lease_is_not_double_booked(scheduler, store):
    """Criterion 8. The lease is gone, so recovery neither keeps it nor charges
    for it twice when the demoted task is dispatched again."""
    task = make_task(resources={"gpu": 3})
    scheduler.submit(task)

    fresh = rebuild(store)
    resume_all(fresh)

    restored = fresh.get("task_mgr").get(task.id)
    assert restored.history[0].outcome is TaskStatus.SUSPENDED  # the cut-short attempt
    assert [e.attempt for e in restored.history] == [0, 1]
    assert fresh.get("resource:gpu").available == 5  # one reservation, not two


def test_a_running_task_whose_inputs_went_stale_lands_back_in_waiting(
    scheduler, runner, store, registry
):
    producer = make_task(outputs=new_handoffs(1))
    scheduler.submit(producer)
    runner.produce(registry, producer.id)
    runner.finish(producer.id)

    consumer = make_task(inputs=producer.outputs, resources={"gpu": 3})
    scheduler.submit(consumer)
    assert registry.get("task_mgr").get(consumer.id).status is TaskStatus.RUNNING

    # a second producer reopens the slot and never seals it
    refresher = make_task(outputs=producer.outputs)
    scheduler.submit(refresher)
    registry.get("handoff_mgr").get(producer.outputs[0]).open_next(
        refresher.id, registry.get("task_mgr").get(refresher.id).current.agent_id
    )
    registry.get("handoff_mgr").persist(producer.outputs[0])

    fresh = rebuild(store)
    resume_all(fresh)

    restored = fresh.get("task_mgr").get(consumer.id)
    assert restored.status is TaskStatus.WAITING_HANDOFF
    assert not restored.is_running
    assert restored.history[-1].outcome is TaskStatus.SUSPENDED
    assert fresh.get("resource:gpu").available == 8  # nothing is held


def test_a_demoted_task_that_is_ready_is_dispatched_again(scheduler, store):
    task = make_task(resources={"gpu": 3})
    scheduler.submit(task)

    fresh = rebuild(store)
    resume_all(fresh)

    assert fresh.get("task_mgr").get(task.id).status is TaskStatus.RUNNING
    assert [e.attempt for e in fresh.get("task_mgr").get(task.id).history] == [0, 1]
    assert fresh.get("runner").started == [task.id]


def test_a_stopping_task_becomes_suspended_because_the_runner_is_gone(scheduler, store):
    task = make_task()
    scheduler.submit(task)
    scheduler.stop(task.id)  # never acknowledged

    fresh = rebuild(store)
    resume_all(fresh)

    assert fresh.get("task_mgr").get(task.id).status is TaskStatus.SUSPENDED


@pytest.mark.parametrize(
    "reach, expected",
    [
        ("succeeded", TaskStatus.SUCCEEDED),
        ("failed", TaskStatus.FAILED),
        ("cancelled", TaskStatus.CANCELLED),
    ],
)
def test_a_settled_task_keeps_its_status(scheduler, runner, store, reach, expected):
    task = make_task(inputs=new_handoffs(1) if reach == "cancelled" else [])
    scheduler.submit(task)
    if reach == "cancelled":
        scheduler.remove_queued(task.id)
    else:
        runner.finish(task.id, TaskStatus[reach.upper()])

    fresh = rebuild(store)
    resume_all(fresh)
    assert fresh.get("task_mgr").get(task.id).status is expected


def test_a_suspended_task_is_not_restarted_by_recovery(scheduler, runner, store):
    """Resuming it is an operator decision, not something a restart makes."""
    task = make_task()
    scheduler.submit(task)
    scheduler.stop(task.id)
    runner.ack_stop(task.id)

    fresh = rebuild(store)
    resume_all(fresh)

    assert fresh.get("task_mgr").get(task.id).status is TaskStatus.SUSPENDED
    assert fresh.get("runner").started == []


# ----------------------------------------------------- verdicts are not redone


def test_handoff_verdicts_are_reloaded_not_re_derived(scheduler, runner, store, registry):
    """Criterion 19."""
    valid_task = make_task(outputs=new_handoffs(1))
    invalid_task = make_task(outputs=new_handoffs(1))
    for task, ok in ((valid_task, True), (invalid_task, False)):
        scheduler.submit(task)
        runner.produce(registry, task.id, valid=ok, content=f"content-{ok}")
        runner.finish(task.id)

    fresh = rebuild(store)
    resume_all(fresh)
    hm = fresh.get("handoff_mgr")

    assert hm.check_if_latest_valid(valid_task.outputs[0])
    assert not hm.check_if_latest_valid(invalid_task.outputs[0])
    assert hm.get(valid_task.outputs[0]).latest.content == "content-True"
    assert hm.get(invalid_task.outputs[0]).latest.status is HandoffStatus.INVALID


def test_a_consumer_of_a_valid_handoff_runs_after_a_restart(scheduler, runner, store, registry):
    producer = make_task(outputs=new_handoffs(1))
    scheduler.submit(producer)
    runner.produce(registry, producer.id)
    runner.finish(producer.id)

    consumer = make_task(inputs=producer.outputs, resources={"gpu": 8})
    scheduler.submit(consumer)

    fresh = rebuild(store)
    resume_all(fresh)
    assert fresh.get("task_mgr").get(consumer.id).status is TaskStatus.RUNNING


def test_a_consumer_of_an_invalid_handoff_still_waits_after_a_restart(
    scheduler, runner, store, registry
):
    producer = make_task(outputs=new_handoffs(1))
    scheduler.submit(producer)
    runner.produce(registry, producer.id, valid=False)
    runner.finish(producer.id)

    consumer = make_task(inputs=producer.outputs)
    scheduler.submit(consumer)

    fresh = rebuild(store)
    resume_all(fresh)
    assert fresh.get("task_mgr").get(consumer.id).status is TaskStatus.WAITING_HANDOFF


# --------------------------------------------------------------- the ordering


def test_resuming_the_scheduler_first_leaves_every_task_blocked(scheduler, runner, store, registry):
    """Criterion 25, which asserts a *failure*. Without this the ordering in
    RESUME_ORDER is a comment."""
    producer = make_task(outputs=new_handoffs(1))
    scheduler.submit(producer)
    runner.produce(registry, producer.id)
    runner.finish(producer.id)

    consumer = make_task(inputs=producer.outputs)
    scheduler.submit(consumer)

    fresh = rebuild(store)
    fresh.get("task_mgr").resume_system()
    fresh.get("scheduler").resume_system()  # HandoffMgr deliberately not resumed

    # Every handoff looks unknown, so nothing is eligible — and no later event
    # will unblock it.
    assert fresh.get("task_mgr").get(consumer.id).status is TaskStatus.WAITING_HANDOFF
    assert fresh.get("runner").started == []

    # In the right order it works.
    right = rebuild(store)
    resume_all(right)
    assert right.get("task_mgr").get(consumer.id).status is TaskStatus.RUNNING


def test_resume_order_puts_the_scheduler_last():
    assert RESUME_ORDER[-1] == "scheduler"
    assert "handoff_mgr" in RESUME_ORDER[:-1]


# ------------------------------------------------------------------- agents


def test_the_audit_trail_resolves_after_a_restart(scheduler, runner, store, registry):
    """Criterion 34."""
    task = make_task(outputs=new_handoffs(1))
    scheduler.submit(task)
    runner.produce(registry, task.id)
    runner.finish(task.id)
    agent_id = registry.get("task_mgr").get(task.id).current.agent_id

    fresh = rebuild(store)
    resume_all(fresh)

    restored = fresh.get("task_mgr").get(task.id)
    agent = fresh.get("agent_mgr").get(restored.current.agent_id)
    assert agent.id == agent_id
    assert agent.task_id == task.id
    assert [ref.handoff_id for ref in agent.handoffs] == task.outputs


# -------------------------------------------------------------- consumables


def test_a_consumable_balance_survives_but_a_renewable_lease_does_not(scheduler, runner, store):
    """Criterion 32, end to end."""
    spent = make_task(resources={"token": 500})
    scheduler.submit(spent)
    runner.finish(spent.id, usage={"token": 300})

    holding = make_task(resources={"gpu": 5, "token": 400})
    scheduler.submit(holding)  # still running when the process dies

    fresh = rebuild(store)
    # The pools alone, before the scheduler recomputes anything: this is purely
    # what recovery restored.
    fresh.get("resource:gpu").resume_system()
    fresh.get("resource:token").resume_system()

    assert fresh.get("resource:token").available == 1_000_000 - 300  # spend is durable
    assert fresh.get("resource:gpu").available == 8  # the lease is not

    # And the reservation that was never settled is not charged: after the full
    # resume the demoted task takes it again from the restored balance.
    resume_all(fresh)
    assert fresh.get("resource:token").available == 1_000_000 - 300 - 400
    assert fresh.get("resource:gpu").available == 3


# ------------------------------------------------- end to end, on real files


def test_a_full_cycle_survives_a_restart_through_the_json_store(tmp_path):
    """Criterion 24, against the filesystem rather than the memory store."""
    store = JsonFileStoreMgr(tmp_path / "state")
    registry = rebuild(store)
    scheduler, runner = registry.get("scheduler"), registry.get("runner")

    (mid,) = new_handoffs(1)
    producer = make_task(outputs=[mid], resources={"token": 200})
    consumer = make_task(inputs=[mid], depends_on=[producer.id], resources={"gpu": 2})
    scheduler.submit(producer)
    scheduler.submit(consumer)
    runner.produce(registry, producer.id, content={"p50": 7.5})
    runner.finish(producer.id, usage={"token": 150})

    # consumer is now running; the process dies here
    assert registry.get("task_mgr").get(consumer.id).status is TaskStatus.RUNNING

    fresh = rebuild(JsonFileStoreMgr(tmp_path / "state"))
    resume_all(fresh)

    assert fresh.get("task_mgr").get(producer.id).status is TaskStatus.SUCCEEDED
    assert fresh.get("handoff_mgr").get(mid).latest.content == {"p50": 7.5}
    assert fresh.get("resource:token").available == 1_000_000 - 150

    restored_consumer = fresh.get("task_mgr").get(consumer.id)
    assert restored_consumer.status is TaskStatus.RUNNING  # demoted, then redispatched
    assert [e.outcome for e in restored_consumer.history] == [TaskStatus.SUSPENDED, None]
    assert restored_consumer.depends_on == [producer.id]

    # and it can still finish
    fresh.get("runner").finish(consumer.id)
    assert fresh.get("task_mgr").get(consumer.id).status is TaskStatus.SUCCEEDED


def test_a_second_restart_is_idempotent(scheduler, runner, store, registry):
    task = make_task(outputs=new_handoffs(1))
    scheduler.submit(task)
    runner.produce(registry, task.id)
    runner.finish(task.id)

    first = rebuild(store)
    resume_all(first)
    second = rebuild(store)
    resume_all(second)

    for name, expected in (("task", 1), ("handoff", 1), ("agent", 1)):
        assert len(store.read_all(name)) == expected, name
    assert second.get("task_mgr").get(task.id).status is TaskStatus.SUCCEEDED
    assert len(second.get("task_mgr").get(task.id).history) == 1


def test_recovery_does_not_need_the_registry_that_wrote_the_state(store, scheduler):
    """Nothing is process-global; the store is the whole of the durable state."""
    scheduler.submit(make_task(outputs=new_handoffs(1)))

    other = Registry()
    del other  # explicitly not reused
    fresh = build_registry(store=store)
    fresh.get("agent_mgr").register("profiler")
    resume_all(fresh)

    assert len(fresh.get("task_mgr").all()) == 1


def test_a_generating_version_is_still_generating_after_a_restart(
    scheduler, runner, store, registry, handoff_mgr, task_mgr
):
    """Criterion 19's other half. A restart must not decide that an abandoned
    version failed — that verdict is the agent's and it was never given."""
    task = make_task(outputs=new_handoffs(1))
    scheduler.submit(task)
    handoff_mgr.get(task.outputs[0]).open_next(task.id, task_mgr.get(task.id).current.agent_id)
    handoff_mgr.persist(task.outputs[0])
    runner.finish(task.id, TaskStatus.FAILED)

    fresh = rebuild(store)
    resume_all(fresh)

    version = fresh.get("handoff_mgr").get(task.outputs[0]).latest
    assert version.status is HandoffStatus.GENERATING
    assert not fresh.get("handoff_mgr").check_if_latest_valid(task.outputs[0])


def test_a_spec_removed_before_a_restart_does_not_take_recovery_down_with_it(scheduler, store):
    """The spec table is deliberately not restored, so a task naming a spec the
    operator has since removed cannot be dispatched. Recovery must fail that
    one task, not abort — otherwise every healthy task behind it stays parked
    with no later event to release it."""
    for spec in ("profiler", "tuner", "profiler"):
        scheduler.submit(make_task(spec=spec, resources={"gpu": 1}))

    fresh = build_registry(store=store)
    fresh.get("agent_mgr").register("tuner")  # "profiler" is gone
    resume_all(fresh)

    statuses = [t.status for t in fresh.get("task_mgr").all()]
    assert statuses.count(TaskStatus.FAILED) == 2
    assert statuses.count(TaskStatus.RUNNING) == 1
    assert len(fresh.get("runner").started) == 1  # the healthy one still ran
    assert fresh.get("resource:gpu").available == 7  # only its lease is held


def test_a_version_left_generating_by_a_crash_deadlocks_its_own_retry(
    scheduler, runner, store, registry, handoff_mgr, task_mgr
):
    """Design open question O10 — a real gap, asserted as it currently behaves.

    Spec §6.4 requires that a GENERATING version is not re-derived by recovery,
    and it is not. But `open_next` refuses a slot that is already open, and
    nothing seals the abandoned one: the agent that opened it is dead, and the
    scheduler is forbidden from writing handoff state. So the task is demoted,
    re-dispatched, and its new agent cannot write its own output.

    This is asserted rather than fixed because the fix changes the state
    machine — `open_next` would need an "adopt an abandoned version" path, and
    that decision belongs in the spec.
    """
    task = make_task(outputs=new_handoffs(1))
    scheduler.submit(task)
    handoff_mgr.get(task.outputs[0]).open_next(task.id, task_mgr.get(task.id).current.agent_id)
    handoff_mgr.persist(task.outputs[0])  # ...and the process dies here

    fresh = rebuild(store)
    resume_all(fresh)

    assert fresh.get("handoff_mgr").get(task.outputs[0]).latest.status is (HandoffStatus.GENERATING)
    assert fresh.get("task_mgr").get(task.id).status is TaskStatus.RUNNING

    with pytest.raises(HandoffStateError):
        fresh.get("runner").produce(fresh, task.id)


def test_a_pool_removed_before_a_restart_does_not_take_recovery_down_with_it(store):
    """The sibling of the removed-spec case. Resolving the pool sits inside the
    dispatch guard, so a pool an operator deleted between restarts fails only
    the tasks that name it — recovery still reaches every healthy task."""
    from agent_sys.resource import GpuMgr

    registry = Registry()
    first = build_registry(store=store, resources=[GpuMgr(registry, capacity=8)])
    first.get("agent_mgr").register("profiler")
    hungry = [make_task(resources={"gpu": 4}) for _ in range(2)]
    healthy = make_task()
    for task in (*hungry, healthy):
        first.get("scheduler").submit(task)

    fresh = build_registry(store=store, resources=[])  # the pool is gone
    fresh.get("agent_mgr").register("profiler")
    resume_all(fresh)

    task_mgr = fresh.get("task_mgr")
    assert [task_mgr.get(t.id).status for t in hungry] == [TaskStatus.FAILED] * 2
    assert task_mgr.get(healthy.id).status is TaskStatus.RUNNING
    assert fresh.get("runner").started == [healthy.id]


def test_a_failure_inside_the_abort_handler_still_parks_the_task(store):
    """`_abort_launch` is the handler; an exception escaping it has nowhere to
    go. It would propagate out of `submit`, leaving the task RUNNING with a
    half-open attempt that only a restart can clear."""
    from agent_sys.resource import GpuMgr
    from agent_sys.runner import FakeRunner

    class WedgedGpu(GpuMgr):
        def give_back(self, amount, actual=None):
            raise RuntimeError("the pool is wedged")

    class DeadRunner(FakeRunner):
        def start(self, task, agent, on_done):
            raise RuntimeError("harness unreachable")

    registry = Registry()
    fresh = build_registry(
        store=store, runner=DeadRunner(), resources=[WedgedGpu(registry, capacity=8)]
    )
    fresh.get("agent_mgr").register("profiler")
    task = make_task(resources={"gpu": 2})

    fresh.get("scheduler").submit(task)  # must not raise

    stored = fresh.get("task_mgr").get(task.id)
    assert stored.status is TaskStatus.FAILED
    assert not stored.is_running  # the attempt was closed
    fresh.get("scheduler").resume_task(task.id)  # and it is recoverable
