"""Dispatch — criteria 3 and 18.

All-or-nothing acquisition is what makes hold-and-wait deadlock structurally
impossible: a task that does not fit takes nothing, so no queued task ever holds
a resource.
"""

from agent_sys.bootstrap import build_registry
from agent_sys.models import HandoffStatus, TaskStatus

from .conftest import gpu, make_task, new_handoffs, token


def test_an_eligible_task_that_fits_starts(scheduler, task_mgr, runner):
    task = make_task(resources={"gpu": 2})
    scheduler.submit(task)

    assert task_mgr.get(task.id).status is TaskStatus.RUNNING
    assert runner.started == [task.id]


def test_starting_takes_the_resources(scheduler, registry):
    scheduler.submit(make_task(resources={"gpu": 2, "token": 500}))
    assert gpu(registry).available == 6
    assert token(registry).available == 1_000_000 - 500


def test_a_task_that_does_not_fit_stays_queued(scheduler, task_mgr, registry, runner):
    scheduler.submit(make_task(resources={"gpu": 8}))
    blocked = make_task(resources={"gpu": 1})
    scheduler.submit(blocked)

    assert task_mgr.get(blocked.id).status is TaskStatus.WAITING_RESOURCE
    assert blocked.id not in runner.started
    assert gpu(registry).available == 0


def test_a_task_that_does_not_fit_takes_nothing(scheduler, registry):
    """Criterion 3: the full set is verified before anything is mutated, so a
    queued task never holds a partial reservation."""
    scheduler.submit(make_task(resources={"gpu": 8}))
    scheduler.submit(make_task(resources={"gpu": 1, "token": 500}))

    assert gpu(registry).available == 0
    assert token(registry).available == 1_000_000  # the affordable half untouched


def test_a_blocked_task_starts_once_the_resource_comes_back(scheduler, task_mgr, runner, registry):
    hog = make_task(resources={"gpu": 8})
    scheduler.submit(hog)
    blocked = make_task(resources={"gpu": 1})
    scheduler.submit(blocked)

    runner.finish(hog.id)

    assert task_mgr.get(blocked.id).status is TaskStatus.RUNNING
    assert gpu(registry).available == 7


def test_a_task_requesting_nothing_always_fits(scheduler, task_mgr):
    scheduler.submit(make_task(resources={"gpu": 8}))
    free = make_task()
    scheduler.submit(free)
    assert task_mgr.get(free.id).status is TaskStatus.RUNNING


# ------------------------------------------------------------------ binding


def test_dispatch_binds_a_fresh_agent(scheduler, task_mgr, agent_mgr, runner):
    task = make_task()
    scheduler.submit(task)

    execution = task_mgr.get(task.id).current
    agent = agent_mgr.get(execution.agent_id)
    assert agent.spec == "profiler"
    assert agent.task_id == task.id
    assert runner.running[task.id][1] is agent


def test_the_first_attempt_is_numbered_zero(scheduler, task_mgr):
    task = make_task()
    scheduler.submit(task)
    assert task_mgr.get(task.id).current.attempt == 0
    assert task_mgr.get(task.id).is_running


def test_dispatch_pins_the_input_versions_it_actually_saw(scheduler, task_mgr, runner, registry):
    """Criterion 18: a later re-run of the producer cannot rewrite history."""
    producer = make_task(outputs=new_handoffs(1))
    scheduler.submit(producer)
    runner.produce(registry, producer.id)
    runner.finish(producer.id)

    consumer = make_task(inputs=producer.outputs)
    scheduler.submit(consumer)
    assert task_mgr.get(consumer.id).current.input_versions == {producer.outputs[0]: 0}

    # something writes the slot again, appending v1
    refresher = make_task(outputs=producer.outputs)
    scheduler.submit(refresher)
    runner.produce(registry, refresher.id)
    runner.finish(refresher.id)

    assert registry.get("handoff_mgr").latest(producer.outputs[0]).version == 1
    assert task_mgr.get(consumer.id).history[0].input_versions == {producer.outputs[0]: 0}


def test_a_task_with_no_inputs_pins_nothing(scheduler, task_mgr):
    task = make_task()
    scheduler.submit(task)
    assert task_mgr.get(task.id).current.input_versions == {}


def test_dispatch_persists_the_running_state(scheduler, store):
    task = make_task()
    scheduler.submit(task)

    record = store.read("task", str(task.id))
    assert record["status"] == "running"
    assert len(record["history"]) == 1
    assert record["history"][0]["ended_at"] is None


# ------------------------------------------------------------------ ordering


def test_dispatch_follows_the_policy_order(scheduler, runner, registry):
    """Two tasks, room for one: the earlier goes first."""
    first, second = make_task(resources={"gpu": 8}), make_task(resources={"gpu": 8})
    scheduler.submit(first)
    scheduler.submit(second)

    assert runner.started == [first.id]
    runner.finish(first.id)
    assert runner.started == [first.id, second.id]


def test_dispatch_keeps_trying_later_entries_after_one_does_not_fit(scheduler, task_mgr, runner):
    """Not a head-of-line block: the pass continues down the ordered list."""
    scheduler.submit(make_task(resources={"gpu": 6}))
    big, small = make_task(resources={"gpu": 4}), make_task(resources={"gpu": 2})
    scheduler.submit(big)
    scheduler.submit(small)

    assert task_mgr.get(big.id).status is TaskStatus.WAITING_RESOURCE
    assert task_mgr.get(small.id).status is TaskStatus.RUNNING


def test_a_chain_flows_without_any_external_prompt(scheduler, task_mgr, runner, registry):
    """The completion of one task is what re-checks eligibility for the next."""
    (mid,) = new_handoffs(1)
    (out,) = new_handoffs(1)
    first = make_task(outputs=[mid])
    second = make_task(inputs=[mid], outputs=[out])
    third = make_task(inputs=[out])
    for task in (third, second, first):  # submitted in reverse dependency order
        scheduler.submit(task)

    assert task_mgr.get(second.id).status is TaskStatus.WAITING_HANDOFF

    runner.produce(registry, first.id)
    runner.finish(first.id)
    assert task_mgr.get(second.id).status is TaskStatus.RUNNING

    runner.produce(registry, second.id)
    runner.finish(second.id)
    assert task_mgr.get(third.id).status is TaskStatus.RUNNING


# ---------------------------------------------------------------- re-entrancy


class SynchronousRunner:
    """Completes inside `start`, which re-enters the scheduler."""

    def __init__(self) -> None:
        self.started: list = []
        self.depth, self.max_depth = 0, 0

    def start(self, task, agent, on_done) -> None:
        self.started.append(task.id)
        self.depth += 1
        self.max_depth = max(self.max_depth, self.depth)
        try:
            on_done(task.id, TaskStatus.SUCCEEDED, {})
        finally:
            self.depth -= 1

    def stop(self, task_id, on_stopped) -> None:
        on_stopped(task_id)


def test_a_synchronous_runner_does_not_recurse_per_task(store):
    """The `_dispatch_again` flag turns re-entry into a loop, not a stack."""
    from agent_sys.bootstrap import build_registry

    runner = SynchronousRunner()
    registry = build_registry(store=store, runner=runner)
    registry.get("agent_mgr").register("profiler")
    scheduler = registry.get("scheduler")

    tasks = [make_task(resources={"gpu": 8}) for _ in range(5)]
    for task in tasks:
        scheduler.submit(task)

    assert runner.started == [t.id for t in tasks]
    assert runner.max_depth == 1
    assert all(registry.get("task_mgr").get(t.id).status is TaskStatus.SUCCEEDED for t in tasks)


# ------------------------------------------------- a launch that never happens


class BrokenRunner:
    """`start` raises for the first task and works thereafter."""

    def __init__(self) -> None:
        self.started: list = []
        self.n = 0

    def start(self, task, agent, on_done) -> None:
        self.n += 1
        if self.n == 1:
            raise RuntimeError("harness unreachable")
        self.started.append(task.id)

    def stop(self, task_id, on_stopped) -> None:
        on_stopped(task_id)


def test_a_runner_that_cannot_start_does_not_leak_the_lease(store, caplog):
    """The reservation is taken before the runner is called, so a launch that
    raises would otherwise shrink the pool permanently — the same shape as the
    negative-amount bug, one step later in the sequence."""
    from agent_sys.bootstrap import build_registry

    registry = build_registry(store=store, runner=BrokenRunner())
    registry.get("agent_mgr").register("profiler")
    scheduler = registry.get("scheduler")

    task = make_task(resources={"gpu": 3, "token": 500})
    with caplog.at_level("ERROR"):
        scheduler.submit(task)  # must not raise into the caller

    assert gpu(registry).available == 8
    assert token(registry).available == 1_000_000  # nothing ran, so nothing spent
    assert registry.get("task_mgr").get(task.id).status is TaskStatus.FAILED
    assert not registry.get("task_mgr").get(task.id).is_running
    assert "failed to launch" in caplog.text


def test_one_failed_launch_does_not_abort_the_pass(store):
    """The other half: a raise must not take the rest of the queue with it."""
    from agent_sys.bootstrap import build_registry

    registry = build_registry(store=store, runner=BrokenRunner())
    registry.get("agent_mgr").register("profiler")
    scheduler = registry.get("scheduler")

    doomed = make_task(resources={"gpu": 3})
    healthy = make_task(resources={"gpu": 6})
    scheduler.submit(doomed)
    scheduler.submit(healthy)

    assert registry.get("task_mgr").get(doomed.id).status is TaskStatus.FAILED
    assert registry.get("task_mgr").get(healthy.id).status is TaskStatus.RUNNING
    assert registry.get("runner").started == [healthy.id]


def test_a_failed_launch_is_not_retried_forever(store):
    """FAILED rather than back in the queue: the next pass would pick it up and
    fail identically. An operator resumes it once the cause is fixed."""
    from agent_sys.bootstrap import build_registry

    registry = build_registry(store=store, runner=BrokenRunner())
    registry.get("agent_mgr").register("profiler")
    scheduler = registry.get("scheduler")

    task = make_task()
    scheduler.submit(task)
    scheduler.try_dispatch()
    scheduler.try_dispatch()

    assert registry.get("task_mgr").get(task.id).status is TaskStatus.FAILED
    assert len(registry.get("task_mgr").get(task.id).history) == 1

    scheduler.resume_task(task.id)  # the operator's call, once it is fixed
    assert registry.get("task_mgr").get(task.id).status is TaskStatus.RUNNING


def test_an_agent_factory_that_is_down_releases_the_lease(store):
    """`instantiate` raises after `take` — the narrowest window there is."""
    from agent_sys.bootstrap import build_registry

    registry = build_registry(store=store)
    registry.get("agent_mgr").register("profiler")
    scheduler = registry.get("scheduler")

    def refuse(spec, task_id):
        raise RuntimeError("agent factory down")

    registry.get("agent_mgr").instantiate = refuse

    task = make_task(resources={"gpu": 3, "token": 500})
    scheduler.submit(task)

    assert gpu(registry).available == 8
    assert token(registry).available == 1_000_000
    assert registry.get("task_mgr").get(task.id).status is TaskStatus.FAILED


def test_an_input_opened_earlier_in_the_same_pass_is_not_dispatched_against(store):
    """Eligibility is re-asked per task, not once per pass.

    Step 1 re-checks every queued task, then the loop starts them one by one —
    and each `start` can run agent code. A producer that opens its output slot
    on start invalidates a handoff that a *later* task in the same pass has
    already been cleared for. Without the per-task re-check that task pins a
    GENERATING version: an input whose content does not exist yet, recorded in
    the audit trail criterion 18 requires.
    """
    from agent_sys.runner import FakeRunner

    (hid,) = new_handoffs(1)

    class OpeningRunner(FakeRunner):
        """A realistic producer: its agent opens the slot as the run starts."""

        registry = None
        armed = False

        def start(self, task, agent, on_done):
            super().start(task, agent, on_done)
            if self.armed and hid in task.outputs:
                self.registry.get("handoff_mgr").get(hid).open_next(task.id, agent.id)
                self.registry.get("handoff_mgr").persist(hid)

    runner = OpeningRunner()
    registry = build_registry(store=store, runner=runner)
    runner.registry = registry
    registry.get("agent_mgr").register("profiler")
    scheduler, task_mgr = registry.get("scheduler"), registry.get("task_mgr")

    seed = make_task(outputs=[hid])
    scheduler.submit(seed)
    runner.produce(registry, seed.id)
    runner.finish(seed.id)
    assert registry.get("handoff_mgr").check_if_latest_valid(hid)

    runner.armed = True
    hog = make_task(resources={"gpu": 8})
    scheduler.submit(hog)
    # Both queue behind the hog; the refresher sorts first and so starts first.
    refresher = make_task(outputs=[hid], resources={"gpu": 1})
    consumer = make_task(inputs=[hid], resources={"gpu": 1})
    scheduler.submit(refresher)
    scheduler.submit(consumer)

    runner.finish(hog.id)  # frees the pool: one pass selects both

    assert registry.get("handoff_mgr").get(hid).latest.status is HandoffStatus.GENERATING
    consumer_task = task_mgr.get(consumer.id)
    assert consumer_task.status is TaskStatus.WAITING_HANDOFF
    assert consumer_task.history == []  # nothing was pinned
    assert registry.get("resource:gpu").available == 7  # only the refresher holds a lease
