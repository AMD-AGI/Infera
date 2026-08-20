"""Dispatch — criteria 3 and 18.

All-or-nothing acquisition is what makes hold-and-wait deadlock structurally
impossible: a task that does not fit takes nothing, so no queued task ever holds
a resource.
"""

from agent_sys.models import TaskStatus

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
