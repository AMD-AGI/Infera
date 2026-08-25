"""TaskMgr — the collection, plus one recovery decision.

There is no `set_status` and no `push_execution` here: those are the `Task`'s
own transitions, with its own guards. The mgr does durability and lookup.
"""

import pytest

from task_graph.ids import AgentId, HandoffId, TaskId
from task_graph.models import Task, TaskStatus
from task_graph.registry import Registry
from task_graph.store import MemoryStoreMgr
from task_graph.task import TaskMgr


@pytest.fixture
def store():
    return MemoryStoreMgr()


@pytest.fixture
def mgr(store):
    registry = Registry()
    registry.register("store_mgr", store)
    manager = TaskMgr(registry)
    registry.register("task_mgr", manager)
    return manager


def rebuild(store) -> TaskMgr:
    """A restart: fresh managers over the same store."""
    registry = Registry()
    registry.register("store_mgr", store)
    fresh = TaskMgr(registry)
    fresh.resume_system()
    return fresh


# --------------------------------------------------------------- collection


def test_add_then_get(mgr):
    task = Task(agent_spec="profiler")
    mgr.add(task)
    assert mgr.get(task.id) is task


def test_add_persists(mgr, store):
    task = Task(agent_spec="profiler")
    mgr.add(task)
    assert store.read("task", str(task.id))["agent_spec"] == "profiler"


def test_get_an_unknown_id_raises(mgr):
    with pytest.raises(KeyError):
        mgr.get(TaskId.new())


def test_add_rejects_a_duplicate_id(mgr):
    task = Task(agent_spec="profiler")
    mgr.add(task)
    with pytest.raises(KeyError):
        mgr.add(Task(id=task.id, agent_spec="tuner"))


def test_add_revives_a_cancelled_id_with_fresh_history(mgr):
    """Forced by `update_task` = remove_queued + submit under the same id."""
    task = Task(agent_spec="profiler")
    mgr.add(task)
    task.push_execution(agent_id=AgentId.new())
    task.close_execution({}, TaskStatus.FAILED)
    task.status = TaskStatus.CANCELLED
    mgr.persist(task.id)

    replacement = Task(id=task.id, agent_spec="tuner")
    mgr.add(replacement)

    assert mgr.get(task.id) is replacement
    assert mgr.get(task.id).history == []
    assert mgr.get(task.id).agent_spec == "tuner"


def test_all(mgr):
    tasks = [Task(agent_spec="profiler") for _ in range(3)]
    for task in tasks:
        mgr.add(task)
    assert sorted(t.id for t in mgr.all()) == sorted(t.id for t in tasks)


def test_by_status(mgr):
    running, waiting = Task(agent_spec="a"), Task(agent_spec="b")
    running.status = TaskStatus.RUNNING
    for task in (running, waiting):
        mgr.add(task)

    assert [t.id for t in mgr.by_status(TaskStatus.RUNNING)] == [running.id]
    assert [t.id for t in mgr.by_status(TaskStatus.WAITING_HANDOFF)] == [waiting.id]
    assert mgr.by_status(TaskStatus.SUCCEEDED) == []


def test_remove(mgr, store):
    task = Task(agent_spec="profiler")
    mgr.add(task)
    mgr.remove(task.id)

    assert mgr.all() == []
    assert not store.exists("task", str(task.id))
    with pytest.raises(KeyError):
        mgr.get(task.id)


def test_remove_an_unknown_id_raises(mgr):
    with pytest.raises(KeyError):
        mgr.remove(TaskId.new())


# -------------------------------------------------------------- persistence


def test_persist_writes_back_a_mutation_made_through_the_task(mgr, store):
    task = Task(agent_spec="profiler")
    mgr.add(task)

    task.status = TaskStatus.RUNNING
    assert store.read("task", str(task.id))["status"] == "waiting_handoff"

    mgr.persist(task.id)
    assert store.read("task", str(task.id))["status"] == "running"


def test_persist_an_unknown_id_raises(mgr):
    with pytest.raises(KeyError):
        mgr.persist(TaskId.new())


def test_history_survives_a_restart_with_attempt_numbering_intact(mgr, store):
    hid = HandoffId.new()
    task = Task(agent_spec="profiler", inputs=[hid])
    mgr.add(task)
    for outcome in (TaskStatus.FAILED, TaskStatus.SUCCEEDED):
        task.push_execution(agent_id=AgentId.new(), input_versions={hid: 0})
        task.close_execution({hid: 1}, outcome)
    mgr.persist(task.id)

    restored = rebuild(store).get(task.id)

    assert [e.attempt for e in restored.history] == [0, 1]
    assert [e.outcome for e in restored.history] == [TaskStatus.FAILED, TaskStatus.SUCCEEDED]
    assert restored.history[0].input_versions == {hid: 0}
    assert restored.created_at == task.created_at


# ------------------------------------------------------------------ resume


def test_resume_closes_a_dangling_stack_top_as_suspended(mgr, store):
    """The restart cut the attempt short; it was not judged."""
    task = Task(agent_spec="profiler")
    mgr.add(task)
    task.push_execution(agent_id=AgentId.new())
    task.status = TaskStatus.RUNNING
    mgr.persist(task.id)

    restored = rebuild(store).get(task.id)

    top = restored.history[-1]
    assert not top.is_open
    assert top.outcome is TaskStatus.SUSPENDED
    assert top.ended_at is not None


def test_resume_leaves_the_task_status_alone(mgr, store):
    """Where the task lands is the scheduler's decision, a moment later."""
    task = Task(agent_spec="profiler")
    mgr.add(task)
    task.push_execution(agent_id=AgentId.new())
    task.status = TaskStatus.RUNNING
    mgr.persist(task.id)

    assert rebuild(store).get(task.id).status is TaskStatus.RUNNING


def test_a_closed_stack_top_is_not_touched(mgr, store):
    task = Task(agent_spec="profiler")
    mgr.add(task)
    task.push_execution(agent_id=AgentId.new())
    task.close_execution({}, TaskStatus.SUCCEEDED, detail="fine")
    mgr.persist(task.id)

    top = rebuild(store).get(task.id).history[-1]
    assert top.outcome is TaskStatus.SUCCEEDED
    assert top.detail == "fine"


def test_closing_the_dangling_top_is_what_lets_the_next_attempt_start(mgr, store):
    """`push_execution` refuses to stack on an open attempt, so leaving it open
    would make the first resume_task after a restart raise."""
    task = Task(agent_spec="profiler")
    mgr.add(task)
    task.push_execution(agent_id=AgentId.new())
    mgr.persist(task.id)

    restored_mgr = rebuild(store)
    restored = restored_mgr.get(task.id)
    restored.push_execution(agent_id=AgentId.new())  # must not raise
    assert [e.attempt for e in restored.history] == [0, 1]


def test_resume_replaces_the_collection_rather_than_merging(mgr, store):
    task = Task(agent_spec="profiler")
    mgr.add(task)
    store.delete("task", str(task.id))

    mgr.resume_system()
    assert mgr.all() == []


def test_resume_is_durable_for_the_close_it_performs(mgr, store):
    """A second restart must not find the same dangling attempt again."""
    task = Task(agent_spec="profiler")
    mgr.add(task)
    task.push_execution(agent_id=AgentId.new())
    mgr.persist(task.id)

    rebuild(store)
    assert store.read("task", str(task.id))["history"][-1]["outcome"] == "suspended"


def test_remove_refuses_a_task_the_scheduler_still_indexes(registry):
    """`remove` is the hard delete, not cancellation. Removing an indexed task
    leaves an id in a pool with no task behind it, and every subsequent
    dispatch pass then raises at the eligibility re-check — permanently."""
    scheduler, mgr = registry.get("scheduler"), registry.get("task_mgr")
    task = Task(agent_spec="profiler", inputs=[HandoffId.new()])
    scheduler.submit(task)

    with pytest.raises(ValueError, match="still indexes it"):
        mgr.remove(task.id)

    scheduler.remove_queued(task.id)  # cancelled, but still indexed
    with pytest.raises(ValueError, match="still indexes it"):
        mgr.remove(task.id)

    scheduler.pools[TaskStatus.CANCELLED].discard(task.id)  # an operator forgets it
    mgr.remove(task.id)
    assert mgr.all() == []
