"""Two-way links and the graph edge — criteria 21, 22 and 31.

`depends_on` is the graph, recorded for traversal and display. Scheduling never
reads it: the dependency question is `inputs` + `check_if_latest_valid`.
"""

import graphlib

from task_graph.models import TaskStatus

from .conftest import DISPATCHED, make_task, new_handoffs

# --------------------------------------------------------- the two-way links


def test_both_directions_resolve(scheduler, task_mgr, agent_mgr, handoff_mgr, runner, registry):
    """Criterion 22."""
    task = make_task(outputs=new_handoffs(2))
    scheduler.submit(task)
    runner.produce(registry, task.id)
    runner.finish(task.id)

    # task -> agent -> handoffs
    execution = task_mgr.get(task.id).current
    agent = agent_mgr.get(execution.agent_id)
    assert agent.task_id == task.id
    assert {ref.handoff_id for ref in agent.handoffs} == set(task.outputs)

    # handoff -> task and agent
    for hid in task.outputs:
        version = handoff_mgr.latest(hid)
        assert version.producer_task_id == task.id
        assert version.producer_agent_id == agent.id

    # agent -> the version it wrote
    for ref in agent.handoffs:
        assert handoff_mgr.get(ref.handoff_id).get(ref.version).producer_agent_id == agent.id


def test_the_live_agent_is_the_stack_top_and_earlier_ones_stay_readable(
    scheduler, task_mgr, agent_mgr, runner
):
    """Criterion 21."""
    task = make_task()
    scheduler.submit(task)
    first = task_mgr.get(task.id).current.agent_id
    runner.finish(task.id, TaskStatus.FAILED)

    scheduler.resume_task(task.id)
    second = task_mgr.get(task.id).current.agent_id

    assert first != second
    assert [e.agent_id for e in task_mgr.get(task.id).history] == [first, second]
    assert {a.id for a in agent_mgr.by_task(task.id)} == {first, second}


def test_an_agent_from_an_earlier_attempt_is_still_resolvable(
    scheduler, task_mgr, agent_mgr, runner, registry
):
    task = make_task(outputs=new_handoffs(1))
    scheduler.submit(task)
    runner.produce(registry, task.id, content="attempt 0")
    runner.finish(task.id, TaskStatus.FAILED)
    first = task_mgr.get(task.id).history[0].agent_id

    scheduler.resume_task(task.id)
    runner.produce(registry, task.id, content="attempt 1")
    runner.finish(task.id)

    old = agent_mgr.get(first)
    assert [ref.version for ref in old.handoffs] == [0]
    assert registry.get("handoff_mgr").get(task.outputs[0]).get(0).content == "attempt 0"


def test_who_wrote_a_given_version_is_answerable_from_the_version_alone(
    scheduler, handoff_mgr, agent_mgr, runner, registry
):
    (hid,) = new_handoffs(1)
    for _ in range(2):
        producer = make_task(outputs=[hid])
        scheduler.submit(producer)
        runner.produce(registry, producer.id)
        runner.finish(producer.id)

    for version in handoff_mgr.get(hid).versions:
        agent = agent_mgr.get(version.producer_agent_id)
        assert agent.task_id == version.producer_task_id


# ------------------------------------------------------------- depends_on


def build_diamond(scheduler):
    """root -> {left, right} -> join, wired through both handoffs and edges."""
    (a,) = new_handoffs(1)
    (b,) = new_handoffs(1)
    (c,) = new_handoffs(1)
    root = make_task(outputs=[a])
    left = make_task(inputs=[a], outputs=[b], depends_on=[root.id])
    right = make_task(inputs=[a], outputs=[c], depends_on=[root.id])
    join = make_task(inputs=[b, c], depends_on=[left.id, right.id])
    for task in (root, left, right, join):
        scheduler.submit(task)
    return root, left, right, join


def test_depends_on_sorts_topologically(scheduler, task_mgr):
    """Criterion 31, first half: the edge is recorded so a display or an
    analysis can traverse it."""
    root, left, right, join = build_diamond(scheduler)

    graph = {t.id: set(t.depends_on) for t in task_mgr.all()}
    order = list(graphlib.TopologicalSorter(graph).static_order())

    assert order.index(root.id) < order.index(left.id) < order.index(join.id)
    assert order.index(root.id) < order.index(right.id) < order.index(join.id)


def test_blanking_depends_on_changes_no_scheduling_behaviour(
    scheduler, task_mgr, runner, registry, store
):
    """Criterion 31, second half. Two identical runs, one with the edges and one
    without, must produce the same dispatch order."""
    from .conftest import rebuild

    def run(with_edges: bool) -> list:
        registry_ = rebuild(type(store)())
        scheduler_ = registry_.get("scheduler")
        (a,) = new_handoffs(1)
        (b,) = new_handoffs(1)
        root = make_task(outputs=[a])
        mid = make_task(inputs=[a], outputs=[b], depends_on=[root.id] if with_edges else [])
        leaf = make_task(inputs=[b], depends_on=[mid.id] if with_edges else [])
        for task in (leaf, mid, root):
            scheduler_.submit(task)

        runner_ = registry_.get("runner")
        for task in (root, mid):
            runner_.produce(registry_, task.id)
            runner_.finish(task.id)
        return [
            registry_.get("task_mgr").get(t.id).status for t in (root, mid, leaf)
        ], runner_.started.index(leaf.id)

    assert run(with_edges=True) == run(with_edges=False)


def test_a_declared_edge_does_not_gate_dispatch(scheduler, task_mgr):
    """The scheduler asks the handoffs, not the edge list. A task depending on
    an unfinished task but consuming nothing from it still runs."""
    blocker = make_task(inputs=new_handoffs(1))
    scheduler.submit(blocker)

    dependent = make_task(depends_on=[blocker.id])
    scheduler.submit(dependent)

    assert task_mgr.get(blocker.id).status is TaskStatus.WAITING_HANDOFF
    assert task_mgr.get(dependent.id).status is DISPATCHED


def test_depends_on_survives_a_restart(scheduler, store):
    from .conftest import rebuild

    root, left, right, join = build_diamond(scheduler)

    fresh = rebuild(store)
    fresh.get("task_mgr").resume_system()
    assert set(fresh.get("task_mgr").get(join.id).depends_on) == {left.id, right.id}


def test_a_task_exposes_no_agent_id_field():
    """Criterion 21: the binding lives in the execution record, nowhere else."""
    from task_graph.models import Task

    assert "agent_id" not in Task.model_fields
    assert "agent" not in Task.model_fields
    assert "agent_spec" in Task.model_fields
