"""Cascading cancel and the graph-level load checks — criteria 49 to 54.

Cancel is an explicit act on tasks that have not run; it decides nothing about
content. That distinction is what makes reversing three earlier "no cascade"
passages consistent rather than contradictory, and criterion 52 is the assertion
that the line holds.
"""

import threading

import pytest
from pydantic import ValidationError

from task_graph.graph import check_graph
from task_graph.ids import TaskId
from task_graph.models import HandoffStatus, TaskStateError, TaskStatus
from task_graph.resource import GpuMgr

from .conftest import (
    DISPATCHED,
    closure_doc,
    make_task,
    new_handoffs,
    rebuild,
    task_specs,
    with_closures,
)

# ------------------------------------------------------------- criterion 49


def chain(scheduler, n: int, *, parent: TaskId | None = None):
    """`t0 -> t1 -> ... -> t(n-1)`, wired through one handoff per link."""
    handoffs = new_handoffs(n - 1)
    tasks = []
    for i in range(n):
        tasks.append(
            make_task(
                inputs=[handoffs[i - 1]] if i else [],
                outputs=[handoffs[i]] if i < n - 1 else [],
                parent=parent,
                resources={"gpu": 8},  # so nothing runs away while we set up
            )
        )
    for task in tasks:
        scheduler.submit(task)
    return tasks


def test_cancel_cascades_downstream_within_its_own_graph(scheduler, task_mgr):
    a, b, c = chain(scheduler, 3)
    report = task_mgr.get(b.id).cancel("no longer wanted")

    assert task_mgr.get(b.id).status is TaskStatus.CANCELLED
    assert task_mgr.get(c.id).status is TaskStatus.CANCELLED
    assert task_mgr.get(a.id).status is DISPATCHED  # upstream is untouched
    assert [tid for tid, _ in report.reached] == [b.id, c.id]
    assert report.reached[1][1] == f"upstream {b.id} cancelled"


def test_a_task_in_another_graph_is_untouched(scheduler, task_mgr):
    """Criterion 49's other half. The outsider consumes the same handoff and is
    in a different subgraph, so the cascade stops at the boundary."""
    parent = TaskId.new()
    (shared,) = new_handoffs(1)
    inside_producer = make_task(outputs=[shared], parent=parent, inputs=new_handoffs(1))
    inside_consumer = make_task(inputs=[shared], parent=parent)
    outside_consumer = make_task(inputs=[shared])  # parent is None: another graph
    for task in (inside_producer, inside_consumer, outside_consumer):
        scheduler.submit(task)

    task_mgr.get(inside_producer.id).cancel()

    assert task_mgr.get(inside_consumer.id).status is TaskStatus.CANCELLED
    assert task_mgr.get(outside_consumer.id).status is TaskStatus.WAITING_HANDOFF


def test_a_diamond_does_not_raise(scheduler, task_mgr):
    """`A -> {B, C} -> D`. Without the visited set `D` is reached twice, and the
    second visit finds it CANCELLED — which is not a waiting state, so
    `cancel()`'s own precondition raises on a graph shape that is not exotic."""
    ab, ac, bd, cd = new_handoffs(4)
    a = make_task(inputs=new_handoffs(1), outputs=[ab, ac])
    b = make_task(inputs=[ab], outputs=[bd])
    c = make_task(inputs=[ac], outputs=[cd])
    d = make_task(inputs=[bd, cd])
    for task in (a, b, c, d):
        scheduler.submit(task)

    report = task_mgr.get(a.id).cancel("root")

    assert all(task_mgr.get(t.id).status is TaskStatus.CANCELLED for t in (a, b, c, d))
    assert [tid for tid, _ in report.reached].count(d.id) == 1


def test_the_walk_is_level_by_level_and_not_depth_first(scheduler, task_mgr):
    """Measured: on a diamond, recursion gives ['A','B','D','E','C','F'] and a
    drained queue gives ['A','B','C','F','D','E']. The spec says level by level,
    so the two are not equal options."""
    ab, ac, af, bd, cd, de = new_handoffs(6)
    a = make_task(inputs=new_handoffs(1), outputs=[ab, ac, af])
    b = make_task(inputs=[ab], outputs=[bd])
    c = make_task(inputs=[ac], outputs=[cd])
    f = make_task(inputs=[af])
    d = make_task(inputs=[bd, cd], outputs=[de])
    e = make_task(inputs=[de])
    for task in (a, b, c, f, d, e):
        scheduler.submit(task)

    order = [tid for tid, _ in task_mgr.get(a.id).cancel().reached]
    names = {a.id: "A", b.id: "B", c.id: "C", f.id: "F", d.id: "D", e.id: "E"}
    assert [names[tid] for tid in order] == ["A", "B", "C", "F", "D", "E"]


def test_a_running_task_is_refused_and_recorded(scheduler, task_mgr):
    """The narrowest behaviour, and therefore the one that presumes least: what
    a cascade does on reaching a live task is an open specification question
    (design O14), so it is reported rather than decided here."""
    (mid,) = new_handoffs(1)
    upstream = make_task(outputs=[mid], inputs=new_handoffs(1))
    live = make_task(inputs=[mid])
    scheduler.submit(upstream)
    scheduler.submit(live)
    scheduler._move(live.id, TaskStatus.RUNNING)  # simulate it having started

    report = task_mgr.get(upstream.id).cancel()
    assert report.refused == ((live.id, TaskStatus.RUNNING),)
    assert task_mgr.get(live.id).status is TaskStatus.RUNNING


def test_the_reason_travels_in_the_report_and_not_onto_the_task(scheduler, task_mgr):
    """Rev. 11 wrote `task.cancel_reason = reason` and there is no such field:
    `Model` sets `extra="forbid"` with `validate_assignment=True`, so the
    cascade raised on its first entry. A field would also be a second record of
    what the report already carries."""
    _, b = chain(scheduler, 2)
    report = task_mgr.get(b.id).cancel("because")
    assert report.reached == ((b.id, "because"),)
    assert "cancel_reason" not in type(b).model_fields
    with pytest.raises(ValidationError):
        b.cancel_reason = "boom"


def test_cancelling_a_live_task_directly_is_rejected(scheduler, task_mgr):
    task = make_task()
    scheduler.submit(task)
    with pytest.raises(TaskStateError, match="expected a waiting state"):
        task_mgr.get(task.id).cancel()


def test_remove_queued_does_not_cascade(scheduler, task_mgr):
    """The operator's verb and the task's transition are two different acts.
    Spec §5.1 gives `remove_queued` one effect and §3.2.3 gives `cancel()` the
    cascade; conflating them would make a single-task cancellation impossible."""
    a, b, c = chain(scheduler, 3)
    scheduler.remove_queued(b.id)
    assert task_mgr.get(b.id).status is TaskStatus.CANCELLED
    assert task_mgr.get(c.id).status is TaskStatus.WAITING_HANDOFF


# ------------------------------------------------------------- criterion 52


def test_a_cascade_changes_no_handoff_verdict(scheduler, task_mgr, handoff_mgr, runner, registry):
    """Cancel decides about tasks, never about content. Airflow actively
    conflates the two — one `clear` both terminates a running task and resets a
    succeeded one — so the line being drawn here is ours."""
    (mid,) = new_handoffs(1)
    producer = make_task(outputs=[mid])
    consumer = make_task(inputs=[mid], outputs=new_handoffs(1))
    scheduler.submit(producer)
    scheduler.submit(consumer)
    runner.produce(registry, producer.id)
    runner.finish(producer.id)

    before = handoff_mgr.get(mid).model_dump(mode="json")
    downstream = make_task(inputs=[mid], resources={"gpu": 8})
    scheduler.submit(downstream)
    scheduler._move(downstream.id, TaskStatus.WAITING_RESOURCE)
    task_mgr.get(downstream.id).cancel("not needed")

    assert handoff_mgr.get(mid).model_dump(mode="json") == before
    assert handoff_mgr.get(mid).latest.status is HandoffStatus.VALID


# ------------------------------------------------------------- criterion 51


CATALOGUE = {
    "pipeline": closure_doc(
        "pipeline",
        outputs=["report"],
        subgraph=[{"closure": "collect"}, {"closure": "summarise"}],
    ),
    "collect": closure_doc("collect", outputs=["trace"]),
    "summarise": closure_doc("summarise", inputs=["trace"], outputs=["report"]),
}


def test_replace_with_instantiates_only_a_declared_closure(registry, scheduler, task_mgr):
    with_closures(registry, CATALOGUE)
    (report,) = new_handoffs(1)
    task = make_task(
        outputs=[report], inputs=new_handoffs(1), kinds={report: "report"}, closure="pipeline"
    )
    scheduler.submit(task)

    with pytest.raises(TaskStateError, match="not a declared closure"):
        task_mgr.get(task.id).replace_with("invented")
    assert sorted(t.closure for t in task_mgr.all() if t.closure) == ["pipeline"]


def test_replace_with_cancels_the_downstream_then_regenerates(registry, scheduler, task_mgr):
    with_closures(registry, CATALOGUE)
    (report,) = new_handoffs(1)
    task = make_task(
        outputs=[report], inputs=new_handoffs(1), kinds={report: "report"}, closure="pipeline"
    )
    consumer = make_task(inputs=[report])
    scheduler.submit(task)
    scheduler.submit(consumer)

    report_out = task_mgr.get(task.id).replace_with("pipeline")

    assert task_mgr.get(consumer.id).status is TaskStatus.CANCELLED
    assert [tid for tid, _ in report_out.reached] == [consumer.id]
    assert task_mgr.get(task.id).status is not TaskStatus.CANCELLED  # itself untouched
    assert sorted(t.closure for t in task_mgr.all() if t.closure) == [
        "collect",
        "pipeline",
        "summarise",
    ]


# --------------------------------------------------- criteria 50 and 53, load


LEAF_ONLY = {
    "optimize_kernel": closure_doc(
        "optimize_kernel",
        resources={"gpu": 2},
        subgraph=[
            {"closure": "a", "froms": []},
            {"closure": "b", "froms": []},
            {"closure": "c", "froms": []},
            {"closure": "d", "froms": []},
        ],
    ),
    "a": closure_doc("a"),
    "b": closure_doc("b"),
    "c": closure_doc("c"),
    "d": closure_doc("d"),
}


def test_a_non_leaf_declaring_resources_is_rejected_at_load():
    problems = check_graph(task_specs(LEAF_ONLY))
    assert [p.keyword for p in problems] == ["leaf_only_acquisition"]
    message = problems[0].message
    assert "optimize_kernel" in message and "{'gpu': 2}" in message
    assert "Move the declaration onto the subtasks" in message


def test_a_leaf_declaring_resources_is_fine():
    assert check_graph(task_specs({"a": closure_doc("a", resources={"gpu": 2})})) == []


def test_a_skipped_spec_is_not_walked():
    assert check_graph(task_specs(LEAF_ONLY), skip={"optimize_kernel"}) == []


CONTAINED = {
    "pipeline": closure_doc(
        "pipeline",
        outputs=["report"],
        subgraph=[
            {"closure": "collect", "is_start": True, "is_end": False, "froms": []},
            {"closure": "summarise", "is_start": False, "is_end": True, "froms": ["collect"]},
        ],
    ),
    "collect": closure_doc("collect", outputs=["trace"]),
    "summarise": closure_doc("summarise", inputs=["trace"], outputs=["report"]),
}


def test_a_handoff_produced_inside_a_subgraph_may_not_be_consumed_outside():
    catalogue = dict(CONTAINED, intruder=closure_doc("intruder", inputs=["trace"]))
    problems = check_graph(task_specs(catalogue))
    assert [p.keyword for p in problems] == ["subgraph_containment"]
    message = problems[0].message
    assert "intruder" in message and "collect" in message and "pipeline" in message


def test_the_end_entry_subtasks_outputs_are_the_declared_boundary():
    """`report` is exported, so consuming it from outside is legal — that is the
    whole point of having a boundary rather than a wall."""
    catalogue = dict(CONTAINED, downstream=closure_doc("downstream", inputs=["report"]))
    assert check_graph(task_specs(catalogue)) == []


def test_a_member_of_the_subgraph_may_consume_from_a_sibling():
    """`summarise` consumes `trace` from inside, which is the ordinary case."""
    assert check_graph(task_specs(CONTAINED)) == []


# `examples/demo2/`'s shape, reduced to the two links that matter: `main` ->
# `grade` -> `review`, with `review` consuming a kind `main`'s own entry
# produces. `review` is a grandchild, so it is strictly inside `main`'s
# subgraph and reachable only through `grade`.
NESTED = {
    "main": closure_doc(
        "main",
        outputs=["scores"],
        subgraph=[
            {"closure": "problems", "froms": []},
            {"closure": "grade", "froms": ["problems"], "is_end": True},
        ],
    ),
    "problems": closure_doc("problems", outputs=["problems"]),
    "grade": closure_doc(
        "grade",
        inputs=["problems"],
        outputs=["scores"],
        subgraph=[
            {"closure": "review", "froms": []},
            {"closure": "score", "froms": ["review"], "is_end": True},
        ],
    ),
    "review": closure_doc("review", inputs=["problems"], outputs=["review"]),
    "score": closure_doc("score", inputs=["review"], outputs=["scores"]),
}


def test_a_grandchild_may_consume_a_kind_produced_in_its_grandparents_subgraph():
    """The false positive `examples/demo2/` measured. `review` consumes
    `problems`, which `main`'s own `problems` entry produces, and `review` is not
    one of `main`'s *direct* entries — but it is `grade`'s, and `grade` is
    `main`'s, so it is inside.

    The check's own message is the proof: "cancelling inside `main` would
    silently block `review`" is false, because cancelling inside `main` cancels
    `grade`, and `grade` cancels `review`. There is no outside observer here to
    protect, so there is nothing to report.
    """
    assert check_graph(task_specs(NESTED)) == []


def test_a_depth_1_violation_is_still_rejected_when_the_subgraph_nests():
    """The test that stops the fix above from being a blanket `return []`.

    Same nested catalogue, plus a genuine outsider consuming `problems` — a kind
    produced one level inside `main` and not exported through `main`'s end entry
    `grade`. Widening "inside" to the descendant set must not widen it to the
    whole catalogue.
    """
    catalogue = dict(NESTED, intruder=closure_doc("intruder", inputs=["problems"]))
    (violation,) = [
        p for p in check_graph(task_specs(catalogue)) if p.keyword == "subgraph_containment"
    ]
    assert violation.origin == "intruder"
    assert "'problems' inside 'main''s subgraph" in violation.message


def test_a_grandchilds_kind_consumed_by_a_real_outsider_is_still_rejected():
    """Why `produced_inside` stays one level deep while `inside` goes all the way.

    `review` (the kind) is produced by a *grandchild* of `main`, and `intruder`
    is outside every subgraph in the catalogue. `main`'s own pass does not see
    the kind at all — its producer set is its direct entries'. The fault is
    caught anyway, because `check_graph` runs this check once per spec and
    `grade`'s pass has `review` as a direct entry.

    The message therefore names `grade`, not `main`, and that is the useful
    report: the boundary the author has to export through is `grade`'s end
    entry. Widening the producer set transitively would add a second, vaguer
    copy of this same problem blamed on `main`.
    """
    catalogue = dict(NESTED, intruder=closure_doc("intruder", inputs=["review"]))
    (violation,) = [
        p for p in check_graph(task_specs(catalogue)) if p.keyword == "subgraph_containment"
    ]
    assert violation.origin == "intruder"
    assert "'review' inside 'grade''s subgraph" in violation.message
    assert "main" not in violation.message, "one fault, one origin, and it is the nearest parent"


def test_a_cycle_in_the_closure_graph_terminates():
    """`a`'s subgraph names `b` and `b`'s names `a`. Nothing rejects that at this
    point in the load — these checks report per spec rather than raise — so the
    descendant walk has to survive being handed it, and it does because the
    membership test is also the visited set.

    Run on a **daemon** thread so that a regression fails the suite instead of
    hanging it. Both halves of that are measured, not assumed: deleting the
    visited set was tried, and the walk is a `while` loop over a work list, so it
    spins for ever rather than raising `RecursionError` — and a runaway worker
    cannot be cancelled, so a `ThreadPoolExecutor` blocks in `shutdown(wait=True)`
    on the way out and hangs anyway. A daemon thread we simply stop joining, and
    the interpreter drops it at exit.
    """
    cyclic = {
        "a": closure_doc("a", subgraph=[{"closure": "b", "froms": []}]),
        "b": closure_doc("b", subgraph=[{"closure": "a", "froms": []}]),
    }
    out: list = []
    walker = threading.Thread(
        target=lambda: out.append(check_graph(task_specs(cyclic))), daemon=True
    )
    walker.start()
    walker.join(timeout=30)
    assert not walker.is_alive(), "the descendant walk did not terminate on a cyclic catalogue"
    assert out == [[]]


def test_the_checks_report_rather_than_raise():
    """A catalogue with two faults reports both: one broken spec must not hide
    the other."""
    catalogue = dict(LEAF_ONLY, **CONTAINED, intruder=closure_doc("intruder", inputs=["trace"]))
    keywords = sorted({p.keyword for p in check_graph(task_specs(catalogue))})
    assert keywords == ["leaf_only_acquisition", "subgraph_containment"]


def test_check_graph_takes_task_specs_and_not_closure_documents():
    """The shape mistake that makes both checks pass **vacuously**, which is why
    it is pinned rather than left to a docstring.

    A task spec is the inner object — `goal`, `body`, `inputs`, `outputs`,
    `resources`, `subgraph`. A closure document wraps it under a `task` key.
    Handed the wrapper, `subgraph_entries` finds no `subgraph`, every task looks
    a leaf, and `check_graph` returns `[]` on a catalogue that violates both
    rules: green, and checking nothing. `interfaces.md` §2 passes
    `views.task_specs`, so the inner object is the shape.
    """
    assert check_graph(task_specs(LEAF_ONLY)), "the real shape must report"
    assert check_graph(LEAF_ONLY) == [], "and the wrapper is silent, which is the hazard"


def test_the_problems_are_spec_loaders_and_not_a_second_shape():
    """One report format across every load-time check. A second shape here would
    make the error output depend on which pass found the fault."""
    from spec_loader.protocols import Problem

    assert all(isinstance(p, Problem) for p in check_graph(task_specs(LEAF_ONLY)))


# ------------------------------------------------------------- criterion 53


def test_a_parent_holds_nothing_while_its_subgraph_runs(registry, scheduler, task_mgr):
    """The same instrument criterion 40 uses for a leaf's single lease, pointed
    at the opposite expectation."""
    with_closures(registry, CATALOGUE)
    (report,) = new_handoffs(1)
    parent = make_task(outputs=[report], kinds={report: "report"}, closure="pipeline")
    scheduler.submit(parent)
    runner = registry.get("runner")

    snapshots = [registry.get("resource:gpu").available]
    runner.advance(registry, parent.id)  # -> RUNNING, which unfolds
    snapshots.append(registry.get("resource:gpu").available)
    runner.advance(registry, parent.id)  # -> OUTPUT_VALIDATING
    snapshots.append(registry.get("resource:gpu").available)

    assert snapshots == [8, 8, 8]
    assert len(task_mgr.children(parent.id)) == 2  # the subgraph really ran


def test_entering_the_main_phase_unfolds_and_submits(registry, scheduler, task_mgr):
    with_closures(registry, CATALOGUE)
    (report,) = new_handoffs(1)
    parent = make_task(outputs=[report], kinds={report: "report"}, closure="pipeline")
    scheduler.submit(parent)
    registry.get("runner").advance(registry, parent.id)

    children = task_mgr.children(parent.id)
    assert sorted(c.closure for c in children) == ["collect", "summarise"]
    assert registry.get("runner").started[-1] == next(c.id for c in children if c.is_start)


# ------------------------------------------------------------- criterion 54


def test_a_parent_and_its_child_on_one_pool_do_not_deadlock(store):
    """A pool sized to satisfy the parent *or* the child, never both. It passes
    because the parent declares nothing, so there is no "both" — and under the
    pre-subgraph model this would hang rather than fail an assertion."""
    catalogue = {
        "wrapper": closure_doc("wrapper", outputs=["report"], subgraph=[{"closure": "worker"}]),
        "worker": closure_doc("worker", outputs=["report"], resources={"gpu": 1}),
    }
    registry = rebuild(store)
    registry.register("resource:gpu", GpuMgr(registry, capacity=1))
    with_closures(registry, catalogue)
    scheduler, runner, task_mgr = (
        registry.get("scheduler"),
        registry.get("runner"),
        registry.get("task_mgr"),
    )

    (report,) = new_handoffs(1)
    parent = make_task(outputs=[report], kinds={report: "report"}, closure="wrapper")
    scheduler.submit(parent)
    runner.advance(registry, parent.id)  # unfolds; the child takes the only slot

    (child,) = task_mgr.children(parent.id)
    assert task_mgr.get(child.id).status is DISPATCHED
    assert registry.get("resource:gpu").available == 0

    runner.produce(registry, child.id)
    runner.finish(child.id)
    runner.advance(registry, parent.id)
    runner.finish(parent.id)

    assert task_mgr.get(parent.id).status is TaskStatus.SUCCEEDED
    assert task_mgr.get(child.id).status is TaskStatus.SUCCEEDED
    assert registry.get("resource:gpu").available == 1
