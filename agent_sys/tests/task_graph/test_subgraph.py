"""Subgraph nesting — criteria 36, 37, 38.

The four structural fields answer "which task did this expand from", "has this
subgraph begun", and "has it finished". They are markers, not gates: the
scheduler does not wait for `is_start` and does not treat `is_end` specially at
completion. `test_structure_blind.py` is the mechanical half of that claim.
"""

import pytest

from task_graph.bootstrap import build_registry
from task_graph.ids import TaskId
from task_graph.models import (
    SUBGRAPH_AGENT_SPEC,
    HandoffStatus,
    Task,
    TaskStateError,
    TaskStatus,
)
from task_graph.runner import FakeRunner

from .conftest import DISPATCHED, closure_doc, make_task, new_handoffs, with_closures

CATALOGUE = {
    "pipeline": closure_doc(
        "pipeline",
        inputs=["seed"],
        outputs=["report"],
        subgraph=[
            {"closure": "collect", "is_start": True, "is_end": False},
            {"closure": "summarise", "is_start": False, "is_end": True},
        ],
    ),
    "collect": closure_doc("collect", inputs=["seed"], outputs=["trace"]),
    "summarise": closure_doc("summarise", agent="tuner", inputs=["trace"], outputs=["report"]),
}


def parent_task(registry):
    """A submitted non-leaf, with its declared inputs and outputs wired."""
    with_closures(registry, CATALOGUE)
    seed, report = new_handoffs(2)
    task = make_task(
        inputs=[seed],
        outputs=[report],
        closure="pipeline",
        kinds={seed: "seed", report: "report"},
    )
    registry.get("scheduler").submit(task)
    return task, seed, report


# ------------------------------------------------------------- criterion 36


def test_a_subtask_names_the_task_it_expanded_from(registry, scheduler, task_mgr):
    task, seed, _ = parent_task(registry)
    subtasks = task.unfold()
    for subtask in subtasks:
        scheduler.submit(subtask)

    assert [s.parent for s in subtasks] == [task.id, task.id]
    assert {c.id for c in task_mgr.children(task.id)} == {s.id for s in subtasks}


def test_exactly_one_task_in_a_graph_has_no_parent(registry, scheduler, task_mgr):
    """The system whole task, whose expansion is the entire graph."""
    task, _, _ = parent_task(registry)
    for subtask in task.unfold():
        scheduler.submit(subtask)

    roots = [t for t in task_mgr.all() if t.parent is None]
    assert roots == [task]


def test_children_of_an_unknown_task_is_empty_and_not_an_error(task_mgr):
    """A dangling edge is treated as absent, which is what keeps submission
    order unconstrained."""
    from task_graph.ids import TaskId

    assert task_mgr.children(TaskId.new()) == []


# ------------------------------------------------------------- criterion 37


def test_the_subgraph_has_begun_is_answerable_without_a_scan(registry, scheduler, task_mgr):
    task, _, _ = parent_task(registry)
    subtasks = task.unfold()
    for subtask in subtasks:
        scheduler.submit(subtask)

    start = next(s for s in subtasks if s.is_start)
    assert not _begun(task_mgr, task.id)
    scheduler.submit(make_task(outputs=start.inputs))  # give the start its input
    registry.get("runner").produce(registry, task_mgr.all()[-1].id)
    registry.get("runner").finish(task_mgr.all()[-1].id)

    assert task_mgr.get(start.id).status is DISPATCHED
    assert _begun(task_mgr, task.id)


def test_the_subgraph_has_finished_is_answerable_without_a_scan(registry, scheduler, task_mgr):
    task, _, _ = parent_task(registry)
    subtasks = task.unfold()
    for subtask in subtasks:
        scheduler.submit(subtask)
    end = next(s for s in subtasks if s.is_end)

    assert not _finished(task_mgr, task.id)
    _force_complete(registry, end)
    assert _finished(task_mgr, task.id)


def _begun(task_mgr, parent) -> bool:
    return any(
        c.is_start and c.status is not TaskStatus.WAITING_HANDOFF for c in task_mgr.children(parent)
    )


def _finished(task_mgr, parent) -> bool:
    return any(c.is_end and c.status is TaskStatus.SUCCEEDED for c in task_mgr.children(parent))


def _force_complete(registry, task) -> None:
    """Run one subtask to SUCCEEDED regardless of what it is waiting for."""
    scheduler = registry.get("scheduler")
    handoff_mgr = registry.get("handoff_mgr")
    agent = registry.get("agent_mgr").instantiate(task.agent_spec, task.id)
    for hid in task.inputs:
        version = handoff_mgr.get(hid).open_next(task.id, agent.id)
        version.seal(HandoffStatus.VALID)
        handoff_mgr.persist(hid)
    scheduler.try_dispatch()
    registry.get("runner").produce(registry, task.id)
    registry.get("runner").finish(task.id)


# ------------------------------------------------------------- criterion 38


def test_a_leaf_is_its_own_start_and_end_and_has_no_subtasks(scheduler, task_mgr):
    leaf = make_task()
    scheduler.submit(leaf)
    assert leaf.is_start and leaf.is_end
    assert task_mgr.children(leaf.id) == []


def test_leafness_is_the_absence_of_children_not_the_marker_pair(registry, scheduler, task_mgr):
    """The spec states both-marks as a *consequence* of leafness, not a test for
    it. A single-entry subgraph's one member is both start and end and is still
    a member, so the pair cannot be the test."""
    catalogue = {
        "solo": closure_doc("solo", outputs=["report"], subgraph=[{"closure": "collect"}]),
        "collect": closure_doc("collect", outputs=["report"]),
    }
    with_closures(registry, catalogue)
    (report,) = new_handoffs(1)
    parent = make_task(outputs=[report], closure="solo", kinds={report: "report"})
    scheduler.submit(parent)
    (only,) = parent.unfold()
    scheduler.submit(only)

    assert only.is_start and only.is_end  # both marks
    assert task_mgr.children(parent.id) == [only]  # and the parent is not a leaf
    assert task_mgr.children(only.id) == []  # the member is


# ------------------------------------------------------------------- unfold


def test_unfold_wires_the_subgraph_by_handoff_kind(registry, scheduler):
    task, seed, report = parent_task(registry)
    collect, summarise = task.unfold()

    assert collect.inputs == [seed]  # a kind nobody inside produces is the parent's
    assert summarise.inputs == collect.outputs  # chained by kind
    assert summarise.outputs == [report]  # the end entry exports through the parent
    assert collect.kinds[collect.outputs[0]] == "trace"
    assert summarise.depends_on == [collect.id]


def test_unfold_inherits_the_agent_spec_from_each_member_closure(registry):
    task, _, _ = parent_task(registry)
    collect, summarise = task.unfold()
    assert (collect.agent_spec, summarise.agent_spec) == ("profiler", "tuner")


def test_unfold_on_a_task_with_no_closure_raises(scheduler):
    task = make_task()
    scheduler.submit(task)
    with pytest.raises(TaskStateError, match="no closure"):
        task.unfold()


def test_unfold_on_an_undeclared_closure_raises_and_names_the_candidates(registry, scheduler):
    with_closures(registry, CATALOGUE)
    task = make_task(closure="not_a_closure")
    scheduler.submit(task)
    with pytest.raises(TaskStateError, match="not a declared closure"):
        task.unfold()


def test_unfold_on_a_leaf_closure_raises(registry, scheduler):
    """A closure with no declared expansion has nothing to unfold, and
    improvising one is what the risk exit is for."""
    with_closures(registry, CATALOGUE)
    task = make_task(closure="collect")
    scheduler.submit(task)
    with pytest.raises(TaskStateError, match="declares no subgraph"):
        task.unfold()


def test_a_task_without_a_registry_cannot_transition(registry):
    """`model_validate` returns a task with no registry under every candidate
    mechanism, so a task obtained any other way has a dead transition path — and
    it says so instead of raising `AttributeError` three frames away."""
    with_closures(registry, CATALOGUE)
    loose = make_task(closure="pipeline")
    with pytest.raises(TaskStateError, match="no registry"):
        loose.unfold()


def test_a_resumed_non_leaf_does_not_build_its_subgraph_twice(registry):
    """A non-leaf's main phase **is** the unfold, so a resume re-enters it.

    `demo` drove `--resume` on the real path — the first time anyone had — and
    measured 2x every subtask and 2x every handoff slot, all parented to the one
    root that had resumed correctly. `has_subgraph()` asks the *declaration*,
    which is still true the second time; nothing asked whether this task had
    already unfolded.
    """
    docs = {
        "main": closure_doc("main", subgraph=[{"closure": "produce"}, {"closure": "consume"}]),
        "produce": closure_doc("produce"),
        "consume": closure_doc("consume"),
    }
    with_closures(registry, docs)
    tasks, runner = registry.get("task_mgr"), registry.get("runner")

    root = Task(agent_spec="profiler", closure="main")
    registry.get("scheduler").submit(root)
    runner.advance(registry, root.id)
    assert len(tasks.children(root.id)) == 2

    # What `TaskMgr.load` does to a task still RUNNING, then `resume_system`.
    task = tasks.get(root.id)
    task.close_execution(TaskStatus.SUSPENDED, detail="interrupted by restart")
    registry.get("scheduler")._move(root.id, TaskStatus.SUSPENDED)
    task.restart()
    runner.advance(registry, root.id)

    children = tasks.children(root.id)
    assert len(children) == 2, "the resumed root built a second subgraph beside the first"
    assert sorted(c.closure for c in children) == ["consume", "produce"]


def test_unfold_on_an_already_unfolded_task_yields_nothing_to_submit(registry):
    """Idempotent like `HandoffMgr.declare`, and for the same reason: the
    children carry attempt history and sealed handoff versions that rebuilding
    would strand."""
    docs = {
        "main": closure_doc("main", subgraph=[{"closure": "produce"}]),
        "produce": closure_doc("produce"),
    }
    with_closures(registry, docs)
    root = Task(agent_spec="profiler", closure="main")
    registry.get("scheduler").submit(root)
    registry.get("runner").advance(registry, root.id)

    first = registry.get("task_mgr").children(root.id)
    assert len(first) == 1
    assert registry.get("task_mgr").get(root.id).unfold() == []
    assert [t.id for t in registry.get("task_mgr").children(root.id)] == [first[0].id]


# ------------------------------------- a non-leaf's agent, main spec §4.8


def _agentless(name: str, **kw) -> dict:
    """A closure document with the `agent` key genuinely absent, not `None`.

    The distinction is the point of these tests: `_agent_spec_for` branches on
    truthiness, so a present-but-`None` key would pass while proving nothing
    about the document an author actually writes.
    """
    doc = closure_doc(name, **kw)
    del doc["agent"]
    return doc


def test_a_nested_non_leaf_with_no_agent_runs_under_the_system_spec(registry, scheduler):
    """§4.8 narrowed to leaf-only at rev. 10, and this is where it lands.

    `Task.agent_spec` stays a required `str` rather than becoming optional,
    because the field is read on a path a non-leaf takes: `submit` gates on
    `is_registered` and `_dispatch_pass` feeds `instantiate(...).id` into a
    required `Execution.agent_id`. A hole would reach the execution record.
    Measured before the fix: `KeyError: 'agent'`
    (`scratch/ui-yaml-2026-08/w5/probe_nested_nonleaf_agent.py`).
    """
    docs = {
        "root": closure_doc("root", subgraph=[{"closure": "mid", "froms": []}]),
        "mid": _agentless("mid", subgraph=[{"closure": "leaf", "froms": []}]),
        "leaf": closure_doc("leaf"),
    }
    with_closures(registry, docs)
    root = Task(agent_spec="profiler", closure="root")
    scheduler.submit(root)

    (mid,) = root.unfold()
    assert mid.agent_spec == SUBGRAPH_AGENT_SPEC
    scheduler.submit(mid)  # the gate at scheduler.py:53 must accept it


def test_a_non_leaf_that_names_an_agent_keeps_it(registry, scheduler):
    """The fallback is for an absent key, not for every non-leaf. Replacing a
    declared spec would be this module overruling the document."""
    docs = {
        "root": closure_doc("root", subgraph=[{"closure": "mid", "froms": []}]),
        "mid": closure_doc("mid", agent="tuner", subgraph=[{"closure": "leaf", "froms": []}]),
        "leaf": closure_doc("leaf"),
    }
    with_closures(registry, docs)
    root = Task(agent_spec="profiler", closure="root")
    scheduler.submit(root)

    (mid,) = root.unfold()
    assert mid.agent_spec == "tuner"


def test_a_leaf_with_no_agent_still_fails_loudly(registry, scheduler):
    """The fallback must not paper over a broken catalogue. A leaf with no agent
    is a load failure (`closure/check.py` check 4, and the schema's `else`); if
    one reaches here anyway, `unfold`'s obligation is to fail loudly rather than
    dispatch under a name that describes something it is not."""
    docs = {
        "root": closure_doc("root", subgraph=[{"closure": "leaf", "froms": []}]),
        "leaf": _agentless("leaf"),
    }
    with_closures(registry, docs)
    root = Task(agent_spec="profiler", closure="root")
    scheduler.submit(root)

    with pytest.raises(KeyError, match="agent"):
        root.unfold()


def test_the_composition_root_registers_the_system_spec(registry):
    """`submit` gates every task on `is_registered`, so the name has to be in
    `AgentMgr`'s table before any non-leaf is submitted. It is registered by
    `build_registry` — unconditionally, because a graph can be built with no
    agent specs admitted at all and a non-leaf in one still has to pass."""
    assert registry.get("agent_mgr").is_registered(SUBGRAPH_AGENT_SPEC)


# ---------------------------------- a non-leaf's zone, against dispatch order


class _Zones:
    """`env_mgr.fs.layout.create`'s one precondition, and nothing else of it.

    *"A subtask's storage is nested inside its parent's"* — `env_mgr` criterion
    2 — so `create` raises when the parent has no zone, with the message a real
    run died on twice:

        task caf4fb37 declares parent bd890c07, which has no zone under <run>/zones

    Modelled rather than mocked: a spy that merely *records* `place_zone` cannot
    fail, and this file's existing depth-2 test is what proved that. It builds a
    root -> mid -> leaf catalogue, ran green throughout, and the same shape in
    `examples/demo2/` did not survive a run — because nothing here creates a
    zone at all.
    """

    def __init__(self) -> None:
        self.placed: list[TaskId] = []
        self.children_when_placed: dict[TaskId, list[TaskId]] = {}

    def create(self, task: Task) -> None:
        if task.parent is not None and task.parent not in self.placed:
            raise ValueError(f"task {task.id} declares parent {task.parent}, which has no zone")
        self.placed.append(task.id)

    # `env_mgr.EnvManager`'s inbound surface, as far as a phase transition
    # reaches it. `place_zone` is the whole of it.
    def place_zone(self, task: Task, execution) -> None:
        self.children_when_placed[task.id] = [
            c.id for c in task._require_registry().get("task_mgr").children(task.id)
        ]
        self.create(task)


class _ZoningRunner(FakeRunner):
    """`FakeRunner`, plus the one filesystem effect a real dispatch has.

    `agent.Runner._main` reaches `env_mgr.prepare` -> `layout.create` for a
    **leaf**, in its main phase. Without that here, a `tests/task_graph` graph
    never needs a parent zone to exist and cannot notice one missing.
    """

    def __init__(self, zones: _Zones) -> None:
        super().__init__()
        self.zones = zones

    def advance(self, registry, task_id) -> None:
        super().advance(registry, task_id)
        task = registry.get("task_mgr").get(task_id)
        if task.status is TaskStatus.RUNNING and not task.has_subgraph():
            self.zones.create(task)


def test_a_nested_non_leaf_is_zoned_before_its_subgraph_is_dispatched(store):
    """Depth 2, through dispatch: root -> mid -> leaf, and `mid` is the shape
    that failed.

    `scratch/demo2-2026-08/runs/full2.log`, and the diagnosis in
    `scratch/demo2-2026-08/zone-ordering.md`. `enter_phase(RUNNING)` unfolds
    **and submits**, and `submit` ends in `try_dispatch` — so every child is
    already running before the call returns. While the container zone was placed
    by the parent's *attempt thread*, which the monitor wakes only afterwards,
    the parent's zone was therefore created after its children's, always; a
    nested non-leaf survived only when it won the race, and it stopped winning
    once `layout.create`'s `os.walk` over the accumulated zones tree grew past a
    child's input-validation phase.

    The two assertions are the two halves of the fix: the zone exists, and it
    exists **before there is anything to nest in it**.
    """
    zones = _Zones()
    registry = build_registry(store=store, runner=_ZoningRunner(zones), env=zones)
    registry.get("agent_mgr").register("profiler")
    docs = {
        "root": closure_doc("root", subgraph=[{"closure": "mid", "froms": []}]),
        "mid": closure_doc("mid", subgraph=[{"closure": "leaf", "froms": []}]),
        "leaf": closure_doc("leaf"),
    }
    with_closures(registry, docs)
    scheduler, task_mgr = registry.get("scheduler"), registry.get("task_mgr")
    runner = registry.get("runner")

    root = Task(agent_spec="profiler", closure="root")
    scheduler.submit(root)
    runner.advance(registry, root.id)  # -> RUNNING: places root, unfolds, submits mid
    (mid,) = task_mgr.children(root.id)
    runner.advance(registry, mid.id)  # -> RUNNING: places mid, unfolds, submits leaf
    (leaf,) = task_mgr.children(mid.id)
    runner.advance(registry, leaf.id)  # -> RUNNING: the grandchild's own zone

    assert zones.placed == [root.id, mid.id, leaf.id]
    assert zones.children_when_placed[mid.id] == [], "placed after its subgraph was dispatched"
