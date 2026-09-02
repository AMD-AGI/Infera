"""Structure drives no scheduling — criteria 42, 43, 44.

The risk in adding structural fields is that scheduling quietly starts reading
them. Criterion 42 is the mechanical guard: blank all three and nothing about
dispatch changes. Criterion 43 then asks for depth-first *anyway*, and the two
are only compatible because the order comes from the pool rather than from a key
computed over a task.
"""

from datetime import timedelta

from task_graph.ids import TaskId
from task_graph.models import Task, TaskStatus
from task_graph.permissions import Access, Grant, Permissions
from task_graph.policy import DepthFirstPolicy, FifoPolicy
from task_graph.store import MemoryStoreMgr

from .conftest import closure_doc, make_task, new_handoffs, rebuild, with_closures

# ------------------------------------------------------------- criterion 42


def scenario(*, structural: bool):
    """A fixed scenario, run once with structure and once with it blanked."""
    registry = rebuild(MemoryStoreMgr())
    scheduler, runner = registry.get("scheduler"), registry.get("runner")
    handoff_mgr = registry.get("handoff_mgr")

    root_id, other_id = TaskId.new(), TaskId.new()
    mid, tail = new_handoffs(2)

    def marks(parent, start, end):
        return (
            {"parent": parent, "is_start": start, "is_end": end}
            if structural
            else {"parent": None, "is_start": True, "is_end": True}
        )

    a = make_task(id=root_id, outputs=[mid], resources={"gpu": 2}, **marks(None, True, False))
    b = make_task(inputs=[mid], outputs=[tail], resources={"gpu": 2}, **marks(root_id, False, True))
    c = make_task(id=other_id, resources={"gpu": 2}, **marks(None, True, True))

    order: list = []
    for task in (a, b, c):
        scheduler.submit(task)
        order.append(list(runner.started))
    runner.produce(registry, a.id)
    runner.finish(a.id)
    order.append(list(runner.started))

    pools = {s.name: sorted(str(t) for t in pool) for s, pool in scheduler.pools.items()}
    assert handoff_mgr.check_if_latest_valid(mid)
    return [len(step) for step in order], pools


def test_blanking_the_structural_fields_changes_nothing():
    """Criterion 31's instrument, reused. It passes trivially because nothing
    reads the fields — which is the point: a policy that started reading
    `parent` would fail here rather than in review."""
    with_structure = scenario(structural=True)
    without = scenario(structural=False)
    assert with_structure[0] == without[0]
    assert sorted(with_structure[1]) == sorted(without[1])
    assert {k: len(v) for k, v in with_structure[1].items()} == {
        k: len(v) for k, v in without[1].items()
    }


def test_the_default_policy_reads_no_structural_field():
    """Statically: the two shipped policies touch `expedited` and `created_at`."""
    import inspect

    source = inspect.getsource(DepthFirstPolicy) + inspect.getsource(FifoPolicy)
    for field in ("parent", "is_start", "is_end", "closure", "kinds", "monitor_spec"):
        assert f".{field}" not in source, f"the policy reads Task.{field}"


def test_kinds_drives_no_scheduling_either(store):
    """Same category as the other three: a label map, not the graph."""
    registry = rebuild(store)
    scheduler, runner = registry.get("scheduler"), registry.get("runner")
    (hid,) = new_handoffs(1)
    labelled = make_task(outputs=[hid], kinds={hid: "trace"})
    scheduler.submit(labelled)
    assert runner.started == [labelled.id]
    assert registry.get("handoff_mgr").get(hid).type == "trace"


def test_kinds_reaches_the_handoff_type(scheduler, handoff_mgr):
    """Without it every `Handoff.type` is "" and a grant naming a kind matches
    no handoff, so the executor is confined to a zone holding none of its own
    inputs and finds out by failing to read one."""
    a, b = new_handoffs(2)
    scheduler.submit(make_task(outputs=[a, b], kinds={a: "trace", b: "profile"}))
    assert (handoff_mgr.get(a).type, handoff_mgr.get(b).type) == ("trace", "profile")


def test_an_unlabelled_output_keeps_the_empty_type(scheduler, handoff_mgr):
    (hid,) = new_handoffs(1)
    scheduler.submit(make_task(outputs=[hid]))
    assert handoff_mgr.get(hid).type == ""


# ------------------------------------------------------------- criterion 43


def test_depth_first_runs_the_subgraph_before_an_unrelated_sibling(store):
    """A parent whose subgraph is dispatchable, and an unrelated sibling of
    equal age competing for the one free slot.

    The trap the design names: a structure-blind key can *appear* to pass by
    accident when the unrelated task happens to sort late. Here the unrelated
    task is submitted **first** and is older, so every flat key — `created_at`,
    submission order, any id tiebreak — picks it. Only promotion order picks the
    child. Without that, the test would certify the bug.
    """
    registry = rebuild(store)
    scheduler, runner = registry.get("scheduler"), registry.get("runner")
    (mid,) = new_handoffs(1)

    # Every contender wants the whole pool, so exactly one runs at a time and
    # each completion is a real choice rather than "everything fits".
    hog = make_task(resources={"gpu": 8})
    scheduler.submit(hog)
    unrelated = make_task(spec="tuner", resources={"gpu": 8})
    scheduler.submit(unrelated)  # oldest contender: every flat key picks it
    parent = make_task(outputs=[mid], resources={"gpu": 8})
    scheduler.submit(parent)
    child = make_task(inputs=[mid], resources={"gpu": 8}, parent=parent.id, is_start=False)
    scheduler.submit(child)

    runner.finish(hog.id)
    assert runner.started[-1] == parent.id, "the newest frontier was not taken"

    runner.produce(registry, parent.id)
    runner.finish(parent.id)  # promotes the child, which is now the frontier
    assert runner.started[-1] == child.id, "depth-first abandoned the subgraph"
    assert unrelated.id not in runner.started  # and the sibling has still not run


def test_depth_first_picks_the_chain_over_a_task_submitted_while_it_ran(store):
    """The case the criterion's wording does not reach, and the one where LIFO
    on `created_at` fails: `P1 -> P2`, with an unrelated `Q` submitted *while*
    `P1` runs. When `P1` completes, depth-first must pick `P2`."""
    registry = rebuild(store)
    scheduler, runner = registry.get("scheduler"), registry.get("runner")
    (mid,) = new_handoffs(1)

    p1 = make_task(outputs=[mid], resources={"gpu": 8})
    p2 = make_task(inputs=[mid], resources={"gpu": 8})
    scheduler.submit(p1)
    scheduler.submit(p2)

    q = make_task(spec="tuner", resources={"gpu": 8})
    scheduler.submit(q)  # submitted later, so LIFO on created_at would pick it

    runner.produce(registry, p1.id)
    runner.finish(p1.id)
    assert runner.started[-1] == p2.id


def test_swapping_back_to_fifo_changes_the_order_and_nothing_else():
    """The seam criterion 10 asks for, pointed at criterion 43's scenario."""
    orders = {}
    for name, policy in (("depth", DepthFirstPolicy()), ("fifo", FifoPolicy())):
        registry = rebuild(MemoryStoreMgr(), policy=policy)
        scheduler, runner = registry.get("scheduler"), registry.get("runner")
        hog = make_task(resources={"gpu": 8})
        scheduler.submit(hog)
        first = make_task(resources={"gpu": 8})
        second = make_task(resources={"gpu": 8})
        second.created_at = first.created_at + timedelta(seconds=1)
        scheduler.submit(first)
        scheduler.submit(second)
        runner.finish(hog.id)
        orders[name] = runner.started[1]
        assert registry.get("task_mgr").get(first.id) is not None  # everything else intact

    assert orders["fifo"] != orders["depth"]


def test_promotion_order_survives_a_second_dispatch_pass(store):
    """The guard on `_move`'s early return. Without it, step 1 re-appends every
    waiting task on every pass and the order dies silently — dispatch still
    works, nothing else fails, and depth-first quietly stops happening."""
    registry = rebuild(store)
    scheduler = registry.get("scheduler")

    hog = make_task(resources={"gpu": 8})
    scheduler.submit(hog)
    queued = [make_task(resources={"gpu": 8}) for _ in range(3)]
    for task in queued:
        scheduler.submit(task)

    before = list(scheduler.pools[TaskStatus.WAITING_RESOURCE])
    scheduler.try_dispatch()  # a second pass, changing nothing
    scheduler.try_dispatch()  # and a third
    assert list(scheduler.pools[TaskStatus.WAITING_RESOURCE]) == before


def test_promotion_order_survives_a_restart(store):
    """The same failure through the recovery path, which `resume_system`
    rebuilding plain `set()` pools would reintroduce where nobody is watching."""
    from task_graph.registry import resume_all

    registry = rebuild(store)
    scheduler = registry.get("scheduler")
    hog = make_task(resources={"gpu": 8})
    scheduler.submit(hog)
    for _ in range(4):
        scheduler.submit(make_task(resources={"gpu": 8}))

    fresh = rebuild(store)
    resume_all(fresh)
    pools = fresh.get("scheduler").pools
    from task_graph.ordered import OrderedIdSet

    assert all(isinstance(pool, OrderedIdSet) for pool in pools.values())


# ------------------------------------------------------------- criterion 44


PERMS = Permissions(
    grants=(
        Grant(path="/z", access=Access.WRITE, kind="trace"),
        Grant(kind="seed"),
    )
)

# Each sub-closure declares **its own** grants, covering its own handoffs and
# nothing else — which is the shape a real package has: `examples/demo`'s three
# sub-closures each name exactly the kinds they touch.
CATALOGUE = {
    "pipeline": closure_doc(
        "pipeline",
        inputs=["seed"],
        outputs=["report"],
        subgraph=[{"closure": "collect"}, {"closure": "summarise"}],
        permissions=PERMS.model_dump(mode="json"),
    ),
    "collect": closure_doc(
        "collect",
        inputs=["seed"],
        outputs=["trace"],
        permissions={"grants": [{"kind": "seed"}, {"kind": "trace", "access": "write"}]},
    ),
    "summarise": closure_doc(
        "summarise",
        inputs=["trace"],
        outputs=["report"],
        permissions={"grants": [{"kind": "trace"}, {"kind": "report", "access": "write"}]},
    ),
}


def test_permissions_are_versioned_with_the_task_and_persist(scheduler, store):
    task = make_task(permissions=PERMS)
    scheduler.submit(task)
    record = store.read("task", str(task.id))
    assert record["permissions"]["grants"][0]["kind"] == "trace"
    assert Task.model_validate(record).permissions == PERMS


def test_a_subtask_carries_its_own_declared_permissions_not_its_parents(registry, scheduler):
    """**This test used to assert the opposite, and the opposite was the defect.**

    It read *"a subtask inherits its parent's permissions"*, and `_instantiate`
    passed `self.permissions` down — so what a sub-closure declared was
    discarded and every subtask received the **root's** full set. Every other
    field on the instantiated `Task` already comes from the sub-closure's own
    `task_spec`; permissions were the one exception, and nothing consumed the
    declaration `closure` check 6 validates at load.

    `demo` measured the cost: `produce` produces only `facts`, inherited the
    root's `summary` grant, and `env_mgr.resolve` refused a kind-named grant
    that matches no slot on the task — so a correctly-declared package could not
    dispatch its first subtask.

    Criterion 44's *"covers its subtasks recursively"* is not lost. It is a
    property of the storage layout `env_mgr` builds — containment — rather than
    of anything copied into this field; design §3.5 says so.
    """
    with_closures(registry, CATALOGUE)
    seed, report = new_handoffs(2)
    parent = make_task(
        inputs=[seed],
        outputs=[report],
        closure="pipeline",
        kinds={seed: "seed", report: "report"},
        permissions=PERMS,
    )
    scheduler.submit(parent)
    collect, summarise = parent.unfold()

    assert {g.kind for g in collect.permissions.grants} == {"seed", "trace"}
    assert {g.kind for g in summarise.permissions.grants} == {"trace", "report"}

    # The assertion the old one could not make: a subtask holds **no grant for
    # a kind it has no slot for**. `collect` never touches `report`, and under
    # inheritance it held a grant for it.
    for subtask in (collect, summarise):
        held = {g.kind for g in subtask.permissions.grants if g.kind}
        assert held <= set(subtask.kinds.values()), (
            f"{subtask.closure} holds a grant for a kind it has no slot for: "
            f"{sorted(held - set(subtask.kinds.values()))}"
        )
    assert collect.permissions != PERMS, "the parent's set must not be copied down"


def test_a_subtask_of_a_closure_that_declares_nothing_gets_an_empty_set(registry, scheduler):
    """Empty, not the parent's. The absent case is where inheritance used to
    look harmless, and it is the same substitution."""
    catalogue = dict(CATALOGUE)
    catalogue["collect"] = closure_doc("collect", inputs=["seed"], outputs=["trace"])
    with_closures(registry, catalogue)
    seed, report = new_handoffs(2)
    parent = make_task(
        inputs=[seed],
        outputs=[report],
        closure="pipeline",
        kinds={seed: "seed", report: "report"},
        permissions=PERMS,
    )
    scheduler.submit(parent)
    collect, _ = parent.unfold()
    assert collect.permissions.grants == ()


def test_an_agent_carries_no_permissions_of_its_own(scheduler, agent_mgr):
    """Criterion 44's other half. The reach is the *task's*, so an agent minted
    for it inherits by construction and has no field of its own to disagree
    with — re-deriving a subtree's reach per agent is what the field placement
    avoids."""
    task = make_task(permissions=PERMS)
    scheduler.submit(task)
    agent = agent_mgr.get(task.current.agent_id)
    assert "permissions" not in agent.model_dump()
    assert "permissions" not in type(agent).model_fields
