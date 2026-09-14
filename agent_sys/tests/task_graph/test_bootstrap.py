"""The composition root, and the import rule that protects the package.

`docs/interfaces.md` §2 is normative for the listing and §4.7 for the imports,
and the two pull against each other: the root constructs six other packages'
types while the package may import pydantic and `spec_loader` only. Deferring
every sibling import to call time is what reconciles them, and
`test_importing_task_graph_reaches_no_sibling_package` is the assertion that it
stays reconciled.
"""

import inspect
import subprocess
import sys

import pytest

from task_graph.bootstrap import RegistryViews, build_registry
from task_graph.models import SUBGRAPH_AGENT_SPEC
from task_graph.policy import DepthFirstPolicy
from task_graph.runner import FakeRunner
from task_graph.store import MemoryStoreMgr

from .conftest import make_task, new_handoffs, rebuild

PARAMETERS = [
    "store",
    "runner",
    "policy",
    "resources",
    "registries",
    "packages",
    "env",
    "strict_level",
    "monitors",
    "config_order",
    "handoff_root",
    "knowledge_root",
]

OWNED = [
    "store_mgr",
    "handoff_mgr",
    "task_mgr",
    "agent_mgr",
    "resource:gpu",
    "resource:token",
    "policy",
    "runner",
    "scheduler",
]


def test_the_signature_is_the_one_the_contract_names():
    signature = inspect.signature(build_registry)
    assert list(signature.parameters) == PARAMETERS
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in signature.parameters.values()), (
        "every parameter is keyword-only: eleven positional arguments is not an interface"
    )


def test_it_registers_everything_this_package_owns():
    r = build_registry(store=MemoryStoreMgr())
    for name in OWNED:
        assert name in r, name


def test_the_default_runner_stays_the_fake():
    """Deliberately. The suite written against a fake whose completion the test
    drives is the regression guard for everything else; swapping the default
    would rewrite it. The real runner is passed in."""
    assert isinstance(build_registry().get("runner"), FakeRunner)


def test_the_default_policy_is_depth_first():
    assert isinstance(build_registry().get("policy"), DepthFirstPolicy)


def test_the_default_pools_are_one_of_each_class():
    r = build_registry()
    assert r.get("resource:gpu").capacity == 8
    assert r.get("resource:token").capacity == 1_000_000


def test_two_registries_share_nothing():
    a, b = build_registry(), build_registry()
    assert a.get("task_mgr") is not b.get("task_mgr")
    assert a.get("resource:gpu") is not b.get("resource:gpu")


def test_a_handoff_store_needs_a_root():
    """An artefact store rooted at a default nobody chose is worse than a loud
    failure at the first resolution."""
    r = build_registry()
    assert "handoff_store" not in r


def test_loading_packages_without_the_spec_registries_fails_loudly():
    """Rather than silently loading nothing. The message names what is missing."""

    class Pkg:
        root = "."

        def discover(self):
            return []

        def config_for(self, source):
            return {}

    r = build_registry()
    if "closures" in r:  # the sibling modules are built; the load path is real
        pytest.skip("the spec registries are available, so there is nothing to refuse")
    with pytest.raises(RuntimeError, match="spec registries"):
        build_registry(packages=[Pkg()])


def test_the_root_reaches_for_the_accessor_handoff_actually_ships():
    """Pinned from this side too, and the reason is a defect this exact line had.

    The call used to be `getattr(handoff_specs, "load_report", lambda: None)()`
    while the registry spelled it `report()`. The mismatch **did not fail**: the
    default produced `None`, `check_closures` returns early on `None`, and an
    escape-hatch admission went unreported in the assembled system — with
    `tests/handoff`, `tests/closure` and `tests/task_graph` all green, because
    each package tested only its own side.

    `handoff` pins the name from their end, which catches a rename by them. This
    catches a rename by *me* — the half their test structurally cannot see.
    """
    import ast

    from handoff import HandoffSpecRegistry

    source = inspect.getsource(sys.modules["task_graph.bootstrap"])
    reached = {
        node.func.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr.endswith("report")
    }
    assert reached == {"load_report"}, reached
    assert hasattr(HandoffSpecRegistry, "load_report")


def test_the_root_does_not_soften_a_missing_accessor_into_none():
    """A `getattr` default is what turned the rename into silence. There is no
    tolerance left in the production path: a registry that cannot answer must
    raise, and a test stub that wants tolerance answers the method.

    Over the AST rather than the text, for the reason criterion 48 gives: the
    token appears in the comment explaining why it is gone, so a substring check
    would fail on the explanation.
    """
    import ast

    source = inspect.getsource(sys.modules["task_graph.bootstrap"])
    tree = ast.parse(source)
    softened = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) > 2
    ]
    assert softened == [], "a getattr default turns a missing accessor into silence"

    # `hasattr` is the same quiet skip in a different spelling, and it slipped
    # past the check above once: a `hasattr(agent_specs, "check_knowledge")`
    # guard was added an hour after the `getattr` default two functions up was
    # removed, for the same defect. A rename would have stopped the pass in
    # silence. Both spellings are forbidden now.
    probed = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "hasattr"
    ]
    assert probed == [], "hasattr guards a sibling's method into a silent skip"


def test_registry_views_resolves_the_five_by_name():
    r = build_registry()
    views = RegistryViews(r)
    with pytest.raises(KeyError, match="no spec registry for kind"):
        views.for_kind("not_a_kind")


def test_importing_task_graph_reaches_no_sibling_package():
    """The import rule, checked in a fresh interpreter because this one has
    already imported half the repository through other tests.

    `spec_loader` is permitted — `graph.py` imports it for `TaskSpec` and
    `Problem`, and it imports nothing of ours in return.
    """
    code = (
        "import sys, task_graph;"
        "forbidden = {'handoff', 'validator', 'agent', 'closure', 'monitor', 'env_mgr'};"
        "print(sorted(forbidden & set(sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "[]", f"importing task_graph pulled in {out}"


def test_building_a_registry_may_reach_a_sibling_and_that_is_the_point():
    """The composition root is the one place a wide fan-in belongs. Deferring it
    to call time is what keeps `import task_graph` free of the edge — not an
    absence of the dependency, which the root genuinely has."""
    source = inspect.getsource(sys.modules["task_graph.bootstrap"])
    tree = __import__("ast").parse(source)
    ast = __import__("ast")
    module_level = {
        alias.name
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    } | {node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.module}
    for forbidden in ("handoff", "validator", "agent", "closure", "monitor", "env_mgr"):
        assert forbidden not in module_level


# --------------------------------------------------------------- KindSource


HANDOFF_KIND = {
    "name": "trace",
    "description": "a profiler trace",
    "content_type": "text",
    "scope": "fixed.required",
    "validators": ["shape"],
}


def wired_store(tmp_path):
    """A root with both stores built, and one admitted handoff kind."""
    r = build_registry(handoff_root=str(tmp_path), knowledge_root=str(tmp_path))
    r.get("handoff_specs").add("trace", HANDOFF_KIND, origin="test")
    return r


def test_a_store_is_built_with_a_kind_source(tmp_path):
    """`interfaces.md` §2.6. Without one a store reads but does not publish —
    `put` checked no README sections, validated `items` against nothing and
    recorded `kind: ""`, with all of `handoff`'s tests green because every one
    injects a resolver."""
    r = wired_store(tmp_path)
    for name in ("handoff_store", "knowledge_store"):
        assert getattr(r.get(name), "_kinds", None) is not None, name


def test_the_kind_source_resolves_a_declared_type_through_both_halves(tmp_path):
    """The mapping no single component holds: `hid -> Handoff.type` is this
    package's and `type -> HandoffKind` is `handoff`'s."""
    from task_graph.bootstrap import _KindSource
    from task_graph.ids import TaskId

    r = wired_store(tmp_path)
    (hid,) = new_handoffs(1)
    r.get("handoff_mgr").declare([hid], producer_task_id=TaskId.new(), types={hid: "trace"})

    assert _KindSource(r).kind_for(hid).name == "trace"


@pytest.mark.parametrize("case", ["untyped", "undeclared", "unadmitted"])
def test_an_unresolvable_id_answers_none_rather_than_raising(tmp_path, case):
    """`None` is what the Protocol asks for, and the store turns it into a loud
    `put`. An exception here would be converted straight back to `None` by
    whoever caught it, one seam further out."""
    from task_graph.bootstrap import _KindSource
    from task_graph.ids import TaskId

    r = wired_store(tmp_path)
    (hid,) = new_handoffs(1)
    types = {hid: "ghost"} if case == "unadmitted" else {}
    if case != "undeclared":
        r.get("handoff_mgr").declare([hid], producer_task_id=TaskId.new(), types=types)

    assert _KindSource(r).kind_for(hid) is None


def test_type_of_is_a_question_and_not_a_mutable_handle(tmp_path):
    """The root asks `type_of` rather than taking `get(hid)` and reading `.type`.
    A live `Handoff` is one call from `open_next` and `seal`, which is the hazard
    `test_authority.py` keeps out of the scheduler — and it is the same hazard
    wherever it happens."""
    import inspect

    from task_graph import bootstrap

    source = inspect.getsource(bootstrap)
    assert 'handoff_mgr").type_of(' in source
    assert 'handoff_mgr").get(' not in source


# ------------------------------------------------------- the caller's five


class Stub:
    """The least a spec registry can be and still be one.

    `object()` was enough until the root gained the `agent_specs` bridge, which
    asks for `names()`. That is the stub being unrealistic rather than the code
    being wrong — a real `Registries` supplies real registries — and it is worth
    the four lines to keep the difference visible.
    """

    def __init__(self, *names: str) -> None:
        self._names = list(names)

    def names(self) -> list[str]:
        return list(self._names)

    def origin_of(self, name: str) -> str:
        return f"pkg/{name}.jsonnet"

    def check_knowledge(self, handoff_specs, *, mandatory: bool = False):
        """`agent`'s pass. The root calls it plainly, so the double answers it —
        the tolerance belongs here and not in a `hasattr` at the call site."""
        return [], []

    def freeze(self) -> None:
        """Only `closures` is frozen, but one stub serves five slots and a
        double that answers less than the phase asks is how the last two of
        these grew a method each."""


class Views:
    """A `Registries` — five attributes, which is all the Protocol declares."""

    def __init__(self, **five) -> None:
        self.__dict__.update(five)


FIVE = ("handoff_specs", "validator_specs", "task_specs", "agent_specs", "closures")


def test_the_caller_can_supply_the_five_registries():
    """The root cannot construct what only the caller can configure — the same
    reason the default runner stays a fake and the real one is passed in."""
    supplied = Views(**{key: Stub() for key in FIVE})
    r = build_registry(registries=supplied)
    for key in FIVE:
        assert r.get(key) is getattr(supplied, key), key


def test_supplying_them_is_how_the_escape_hatch_gets_a_route():
    """`handoff` spec §5.3 gives the escape hatch a command-line switch, and
    until `registries` existed nothing in the assembled system could turn it on:
    the root built `HandoffSpecRegistry()` with no arguments. A capability with
    no route to it is the same family as `put` having no caller.

    Asserted through the real registry rather than a stub, because the point is
    that the flag survives the trip."""
    from handoff import HandoffSpecRegistry

    loose = HandoffSpecRegistry(allow_no_validator=True)
    supplied = Views(**{key: loose if key == "handoff_specs" else Stub() for key in FIVE})

    r = build_registry(registries=supplied)
    assert r.get("handoff_specs") is loose
    assert r.get("handoff_specs")._allow_no_validator is True


def test_omitting_it_builds_exactly_what_it_built_before():
    """Additive, so no caller has to move. Changing a signature under the first
    real caller is the shape §4.11 names."""
    r = build_registry()
    for key in FIVE:
        assert key in r
    assert r.get("handoff_specs")._allow_no_validator is False


def test_an_incomplete_view_raises_rather_than_registering_four_of_five():
    """`Registries` declares all five. A view that cannot answer is a wiring
    fault, not a tolerable absence — and a plain `getattr` is what says so."""
    with pytest.raises(AttributeError):
        build_registry(registries=Views(handoff_specs=Stub()))


# ---------------------------------------------- the agent_specs bridge, F-D2


def test_every_admitted_agent_spec_is_instantiable(tmp_path):
    """`demo` F-D2. Two tables and nothing joined them: `agent_specs` holds the
    admitted documents, `AgentMgr`'s table is what `submit` checks — so a graph
    whose spec loaded cleanly still got `unknown agent spec 'collect'`.

    A root that has run both whole-catalogue passes and raised on nothing, then
    returns a registry that cannot dispatch anything, has not finished
    composing.
    """
    supplied = Views(
        **{k: Stub("collect", "summarise") if k == "agent_specs" else Stub() for k in FIVE}
    )
    r = build_registry(registries=supplied)

    agent_mgr = r.get("agent_mgr")
    # `subgraph` is not admitted from anywhere — it is the name the system
    # supplies for a non-leaf, registered by the same function and for the same
    # reason: `submit` gates on `is_registered`, and a non-leaf has no author to
    # write one. Spelled out rather than filtered away, because the exactness is
    # what makes this assertion say "the bridge registered everything".
    assert sorted(agent_mgr.specs()) == ["collect", SUBGRAPH_AGENT_SPEC, "summarise"]
    assert agent_mgr.is_registered("collect")


def test_the_bridge_carries_the_name_and_not_the_document():
    """`Agent.config` is left empty by the task definition. Passing the admitted
    spec would copy the whole document into every minted `Agent`; the document
    stays in `agent_specs`, resolved by name at use time."""
    from task_graph.ids import TaskId

    supplied = Views(**{k: Stub("collect") if k == "agent_specs" else Stub() for k in FIVE})
    r = build_registry(registries=supplied)

    assert r.get("agent_mgr").instantiate("collect", TaskId.new()).config == {}


def test_bridging_twice_is_harmless():
    """`demo` has the same bridge in `build.py` and will delete it when ready.
    Until then both run, so this pins that the second is a no-op rather than a
    duplicate-registration raise — measured, because that is what makes the
    handover safe to do in either order."""
    supplied = Views(**{k: Stub("collect") if k == "agent_specs" else Stub() for k in FIVE})
    r = build_registry(registries=supplied)

    for name in r.get("agent_specs").names():  # what cli/build.py does
        r.get("agent_mgr").register(name)
    assert sorted(r.get("agent_mgr").specs()) == ["collect", SUBGRAPH_AGENT_SPEC]


def test_the_public_id_base_is_exported_and_is_what_monitor_subclasses():
    """`interfaces.md` §4.7 ruled it public. A leading underscore says "named in
    one package" (§1.2) and it was named in two — `monitor.record.EventId`
    subclasses it, which is right, because a fourth id class here would make
    this package carry a monitor concept."""
    import task_graph
    import task_graph.ids
    from monitor import EventId

    assert "Id" in task_graph.__all__
    assert issubclass(EventId, task_graph.Id)
    # The transitional `_Id = Id` alias is gone: `monitor` moved to `Id`, and an
    # alias kept past the migration it existed for is the shape this package has
    # now reported four times.
    assert not hasattr(task_graph.ids, "_Id")


# ------------------------------------------- set_task at dispatch, demo F-D8


class WatchingMonitor:
    """The inbound half of `Monitor`, which is `set_task` and `report`."""

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self.watched: list = []

    def set_task(self, task_id) -> None:
        self.watched.append(task_id)


def watched_system(store=None, **kw):
    from task_graph.store import MemoryStoreMgr

    watcher = WatchingMonitor()
    r = rebuild(store or MemoryStoreMgr(), monitors=[watcher], **kw)
    return r, watcher


def test_a_dispatched_task_is_given_to_its_monitor(store):
    """`demo` F-D8: nobody called `set_task`, so no task was watched and every
    planned advance raised `ScopeViolation`."""
    r, watcher = watched_system(store)
    task = make_task()
    r.get("scheduler").submit(task)

    assert watcher.watched == [task.id]


def test_a_queued_task_is_not_watched_until_it_runs(store):
    """At dispatch, not at birth. A `WAITING_HANDOFF` task has no planned
    advance to make and no agent to poll for a stall."""
    r, watcher = watched_system(store)
    blocked = make_task(inputs=new_handoffs(1))
    r.get("scheduler").submit(blocked)

    assert watcher.watched == []


def test_subtasks_born_inside_unfold_are_watched(registry, store):
    """The case `demo` could not reach: they submit the root only, and `unfold`
    instantiates the subtasks inside `enter_phase(RUNNING)`."""
    from .conftest import closure_doc, with_closures

    r, watcher = watched_system(store)
    with_closures(
        r,
        {
            "pipeline": closure_doc(
                "pipeline", outputs=["report"], subgraph=[{"closure": "a"}, {"closure": "b"}]
            ),
            "a": closure_doc("a", outputs=["mid"]),
            "b": closure_doc("b", inputs=["mid"], outputs=["report"]),
        },
    )
    (report,) = new_handoffs(1)
    root = make_task(outputs=[report], kinds={report: "report"}, closure="pipeline")
    r.get("scheduler").submit(root)
    r.get("runner").advance(r, root.id)  # -> RUNNING, which unfolds and submits

    children = r.get("task_mgr").children(root.id)
    assert root.id in watcher.watched
    assert any(c.id in watcher.watched for c in children), "the start subtask ran unwatched"
    assert registry is not None  # the fixture is unused; kept for parity with the file


def test_a_restored_task_is_watched_again(store):
    """The measurement that ruled out the birth sites. `TaskMgr.resume_system`
    rebuilds the collection without going through `add` — it cannot, `add` is
    the new-task path and raises on a duplicate id — so watching at birth is
    silently lost across a restart. Dispatch covers it because resume
    re-dispatches."""
    from task_graph.registry import resume_all

    first, _ = watched_system(store)
    task = make_task()
    first.get("scheduler").submit(task)

    fresh, watcher = watched_system(store)
    assert watcher.watched == []
    resume_all(fresh)
    assert watcher.watched == [task.id]


def test_an_unresolvable_monitor_name_fails_the_launch_rather_than_stalling(store, caplog):
    """A name that will not resolve is "a task that never advances a phase"
    (`interfaces.md` §2.1). It raises into the launch guard, which releases the
    lease and parks the task in `FAILED` — loud at dispatch beats a silent stall
    for ever."""
    import logging

    from task_graph.models import TaskStatus

    r, _ = watched_system(store)
    task = make_task(monitor_spec="nobody", resources={"gpu": 2})
    # `caplog`, not `logging.getLogger(...).setLevel` — the launch failure is
    # logged at ERROR and expected here, but silencing the logger directly
    # leaks into every later test in the session. Measured: it turned three
    # warning-assertions elsewhere red.
    caplog.set_level(logging.CRITICAL, logger="task_graph.scheduler")
    r.get("scheduler").submit(task)

    assert r.get("task_mgr").get(task.id).status is TaskStatus.FAILED
    assert r.get("resource:gpu").available == 8, "and the lease did not leak"


# ------------------------------------- the whole-catalogue phase's two kinds


class OnePackage:
    """A `TaskPackage` that discovers nothing — enough to reach the passes."""

    root = "."

    def discover(self):
        return []

    def config_for(self, source):
        return {}


def test_a_non_fatal_problem_is_reported_rather_than_dropped(caplog, monkeypatch):
    """`Problem.fatal=False` has one producer — `closure`'s check 3, for a kind
    admitted under the escape hatch — and reporting it is what `closure`
    criterion 6 *is*. This root computed those problems and filtered them away,
    so the admission reached nobody even after `handoff` renamed the accessor
    and the `getattr` default came out. Three fixes on one path and the value
    still went nowhere at the end of it.
    """
    import logging

    import task_graph.bootstrap as boot
    from spec_loader.protocols import LoadReport, Problem

    noted = Problem(
        origin="pkg/handoffs/loose.jsonnet",
        path="$.validators",
        keyword="no_validator",
        message="admitted under the escape hatch",
        fatal=False,
    )
    monkeypatch.setattr(
        boot,
        "_optional",
        lambda wanted: {"load_package": lambda pkg, views: LoadReport((), (noted,))},
    )

    caplog.set_level(logging.WARNING, logger="task_graph.bootstrap")
    build_registry(registries=Views(**{k: Stub() for k in FIVE}), packages=[OnePackage()])

    assert "escape hatch" in caplog.text
    assert "admitted with reservations" in caplog.text


def test_a_fatal_problem_still_raises_and_says_which(monkeypatch):
    """The other half, so the repair above cannot have softened the gate."""
    import task_graph.bootstrap as boot
    from spec_loader import SpecInvalid
    from spec_loader.protocols import LoadReport, Problem

    bad = Problem(origin="pkg/x.jsonnet", path="$", keyword="required", message="boom")
    monkeypatch.setattr(
        boot,
        "_optional",
        lambda wanted: {"load_package": lambda pkg, views: LoadReport((), (bad,))},
    )

    with pytest.raises(SpecInvalid, match="boom"):
        build_registry(registries=Views(**{k: Stub() for k in FIVE}), packages=[OnePackage()])
