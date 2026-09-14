"""`cli/build.py`, and the graph it makes. Criteria 2, 3, 7, 12.

`docs/interfaces.md` §5.3: turning a closure into the root `Task` is owned by
this package knowingly, so it is tested here knowingly too — and when it moves
to the whole-system CLI these tests move with it unchanged.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from cli import build, package
from task_graph import HandoffId, TaskStatus
from tests.cli.conftest import by_closure

# --------------------------------------------------------------------------- #
# Criterion 2 — a root with `parent = None`, a subtask with a non-`None` parent


def test_root_and_subtasks(graph: list[Any]) -> None:
    root = by_closure(graph, "main")
    assert root.parent is None
    assert root.closure == "main"

    subtasks = [task for task in graph if task is not root]
    assert len(subtasks) == 3
    assert {task.closure for task in subtasks} == {"produce", "describe", "consume"}
    # Not "some subtask has a parent": every one of them does, and the parent is
    # this root rather than merely non-None.
    assert all(task.parent == root.id for task in subtasks)

    # The marks are their own test — `test_the_marks_say_what_they_come_from`
    # below — because *where each one comes from* is the part that is easy to
    # assert falsely.
    assert by_closure(graph, "produce").is_start
    assert by_closure(graph, "consume").is_end
    assert not by_closure(graph, "describe").is_start
    assert not by_closure(graph, "describe").is_end


def test_the_marks_say_what_they_come_from(registry: Any, graph: list[Any]) -> None:
    """**The two marks have different provenance, and a test that says otherwise
    is green while its claim is false.**

    This is the correction of a specific defect. The old assertion read

        # `is_start` / `is_end` come from the closure's declared marks, and the
        # demo declares both explicitly so the defaulting rule is not what is
        # under test.
        assert by_closure(graph, "produce").is_start
        assert by_closure(graph, "consume").is_end

    and **deleting both marks from the package leaves it passing**, off the
    positional defaults: absent, the first entry is the start and the last is the
    end. So the comment could stop being true without the test noticing, and the
    conversion to YAML dropped `is_start` for exactly the reason the comment
    claimed it was there.

    The two are asserted separately now, each against its declaration:

    - **`is_start` is not declared, and the demo does not declare it.** Index 0
      is always a root — the listing order is required to be a valid topological
      order, so the first entry can have no predecessor — and a mark that can
      never carry information is one the reference package should not write.
      This asserts the *absence* as well as the effect, so reintroducing the mark
      fails here and has to be argued for.
    - **`is_end` is declared, and this fails if it is deleted.** Its default is
      positional, so appending a fourth entry would move the end silently, and
      `monitor/base.py:663` is where that lands. The declaration is what stops an
      append from being a semantic change nobody wrote.
    """
    entries = registry.get("closures").get("main")["task"]["subgraph"]
    assert [e["closure"] for e in entries] == ["produce", "describe", "consume"]

    assert "is_start" not in entries[0], (
        "the first entry declares is_start. Index 0 is always a root, so the "
        "mark cannot carry information; drop it rather than writing it."
    )
    assert by_closure(graph, "produce").is_start, "the positional default did not apply"

    assert entries[-1].get("is_end") is True, (
        "the last entry no longer declares is_end. Deleting it leaves every "
        "assertion in this file green — the positional default puts the end on "
        "the last entry — and appending a fourth step then moves it silently."
    )
    assert by_closure(graph, "consume").is_end

    # `froms` is mandatory on every entry, and `[]` is an answer rather than an
    # omission. The schema separates the two, so this pins the demo's side of it:
    # a chain, written out, with the head saying it has no predecessor.
    assert [e["froms"] for e in entries] == [[], ["produce"], ["describe"]]


def test_a_flat_graph_would_not_have_proved_this(graph: list[Any]) -> None:
    """Spec §2 item 1: *a flat graph exercises none of the nesting.*

    Stated as a test because the shape is the requirement — the demo is not
    allowed to satisfy criterion 2 by declaring a second root.
    """
    roots = [task for task in graph if task.parent is None]
    assert len(roots) == 1
    assert len(graph) > 1


# --------------------------------------------------------------------------- #
# Criterion 3 — one input phase empty and runs nothing, another populated


def _phase_validators(registry: Any, kinds: list[str]) -> list[str]:
    specs = registry.get("handoff_specs")
    return [name for kind in kinds for name in specs.validators_for(kind)]


def test_produce_input_phase_empty(registry: Any, graph: list[Any]) -> None:
    """Empty is the normal case, and the demo shows it being normal.

    `produce` has no inputs at all, so there is nothing for its input phase to
    select — and `PhaseOutcome.fold(kind)` with nothing folded is `empty`, which
    is its own outcome and **not a pass**.
    """
    produce = by_closure(graph, "produce")
    assert produce.inputs == []
    assert _phase_validators(registry, [produce.kinds[h] for h in produce.inputs]) == []


def test_describe_input_phase_runs(registry: Any, graph: list[Any]) -> None:
    """And the same validator runs in the other phase, which is the point.

    `check_facts` is `produce`'s **output** phase and `describe`'s **input**
    phase. A phase is a position, not a kind of validator, and that is the
    cheapest possible demonstration of it.
    """
    describe = by_closure(graph, "describe")
    produce = by_closure(graph, "produce")
    incoming = [describe.kinds[h] for h in describe.inputs]
    outgoing = [produce.kinds[h] for h in produce.outputs]

    assert incoming == ["facts"]
    assert _phase_validators(registry, incoming) == ["check_facts"]
    assert _phase_validators(registry, outgoing) == ["check_facts"]


def test_the_same_handoff_slot_is_on_both_sides(graph: list[Any]) -> None:
    """`produce`'s output and `describe`'s input are one `HandoffId`.

    Identity, not equality of kind: two slots of the same kind would satisfy a
    kind comparison and would be a graph in which nothing is handed over.
    """
    assert by_closure(graph, "produce").outputs == by_closure(graph, "describe").inputs
    assert by_closure(graph, "describe").outputs == by_closure(graph, "consume").inputs


# --------------------------------------------------------------------------- #
# Criterion 7 — one SDK node, one program node, handoff state the same in kind


def test_one_node_is_a_program_and_one_is_an_sdk_agent(registry: Any, graph: list[Any]) -> None:
    specs = registry.get("agent_specs")
    assert specs.spec(by_closure(graph, "produce").agent_spec).kind.value == "program"
    assert specs.spec(by_closure(graph, "describe").agent_spec).kind.value == "ai"
    # And the AI one names a real backend rather than a fake: criterion 6's
    # second half is structural here, not a run-time observation.
    backends = [decl.key for decl in specs.spec("describe").backends]
    assert backends == ["claude_code_sdk"]


def test_program_and_sdk_handoff_state_identical(registry: Any, submitted: Any) -> None:
    """The half criterion 7 is really about.

    `agent` design §9 makes this nearly a tautology at the *type* level, because
    the runner holds level 1 only and a `ProgramExecutor` has no level 2. The
    criterion is written about **handoffs**, so it is tested where a regression
    would actually show: two nodes, two executors, and records that differ only
    in which kind they carry.

    Driven through `FakeRunner`, which is the only thing in the suite that
    writes handoff state — a model call here would need a credential.
    """
    handoff_mgr, task_mgr = registry.get("handoff_mgr"), registry.get("task_mgr")
    _advance_to_running(registry, submitted)  # the unfold
    tasks = task_mgr.all()
    produce, describe = by_closure(tasks, "produce"), by_closure(tasks, "describe")

    _run_leaf(registry, produce, valid=True)
    _run_leaf(registry, describe, valid=False)

    facts = handoff_mgr.get(produce.outputs[0])
    summary = handoff_mgr.get(describe.outputs[0])

    # Same shape, same version numbering, same binding record. Different kind,
    # different verdict — and nothing else.
    assert facts.type == "facts" and summary.type == "summary"
    assert [v.version for v in facts.versions] == [v.version for v in summary.versions] == [0]
    assert facts.latest.producer_task_id == produce.id
    assert summary.latest.producer_task_id == describe.id
    assert facts.latest.producer_agent_id is not None
    assert summary.latest.producer_agent_id is not None
    assert set(facts.model_dump()) == set(summary.model_dump())
    assert len(produce.history) == len(describe.history) == 1


def _advance_to_running(registry: Any, task: Any) -> None:
    """A non-leaf's main phase is its unfold, so it never reaches an executor."""
    registry.get("runner").advance(registry, task.id)  # INPUT_VALIDATING -> RUNNING


def _run_leaf(registry: Any, task: Any, *, valid: bool) -> None:
    runner = registry.get("runner")
    registry.get("scheduler").try_dispatch()
    assert task.status is TaskStatus.INPUT_VALIDATING, task.status
    runner.advance(registry, task.id)  # -> RUNNING
    runner.produce(registry, task.id, valid=valid)
    runner.advance(registry, task.id)  # -> OUTPUT_VALIDATING
    runner.finish(task.id, TaskStatus.SUCCEEDED)


# --------------------------------------------------------------------------- #
# `wire`, and the warning it exists to silence


def test_depends_on_is_derived_and_the_log_is_silent(
    registry: Any, graph: list[Any], caplog: Any
) -> None:
    """§6.2 and `materials/08-demo.md` §5.

    `scheduler._warn_depends_on` logs on every dispatch whose `depends_on` omits
    the producer of one of its inputs. It is a warning by design — *rejecting
    would make declaration order matter* — but `depends_on` is `list[TaskId]`,
    runtime ids, so **no spec document can carry one**. Without `wire` the
    reference example of this system prints a warning on every run, which is
    exactly the accepted noise a demo exists to prevent.
    """
    task_mgr, scheduler = registry.get("task_mgr"), registry.get("scheduler")
    assert by_closure(graph, "describe").depends_on == [by_closure(graph, "produce").id]
    assert by_closure(graph, "consume").depends_on == [by_closure(graph, "describe").id]

    with caplog.at_level("WARNING", logger="task_graph.scheduler"):
        for task in graph:
            task_mgr.remove(task.id)
        for task in graph:
            scheduler.submit(task)
    assert [r for r in caplog.records if "depends_on omits" in r.getMessage()] == []


def test_wire_is_idempotent(graph: list[Any]) -> None:
    """`Task.unfold` already derives `depends_on` for a subgraph it instantiates
    in one pass, so `wire` runs over work already done — every time, on every
    graph this builder makes. If it were not idempotent the demo would grow a
    duplicate edge per call and nothing would notice."""
    before = {task.id: list(task.depends_on) for task in graph}
    build.wire(graph)
    build.wire(graph)
    assert {task.id: list(task.depends_on) for task in graph} == before


def test_handoff_ids_covers_every_declared_kind(registry: Any) -> None:
    """A closure may declare a kind its subgraph uses internally, and `main`
    does — both of them. The map is what a caller that needs to *name* one
    before the unfold uses, so it covers the closure's declaration and not only
    the root task's own inputs and outputs."""
    ids = build.handoff_ids("main", registry)
    assert sorted(ids) == ["facts", "summary"]
    assert all(isinstance(hid, HandoffId) for hid in ids.values())
    assert len(set(ids.values())) == 2  # fresh, and distinct


def test_root_task_names_its_closure_and_carries_its_permissions(registry: Any) -> None:
    """The root's permissions are inherited by every subtask — `_instantiate`
    passes `self.permissions` down — so the root's set has to cover the whole
    subgraph, and `WRITE` covering `READ` is what makes two grants enough."""
    root = build.root_task("main", registry)
    assert root.closure == "main"
    assert {grant.kind for grant in root.permissions.grants} == {"facts", "summary"}
    assert root.permissions.covers("facts", _read())
    assert root.permissions.covers("summary", _read())


def _read() -> Any:
    from task_graph import Access

    return Access.READ


def test_the_root_of_a_non_leaf_runs_on_the_built_in_and_a_leaf_does_not(registry: Any) -> None:
    """**`root_task` builds the one `Task` that `unfold` does not, so the
    non-leaf rule has to hold on this path too.**

    `cli/build.py` read `doc["agent"]` unconditionally and raised `KeyError` on
    the demo's own root the moment `main` stopped declaring one. It now calls
    `task_graph.models.agent_spec_for`, which is the same function `unfold` uses
    — so this is the test that the *demo's* entry point is on that function and
    not on a copy of its rule.

    **The second half is the one that matters and is easy to lose.** The
    fallback is conditional on being a **non-leaf**, not on the key being
    absent: a leaf with no agent is a load failure (`closure/check.py` check 4
    and the schema's `else`), and filling it with the built-in would let a
    broken catalogue dispatch under a name that describes something it is not.
    A `doc.get("agent") or SUBGRAPH_AGENT_SPEC` reads identically on every
    document the loader admits and differs on exactly that one, so the
    difference is invisible without a document the loader would have rejected —
    which is what is built here by hand.
    """
    from task_graph.models import SUBGRAPH_AGENT_SPEC, agent_spec_for

    # The demo's own root, through the real registry.
    assert build.root_task("main", registry).agent_spec == SUBGRAPH_AGENT_SPEC
    assert "agent" not in registry.get("closures").get("main")
    # And every leaf still runs on what it declares.
    for leaf in ("produce", "describe", "consume"):
        declared = registry.get("closures").get(leaf)["agent"]
        assert build.root_task(leaf, registry).agent_spec == declared

    # A leaf with no agent, which no loaded package can contain. `KeyError`, not
    # a built-in: this is the assertion that fails if the branch is ever
    # simplified to `or SUBGRAPH_AGENT_SPEC`.
    with pytest.raises(KeyError):
        agent_spec_for({"name": "orphan"}, {"goal": "x"})
    # A declared agent wins even on a non-leaf — the author said so and this
    # must not silently replace it.
    assert agent_spec_for({"agent": "mine"}, {"subgraph": [{"closure": "a"}]}) == "mine"


def test_root_task_refuses_an_undeclared_closure(registry: Any) -> None:
    with pytest.raises(KeyError, match="not a declared closure"):
        build.root_task("no-such-closure", registry)


# --------------------------------------------------------------------------- #
# Criterion 12 — interrupt and restart continues from persisted state


_CHILD = textwrap.dedent(
    """
    import json, os, sys
    from pathlib import Path
    from task_graph import JsonFileStoreMgr, TaskStatus, build_registry
    from cli import build, package

    root, store, kill_at = sys.argv[1], sys.argv[2], int(sys.argv[3])
    r = build_registry(
        store=JsonFileStoreMgr(store),
        packages=[package.task_package(Path(root))],
        handoff_root=store + "/h",
        knowledge_root=store + "/k",
    )
    task = build.root_task("main", r)
    r.get("task_mgr").add(task)
    tasks = [task, *task.unfold()]
    for t in tasks[1:]:
        r.get("task_mgr").add(t)
    build.wire(tasks)
    for t in tasks:
        r.get("task_mgr").remove(t.id)
    for t in tasks:
        r.get("scheduler").submit(t)

    runner = r.get("runner")
    runner.advance(r, task.id)                     # main -> RUNNING, unfolds
    produce = [t for t in r.get("task_mgr").all() if t.closure == "produce"][0]
    r.get("scheduler").try_dispatch()
    runner.advance(r, produce.id)                  # produce -> RUNNING
    if kill_at == 1:
        # os._exit: no atexit, no flush. The store's per-record
        # `tmp.replace(path)` is the only thing standing here.
        os._exit(9)
    runner.produce(r, produce.id, valid=True)
    print(json.dumps({"produce": str(produce.id)}))
    """
).strip()


def _child(tmp_path: Path, package_root: Path, kill_at: int) -> subprocess.CompletedProcess:
    script = tmp_path / "child.py"
    script.write_text(_CHILD)
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    return subprocess.run(  # noqa: S603 — our own interpreter, our own script
        [sys.executable, str(script), str(package_root), str(tmp_path / "store"), str(kill_at)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


def test_resume_continues_from_disk(tmp_path: Path, package_root: Path) -> None:
    """Two processes, `os._exit` mid-run, and the graph continues.

    Stronger than `tests/task_graph/test_recovery.py`, which restarts with fresh
    managers over the **same live store object** in one process. Here the writer
    is killed with `os._exit(9)` — no `atexit`, no flush — and a *new*
    interpreter rebuilds from the files.

    What makes "continued from persisted state" observable rather than asserted
    is the attempt numbering: attempt 0 recorded `SUSPENDED`, attempt 1 open.
    """
    killed = _child(tmp_path, package_root, kill_at=1)
    assert killed.returncode == 9, killed.stderr

    from task_graph import JsonFileStoreMgr, build_registry

    store = JsonFileStoreMgr(tmp_path / "store")
    revived = build_registry(
        store=store,
        # Reloaded on a resume, and that is not optional: the spec registries
        # come back empty, and `agent_mgr` with them, so every dispatch would
        # raise `unknown agent spec`. F-D2 in `demo/README.md`.
        packages=[package.task_package(package_root)],
        handoff_root=str(tmp_path / "store" / "h"),
        knowledge_root=str(tmp_path / "store" / "k"),
    )
    from task_graph import resume_all

    resume_all(revived)
    records = list(revived.get("task_mgr").all())
    tasks = {task.closure: task for task in records}

    # **Counted, not collapsed.** `{task.closure: task}` and `set(...)` over its
    # keys both fold duplicates away, so seven records under four names would
    # satisfy the name assertion — and `--resume` really did produce seven for a
    # while (`task_graph` `cc23f98`: a non-leaf's main phase *is* the unfold, and
    # `has_subgraph()` asked the declaration rather than the graph, so resuming
    # rebuilt the whole subgraph beside the first).
    #
    # **This assertion would not have caught that, and the limit is worth more
    # than the guard.** Measured: `resume_all` alone yields four records even on
    # the tree that duplicated — the second subgraph appeared when the reloaded
    # root was *dispatched*, which this test never does. It stops at reload, so
    # it shows the state came back, not that the run continues. Criterion 12
    # says *continues*, and the part that continues is not exercised here;
    # driving it needs a real run, which the suite must not require
    # (spec §14.3). `scratch/impl-2026-08/demo/p6_where_does_resume_duplicate.py`
    # is that drive, and confirms four records and two slots after `run` plus
    # `run --resume` on the current tree.
    assert len(records) == 4, sorted(t.closure for t in records)
    assert set(tasks) == {"main", "produce", "describe", "consume"}
    produce = tasks["produce"]
    # The evidence, and it is what `materials/08-demo.md` §5 measured:
    # `attempts=[(0, 'SUSPENDED'), (1, None)]`. The interrupted attempt is
    # demoted rather than lost, and a **second** one is open — which is the
    # observable form of "continued from persisted state", and also why the
    # README tells a reviewer to interrupt during `produce`: a resume re-runs
    # the interrupted attempt, so interrupting during `describe` pays for a
    # second model call.
    assert [e.outcome for e in produce.history] == [TaskStatus.SUSPENDED, None]
    assert tasks["consume"].status is TaskStatus.WAITING_HANDOFF
    assert tasks["consume"].history == []


def test_the_named_work_root_outranks_the_xdg_base(tmp_path: Path, monkeypatch: Any) -> None:
    """`INFERA_AGENT_SYSTEM_WORKROOT` names the run root; `XDG_STATE_HOME` a base.

    The two are not two spellings of one setting and the test says which is
    which: with both set the specific name wins **whole**, and the loser's value
    does not appear in the answer even as a prefix. A first version appended
    `agent-sys-demo` to it too, which reads the same in a passing test and puts
    the run somewhere the operator did not name.
    """
    from cli.environment import WORKROOT_ENV_VAR, default_root

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv(WORKROOT_ENV_VAR, str(tmp_path / "work"))
    assert default_root() == tmp_path / "work"

    monkeypatch.delenv(WORKROOT_ENV_VAR)
    assert default_root() == tmp_path / "xdg" / "agent-sys-demo"


@pytest.mark.parametrize("value", ["", "   ", "relative/path", "./runs"])
def test_an_unusable_work_root_reads_as_unset(
    value: str, tmp_path: Path, monkeypatch: Any
) -> None:
    """Empty or relative falls back rather than being obeyed.

    A relative run root resolves against whatever `cwd` a body inherited, and
    `<run root>` is the one path that may not depend on that: the bodies reach
    their own staged package by absolute path and
    `remote.sh:require_visible_on_node` asserts the compute node resolves the
    same string. Obeying `./runs` would put the run somewhere that reads
    correctly here and is not findable from the other side.

    Its non-vacuity control is the test above, which must still return the
    named path — otherwise this one would pass against a function that had
    stopped reading the variable at all.
    """
    from cli.environment import WORKROOT_ENV_VAR, default_root

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv(WORKROOT_ENV_VAR, value)
    assert default_root() == tmp_path / "xdg" / "agent-sys-demo"


def test_an_explicit_root_outranks_the_environment(tmp_path: Path, monkeypatch: Any) -> None:
    """`--demo-root` wins over both. The flag is the most specific statement."""
    from cli.environment import WORKROOT_ENV_VAR, layout_for

    monkeypatch.setenv(WORKROOT_ENV_VAR, str(tmp_path / "work"))
    assert layout_for(tmp_path / "explicit").root == tmp_path / "explicit"
    # And with no argument the variable is what answers, so the assertion above
    # is about precedence rather than about `layout_for` ignoring the environment.
    assert layout_for().root == tmp_path / "work"


def test_two_runs_under_one_root_get_separate_stores(tmp_path: Path) -> None:
    """**Criterion 13's real case: two runs, one root, back to back.**

    `test_two_runs_do_not_collide` below gives each run its own **root**, so it
    proves two roots do not collide — which nothing threatens. The collision is
    between two runs under *one* root, which is what a reviewer creates by
    typing the command twice, and it was live and silent: the run id was
    `%Y%m%dT%H%M%S`, `create()` uses `mkdir(exist_ok=True)`, so two runs inside
    one second shared a directory and the second adopted the first's store.

    No sleeping and no clock control — the two layouts are built back to back
    on purpose, because that *is* the failing case. Before the fix this failed
    on the first assertion every time it happened to straddle no second
    boundary, and passed otherwise: a flaky guard is the shape that gets
    re-run rather than read.
    """
    from cli.environment import layout_for

    first = layout_for(tmp_path).create()
    second = layout_for(tmp_path).create()

    assert first.run != second.run, "two runs under one root share a directory"
    (first.store / "state.json").write_text("first run")
    assert not (second.store / "state.json").exists(), (
        "the second run reads the first run's store; criterion 12's resume "
        "would continue the wrong graph and criterion 13's second run is not a "
        "second run at all"
    )
    # `latest` follows the newest, which is what `--resume` needs.
    assert (tmp_path / "latest").resolve() == second.run.resolve()


def test_two_runs_do_not_collide(tmp_path: Path, package_root: Path) -> None:
    """Criterion 13, from this side: the same graph twice over one demo root.

    Two runs produce two sets of `TaskId`s and `HandoffId`s, so nothing in the
    graph can collide; what could collide is the store, and each run gets its
    own directory under `runs/`.

    **Two separate roots, so this does not test what its name suggests** — see
    `test_two_runs_under_one_root_get_separate_stores` above, which is the case
    that was broken. Kept because it does prove the *graph* half: two runs mint
    disjoint `TaskId`s.
    """
    first = _child(tmp_path / "a", package_root, kill_at=0)
    (tmp_path / "b").mkdir(parents=True, exist_ok=True)
    second = _child(tmp_path / "b", package_root, kill_at=0)
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert json.loads(first.stdout)["produce"] != json.loads(second.stdout)["produce"]


@pytest.fixture(autouse=True)
def _child_dirs(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir(parents=True, exist_ok=True)
