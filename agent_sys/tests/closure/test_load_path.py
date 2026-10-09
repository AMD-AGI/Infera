"""The real load path, end to end — the test that makes `task_specs` stay fixed.

`main`'s framing, and it is the right one:

> Not "`check_graph` returns `[]` on a good catalogue" — that is what passed
> while inert. A test that a violating catalogue, loaded through the real
> `load_package` path, produces a problem.

Nothing wrote `task_specs` until `2215158`. `check_graph` walked an empty
catalogue, so `task_graph` criteria 50 and 53 returned `[]` on inputs that
violate both — green, and inert. The reason it went unnoticed is sharper than
the defect: **the fixture supplied what production did not.** `conftest.py` used
to do the `add` that the real admission path never did, so every unit test saw a
populated registry and the assembled system did not.

So this test uses no fixture registries. Real YAML on disk, a real
`YamlPackage`, the real `load_package`, then the real pass, then `task_graph`'s
real `check_graph`. A test may import all three; `tests/` is not under
`docs/interfaces.md` §4's rule.

**The fixture is written with `ruamel.yaml`, deliberately, and not with
`json.dumps` as it was.** JSON is a subset of YAML, so the old serialiser would
still have produced loadable files — and every scalar would have arrived quoted,
which is the one shape that cannot expose a parser disagreement. Reading and
writing through the same parser the package uses keeps this test's fixture out
of the YAML 1.1 / 1.2 question entirely rather than dodging it by accident. The
question itself belongs to `tests/spec_loader`, which measures it.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path

import pytest
from ruamel.yaml import YAML

from closure.check import check_closures
from closure.registry import ClosureRegistry
from closure.task_registry import TaskSpecRegistry
from spec_loader import YamlPackage, load_package
from task_graph.graph import check_graph

from .conftest import NO_ESCAPE_HATCH, DictRegistry, Regs

AGENT = {"name": "profiler", "kind": "program", "version": "1"}

LEAF = {
    "name": "leaf",
    "description": "does the work",
    "agent": "profiler",
    "handoffs": [],
    "validators": [],
    "task": {
        "goal": "do the work",
        "version": "1",
        "inputs": [],
        "outputs": [],
        "resources": {"gpu": 1},
    },
}

#: The violation. A non-leaf declaring `resources` is `task_graph` criterion 53:
#: a parent holding a lease while its subtasks queue for the same pool is
#: hold-and-wait. Load-time detectable, and detected by nothing until the
#: catalogue had contents.
PARENT = {
    "name": "parent",
    "description": "expands into a subgraph",
    "agent": "profiler",
    "handoffs": [],
    "validators": [],
    "task": {
        "goal": "run the subgraph",
        "version": "1",
        "inputs": [],
        "outputs": [],
        "resources": {"gpu": 1},
        "subgraph": [{"closure": "leaf", "is_start": True, "is_end": True, "froms": []}],
    },
}


def _kind(name: str) -> dict:
    return {
        "name": name,
        "description": f"the {name} artefact",
        "content_type": "structured_text",
        "scope": "fixed.required",
        "version": "1",
    }


def _closure(name: str, *, inputs=(), outputs=(), subgraph=None, resources=None) -> dict:
    """A closure that passes every one of the seven checks unless told otherwise.

    Grants cover exactly the declared inputs and outputs, so a test that wants a
    coverage failure has to say what it removed.
    """
    task: dict = {
        "goal": f"{name}: a step",
        "version": "1",
        "inputs": list(inputs),
        "outputs": list(outputs),
        "permissions": {
            "grants": [{"path": f"h/{k}", "access": "read", "kind": k} for k in inputs]
            + [{"path": f"h/{k}", "access": "write", "kind": k} for k in outputs]
        },
    }
    if subgraph is not None:
        task["subgraph"] = list(subgraph)
    if resources is not None:
        task["resources"] = dict(resources)
    return {
        "name": name,
        "description": f"the {name} step",
        "agent": "profiler",
        "handoffs": sorted({*inputs, *outputs}),
        "validators": [],
        "task": task,
    }


_yaml = YAML(typ="rt")


def _write(path: Path, module: str, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO()
    _yaml.dump({"module": module, **doc}, stream)
    path.write_text(stream.getvalue())


def _package(root: Path, closures: list[dict], kinds: Sequence[str] = ()) -> YamlPackage:
    """A real task package: `assets/`, a `main.yaml`, one object per file.

    Three things about the shape, each of which is the format rather than a
    choice this test made:

    - **A closure is written `module: task`.** Users write `task` and the closure
      is what comes out (`closure` spec §2); `closure` is not one of the four
      modules a package may declare.
    - **The outermost graph goes in `main.yaml`**, which is mandatory (main spec
      §4.3). The entry is the closure that declares a subgraph — every caller
      here has exactly one — and it also lands where the definition-order check
      wants it, since `_scan` sorts `main.yaml` last and the objects it
      references are therefore already defined.
    - **Bodies are found by convention, not bound by hand.** An explicit
      `body.readme` is legal and *warns*, and these tests assert on an empty
      problem list, so a bound path would make every one of them fail for a
      reason that has nothing to do with what they test. `assets/<name>.md` and
      `assets/<name>.entry.sh` are what the old `body` keys became.
    """
    (root / "assets").mkdir(parents=True, exist_ok=True)
    _write(root / "agents" / "profiler.yaml", "agent", AGENT)
    (root / "assets" / "profiler.md").write_text("what profiler is")
    for name in kinds:
        _write(root / "handoffs" / f"{name}.yaml", "handoff", _kind(name))
        (root / "assets" / f"{name}.md").write_text(f"what {name} is")

    entry = next((c for c in closures if "subgraph" in c["task"]), closures[-1])
    for doc in closures:
        name = doc["name"]
        (root / "assets" / f"{name}.md").write_text(f"what {name} is")
        if "subgraph" not in doc["task"]:
            # A leaf does the work itself, so it has an entry; a non-leaf's work
            # *is* its subgraph and the two are mutually exclusive (check 7).
            (root / "assets" / f"{name}.entry.sh").write_text("#!/bin/sh\n")
        where = root / "main.yaml" if doc is entry else root / "tasks" / f"{name}.yaml"
        _write(where, "task", doc)
    return YamlPackage(root=root, variables={})


@pytest.fixture
def regs() -> Regs:
    """Five real registries, and no fixture doing production's job.

    `handoff_specs`, `validator_specs` and `agent_specs` are the Protocol stub —
    this package may not import the modules that own them — but `closures` and
    `task_specs` are the real classes, because they are the two under test.
    """
    return Regs(
        handoff_specs=DictRegistry("handoff"),
        validator_specs=DictRegistry("validator"),
        task_specs=TaskSpecRegistry(),
        agent_specs=DictRegistry("agent"),
        closures=ClosureRegistry(),
    )


def test_a_real_load_populates_the_catalogue_check_graph_walks(tmp_path, regs) -> None:
    """The regression guard. Load two closures for real; `check_graph` must see
    both task specs and report the violation."""
    report = load_package(_package(tmp_path, [LEAF, PARENT]), regs)
    assert [p.message for p in report.problems] == []
    assert sorted(report.admitted) == ["leaf", "parent", "profiler"]

    assert regs.task_specs.names() == [], "load_package does not admit a task spec, and must not"

    assert check_closures(regs, NO_ESCAPE_HATCH) == []
    assert regs.task_specs.names() == ["leaf", "parent"]

    catalogue = {name: regs.task_specs.get(name) for name in regs.task_specs.names()}
    problems = check_graph(catalogue)

    assert problems, (
        "check_graph reported nothing on a catalogue that violates criterion 53 — "
        "which is exactly the silence this test exists to prevent"
    )
    (violation,) = [p for p in problems if "parent" in p.message]
    assert "resources" in violation.message


def test_the_catalogue_holds_inner_task_specs_and_not_closure_documents(tmp_path, regs) -> None:
    """The other half of the same silence.

    Handed a closure document, `subgraph_of` finds no `subgraph`, every task looks
    like a leaf, and both graph checks return `[]` — no crash, no wrong answer.
    `task_graph` shipped with that bug for the same reason: a hand-built fixture
    that happened to be the shape the code expected.
    """
    load_package(_package(tmp_path, [LEAF, PARENT]), regs)
    check_closures(regs, NO_ESCAPE_HATCH)

    stored = regs.task_specs.get("parent")
    assert "task" not in stored, "the inner task spec, not the closure document"
    assert stored["goal"] == "run the subgraph"
    assert stored["subgraph"][0]["closure"] == "leaf"


def test_the_origin_survives_the_whole_path(tmp_path, regs) -> None:
    """A task spec's origin is the file the author wrote, so a problem
    `check_graph` raises names something openable.

    **It no longer asserts a filename, and the reason is a real one rather than
    churn.** It used to say `parent.jsonnet`, one file per object named after the
    object. A package may now put several objects in one file and must put its
    outermost graph in `main.yaml`, so the file an object came from is not
    derivable from its name — asserting one would pin this fixture's layout and
    call it the contract.

    What the test is actually for survives intact, and is the failure
    `closure/check.py`'s `origin_of` comment records: a degraded origin labels a
    `Problem` with the closure's *name* where a path belongs, which is
    indistinguishable from a real origin in a message. So: it opens, and it is
    not the name.
    """
    load_package(_package(tmp_path, [LEAF, PARENT]), regs)
    check_closures(regs, NO_ESCAPE_HATCH)

    origin = regs.task_specs.origin_of("parent")
    assert Path(origin).is_file(), f"{origin!r} does not open"
    assert origin != "parent", "the origin degraded to the spec name"
    assert origin == regs.closures.origin_of("parent")


# --------------------------------------------------------------------------- #
# Criterion 50 — a handoff produced inside a subgraph must not be consumed
# outside it, except through the end entry subtask's outputs.
#
# The shape is `task_graph`'s, handed over measured rather than reasoned, and it
# lives here rather than in `tests/task_graph/` for one reason: the value is that
# it runs through the real `load_package`, and duplicating the on-disk fixture to
# get that would be a second copy of the more expensive half.

#: **Four closures, and the fourth is not padding.** A single-member subgraph's
#: one member is both start and end, so its outputs *are* the declared boundary
#: and nothing is contained. It takes two members for one of them to be non-end,
#: and only then is there an inside to escape from.
CONTAINMENT = [
    _closure(
        "pipeline",
        outputs=["report"],
        subgraph=[
            {"closure": "collect", "is_start": True, "is_end": False, "froms": []},
            {"closure": "summarise", "is_start": False, "is_end": True, "froms": ["collect"]},
        ],
    ),
    _closure("collect", outputs=["trace"]),  # produced INSIDE, not exported
    _closure("summarise", inputs=["trace"], outputs=["report"]),  # the end entry exports
]


def _load(tmp_path: Path, regs: Regs, closures: list[dict]) -> list:
    kinds = sorted({k for c in closures for k in c["handoffs"]})
    report = load_package(_package(tmp_path, closures, kinds), regs)
    assert [p.message for p in report.problems] == []
    assert check_closures(regs, NO_ESCAPE_HATCH) == []
    return check_graph({n: regs.task_specs.get(n) for n in regs.task_specs.names()})


def test_a_handoff_escaping_a_subgraph_is_reported(tmp_path, regs) -> None:
    """`intruder` consumes `trace`, which `collect` produces inside `pipeline`'s
    subgraph and which the end entry does not export."""
    problems = _load(tmp_path, regs, [*CONTAINMENT, _closure("intruder", inputs=["trace"])])

    (violation,) = [p for p in problems if p.keyword == "subgraph_containment"]
    assert violation.origin == "intruder"
    assert "'trace'" in violation.message
    assert "collect" in violation.message and "pipeline" in violation.message


def test_consuming_what_the_end_entry_exports_is_legal(tmp_path, regs) -> None:
    """The negative row, and it is the half that says the check discriminates.

    `report` is produced inside the subgraph too — by `summarise` — but
    `summarise` is the `is_end` subtask, so `report` crosses the boundary through
    the door the boundary exists to provide. Without this row the test would pass
    equally against a check that rejected *every* cross-subgraph read.
    """
    problems = _load(tmp_path, regs, [*CONTAINMENT, _closure("intruder", inputs=["report"])])
    assert problems == []


# --------------------------------------------------------------------------- #
# `docs/interfaces.md` §8.7 — the wrong-on-the-day check.
#
# Distinct from drift, and `handoff` drew the line: drift has a trigger, a
# neighbour changed. **Wrong-on-the-day has no trigger at all** — the stub is as
# wrong on day 300 as on day 1, and the only thing that catches it is running the
# stub's *subject* rather than the stub, once, on purpose.
#
# This package's doubles are `conftest.DictRegistry` (standing in for three real
# registries) and `conftest.Report` (for `handoff.HandoffLoadReport`). Two of the
# three subjects were already driven — `ValidatorSpecRegistry` by
# `test_the_real_users_of_stops_under_reporting`, and all five by the criterion-6
# probe in `scratch/`. These are the ones no *test* had driven.


def test_the_real_handoff_load_report_drives_check_three(tmp_path, regs) -> None:
    """`conftest.Report` stands in for `handoff.HandoffLoadReport`. This runs the
    subject.

    Criterion 6 is *"loads, and reports that it did"*, and check 3 is an
    intersection against `report.without_validator`. A stub whose field were
    named differently, or held anything but bare kind names, would make that
    intersection empty — and an empty intersection is indistinguishable from
    "no escape-hatch admissions", which is the whole failure the criterion exists
    to prevent.
    """
    from handoff.registry import HandoffSpecRegistry

    real = HandoffSpecRegistry(allow_no_validator=True)
    regs.handoff_specs = real
    load_package(_package(tmp_path, [_closure("collect", outputs=["loose"])], ["loose"]), regs)

    report = real.load_report()  # the subject, not the double
    assert report.without_validator == ["loose"], "the kind was admitted with no validator"

    problems = check_closures(regs, report)
    (reported,) = [p for p in problems if p.keyword == "escape_hatch"]
    assert not reported.fatal, "an escape-hatch admission is a report, not a failure"
    assert "'loose'" in reported.message


def test_the_real_agent_registry_answers_check_four(tmp_path, regs) -> None:
    """`DictRegistry` stands in for `AgentSpecRegistry` in every other test here.

    Check 4 asks it two questions — `name in registry` and `names()` for the
    candidate list — and both are answered by the double everywhere else.
    """
    from agent.registry import AgentSpecRegistry

    regs.agent_specs = AgentSpecRegistry()
    load_package(_package(tmp_path, [_closure("collect")], []), regs)

    assert check_closures(regs, NO_ESCAPE_HATCH) == [], "the real registry resolves 'profiler'"

    regs.closures._specs["collect"]["agent"] = "ghost"  # type: ignore[index]
    (problem,) = [p for p in check_closures(regs, NO_ESCAPE_HATCH) if p.path == "$.agent"]
    assert "ghost" in problem.message
    assert "profiler" in problem.message, "the candidate list comes from the real names()"


def test_a_closures_registry_without_origin_of_fails_loudly(regs) -> None:
    """The two `getattr(closures, ...)` guards are gone, and this is why.

    They were dead — no test ever supplied a `closures` lacking either — and not
    harmlessly so: `build_registry` now takes `registries=` from the caller, so a
    `closures` without `origin_of` became reachable, and the fallback labelled
    every `Problem` with the closure's *name* where a file path belongs. That is
    indistinguishable from a real origin in a message, and the file path is the
    one thing `docs/design.md` §6.2 asks these messages to carry.
    """

    class WithoutIt:
        kind = "closure"
        _specs: dict = {}

        def names(self) -> list[str]:
            return []

    regs.closures = WithoutIt()
    with pytest.raises(AttributeError, match="origin_of"):
        check_closures(regs, NO_ESCAPE_HATCH)
