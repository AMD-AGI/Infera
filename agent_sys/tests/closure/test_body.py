"""Check 7 — the body. Not a criterion of its own; spec §2.6 and design §3.6.

`readme.md` is required of every task, leaf or not; `entry.sh` is required iff
the body is programmatic; `entry.sh` and a subgraph are mutually exclusive.
"""

from __future__ import annotations

from closure.check import check_closures
from closure.model import body_of, has_subgraph

from .conftest import NO_ESCAPE_HATCH, Regs, make_closure


def body_problems(problems) -> list:
    return [p for p in problems if p.path.startswith("$.task.body")]


def test_readme_is_required_of_a_leaf(regs: Regs) -> None:
    regs.with_agents("profiler")
    regs.with_closure(make_closure(body={"entry": "run.sh"}))

    (problem,) = body_problems(check_closures(regs, NO_ESCAPE_HATCH))
    assert problem.keyword == "required"
    assert "readme.md" in problem.message


def test_readme_is_required_of_a_non_leaf_too(regs: Regs) -> None:
    """A task nobody can read is a step nobody can review, and that holds for a
    non-leaf: its work is its subgraph, but a reviewer still has to know why the
    step exists."""
    regs.with_agents("profiler")
    regs.with_closure(make_closure(body={}, subgraph=[{"closure": "inner"}]))

    (problem,) = body_problems(check_closures(regs, NO_ESCAPE_HATCH))
    assert problem.keyword == "required"


def test_entry_and_a_subgraph_are_mutually_exclusive(regs: Regs) -> None:
    """A task contains a task graph, or it is a leaf that does the work itself."""
    regs.with_agents("profiler")
    regs.with_closure(
        make_closure(body={"readme": "r.md", "entry": "run.sh"}, subgraph=[{"closure": "inner"}])
    )

    (problem,) = body_problems(check_closures(regs, NO_ESCAPE_HATCH))
    assert problem.keyword == "oneOf"
    assert "mutually exclusive" in problem.message


def test_a_body_with_a_subgraph_and_no_entry_loads(regs: Regs) -> None:
    """The exclusion is between `entry.sh` and the subgraph, not between `body`
    and the subgraph."""
    regs.with_agents("profiler")
    # `inner` is declared, because check 8 now insists a subgraph entry names a
    # closure the catalogue holds.
    regs.with_closure(make_closure("inner"))
    regs.with_closure(make_closure(body={"readme": "r.md"}, subgraph=[{"closure": "inner"}]))
    assert check_closures(regs, NO_ESCAPE_HATCH) == []


def test_a_programmatic_leaf_loads(regs: Regs) -> None:
    regs.with_agents("run_pytest")
    regs.with_closure(
        make_closure(agent="run_pytest", body={"readme": "r.md", "entry": "entry.sh"})
    )
    assert check_closures(regs, NO_ESCAPE_HATCH) == []


def test_the_accessors_read_a_missing_body_as_empty() -> None:
    """A checker must run over a document that never reached the schema, because
    problems are collected rather than raised."""
    assert body_of({}) == {}
    assert not has_subgraph({})
    assert not has_subgraph({"subgraph": []})
    assert has_subgraph({"subgraph": [{"closure": "inner"}]})


# --------------------------------------------------------------------------- #
# Check 8 — a subgraph entry names a declared closure. Asked for by `task_graph`.


def target_problems(problems) -> list:
    return [p for p in problems if p.path.startswith("$.task.subgraph")]


def test_a_subgraph_entry_naming_an_undeclared_closure_is_rejected(regs: Regs) -> None:
    """`Task.unfold` raises on the same fault, hours into a run for a subgraph
    nested three deep. This moves the declared case to load, where the author
    fixes it alongside every other reason the graph is not admissible."""
    regs.with_agents("profiler")
    regs.with_closure(make_closure("inner"))
    regs.with_closure(make_closure("outer", subgraph=[{"closure": "innr"}]))

    (problem,) = target_problems(check_closures(regs, NO_ESCAPE_HATCH))
    assert problem.keyword == "resolves"
    assert "'innr'" in problem.message
    assert "known closures: inner, outer" in problem.message
    assert "hint: 'inner' is close." in problem.message, (
        "the catalogue is enumerated, which is the difference between 'no such "
        "closure' and 'you wrote collect_trace and the catalogue has collect_traces'"
    )


def test_a_declared_subgraph_target_loads(regs: Regs) -> None:
    regs.with_agents("profiler")
    regs.with_closure(make_closure("inner"))
    regs.with_closure(make_closure("outer", subgraph=[{"closure": "inner"}]))
    assert check_closures(regs, NO_ESCAPE_HATCH) == []


def test_an_entry_naming_no_closure_is_rejected(regs: Regs) -> None:
    """The entry shape is `task_graph`'s convention and no spec fixes it, so the
    message says what an entry is rather than citing one."""
    regs.with_agents("profiler")
    regs.with_closure(make_closure("outer", subgraph=[{"is_start": True}]))

    (problem,) = target_problems(check_closures(regs, NO_ESCAPE_HATCH))
    assert problem.keyword == "required"
    assert "naming no closure" in problem.message


def test_each_bad_entry_is_reported_with_its_index(regs: Regs) -> None:
    regs.with_agents("profiler")
    regs.with_closure(
        make_closure("outer", subgraph=[{"closure": "a"}, {"closure": "b"}, {"closure": "outer"}])
    )
    problems = target_problems(check_closures(regs, NO_ESCAPE_HATCH))
    assert [p.path for p in problems] == ["$.task.subgraph[0]", "$.task.subgraph[1]"]
