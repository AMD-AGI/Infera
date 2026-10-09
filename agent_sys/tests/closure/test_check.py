"""The load checks — criteria 1, 2, 3, 4 and 6."""

from __future__ import annotations

from closure.check import check_closures

from .conftest import NO_ESCAPE_HATCH, Regs, Report, make_closure


def messages(problems) -> str:
    return "\n".join(p.message for p in problems)


# --------------------------------------------------------------------------- #
# Criterion 1


def test_unresolved_kind_names_it(regs: Regs) -> None:
    """A closure naming a handoff kind that does not resolve is rejected at load,
    with the kind name in the message.

    It also asserts the ordering of the check: a kind that neither resolves *nor*
    is declared must report the RESOLUTION failure, not the declaration one. The
    author's next action differs — "you typed it wrong, or the file is missing"
    sends them somewhere else than "add it to `handoffs`" — and reporting the
    second when the first is true sends them to the wrong file.
    """
    regs.with_kinds("trace", "kernel_ir").with_agents("profiler")
    regs.with_closure(make_closure(inputs=["trace_v2"], handoffs=[]))

    problems = check_closures(regs, NO_ESCAPE_HATCH)
    resolution = [p for p in problems if p.keyword == "resolves"]

    assert len(resolution) == 1
    assert "trace_v2" in resolution[0].message
    assert "trace" in resolution[0].message  # the candidate list
    assert resolution[0].fatal
    assert not [p for p in problems if p.keyword == "declared"], (
        "an unresolved kind must not also be reported as undeclared"
    )


def test_unresolved_kind_offers_a_computed_repair(regs: Regs) -> None:
    """The message names three things, not two.

    Everyone clears "name both sides". What users called actionable is a repair
    enumerated from what is actually in scope, which is Dagster's model.
    """
    regs.with_kinds("trace").with_agents("profiler")
    regs.with_closure(make_closure(inputs=["trcae"], handoffs=["trcae"]))

    (problem,) = [p for p in check_closures(regs, NO_ESCAPE_HATCH) if p.keyword == "resolves"]
    assert "known kinds: trace" in problem.message
    assert "hint: 'trace' is close." in problem.message


def test_the_origin_is_the_file_the_author_wrote(regs: Regs) -> None:
    regs.with_kinds("trace").with_agents("profiler")
    regs.with_closure(
        make_closure(inputs=["nope"]), origin="packages/perf/closures/collect_trace.jsonnet"
    )
    assert all(
        p.origin == "packages/perf/closures/collect_trace.jsonnet"
        for p in check_closures(regs, NO_ESCAPE_HATCH)
    )


# --------------------------------------------------------------------------- #
# Criterion 2


def test_declared_handoffs_one_directional(regs: Regs) -> None:
    """A closure whose task names an input absent from its declared `handoffs` is
    rejected; the reverse — a declared kind the task does not name — loads.

    The reverse is legal because a closure may declare a kind its subgraph uses
    internally.
    """
    regs.with_kinds("trace", "kernel_ir").with_agents("profiler")

    regs.with_closure(make_closure("forgot", inputs=["trace"], handoffs=[]))
    (problem,) = [p for p in check_closures(regs, NO_ESCAPE_HATCH) if p.keyword == "declared"]
    assert "trace" in problem.message
    assert problem.path == "$.handoffs"

    extra = Regs().with_kinds("trace", "kernel_ir").with_agents("profiler")
    extra.with_closure(make_closure("spare", inputs=["trace"], handoffs=["trace", "kernel_ir"]))
    assert check_closures(extra, NO_ESCAPE_HATCH) == []


# --------------------------------------------------------------------------- #
# Criterion 3


def test_missing_agent_spec_rejected(regs: Regs) -> None:
    regs.with_kinds("trace").with_agents("profiler")
    regs.with_closure(make_closure(inputs=["trace"], agent="ghost"))

    (problem,) = [p for p in check_closures(regs, NO_ESCAPE_HATCH) if p.path == "$.agent"]
    assert problem.keyword == "resolves"
    assert "ghost" in problem.message
    assert "profiler" in problem.message


def test_no_agent_key_rejected(regs: Regs) -> None:
    """A **leaf** naming no agent spec at all is rejected too.

    `agent` is required of a leaf and the loader synthesises nothing: a
    `kind: program` spec is written by the package author and admitted by the
    ordinary registry. Narrowed to leaf-only at rev. 10 — see the non-leaf case
    below, which is the half that changed.
    """
    regs.with_kinds("trace").with_agents("profiler")
    regs.with_closure(make_closure(inputs=["trace"], agent=None))

    (problem,) = [p for p in check_closures(regs, NO_ESCAPE_HATCH) if p.path == "$.agent"]
    assert problem.keyword == "required"
    assert "names no agent spec" in problem.message
    assert "program" in problem.message


def test_a_non_leaf_needs_no_agent_key(regs: Regs) -> None:
    """Main spec §4.8, narrowed at rev. 10: a non-leaf's work *is* its subgraph.

    The schema had already been narrowed by `fbac040` — `agent` left `required`
    and an if/else reinstates it unless `task.subgraph` is present and non-empty
    — and this check had not, so the two disagreed and a non-leaf that the
    schema admitted was rejected here. Pinned against the schema's own
    condition: `has_subgraph` is `bool(subgraph_of(task))`, which is
    `required` + `minItems: 1` said in Python.
    """
    regs.with_kinds("trace").with_agents("profiler")
    regs.with_closure(make_closure("leaf"))
    regs.with_closure(
        make_closure("parent", agent=None, subgraph=[{"closure": "leaf", "froms": []}])
    )
    assert [p for p in check_closures(regs, NO_ESCAPE_HATCH) if p.path == "$.agent"] == []


def test_a_non_leaf_naming_an_agent_that_does_not_resolve_is_still_rejected(regs: Regs) -> None:
    """Only *absent* narrowed. An author may still name one on a non-leaf, and a
    name that does not resolve is a typo whether or not the task is a leaf."""
    regs.with_kinds("trace").with_agents("profiler")
    regs.with_closure(make_closure("leaf"))
    regs.with_closure(
        make_closure("parent", agent="ghost", subgraph=[{"closure": "leaf", "froms": []}])
    )

    (problem,) = [p for p in check_closures(regs, NO_ESCAPE_HATCH) if p.path == "$.agent"]
    assert problem.keyword == "resolves"
    assert "ghost" in problem.message


def test_a_program_agent_is_an_ordinary_spec(regs: Regs) -> None:
    """The `kind: program` case is not special-cased anywhere in this package."""
    regs.with_kinds("trace").with_agents("run_pytest")
    regs.with_closure(make_closure(inputs=["trace"], agent="run_pytest"))
    assert check_closures(regs, NO_ESCAPE_HATCH) == []


# --------------------------------------------------------------------------- #
# Criterion 4


def test_phase_validator_is_not_a_task(regs: Regs) -> None:
    """A closure naming a phase validator that resolves to a general task is
    rejected, and the message says which mistake it was.

    Resolves-then-is-the-right-kind: the second message is only reachable when
    the first passed, which is what makes it specific.
    """
    regs.with_kinds("trace").with_agents("profiler").with_validators("check_trace_shape")
    regs.with_closure(make_closure("collect_trace", inputs=["trace"], validators=["prepare_e2e"]))
    regs.with_closure(make_closure("prepare_e2e"))

    (problem,) = [p for p in check_closures(regs, NO_ESCAPE_HATCH) if p.keyword == "kind"]
    assert "prepare_e2e" in problem.message
    assert "general task, not a validator" in problem.message


def test_unresolved_phase_validator_is_a_different_message(regs: Regs) -> None:
    regs.with_kinds("trace").with_agents("profiler").with_validators("check_trace_shape")
    regs.with_closure(make_closure(inputs=["trace"], validators=["check_trace_shpe"]))

    (problem,) = [p for p in check_closures(regs, NO_ESCAPE_HATCH) if p.path == "$.validators"]
    assert problem.keyword == "resolves"
    assert "hint: 'check_trace_shape' is close." in problem.message


# --------------------------------------------------------------------------- #
# Criterion 6


def test_escape_hatch_reported(regs: Regs) -> None:
    """A closure assembled from a handoff kind admitted under the escape-hatch
    flag loads, and reports that it did.

    Report severity, not error: a kind with no validator is unadmittable in the
    first place, so anything in `without_validator` is already known and already
    permitted. This does not re-derive the coverage.
    """
    regs.with_kinds("trace", "kernel_ir").with_agents("profiler")
    regs.with_closure(make_closure(inputs=["trace"], outputs=["kernel_ir"]))

    report = Report(admitted=["trace", "kernel_ir"], without_validator=["kernel_ir"])
    problems = check_closures(regs, report)

    assert [p.fatal for p in problems] == [False], "an escape-hatch admission is not a failure"
    assert "kernel_ir" in problems[0].message
    assert "escape-hatch" in problems[0].message
    assert "'trace'" not in problems[0].message, "only the escape-hatch kind is named"

    assert check_closures(regs, NO_ESCAPE_HATCH) == [], "no report means nothing to intersect"


# --------------------------------------------------------------------------- #
# Properties of the pass itself


def test_every_check_appends_and_none_raises(regs: Regs) -> None:
    """A closure with a typo'd kind AND a missing agent reports both.

    A loader that dies on the first bad spec makes fixing a package an N-round
    trip.
    """
    regs.with_kinds("trace").with_agents("profiler").with_validators("check_trace_shape")
    regs.with_closure(
        make_closure(
            inputs=["trace_v2"],
            handoffs=["trace_v2"],
            agent="ghost",
            validators=["nobody"],
            grants=[],
            body={},
        )
    )
    problems = check_closures(regs, NO_ESCAPE_HATCH)
    assert {p.path for p in problems} >= {
        "$.task",
        "$.agent",
        "$.validators",
        "$.task.permissions.grants",
        "$.task.body.readme",
    }


def test_problems_are_in_sorted_closure_order(regs: Regs) -> None:
    """Determinism: a package with two broken closures reports them the same way
    twice."""
    regs.with_agents("profiler")
    for name in ("zeta", "alpha", "mid"):
        regs.with_closure(make_closure(name, inputs=["ghost"], handoffs=["ghost"]))

    order = [p.message.split("'")[1] for p in check_closures(regs, NO_ESCAPE_HATCH)]
    assert order == ["alpha", "mid", "zeta"]


def test_skip_is_the_layering_gate(regs: Regs) -> None:
    """A closure whose own spec already failed is not checked again.

    "Your task's handoff kind does not resolve" on top of "your schema is broken"
    is noise, which is the reason Kubernetes CRD validation gives for the same
    gate: error messages that are not actionable.
    """
    regs.with_agents("profiler")
    regs.with_closure(make_closure("broken", inputs=["ghost"], handoffs=["ghost"]))
    regs.with_closure(make_closure("fine"))

    assert check_closures(regs, NO_ESCAPE_HATCH) != []
    assert check_closures(regs, NO_ESCAPE_HATCH, skip={"broken"}) == []
