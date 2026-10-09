"""The two registries — criterion 7, and the collision policy they share."""

from __future__ import annotations

import pytest

from closure.check import check_closures
from closure.registry import ClosureRegistry
from closure.task_registry import TaskSpecRegistry
from spec_loader.protocols import SpecInconsistent, SpecNotFound

from .conftest import NO_ESCAPE_HATCH, Regs, make_closure


def test_sharing_is_legal(regs: Regs) -> None:
    """Two closures may share a handoff kind, an agent spec, and a validator;
    none is exclusive to one closure."""
    regs.with_kinds("trace").with_agents("profiler").with_validators("check_trace_shape")
    regs.with_closure(
        make_closure("collect_trace", outputs=["trace"], validators=["check_trace_shape"])
    )
    regs.with_closure(
        make_closure("analyse_trace", inputs=["trace"], validators=["check_trace_shape"])
    )

    assert check_closures(regs, NO_ESCAPE_HATCH) == []
    assert regs.closures.closures_using_kind("trace") == ("analyse_trace", "collect_trace")
    assert regs.closures.closures_using_agent("profiler") == ("analyse_trace", "collect_trace")
    assert regs.closures.closures_using_validator("check_trace_shape") == (
        "analyse_trace",
        "collect_trace",
    )


def test_a_duplicate_name_is_an_error_not_an_overwrite() -> None:
    """The opposite of the component `Registry`, deliberately. That one
    overwrites so a test can swap an implementation after wiring; a spec registry
    is a name table, and two specs claiming one name is a fault."""
    registry = ClosureRegistry()
    doc = make_closure()
    registry.add("collect_trace", doc, origin="a.jsonnet")

    registry.add("collect_trace", dict(doc), origin="a.jsonnet")  # identical: a no-op

    with pytest.raises(SpecInconsistent) as caught:
        registry.add("collect_trace", make_closure(agent="other"), origin="b.jsonnet")
    assert "a.jsonnet" in str(caught.value)
    assert "b.jsonnet" in str(caught.value)


def test_an_unknown_name_enumerates_the_candidates() -> None:
    registry = ClosureRegistry()
    registry.add("collect_trace", make_closure(), origin="a.jsonnet")

    with pytest.raises(SpecNotFound) as caught:
        registry.get("collect_trcae")
    assert "collect_trace" in str(caught.value)


def test_the_pass_keys_each_nested_task_spec(regs: Regs) -> None:
    """`check_closures` is what populates `task_specs`, and nothing else can be.

    A task spec is nested inside its closure and carries no `name`, so
    `spec_loader` does not discover one — `task` is deliberately not a
    discoverable kind. Until this pass did it, `task_specs` was empty after a
    load and `check_graph` walked an empty catalogue: `task_graph` criteria 50
    and 53 were green and inert. Found by `task_graph`, and this test is what
    stops it recurring.

    It is the *inner* task spec, not the closure document. Handed the wrong one,
    `subgraph_of` finds no `subgraph`, every task looks like a leaf, and the
    graph checks return `[]` — no crash and no wrong answer, just silence.
    """
    regs.with_agents("profiler")
    regs.with_closure(make_closure("collect_trace"))
    assert regs.task_specs.names() == [], "nothing but the pass may key a task spec"

    assert check_closures(regs, NO_ESCAPE_HATCH) == []

    assert regs.closures.names() == regs.task_specs.names() == ["collect_trace"]
    task = regs.task_specs.get("collect_trace")
    assert "task" not in task, "the inner task spec, not the closure document"
    assert task["goal"].startswith("collect_trace")
    assert regs.task_specs.origin_of("collect_trace") == regs.closures.origin_of("collect_trace")


def test_a_skipped_closure_keys_no_task_spec(regs: Regs) -> None:
    """The layering gate covers admission too: a closure whose own spec failed
    should not put a half-understood task into the catalogue `check_graph`
    walks."""
    regs.with_agents("profiler")
    regs.with_closure(make_closure("broken"))
    regs.with_closure(make_closure("fine"))

    check_closures(regs, NO_ESCAPE_HATCH, skip={"broken"})
    assert regs.task_specs.names() == ["fine"]


def test_two_closures_cannot_share_one_task_spec(regs: Regs) -> None:
    """Design O6, surfacing as a message rather than as a silent second key.

    The shared base rejects the reverse collision — one spec under two names — so
    two closures whose tasks are byte-identical land here. The pass reports it and
    raises nothing.
    """
    regs.with_agents("profiler")
    shared = make_closure("first")["task"]
    for name in ("first", "second"):
        doc = make_closure(name)
        doc["task"] = dict(shared)
        regs.with_closure(doc)

    (problem,) = [p for p in check_closures(regs, NO_ESCAPE_HATCH) if p.keyword == "duplicate"]
    assert "second" in problem.message
    assert "two closures cannot share one task" in problem.message


def test_task_spec_registry_adds_nothing_to_the_base() -> None:
    added = {n for n in vars(TaskSpecRegistry) if not n.startswith("__")} - {"kind"}
    assert not added, f"TaskSpecRegistry grew {sorted(added)}; it exists to be separate, not to do"


def test_freeze_before_the_pass_is_a_programming_error() -> None:
    with pytest.raises(RuntimeError, match="check_closures"):
        ClosureRegistry().freeze()


# --------------------------------------------------------------------------- #
# The third edge kind — `docs/interfaces.md` §5.4, wired from the pass.


def test_the_pass_records_the_closure_to_phase_validator_edge(regs: Regs) -> None:
    """`validator.users_of` cannot see this edge and the pass is where it is seen.

    A closure's phase validators are a property of the task, so the handoff specs
    cannot carry them. Without this, a validator two closures run in every output
    phase is reported as used by nothing — Airflow #58058's false-negative
    deadness, with dbt#14436 as a second instance.
    """
    regs.with_agents("profiler").with_validators("check_shape", "check_cov")
    regs.with_closure(make_closure("collect", validators=["check_shape"]))
    regs.with_closure(make_closure("analyse", validators=["check_shape", "check_cov"]))

    assert regs.validator_specs.phase_edges == {}, "nothing but the pass records the edge"
    assert check_closures(regs, NO_ESCAPE_HATCH) == []

    assert regs.validator_specs.phase_edges == {
        "analyse": ["check_shape", "check_cov"],
        "collect": ["check_shape"],
    }


def test_a_skipped_closure_records_no_edge(regs: Regs) -> None:
    """The layering gate covers the edge too, and it comes free from running in
    the pass — the composition root would have had to restate it."""
    regs.with_agents("profiler").with_validators("check_shape")
    regs.with_closure(make_closure("broken", validators=["check_shape"]))
    regs.with_closure(make_closure("fine", validators=["check_shape"]))

    check_closures(regs, NO_ESCAPE_HATCH, skip={"broken"})
    assert regs.validator_specs.phase_edges == {"fine": ["check_shape"]}


def test_an_unresolvable_phase_validator_is_still_recorded_as_declared(regs: Regs) -> None:
    """The edge is what the closure *names*, not what resolves.

    Check 5 reports the unresolvable name; `users_of` is asked "who names this",
    and an index that hid unresolvable users would answer a different question
    from the one it was asked.
    """
    regs.with_agents("profiler")
    regs.with_closure(make_closure("collect", validators=["ghost"]))

    assert [p.keyword for p in check_closures(regs, NO_ESCAPE_HATCH)] == ["resolves"]
    assert regs.validator_specs.phase_edges == {"collect": ["ghost"]}


def test_a_registry_without_bind_phase_fails_loudly(regs: Regs) -> None:
    """Called directly, never through `getattr(..., None)`.

    A guard would make a missing binding silent in the assembled system, and the
    symptom would be an *under-reported* `users_of` rather than an error — which
    is the shape that hid `handoff`'s `load_report` mismatch. `docs/interfaces.md`
    §4.11 is about a parameter; this is the same rule for a collaborator.
    """

    class WithoutIt:
        kind = "validator"

        def names(self) -> list[str]:
            return []

        def __contains__(self, name: object) -> bool:
            return False

    regs.validator_specs = WithoutIt()
    regs.with_agents("profiler")
    regs.with_closure(make_closure("collect"))

    with pytest.raises(AttributeError, match="bind_phase"):
        check_closures(regs, NO_ESCAPE_HATCH)


def test_the_real_users_of_stops_under_reporting(regs: Regs) -> None:
    """The guard that is actually worth having, against the real registry.

    The three tests above assert on this package's *stub*, which proves the call
    happens. It does not prove the thing the call exists for. `main`'s note is
    the sharp one: **a reverse index that is merely incomplete does not raise**,
    so a stub assertion is one step removed from the failure — the same distance
    that let `check_graph` walk an empty catalogue while every unit test passed.

    So this uses `validator`'s real `ValidatorSpecRegistry` and asks it the
    question a user asks: *what breaks if I change `check_shape`?* Two closures
    run it in every output phase and no handoff kind names it, which is exactly
    Airflow #58058's shape — the answer must not be "nothing".

    A test may import `validator`; `docs/interfaces.md` §4's rule is about what
    leaves a package.
    """
    from validator.registry import ValidatorSpecRegistry

    regs.validator_specs = ValidatorSpecRegistry()
    regs.with_agents("profiler")
    regs.with_closure(make_closure("collect", validators=["check_shape"]))
    regs.with_closure(make_closure("analyse", validators=["check_shape"]))

    assert regs.validator_specs.users_of("check_shape") == [], "nothing has run yet"

    check_closures(regs, NO_ESCAPE_HATCH)

    assert regs.validator_specs.users_of("check_shape") == [
        "closure:analyse",
        "closure:collect",
    ], "a validator two closures run in every output phase reported as used by nothing"


# --------------------------------------------------------------------------- #
# `validator` §8.2 row 1 — a validator's agent spec must resolve. Step 4 of a
# four-package sequence; `validator` asserts both halves at run time and this is
# the load-time layer, which is what turns a mid-run failure into a report.


def _validator(name: str, agent: str | None = None) -> dict:
    rec: dict = {
        "name": name,
        "brief": "checks the shape",
        "dimension": "completeness",
        "strength": "weak",
        "inputs": ["trace"],
        "version": "1",
        "body": {"readme": f"{name}/readme.md", "entry": f"{name}/entry.sh"},
        "tags": {"logic_source": "external_static", "cost": "seconds", "domain": ["trace"]},
    }
    if agent is not None:
        rec["agent"] = agent
    return rec


def _with_real_validators(regs: Regs, *specs: dict) -> Regs:
    """The real `ValidatorSpecRegistry`, because this check reads a document it
    does not own and the stub cannot tell me `get()` hands back a Mapping."""
    from validator.registry import ValidatorSpecRegistry

    regs.validator_specs = ValidatorSpecRegistry()
    for rec in specs:
        regs.validator_specs.add(rec["name"], rec, origin=f"validators/{rec['name']}.jsonnet")
    return regs


def test_a_validator_naming_an_unresolvable_agent_is_rejected(regs: Regs) -> None:
    """The author wanted row 1 and would get row 4 — silently, with a working
    environment that is not the one they configured."""
    regs.with_agents("profiler")
    _with_real_validators(regs, _validator("shape", agent="profilr"))

    (problem,) = [p for p in check_closures(regs, NO_ESCAPE_HATCH) if p.path == "$.agent"]
    assert problem.fatal
    assert "'profilr'" in problem.message
    assert "known agent specs: profiler" in problem.message
    assert "hint: 'profiler' is close." in problem.message
    assert problem.origin == "validators/shape.jsonnet", (
        "keyed to the validator file the author opens, not to a closure's origin"
    )


def test_a_validator_naming_no_agent_is_not_a_fault(regs: Regs) -> None:
    """Absent is legal and ordinary: it takes the global row, which is the
    designed answer rather than a fallback."""
    regs.with_agents("profiler")
    _with_real_validators(regs, _validator("shape"), _validator("cover", agent="profiler"))
    assert [p for p in check_closures(regs, NO_ESCAPE_HATCH) if p.path == "$.agent"] == []


def test_a_validator_no_closure_names_is_still_checked(regs: Regs) -> None:
    """The case the per-closure form would miss entirely.

    A validator bound only to a handoff kind is named by no closure and has
    exactly the same defect — its environment falls back just as silently. This
    is why the check is a whole-catalogue pass rather than a row in the loop over
    closures.
    """
    regs.with_agents("profiler")
    _with_real_validators(regs, _validator("orphan", agent="ghost"))
    assert regs.closures.names() == [], "no closure names it, or anything"

    (problem,) = [p for p in check_closures(regs, NO_ESCAPE_HATCH) if p.path == "$.agent"]
    assert "'orphan'" in problem.message


def test_one_validator_two_closures_reports_once(regs: Regs) -> None:
    """The fault is a property of the validator spec, so it is reported once.

    In the per-closure form this would be two problems, each keyed to a closure's
    origin — the wrong file, twice.
    """
    regs.with_agents("profiler")
    _with_real_validators(regs, _validator("shape", agent="ghost"))
    regs.with_closure(make_closure("collect", validators=["shape"]))
    regs.with_closure(make_closure("analyse", validators=["shape"]))

    reported = [p for p in check_closures(regs, NO_ESCAPE_HATCH) if p.path == "$.agent"]
    assert len(reported) == 1
    assert reported[0].origin == "validators/shape.jsonnet"
