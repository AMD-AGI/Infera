"""The read-only query helpers — criterion 9."""

from __future__ import annotations

import pytest

from closure.check import check_closures
from spec_loader.protocols import SpecNotFound

from .conftest import NO_ESCAPE_HATCH, Regs, make_closure


@pytest.fixture
def loaded() -> Regs:
    """Two closures over a shared world, loaded and indexed."""
    regs = Regs()
    regs.with_agents("profiler", "analyst")
    regs.with_validators("check_trace_shape", "check_summary", "phase_inputs_present")
    regs.handoff_specs.add(
        "trace",
        {"name": "trace", "version": "1", "validators": ["check_trace_shape"]},
        origin="handoff/trace.jsonnet",
    )
    regs.handoff_specs.add(
        "summary",
        {"name": "summary", "version": "1", "validators": ["check_summary"]},
        origin="handoff/summary.jsonnet",
    )
    regs.handoff_specs.add(
        "unused_kind",
        {"name": "unused_kind", "version": "1", "validators": ["check_summary"]},
        origin="handoff/unused_kind.jsonnet",
    )
    regs.with_closure(
        make_closure(
            "collect_trace",
            outputs=["trace"],
            agent="profiler",
            validators=["phase_inputs_present"],
        )
    )
    regs.with_closure(
        make_closure("analyse_trace", inputs=["trace"], outputs=["summary"], agent="analyst")
    )
    assert check_closures(regs, NO_ESCAPE_HATCH) == []
    regs.closures.freeze()
    return regs


def test_queries_need_no_join(loaded: Regs) -> None:
    """Every read-only query answers without a caller-written join.

    `validators_for` is the one that would otherwise force one: the phase
    validators live on the closure and the per-handoff ones live on the handoff
    specs, and a caller wanting "every validator that will run" would have to
    walk both.
    """
    closures = loaded.closures

    assert closures.handoff_kinds("analyse_trace") == ("trace", "summary")
    assert closures.agent_of("analyse_trace") == "analyst"
    assert closures.validators_for("analyse_trace") == ("check_trace_shape", "check_summary")
    assert closures.validators_for("collect_trace") == (
        "phase_inputs_present",
        "check_trace_shape",
    )
    assert closures.closures_using_kind("trace") == ("analyse_trace", "collect_trace")
    assert closures.closures_using_agent("profiler") == ("collect_trace",)
    assert closures.closures_using_validator("phase_inputs_present") == ("collect_trace",)


def test_not_found_and_used_by_nothing_are_different_answers(loaded: Regs) -> None:
    """dbt has this right in the data and loses it at every call site, and its
    user-facing message then hedges across typo, unused, and disabled alike.

    An empty answer is the one a caller acts on, because the question these
    queries exist to answer is "what breaks if I change this".
    """
    assert loaded.closures.closures_using_kind("unused_kind") == ()

    with pytest.raises(SpecNotFound, match="unused_knid"):
        loaded.closures.closures_using_kind("unused_knid")
    with pytest.raises(SpecNotFound):
        loaded.closures.closures_using_agent("nobody")
    with pytest.raises(SpecNotFound):
        loaded.closures.closures_using_validator("nobody")


def test_closures_using_validator_answers_within_one_edge_kind(loaded: Regs) -> None:
    """The sixth query, and the reason it *now* exists — which is not the reason
    it was added.

    It was added because `users_of` could not see the closure edge at all. Wiring
    `bind_phase` removed that, and the query was withdrawn on the reasoning that
    it had become a filter over `users_of`'s output; the withdrawal was reversed
    (`docs/interfaces.md` §4.5). What keeps it is that the two answer different
    questions: `users_of` spans every edge kind and tags each entry, so recovering
    this answer from it means one package parsing another's display format.

    Both halves are asserted, and the second is the one that matters: a phase
    validator has closures, and a validator reached only through a handoff kind
    has **none** — `()` rather than an error, because it is a known validator that
    no closure names.
    """
    assert loaded.closures.closures_using_validator("phase_inputs_present") == ("collect_trace",)
    assert loaded.closures.closures_using_validator("check_trace_shape") == ()


def test_queries_do_not_mutate(loaded: Regs) -> None:
    """A structural test, not a behavioural one.

    The claim is that post-load mutation is *impossible*; a test that merely
    observed no mutation would not distinguish that from "nobody happened to".
    Sphinx is the argument for making it impossible rather than discouraged.
    """
    before = {name: loaded.closures.get(name) for name in loaded.closures.names()}

    for name in loaded.closures.names():
        loaded.closures.handoff_kinds(name)
        loaded.closures.validators_for(name)
        loaded.closures.agent_of(name)
    loaded.closures.closures_using_kind("trace")

    assert {name: loaded.closures.get(name) for name in loaded.closures.names()} == before

    with pytest.raises(RuntimeError, match="frozen"):
        loaded.closures.add("late", make_closure("late"), origin="late.jsonnet")


def test_a_query_before_the_index_says_so() -> None:
    regs = Regs().with_agents("profiler")
    regs.with_closure(make_closure())
    with pytest.raises(RuntimeError, match="index"):
        regs.closures.closures_using_kind("trace")


def test_the_index_is_derived_from_the_accessors(loaded: Regs) -> None:
    """Membership is derived, never restated.

    dbt's `build_parent_and_child_maps` chains seven collections by hand and
    `build_node_edges` silently drops any edge whose target is outside that set.
    Here every edge's source is one of the accessors in `model.py`, so the two
    answers cannot disagree.
    """
    for name in loaded.closures.names():
        for kind in loaded.closures.handoff_kinds(name):
            assert name in loaded.closures.closures_using_kind(kind)
        assert name in loaded.closures.closures_using_agent(loaded.closures.agent_of(name))
