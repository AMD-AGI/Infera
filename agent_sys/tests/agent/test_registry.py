"""Criteria 1 and 2 — design §12."""

from __future__ import annotations

import pytest

from agent.registry import AgentSpecRegistry, KnowledgeWarning
from spec_loader.protocols import SpecInconsistent, SpecInvalid, SpecNotFound

from .conftest import ai_spec


def test_load_rejects_unknown_backend() -> None:
    """Criterion 1, first half: **with the offending value named.**"""
    registry = AgentSpecRegistry()
    with pytest.raises(SpecInvalid) as caught:
        registry.add(
            "writer",
            ai_spec(backends=[{"key": "nope", "backend_entry": "no.such.module:Thing"}]),
            origin="pkg/writer.jsonnet",
        )
    message = str(caught.value)
    assert "no.such.module:Thing" in message
    assert "nope" in message
    assert "pkg/writer.jsonnet" in message
    assert "writer" not in registry


def test_load_rejects_unresolvable_knowledge() -> None:
    """Criterion 1, second half. A `required` knowledge ref whose kind resolves
    nowhere is fatal in **both** modes; see `check_knowledge`'s docstring for
    why that is what reconciles criteria 1 and 2."""
    registry = AgentSpecRegistry()
    registry.add(
        "writer",
        ai_spec(knowledge=[{"kind": "absent", "knowledge_type": "few_shot", "required": True}]),
        origin="tests",
    )
    with pytest.warns(KnowledgeWarning):
        _, problems = registry.check_knowledge(handoff_specs=set())
    assert [p.fatal for p in problems] == [True]
    assert "absent" in problems[0].message


def test_knowledge_missing_warns_then_fatal() -> None:
    """Criterion 2: **the same spec loads in one mode and is rejected in the
    other**, and both modes read the same value, so they cannot drift into
    disagreeing about what is missing."""
    registry = AgentSpecRegistry()
    registry.add(
        "writer",
        ai_spec(knowledge=[{"kind": "absent", "knowledge_type": "runnable"}]),
        origin="tests",
    )

    with pytest.warns(KnowledgeWarning) as warned:
        lenient_reports, lenient = registry.check_knowledge(handoff_specs=set())
    strict_reports, strict = registry.check_knowledge(handoff_specs=set(), mandatory=True)

    assert [p.fatal for p in lenient] == [False]
    assert [p.fatal for p in strict] == [True]
    assert lenient_reports[0].missing == strict_reports[0].missing
    assert "absent" in str(warned[0].message)


def test_knowledge_that_resolves_is_neither() -> None:
    registry = AgentSpecRegistry()
    registry.add(
        "writer",
        ai_spec(knowledge=[{"kind": "notes", "knowledge_type": "runnable"}]),
        origin="tests",
    )
    reports, problems = registry.check_knowledge(handoff_specs={"notes"}, mandatory=True)
    assert problems == []
    assert reports[0].complete
    assert "runnable" not in reports[0].types_absent


def test_types_absent_is_advisory_and_never_fatal() -> None:
    """Design §4.1. A spec that declares no `runnable` knowledge is not
    malformed; treating a checklist as a schema is how "strongly suggested"
    becomes "hardcoded mandatory"."""
    registry = AgentSpecRegistry()
    registry.add("writer", ai_spec(), origin="tests")
    reports, problems = registry.check_knowledge(handoff_specs=set(), mandatory=True)
    assert problems == []
    assert len(reports[0].types_absent) == 6


def test_duplicate_policy() -> None:
    registry = AgentSpecRegistry()
    registry.add("writer", ai_spec(), origin="a")
    registry.add("writer", ai_spec(), origin="b")  # byte-identical is a no-op
    with pytest.raises(SpecInconsistent):
        registry.add("writer", ai_spec(description="different"), origin="c")


def test_unknown_name_enumerates_candidates() -> None:
    registry = AgentSpecRegistry()
    registry.add("writer", ai_spec(), origin="tests")
    with pytest.raises(SpecNotFound) as caught:
        registry.get("typo")
    assert "writer" in str(caught.value)


def test_resolution_is_not_availability() -> None:
    """Design §4, check 2. Resolution answers "does this name denote
    something"; availability answers "can it run here", and the second changes
    after `env_mgr` deploys. A backend that refuses to construct still
    *resolves*, so it loads."""
    registry = AgentSpecRegistry()
    registry.add(
        "writer",
        ai_spec(
            backends=[
                {
                    "key": "scripted",
                    "backend_entry": "tests.agent.conftest:ScriptedBackend",
                    "config": {"unavailable": "no CLI on PATH"},
                }
            ]
        ),
        origin="tests",
    )
    assert "writer" in registry


def test_the_same_spec_under_two_names_is_rejected() -> None:
    """`pluggy`'s failure transfers: one spec under two names runs twice. Here
    it would mint two `Agent` records from one declaration.

    `BaseSpecRegistry` rejects it, and for an agent spec `_validate` gets there
    first with a **more specific** message — the spec carries its own `name`, so
    the fault is nameable as "registered as X, names itself Y" rather than as a
    byte-identical twin. The test asserts the specific one on purpose: if the
    ordering is ever changed the message degrades and this says so.
    """
    registry = AgentSpecRegistry()
    registry.add("writer", ai_spec(name="writer"), origin="a")
    with pytest.raises(SpecInvalid) as caught:
        registry.add("scribe", ai_spec(name="writer"), origin="b")
    assert "scribe" in str(caught.value)
    assert "writer" in str(caught.value)
    assert registry.names() == ["writer"]


def test_a_rejected_spec_leaves_no_parsed_model_behind() -> None:
    """`_validate` runs **before** the collision check, so a hook that recorded
    anything there would leave a model for a spec that was then rejected. The
    parse is cached in `_admitted`, which runs only on the branch that stores."""
    registry = AgentSpecRegistry()
    registry.add("writer", ai_spec(), origin="a")
    with pytest.raises(SpecInconsistent):
        registry.add("writer", ai_spec(description="different"), origin="b")
    assert registry.spec("writer").description == "writes"
    assert registry.names() == ["writer"]
