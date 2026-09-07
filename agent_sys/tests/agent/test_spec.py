"""Criteria 5 and 13's testable half — design §12."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.spec import KNOWLEDGE_TYPES, AgentSpec, Kind, types_absent

from .conftest import ai_spec


def test_agent_spec_has_no_permissions_field() -> None:
    """Criterion 5, **structurally**. A structural criterion tested
    behaviourally passes for the wrong reason as soon as someone adds the
    field, so this asserts over the model rather than over a run."""
    assert "permissions" not in AgentSpec.model_fields
    with pytest.raises(ValidationError):
        AgentSpec.of({**ai_spec(), "permissions": {"grants": []}})


def test_same_spec_two_tasks_two_reaches() -> None:
    """Criterion 5's second half: reach is decided by the *task*'s versioned
    permissions, and the same agent spec on two tasks reaches two sets.

    Nothing in this package carries a permission, so the assertion is that the
    spec is identical while the tasks differ — the agent contributes nothing to
    the difference, which is the criterion's content.
    """
    from task_graph.models import Task
    from task_graph.permissions import Grant, Permissions

    spec = AgentSpec.of(ai_spec())
    one = Task(agent_spec=spec.name, permissions=Permissions(grants=[Grant(kind="a")]))
    two = Task(agent_spec=spec.name, permissions=Permissions(grants=[Grant(kind="b")]))
    assert one.agent_spec == two.agent_spec == spec.name
    assert one.permissions != two.permissions


def test_material_stored_canonically() -> None:
    """Criterion 13's testable half. `rules`, `hooks` and `skills` hold **paths
    into the task package**, in Claude Code's format, and this module does not
    parse them — it hands them on."""
    spec = AgentSpec.of(
        ai_spec(
            rules=[".claude/rules/style.md"],
            hooks=[".claude/settings.json"],
            skills=[".claude/skills/review"],
        )
    )
    assert spec.rules == [".claude/rules/style.md"]
    assert spec.hooks == [".claude/settings.json"]
    assert spec.skills == [".claude/skills/review"]
    assert all(isinstance(value, str) for value in spec.rules + spec.hooks + spec.skills)


def test_backends_is_a_list_and_a_mapping_normalises() -> None:
    """Design D2. The order is load-bearing, and a dict that carries an ordering
    is a dict whose ordering nobody can see at the call site."""
    spec = AgentSpec.of(
        ai_spec(
            backends={
                "first": {"backend_entry": "a:B"},
                "second": {"backend_entry": "c:D"},
            }
        )
    )
    assert [decl.key for decl in spec.backends] == ["first", "second"]


def test_identical_duplicate_key_is_tolerated_and_a_differing_one_is_not() -> None:
    """matplotlib's entry-point validation, adopted — design §6.3."""
    same = {"key": "x", "backend_entry": "a:B"}
    assert len(AgentSpec.of(ai_spec(backends=[same, dict(same)])).backends) == 1
    with pytest.raises(ValidationError):
        AgentSpec.of(ai_spec(backends=[same, {"key": "x", "backend_entry": "c:D"}]))


def test_human_is_declarable() -> None:
    """Design §3.1: a `kind: human` spec **loads** and fails at selection. The
    spec is well-formed and this alpha cannot run it; rejecting it at load
    would make it ill-formed, which it is not."""
    assert AgentSpec.of({**ai_spec(), "kind": "human", "backends": []}).kind is Kind.HUMAN


def test_knowledge_type_is_a_string_not_an_enum() -> None:
    """Design D3: spec §3.4 says the list is extensible, and an enum would make
    the seventh type a code change in this module."""
    spec = AgentSpec.of(ai_spec(knowledge=[{"kind": "notes", "knowledge_type": "a_seventh_kind"}]))
    assert spec.knowledge[0].knowledge_type == "a_seventh_kind"
    assert len(KNOWLEDGE_TYPES) == 6
    assert "a_seventh_kind" not in KNOWLEDGE_TYPES
    assert types_absent(spec.knowledge) == list(KNOWLEDGE_TYPES)
