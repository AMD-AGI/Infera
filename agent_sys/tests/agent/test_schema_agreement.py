"""The schema and the model are two records of one shape, so something checks them.

`spec_loader` owns `agent.schema.json`; this package owns its content and
`AgentSpec` is the validated form behind it. **Two declarations of one shape**,
which is what `engineer_principle.md` §1 names — and the cost was not
hypothetical: the first version of the schema invented `{type, handoff}` for a
knowledge reference against the model's `{kind, knowledge_type, required}`, and
with `additionalProperties: false` on both sides **every knowledge-bearing agent
spec in the system would have been rejected**, by a file neither reader wrote.

It was caught by `spec-loader` asking rather than shipping. This is what makes
the next one mechanical.

Validation goes through `spec_loader.validate`, not a bare `Draft202012Validator`:
the schema `$ref`s `_common.schema.json`, and only the real pipeline resolves it.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.spec import AgentSpec
from spec_loader import schema_for, validate

from .conftest import ai_spec, program_spec

#: Every one must be admitted by **both** the schema and the model.
ADMITTED: list[tuple[str, dict]] = [
    ("an ai spec with a list of backends", ai_spec()),
    ("a program spec, which declares none", program_spec()),
    (
        "knowledge in the model's shape",
        ai_spec(knowledge=[{"kind": "notes", "knowledge_type": "runnable", "required": True}]),
    ),
    ("backends as a mapping — design D2", ai_spec(backends={"a": {"backend_entry": "m:C"}})),
    (
        "material as package-relative paths",
        ai_spec(
            rules=[".claude/rules/x.md"],
            hooks=[".claude/settings.json"],
            skills=[".claude/skills/s"],
        ),
    ),
    (
        "a backend carrying config and its own err",
        ai_spec(backends=[{"key": "k", "backend_entry": "m:C", "config": {"x": 1}, "err": "no"}]),
    ),
    (
        "kind: human, which loads and fails at selection",
        {**ai_spec(), "kind": "human", "backends": []},
    ),
]

#: Every one must be rejected by **both**, and the schema names where.
REJECTED: list[tuple[str, dict, str]] = [
    (
        "permissions on an agent spec — criterion 5",
        {**ai_spec(), "permissions": {"grants": []}},
        "$",
    ),
    (
        "the knowledge shape the schema first guessed",
        ai_spec(knowledge=[{"type": "few_shot", "handoff": "notes"}]),
        "$.knowledge[0]",
    ),
    (
        "a backend with no entry to resolve",
        ai_spec(backends=[{"key": "k"}]),
        "$.backends[0]",
    ),
]


@pytest.mark.parametrize(("label", "doc"), [(label, doc) for label, doc in ADMITTED], ids=str)
def test_the_schema_and_the_model_both_admit_it(label: str, doc: dict) -> None:
    problems = validate(doc, schema_for("agent"), origin=label)
    assert not problems, f"{label}: schema rejected it — {[p.message for p in problems]}"
    assert AgentSpec.of(doc).name == doc["name"]


@pytest.mark.parametrize(
    ("label", "doc", "path"), [(label, doc, path) for label, doc, path in REJECTED], ids=str
)
def test_the_schema_and_the_model_both_reject_it(label: str, doc: dict, path: str) -> None:
    """**The schema is the enforcement point and the model is the backstop.**

    Both must refuse, and the schema must name *where* — a rejection whose path
    is `$` when the fault is one key deep is a rejection an author cannot act
    on.
    """
    problems = validate(doc, schema_for("agent"), origin=label)
    assert problems, f"{label}: the schema admitted it"
    assert problems[0].path == path
    with pytest.raises(ValidationError):
        AgentSpec.of(doc)


def test_the_schema_declares_no_permissions_key() -> None:
    """Criterion 5, at the layer that enforces it. `extra="forbid"` on the model
    is the second of two; this is the first, and it is the one an author's
    editor shows them."""
    assert "permissions" not in schema_for("agent")["properties"]
    assert schema_for("agent")["additionalProperties"] is False
