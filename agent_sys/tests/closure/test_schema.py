"""The two schemas — criterion 11, and the idiom the schemas are the enforcement of."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from spec_loader import schema_for
from spec_loader.bundled import bundled_registry

from .conftest import make_closure

ROOT = Path(__file__).resolve().parents[2]


def closure_schema():
    return schema_for("closure")


def task_schema():
    return schema_for("task")


def closure_validator() -> Draft202012Validator:
    """The bundled closure schema, with the task schema resolvable from it.

    `closure` owns what is *in* these two documents; `spec_loader` owns that all
    five live in one installable package and that `$ref` resolves without a
    network fetch. This test reads them through the accessor rather than by
    path, which is the whole reason the accessor is exported.
    """
    return Draft202012Validator(closure_schema(), registry=bundled_registry())


def errors(doc) -> list[str]:
    return [e.message for e in closure_validator().iter_errors(doc)]


def test_the_schemas_are_valid_schemas() -> None:
    """A schema is only the enforcement point if it is itself well-formed."""
    Draft202012Validator.check_schema(closure_schema())
    Draft202012Validator.check_schema(task_schema())


def test_a_well_formed_closure_validates() -> None:
    assert errors(make_closure(inputs=["trace"], outputs=["summary"])) == []


# --------------------------------------------------------------------------- #
# Criterion 11


def test_closure_version_rejected() -> None:
    """A `version` key on a closure is rejected at load.

    Expressed as a schema *difference* — the closure schema omits `version`, the
    member schemas declare it — rather than as a hand-written check. Measured
    against jsonschema 4.26.0: `additionalProperties: false` alone names the
    offending key and is actionable, while adding `not: {required: [version]}`
    produces that message plus a second one nobody can act on.
    """
    doc = make_closure()
    doc["version"] = "3"
    assert errors(doc) == ["Additional properties are not allowed ('version' was unexpected)"]


def test_each_member_declares_its_own_version() -> None:
    """The nested task spec requires one; a closure has none to require."""
    assert "version" in task_schema()["required"]
    assert "version" not in closure_schema()["properties"]

    doc = make_closure()
    del doc["task"]["version"]
    assert "'version' is a required property" in errors(doc)


def test_no_runtime_version_read() -> None:
    """Criterion 11's second half: nothing at runtime reads a member's `version`.

    A structural test rather than a behavioural one — the same shape as
    `task_graph`'s criterion-42 test. A spec `version` exists so a reviewer can
    see that a handoff kind changed between two branches; nothing pins to it, and
    loading two versions of one spec at once is not supported.
    """
    offenders: list[str] = []
    for path in sorted((ROOT / "closure").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            reads_key = (
                isinstance(node, ast.Subscript)
                and isinstance(node.slice, ast.Constant)
                and node.slice.value == "version"
            )
            reads_get = (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "version"
            )
            if reads_key or reads_get:
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not offenders, f"a member spec's `version` is read at {offenders}"


# --------------------------------------------------------------------------- #
# The idiom — the schema is the only enforcement point


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda d: d.pop("agent"), "'agent' is a required property"),
        (lambda d: d.pop("handoffs"), "'handoffs' is a required property"),
        (lambda d: d.pop("validators"), "'validators' is a required property"),
        (lambda d: d["task"].pop("body"), "'body' is a required property"),
        (lambda d: d["task"].pop("goal"), "'goal' is a required property"),
        (lambda d: d.update(sneak=1), "Additional properties are not allowed"),
        (lambda d: d["task"].update(sneak=1), "Additional properties are not allowed"),
        (lambda d: d["task"]["body"].pop("readme"), "'readme' is a required property"),
    ],
)
def test_the_schema_rejects_what_it_says_it_rejects(mutate, expected: str) -> None:
    doc = make_closure(inputs=["trace"])
    mutate(doc)
    assert any(expected in message for message in errors(doc)), errors(doc)


def test_goal_is_one_sentence() -> None:
    """At most 100 characters. A goal is a sentence and a body is an artefact,
    and the split is the point."""
    doc = make_closure()
    doc["task"]["goal"] = "x" * 101
    assert any("is too long" in m for m in errors(doc))


def test_a_grant_admits_no_wildcard_syntax_it_cannot_define() -> None:
    """The grammar the covering relation is total over is exact equality, so the
    schema admits a plain string and nothing that looks like a pattern."""
    grant = task_schema()["properties"]["permissions"]["properties"]["grants"]["items"]
    fields = grant["properties"]
    assert fields["kind"]["type"] == "string"
    assert "pattern" not in fields["kind"]
    assert "pattern" not in fields["path"]
    assert set(fields["access"]["enum"]) == {"read", "write"}
    assert grant["additionalProperties"] is False
