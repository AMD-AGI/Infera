"""`Body`, `body_of`, `subgraph_of` — one declaration for shapes that had three.

`main` ruled these into this package because `schemas/_common.schema.json` holds
**one** `$defs.body`, `$ref`ed by `task.schema.json` and `validator.schema.json`,
while Python declared it three times: `closure/model.py`, `agent/body.py`,
`validator/protocols.py`.

The tests that matter here are the two that pin *semantics the three versions
disagreed about*, not the happy path.
"""

from __future__ import annotations

import pytest

import spec_loader.access as _access
from spec_loader import (
    Body,
    body_of,
    schema_for,
    subgraph_of,
    task_of,
    validate,
    validator_agent_of,
)


def test_a_declared_body_comes_back_as_written() -> None:
    task = {"body": {"readme": "readme.md", "entry": "entry.sh", "materials": ["m/a"]}}

    assert body_of(task) == {"readme": "readme.md", "entry": "entry.sh", "materials": ["m/a"]}


def test_no_body_is_falsy() -> None:
    """The semantic the three versions disagreed on, and the reason for a TypedDict.

    `closure.body_of({})` returned `{}`; `agent.body_of({})` returned
    `Body(readme="")`, **which is truthy** — an object reporting a body that is
    present and empty where the document had none. A dataclass has to construct,
    and constructing means inventing a value for a field the document does not
    have. `{}` is falsy, so `if body_of(task):` means what it looks like.
    """
    assert body_of({}) == {}
    assert not body_of({})
    assert not body_of({"body": None})


def test_the_accessors_are_tolerant_of_a_document_that_never_passed_the_schema() -> None:
    """Problems are collected, not raised, so a checker runs over broken documents.

    An accessor that raised would take the checker down on the first bad spec —
    the failure mode `docs/design.md` §3.6 exists to avoid. The cost is that an
    empty result is **not evidence of absence** before the schema has run, which
    is the silence that made `task_graph`'s `check_graph` bug survivable. The
    schema is the gate; this is not a second one.
    """
    for malformed in ({"body": "oops"}, {"body": 3}, {"body": ["r"]}):
        assert body_of(malformed) == {}
    for malformed in ({"subgraph": "oops"}, {"subgraph": 3}, {"subgraph": None}):
        assert subgraph_of(malformed) == ()


def test_subgraph_entries_are_returned_unnormalised() -> None:
    """No mark is defaulted here, and that split is deliberate.

    `is_start` / `is_end` mean something only once an entry has become a
    `task_graph.SubgraphEntry` — a type this package may not name. So this owns
    that the key exists and is called `subgraph`; `task_graph` owns what an entry
    means and defaults the marks on top. `engineer_principle.md` §3.
    """
    entries = [{"closure": "a"}, {"closure": "b", "is_end": True}]

    assert subgraph_of({"subgraph": entries}) == tuple(entries)
    assert "is_start" not in subgraph_of({"subgraph": entries})[0]
    assert subgraph_of({}) == ()
    assert subgraph_of({"subgraph": []}) == ()


def test_body_matches_the_schema_it_mirrors() -> None:
    """One `$defs.body` and one `Body`, and this is what keeps them in step.

    The whole reason these moved here is that the schema had one declaration and
    Python had three. A fourth divergence — between the schema and the type that
    mirrors it — would be the same defect one level in.
    """
    definition = schema_for("task")["properties"]["body"]
    assert definition["$ref"] == "_common.schema.json#/$defs/body"

    keys = set(Body.__annotations__)
    assert keys == {"readme", "entry", "materials"}
    assert Body.__required_keys__ == frozenset({"readme"})
    assert Body.__optional_keys__ == frozenset({"entry", "materials"})

    full: Body = {"readme": "r", "entry": "e", "materials": ["m"]}
    task = {"goal": "g", "version": "1", "inputs": [], "outputs": [], "body": full}
    assert validate(task, schema_for("task"), origin="x") == []


@pytest.mark.parametrize("kind", ["task", "validator"])
def test_both_kinds_ref_the_same_body(kind: str) -> None:
    """`closure` §2.6 and `validator` §6.1: a task's body and a validator's are
    deliberately the same thing — *"one answer in the system, not two"*."""
    assert schema_for(kind)["properties"]["body"]["$ref"] == "_common.schema.json#/$defs/body"


def test_task_of_returns_the_nested_object_itself() -> None:
    """Identity, not equality — every caller reads further into what it gets back.

    `closure`'s checks and `task_graph.Task.unfold` both do, and `unfold`
    compares identity. A reader that copied would be correct by `==` and wrong
    in use, which is the worst combination available.
    """
    task = {"goal": "collect a trace", "body": {"readme": "r"}, "inputs": [], "outputs": []}
    doc = {"name": "collect_trace", "agent": "tracer", "handoffs": [], "task": task}

    assert task_of(doc) is task


def test_task_of_reads_the_key_named_task() -> None:
    """Pinned separately, because a document with one nested object proves less
    than it looks: a reader under the wrong key would pass the identity test by
    finding the only mapping there is."""
    doc = {
        "name": "c",
        "agent": "a",
        "handoffs": [],
        "task": {"inputs": ["real"]},
        "spec": {"inputs": ["decoy"]},
    }

    assert task_of(doc) == {"inputs": ["real"]}


def test_task_of_is_tolerant_and_that_is_the_dangerous_direction() -> None:
    """`{}` for absent or wrong-typed, and the failure it must not hide is on record.

    `task_graph`'s `check_graph` shipped reading `doc["task"][...]` where it was
    already handed the inner spec: it found nothing under the wrong key, so every
    task looked like a leaf and both of its checks returned no problems for a
    catalogue that violated both. Green, and inert. An empty result from an
    accessor is not evidence of absence — the schema is the gate.
    """
    assert task_of({}) == {}
    assert task_of({"task": None}) == {}
    assert task_of({"task": "oops"}) == {}


def test_the_closure_schema_is_where_the_task_key_is_declared() -> None:
    """One writer for the key, and the schema is the other half of it.

    `closure.schema.json` `$ref`s `task.schema.json` through exactly this key,
    which is why reading it here adds no interpretation of a package's content:
    this package already declares that the key exists and what is under it.
    """
    assert schema_for("closure")["properties"]["task"]["$ref"] == "task.schema.json"
    assert "task" in schema_for("closure")["required"]


@pytest.mark.parametrize(
    ("label", "spec", "expected"),
    [
        ("no agent — the ordinary case", {}, None),
        ("a named agent spec", {"agent": "profiler"}, "profiler"),
        # `""` and a non-string are absences, not declarations nobody made.
        # `minLength: 1` means neither survives a document that passed the
        # schema, and `body`'s `entry: ""` is on record as the bug this shape
        # causes: an empty path read as *no entry*, and a programmatic validator
        # silently ran as agent-bodied.
        ("an empty name is an absence", {"agent": ""}, None),
        ("a non-string is an absence", {"agent": 3}, None),
    ],
)
def test_validator_agent_of(label: str, spec: dict, expected: str | None) -> None:
    """`validator` spec §8.2 row 1's reader, hosted here because `closure`
    resolves the name and may import only this package (§4.5).

    Not the forced-duplication criterion, despite looking like it: `validator`
    has **no caller** of their own accessor — `_bound_environment` reads
    `spec.agent` off the pydantic model. So there is one reader, not two. The
    argument is `body_of`'s instead: this package declares the key in
    `validator.schema.json`, so hosting its reader adds no interpretation of a
    package's content, and the alternative is `closure` hardcoding a key of
    `validator`'s document.
    """
    assert validator_agent_of(spec) == expected, label


def test_the_name_breaks_the_pattern_because_the_key_is_ambiguous() -> None:
    """`body_of`, `subgraph_of` and `task_of` are `<key>_of`; this one is not.

    `closure.agent_of` reads a **closure document** and returns `str`; this
    reads a **validator spec** and returns `str | None`. Both take a `Mapping`
    and neither raises on the wrong document, so a single `agent_of` exported
    from the leaf would hand back a plausible string for the wrong input. Inside
    one package that collision is aliasable; from the package everyone imports
    it is not.

    The `str | None` is the two specs disagreeing rather than an inconsistency:
    `agent` is required on a closure document and optional on a validator spec.
    """
    import spec_loader

    assert not hasattr(spec_loader, "agent_of"), (
        "an `agent_of` in the leaf would collide with `closure.agent_of`, which "
        "reads a different document and cannot fail on this one"
    )
    # `minLength: 1` on this key is asserted in `tests/interfaces/`, not here:
    # two other packages' correctness rests on it and neither of their suites
    # would notice it relaxed.
    assert "agent" not in schema_for("validator")["required"]

    # **`agent` left `closure`'s top-level `required` at rev. 10, and the
    # asymmetry this test names survives it.** A non-leaf has no executor, so
    # the key is reinstated by an `if`/`else` on `task.subgraph` rather than by
    # a flat `required` (`closure` spec §2.2). What made the two accessors
    # different was never that one key was required and the other optional — it
    # is that they read *different documents* and neither raises on the wrong
    # one. `closure.agent_of` still returns `str` for a leaf; this still returns
    # `str | None` for a validator.
    assert "agent" not in schema_for("closure")["required"]
    assert "agent" in schema_for("closure")["else"]["required"]


def test_an_accessor_is_annotated_with_the_document_it_reads() -> None:
    """`validator`'s finding, and it is this module's own argument one level down.

    `validator_agent_of` exists under that name because `closure.agent_of` reads
    a *different document* and neither raises on the wrong one — so the name has
    to say which. **The annotation is the other half of that job**, and it is the
    half a reader meets first, in a hover or a generated signature, before any
    docstring. It said `TaskSpec`.

    `body_of` had the same defect and nobody had named it: its own first line is
    *"a task's **or a validator's** declared body"*, because `_common.schema.json`
    has one `$defs.body` that both `$ref`. `TaskSpec` was the wrong half of the
    pair it serves.

    `Mapping[str, Any]` rather than a new alias, and that is not a shrug. A
    `ValidatorSpec` alias would collide with `validator.ValidatorSpec`, which is
    a **pydantic model** — the leaf would export a second importable name for a
    different thing, which is exactly the collision `validator_agent_of`'s name
    exists to avoid, one level up.
    """
    from spec_loader import protocols

    serves_one_document = {"subgraph_of": "TaskSpec", "task_of": "ClosureDoc"}
    serves_more_than_one = {"body_of", "validator_agent_of"}

    for name, alias in serves_one_document.items():
        annotation = _sole_parameter_annotation(getattr(protocols, name))
        assert annotation == alias, f"{name} reads only a {alias}"

    for name in serves_more_than_one:
        annotation = _sole_parameter_annotation(getattr(protocols, name))
        assert annotation == "Mapping[str, Any]", (
            f"{name} reads more than one kind of document, or one this package "
            f"does not model; an alias here would name the wrong one"
        )
        assert _sole_parameter_annotation(getattr(_access, name)) == annotation, (
            f"{name}: the declaration and the implementation disagree"
        )
    assert not hasattr(protocols, "ValidatorSpec"), (
        "a ValidatorSpec alias would collide with validator.ValidatorSpec, a pydantic model"
    )


def _sole_parameter_annotation(fn) -> str:
    import inspect

    (parameter,) = inspect.signature(fn).parameters.values()
    return str(parameter.annotation)
