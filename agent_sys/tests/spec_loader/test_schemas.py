"""Main spec criterion 3 — the schema is the only enforcement point.

*"A `const` field given another value is rejected, and an undeclared field
smuggled in is rejected — whatever the package did to produce the document
(§4.4)."*

**Amended at rev. 10, and the tests moved with it.** They said *"a template's
`const` field overridden after rendering"* and rendered real jsonnet, because
the property was about an ordering: render first, then check. There is no
render. The property being pinned never depended on one — it is that the check
runs over the **delivered document**, whatever produced it — so both go through
the real YAML front end instead, which is what a package delivers now.

The expected messages are `docs/design.md` §4.2's, measured against `jsonschema`
4.26.0. They are asserted verbatim because the criterion names them.
"""

from __future__ import annotations

from spec_loader import load_package, validate

from .conftest import FakeRegistries, PackageBuilder

#: A contract that seals `kind`. `const` is what a repository-owned schema uses
#: for a field a package must not change; a per-kind schema in `schemas/` uses a
#: tight `enum` instead, because `content_type` legitimately varies per kind and
#: `const` cannot express a per-document decision in a per-kind file.
SEALED = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "kind": {"const": "reproducible", "description": "Sealed by the contract."},
        "name": {"type": "string"},
    },
    "required": ["kind", "name"],
}


def test_const_override_rejected() -> None:
    """A sealed field given another value does not get through.

    The document is what a package delivered — it is well formed, it parsed, and
    nothing upstream objected. The schema is the gate and the only one.
    """
    problems = validate({"kind": "text", "name": "trace"}, SEALED, origin="steps.yaml#/0")

    (problem,) = problems
    assert problem.path == "$.kind"
    assert problem.keyword == "const"
    assert problem.message == "'reproducible' was expected"


def test_undeclared_field_rejected(builder: PackageBuilder, registries: FakeRegistries) -> None:
    """A field nobody declared is rejected.

    Through the whole pipeline and against the real `handoff.schema.json`, so
    this is also the check that every shipped schema carries
    `additionalProperties: false` where it counts.

    **The smuggling route changed with the format and the test is stronger for
    it.** It used to be a jsonnet overlay — `(import "t.libsonnet") + {sneak: …}`
    — which is a thing only jsonnet could do. Now it is an ordinary key in an
    ordinary document, which is the case every package can produce, so what is
    demonstrated is no longer "the overlay did not escape the render" but "the
    schema is what stops it".
    """
    path = builder.write(
        "kinds.yaml",
        "module: handoff\nname: trace\ndescription: d\n"
        "content_type: text\nscope: fixed.required\nsneak: past the schema\n",
    )

    result = load_package(builder.package(), registries)

    assert "trace" not in result.admitted
    (problem,) = [p for p in result.problems if p.keyword == "additionalProperties"]
    assert problem.origin == str(path)
    assert problem.message == "Additional properties are not allowed ('sneak' was unexpected)"


def test_a_missing_required_field_is_rejected_naming_it() -> None:
    """`docs/design.md` §4.2's third row, and §4.5's honest limit.

    `required` catches a field a package left *absent*. It does not catch one
    filled with `"TODO"` or `""` — the document is then structurally valid and
    the loader admits it. That is main design O9, and it is open: a presence
    check cannot tell a value from a placeholder, as Hugging Face measured with
    `[More Information Needed]` in 636,321 repositories.
    """
    problems = validate({"kind": "reproducible"}, SEALED, origin="x.yaml")

    assert [p.message for p in problems] == ["'name' is a required property"]


def test_a_placeholder_is_admitted_and_that_is_open() -> None:
    """The other half of O9, asserted so the limit is visible rather than assumed.

    If this test ever fails, somebody closed O9 — and the schemas, not this
    file, are where that decision belongs.
    """
    assert validate({"kind": "reproducible", "name": "TODO"}, SEALED, origin="x") == []


# --------------------------------------------------------------------------- #
# Shapes another module owns.
#
# The five schemas live here and four modules own what is in them. These pin the
# shapes those owners settled, so that a later drift shows up as a failing test
# naming the owner rather than as a package that mysteriously stops loading.

import pytest  # noqa: E402

from spec_loader import report, schema_for  # noqa: E402

VALIDATOR = {
    "name": "check_trace_shape",
    "brief": "every kernel in the trace has a recorded shape",
    "dimension": "completeness",
    "strength": "strong",
    "inputs": ["trace"],
    "tags": {"logic_source": "external_static", "cost": "seconds"},
}
TASK = {"goal": "g", "body": {"readme": "readme.md"}, "version": "1", "inputs": [], "outputs": []}


def _check(kind: str, doc: dict) -> list:
    return validate(doc, schema_for(kind), origin="x.yaml")


@pytest.mark.parametrize(
    ("label", "extra"),
    [
        # `validator` criterion 12: two validators differing only in a parameter
        # are two registry entries over one implementation. Design §10.6 puts the
        # parameters in the spec, reaching the body as `args.json` in its zone —
        # so under `additionalProperties: false` an absent `args` fails every
        # parameterised validator there is.
        ("args", {"body": {"readme": "r"}, "args": {"threshold": 0.95, "n": {"k": [1, 2]}}}),
        # Top-level beside `reduce`, per `ValidatorSpec` (design §3.2). Spec
        # §6.2's `composite(validators=[...])` is function-call prose.
        ("members", {"members": ["a", "b"], "reduce": "at_least(2)"}),
        # Design §3.5: a tag dictionary is where a site adds its own key, while a
        # stray top-level field is a typo. The asymmetry is the point.
        (
            "a site's own tag",
            {
                "body": {"readme": "r"},
                "tags": {
                    "logic_source": "external_static",
                    "cost": "seconds",
                    "site_owner": "perf",
                },
            },
        ),
    ],
)
def test_the_validator_schema_admits_what_validator_owns(label: str, extra: dict) -> None:
    assert _check("validator", {**VALIDATOR, **extra}) == [], label


@pytest.mark.parametrize(
    ("label", "extra", "path"),
    [
        # A bare string is iterable, so it passes a presence check and then
        # iterates as characters; `Tags.domain` is `tuple[str, ...]`, which
        # pydantic refuses a string for. Admitting one here would let a document
        # through this gate that the next gate rejects.
        (
            "domain must be a list",
            {"tags": {"logic_source": "external_static", "cost": "seconds", "domain": "trace"}},
            "$.tags.domain",
        ),
        # §5.3 orders a phase's validators by cost, cheap-first. Free text cannot
        # be ordered, so a typo would become an unordered validator rather than a
        # load error.
        (
            "cost is a closed set",
            {"tags": {"logic_source": "external_static", "cost": "a while"}},
            "$.tags.cost",
        ),
        ("a composite of one is a leaf", {"members": ["a"], "reduce": "all"}, "$.members"),
        ("a stray top-level field", {"sneak": 1}, "$"),
    ],
)
def test_the_validator_schema_rejects_what_validator_forbids(
    label: str, extra: dict, path: str
) -> None:
    problems = _check("validator", {**VALIDATOR, "body": {"readme": "r"}, **extra})
    assert problems, label
    assert problems[0].path == path, report(problems, verbose=True)


@pytest.mark.parametrize(
    ("label", "subgraph", "path"),
    [
        # `is_start` / `is_end` are optional: absent, the first entry is the
        # start and the last is the end — the only reading that makes a
        # one-entry subgraph well formed, so a chain needs no marks at all.
        # `froms` is required on every entry (rev. 10, `closure` spec §2.7):
        # `[]` is how an entry says it has no predecessor, and omitting the key
        # is how an author forgets to think about it.
        (
            "a chain needs no marks",
            [{"closure": "a", "froms": []}, {"closure": "b", "froms": ["a"]}],
            None,
        ),
        ("marks are allowed", [{"closure": "a", "is_start": True, "froms": []}], None),
        ("froms is required", [{"closure": "a"}], "$.subgraph[0]"),
        # `additionalProperties: false` matches `Model`'s `extra="forbid"`. A
        # misspelled mark must be a load-time problem *with a path*, not a
        # silently unmarked entry.
        ("a misspelled mark", [{"closure": "a", "froms": [], "is_stat": True}], "$.subgraph[0]"),
        ("closure is required", [{"is_start": True, "froms": []}], "$.subgraph[0]"),
    ],
)
def test_the_subgraph_entry_shape_is_task_graphs(
    label: str, subgraph: list, path: str | None
) -> None:
    problems = _check("task", {**TASK, "subgraph": subgraph})
    if path is None:
        assert problems == [], label
    else:
        assert problems and problems[0].path == path, report(problems, verbose=True)


AGENT = {"name": "scribe", "kind": "ai"}


@pytest.mark.parametrize(
    ("label", "extra"),
    [
        # `agent` design §3.2. `kind` is the knowledge handoff KIND's name, not a
        # handoff's — the earlier guess here spelled it `handoff`, which reads as
        # "the handoff called trace" and means the wrong thing. `required` is the
        # only key that reconciles criteria 1 and 2: a required ref whose kind
        # does not resolve is fatal in both modes, a non-required one only under
        # the run-config flag.
        (
            "knowledge",
            {
                "knowledge": [
                    {"kind": "trace_kb", "knowledge_type": "few_shot", "required": True},
                    {"kind": "docs", "knowledge_type": "official_reference"},
                ]
            },
        ),
        ("backends as a list", {"backends": [{"key": "sdk", "backend_entry": "a.b:C"}]}),
        # Spec §3.1 says "a list or dict"; design D2 normalises the mapping to a
        # list preserving declaration order. An array-only schema would reject it
        # before the normaliser ever ran — the feature dead on arrival.
        ("backends as a mapping", {"backends": {"sdk": {"backend_entry": "a.b:C"}}}),
        # matplotlib's entry-point rule: an identical duplicate is tolerated and a
        # differing one is an error, because duplicates arise from packaging
        # outside the declarer's control. `uniqueItems` would reject the
        # identical case too, so `backends` deliberately has none.
        (
            "an identical duplicate key",
            {
                "backends": [
                    {"key": "k", "backend_entry": "a:B"},
                    {"key": "k", "backend_entry": "a:B"},
                ]
            },
        ),
        ("a program agent needs no backends", {"kind": "program"}),
    ],
)
def test_the_agent_schema_admits_what_agent_owns(label: str, extra: dict) -> None:
    assert _check("agent", {**AGENT, **extra}) == [], label


@pytest.mark.parametrize(
    ("label", "extra", "path"),
    [
        (
            "the old guess at knowledge",
            {"knowledge": [{"handoff": "t", "type": "few_shot"}]},
            "$.knowledge[0]",
        ),
        ("knowledge without a type", {"knowledge": [{"kind": "trace_kb"}]}, "$.knowledge[0]"),
        # `backend_entry` is what makes a declaration resolvable at all. A spec
        # with `key` alone would pass here and fail the model, putting the error
        # one layer past the enforcement point.
        ("a backend without an entry", {"backends": [{"key": "k"}]}, "$.backends[0]"),
        (
            "a mapping entry repeating its key",
            {"backends": {"k": {"key": "k", "backend_entry": "a:B"}}},
            "$.backends.k",
        ),
    ],
)
def test_the_agent_schema_rejects_what_agent_forbids(label: str, extra: dict, path: str) -> None:
    problems = _check("agent", {**AGENT, **extra})
    assert problems, label
    assert problems[0].path == path, report(problems, verbose=True)


def test_permissions_are_not_on_the_agent_spec() -> None:
    """`agent` criterion 5, and it is structural rather than checked.

    Permission is a runtime fact about a particular piece of work, not a property
    of a kind of executor (`agent` spec §3.2) — the same agent spec running two
    tasks should reach two different sets of files. `additionalProperties: false`
    here is half of what enforces that; `extra="forbid"` on the model is the
    other half.
    """
    assert "permissions" not in schema_for("agent")["properties"]
    problems = _check("agent", {**AGENT, "permissions": {"grants": []}})
    assert problems and problems[0].keyword == "additionalProperties"


@pytest.mark.parametrize(
    ("label", "doc", "path"),
    [
        ("no tags at all", {}, "$"),
        ("tags without logic_source", {"tags": {"cost": "seconds"}}, "$.tags"),
        ("tags without cost", {"tags": {"logic_source": "external_static"}}, "$.tags"),
    ],
)
def test_tags_and_two_of_its_fields_are_required(label: str, doc: dict, path: str) -> None:
    """`validator` settled this, and the reason is stronger than the one I gave.

    I argued §5.3's cheap-first ordering needs `cost`. The decisive argument is
    what optional would *cost*: `Tags` cannot be constructed empty, because
    `logic_source` and `cost` have no defaults. So an optional container means
    either a default `logic_source` — a default **trust** claim, which §9.3
    check 2 forbids in the words *"an unlabelled validator would default to
    being trusted"* — or `None`, which makes §5.3's ordering crash rather than
    lose a value.

    The two required fields are checked here too, because requiring the
    container while leaving its required fields optional moves the same
    two-gates gap one level deeper instead of closing it: a document with
    `tags: {domain: [...]}` would pass this schema and fail the model.
    """
    base = dict(VALIDATOR, body={"readme": "r"})
    base.pop("tags")
    problems = _check("validator", {**base, **doc})

    assert problems, label
    assert problems[0].path == path, report(problems, verbose=True)


@pytest.mark.parametrize(
    ("label", "doc", "path"),
    [
        # Absent is legal and takes §8.2's global row — three of the four rows
        # are reached by *not* naming one, so requiring it would make them
        # unreachable, which is the opposite of the gap this key closes.
        ("absent takes the global row", {}, None),
        ("a named agent spec", {"agent": "reviewer"}, None),
        # `minLength: 1` because an empty path was already a live bug in this
        # schema's `body`: `entry: ""` read as *no entry*, and a programmatic
        # validator silently ran as agent-bodied. An empty name is a fault, not
        # an absence.
        ("an empty name is a fault", {"agent": ""}, "$.agent"),
        ("a list is not a name", {"agent": ["reviewer"]}, "$.agent"),
    ],
)
def test_a_validator_may_name_an_agent_spec(label: str, doc: dict, path: str | None) -> None:
    """`validator` spec §8.2 row 1 — *"bound to a real agent with a declared
    environment: that one"*.

    A **name**, not an inline environment block: `agent.env` already declares
    the environment, so a second copy here would be two writers of one fact.
    That reading was mine, `agent-mod` assented, and `closure` owns the resolve
    because the agent registry may not be loaded when this document is.

    Step 2 of three. It could not land until `validator` added `_PENDING` to
    their conformance test, which asserts exact set equality between this
    schema's keys and their model's fields — so either side moving alone is a
    red shared suite, symmetrically.
    """
    base = dict(VALIDATOR, body={"readme": "r"})
    problems = _check("validator", {**base, **doc})

    if path is None:
        assert problems == [], label
    else:
        assert problems and problems[0].path == path, report(problems, verbose=True)


def test_agent_is_not_required_and_the_absence_is_the_point() -> None:
    """A required `agent` would force every validator to declare a binding it may
    not have, and would make §8.2's consumer, producer and global rows
    unreachable."""
    assert "agent" not in schema_for("validator")["required"]
