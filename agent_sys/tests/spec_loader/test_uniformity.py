"""Main spec criterion 1 — the four objects are uniform.

*"Each of handoff, validator, task, and agent has a static spec kind, a
uuid-identified runtime object, a runtime manager, a spec registry, and a JSON
Schema — demonstrated by instantiating one of each from a spec on disk."*

The loader owns two of those five: the static spec kind and the JSON Schema.
The runtime object and its manager are `task_graph`'s and each module's, so what
is demonstrated here is the half that is this package's — one spec of each kind,
on disk, through the whole pipeline, into its registry.
"""

from __future__ import annotations

import pytest

from spec_loader import KINDS, load_package, schema_for

from .conftest import ONE_OF_EACH, FakeRegistries, PackageBuilder


def test_four_kinds_instantiate_from_disk(
    builder: PackageBuilder, registries: FakeRegistries
) -> None:
    for module, (name, source) in ONE_OF_EACH.items():
        builder.one(module, name, source)

    report = load_package(builder.package(), registries)

    assert not report.problems, report.problems
    assert sorted(report.admitted) == [
        "check_trace_shape",
        "collect_trace",
        "main",
        "trace",
        "tracer",
    ]
    assert registries.handoff_specs.get("trace")["content_type"] == "reproducible"
    assert registries.validator_specs.get("check_trace_shape")["strength"] == "strong"
    assert registries.agent_specs.get("tracer")["kind"] == "program"
    assert registries.closures.get("collect_trace")["agent"] == "tracer"


def test_the_fifth_schema_is_the_closures_and_task_is_nested() -> None:
    """Five schemas, four discoverable kinds.

    A task spec is not independently loadable — `closure` spec §2 declares it as
    the closure's `task` key — so `closure.schema.json` `$ref`s the task schema
    and no package ships one alone. `docs/design.md` D2 records the same shape
    from the package side: four registry *objects* in three packages.

    **Four discoverable kinds and four words a user writes, and they are not the
    same four.** A package author writes `module: task` and the *closure* is what
    comes out; `closure` is a schema kind nobody types. The count matching is a
    coincidence worth naming, because it is the one that makes the mapping look
    like an identity.
    """
    assert set(KINDS) == {"handoff", "validator", "task", "agent", "closure"}
    assert schema_for("closure")["properties"]["task"]["$ref"] == "task.schema.json"


@pytest.mark.parametrize("kind", KINDS)
def test_every_schema_is_itself_a_valid_schema(kind: str) -> None:
    """The schemas are the enforcement, so an unusable one is a hole.

    `docs/design.md` §10 step 3: *"the schemas ARE the enforcement, so an
    over-permissive one is a hole nothing downstream can close."* An invalid one
    is worse — `jsonschema` would accept every document against it.
    """
    from jsonschema import Draft202012Validator

    Draft202012Validator.check_schema(schema_for(kind))


@pytest.mark.parametrize("kind", KINDS)
def test_every_schema_forbids_undeclared_keys(kind: str) -> None:
    """`additionalProperties: false` on every one of the five.

    Main spec §4.4 makes it one of the three mechanisms the schema layer is
    built from, and a schema that omits it silently admits whatever a template
    smuggled through.
    """
    assert schema_for(kind)["additionalProperties"] is False
