"""Main spec criterion 2, plus the two `docs/design.md` §9.2 tests `validate` owns.

Criterion 2: *"A spec that violates its schema is rejected at load time, with the
file path and the offending field in the message. It does not fail later, at
use."*
"""

from __future__ import annotations

import inspect
from pathlib import Path

from spec_loader import load_package, report, validate

from .conftest import HANDOFF, ONE_OF_EACH, FakeRegistries, PackageBuilder

# --------------------------------------------------------------------------- #
# Criterion 2


def test_rejects_with_path_and_field(builder: PackageBuilder, registries: FakeRegistries) -> None:
    """The rejection names the file and the field, and happens at load.

    `$.scope` is where the fault is and `scope` is the field, so both halves of
    the criterion are in one message. Nothing is admitted, which is the "does not
    fail later, at use" half: a spec that never entered a registry cannot be
    reached by anything downstream.
    """
    path = builder.write(
        "kinds.yaml",
        "module: handoff\nname: trace\ndescription: d\ncontent_type: text\nscope: sideways\n",
    )

    result = load_package(builder.package(), registries)

    assert "trace" not in result.admitted
    (problem,) = [p for p in result.problems if p.origin == str(path)]
    assert problem.origin == str(path)
    assert problem.path == "$.scope"
    assert problem.keyword == "enum"
    assert "trace" not in registries.handoff_specs


def test_a_broken_spec_does_not_hide_the_others(
    builder: PackageBuilder, registries: FakeRegistries
) -> None:
    """Failures are collected, not raised (`docs/design.md` §3.6).

    A loader that dies on the first bad spec makes fixing a package an N-round
    trip. The three good specs are still admitted, and the bad one is still
    reported.
    """
    for module, (name, source) in ONE_OF_EACH.items():
        builder.one(module, name, source)
    builder.write(
        "broken.yaml", "module: handoff\nname: broken\ndescription: d\ncontent_type: text\n"
    )

    result = load_package(builder.package(), registries)

    assert sorted(result.admitted) == [
        "check_trace_shape",
        "collect_trace",
        "main",
        "trace",
        "tracer",
    ]
    assert [p.keyword for p in result.problems] == ["required"]
    assert "broken" not in registries.handoff_specs


def test_a_parse_failure_travels_on_its_own_keyword(
    builder: PackageBuilder, registries: FakeRegistries
) -> None:
    """ "This is not a document" and "this is the wrong document" are different.

    `check-jsonschema` returns parse failure as a value for the same reason, so
    the two never share a channel.

    **It moved across the seam and kept its keyword.** `validate` used to parse
    and owned this; the package parses now, so the test goes through
    `load_package` and asserts that the distinction survived the move rather
    than that `validate` still makes it — which it cannot, having nothing to
    parse.
    """
    broken = builder.write("broken.yaml", "{: not yaml")
    builder.asset("trace.md", "x")
    builder.write("good.yaml", f"module: handoff\nname: trace\n{HANDOFF}")

    result = load_package(builder.package(), registries)

    assert [p.keyword for p in result.problems if p.origin == str(broken)] == ["parse"]
    assert "trace" in result.admitted


# --------------------------------------------------------------------------- #
# docs/design.md §9.2 — the boundary is the signature, so the signature is tested


def test_validate_takes_no_path() -> None:
    """`docs/design.md` §3.1, and the one test in this suite about a type signature.

    Main spec §4.4 says the loader does not read, audit, or constrain a package's
    source. This function is where that stops being an assertion: there is no
    parameter through which a path could arrive, so a maintainer who wants to
    read a package's source cannot do it by accident.

    It exists because a claim of that kind should fail loudly when someone adds a
    convenience overload. If you are here because it failed, the fix is not to
    widen the assertion.

    **`data: bytes` became `doc: Any` at rev. 10 and the assertion followed.**
    Path-free is what criterion 4 rests on and is what is checked below; `bytes`
    was never the point, and a parsed document is one step *further* from a path
    than bytes were. Keeping `bytes` would have meant serialising a parsed
    document so this function could parse it again with a **different** parser —
    PyYAML is YAML 1.1, `ruamel.yaml` is 1.2, and they disagree on ordinary
    scalars.
    """
    signature = inspect.signature(validate)

    assert list(signature.parameters) == ["doc", "schema", "origin"]
    assert signature.parameters["doc"].annotation == "Any"
    assert signature.parameters["origin"].annotation == "str"
    assert signature.parameters["origin"].kind is inspect.Parameter.KEYWORD_ONLY

    for name, parameter in signature.parameters.items():
        annotation = str(parameter.annotation)
        assert "Path" not in annotation, f"{name}: a path can reach validate through {annotation}"
        assert "PathLike" not in annotation, f"{name}: a path can reach validate"
        assert parameter.kind is not inspect.Parameter.VAR_KEYWORD, (
            f"{name}: **kwargs is a channel through which a path could arrive"
        )
    assert Path not in (p.annotation for p in signature.parameters.values())


def test_nested_schema_error_is_one_line() -> None:
    """`docs/design.md` §3.5 — one actionable message, not eight.

    A field carrying a nested user-supplied schema `$ref`-ed to the 2020-12
    metaschema produces one error per `anyOf` branch, and they are identical: a
    package author reading eight copies of `is not of type 'object', 'boolean'`
    learns nothing. Both `best_match` and dedupe-by-`(path, keyword, message)`
    collapse it to one.

    This is why the real `handoff.schema.json` types `items_schema` as
    `{"type": "object"}` and leaves its validity to a named `check_schema` call.
    The `$ref` form is reproduced here because the collapsing is the property
    being guarded, and it must keep working for every other `anyOf` in the set.
    """
    metaschema_ref = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"items_schema": {"$ref": "https://json-schema.org/draft/2020-12/schema"}},
    }

    problems = validate(
        {"items_schema": "notaschema"}, metaschema_ref, origin="handoffs/trace.yaml"
    )

    assert len(problems) == 1, report(problems, verbose=True)
    assert problems[0].path == "$.items_schema"
    assert report(problems).count("\n") == 0
