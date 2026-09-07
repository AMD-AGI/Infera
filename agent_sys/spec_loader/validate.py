"""A document plus a schema to a list of faults. **No path reaches this module.**

`docs/design.md` §3.1: this signature is where main spec §4.4's *"the loader does
not read, audit, or constrain a package's"* source stops being an assertion and
becomes a property of the code. There is no parameter through which a path could
arrive, so a future maintainer who wants to read a package's source cannot do it
by accident. `tests/spec_loader/test_validate.py::test_validate_takes_no_path`
guards it, because a claim of that kind should fail loudly when someone adds a
convenience overload.

**It parsed, and it no longer does.** Through rev. 9 the parameter was `bytes`
and this module ran PyYAML's `safe_load` over it, on the argument that *"rendered
jsonnet is a YAML subset by construction"*. There is no render, the package
parses its own documents with `ruamel.yaml`, and a second parse here would be a
second reading of the same file: measured, the two disagree on ordinary scalars —
`12:30` is the string under `ruamel`'s YAML 1.2 and the integer 750 under
PyYAML's 1.1 (`scratch/ui-yaml-2026-08/w3/probe_ruamel_semantics.py`). One
document must have one reading, so the parse lives in exactly one place and it is
not here.

Path-free is what criterion 4 rests on and is unchanged; `bytes` was never the
point, and a parsed document is one step *further* from a path than bytes were.

`origin` is an opaque label used only in messages, never opened. cdk8s solves the
same problem the same way — provenance joined to violations after the plugin
returns.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError, best_match, relevance

from .bundled import bundled_registry
from .protocols import Problem

__all__ = ["validate"]


def validate(doc: Any, schema: Mapping[str, Any], *, origin: str) -> list[Problem]:
    """Check `doc` against `schema`, and return every distinct fault.

    Faults are *returned*, never raised: `check-jsonschema` returns parse failure
    as a value for the same reason, and a loader that dies on the first bad spec
    makes fixing a package an N-round trip (`docs/design.md` §3.6).

    The document may be a `ruamel.yaml` `CommentedMap` and is validated as it
    stands, not converted. Measured: `CommentedMap` and `CommentedSeq` subclass
    `dict` and `list`, so every `type` keyword and every `json_path` is correct
    over the position-carrying tree — which is why nothing is lost between the
    parse and here.
    """
    validator = Draft202012Validator(schema, registry=bundled_registry())
    errors = list(validator.iter_errors(doc))
    if not errors:
        return []

    return _to_problems(errors, origin=origin)


def _to_problems(errors: Sequence[ValidationError], *, origin: str) -> list[Problem]:
    """Best match first, the deep match second when it differs, then the rest.

    `docs/design.md` §3.5 measured why de-duplication is not cosmetic. Against a
    schema shaped like `handoff`'s, `$ref`-ing the 2020-12 metaschema to check a
    nested user-supplied schema produced **eight identical errors** for
    `"notaschema"` — the metaschema's `anyOf` branches each failing the same way
    — and a package author reading eight copies of `is not of type 'object',
    'boolean'` learns nothing. Both `best_match` and dedupe-by-`(path, keyword,
    message)` collapse it to one, and `best_match` picked the correct error in
    every case measured.

    Two heuristics rather than one, and that is adopted rather than invented:
    `check-jsonschema` shipped `_deep_match_relevance` *after* finding stock
    `best_match` insufficient. Stock relevance minimises depth at the top level
    and then descends; the deep one maximises `len(absolute_path)` over a
    flattened recursive walk. Two guesses plus an escape hatch is the state of
    the art, and a third guess invented here would not be an improvement.
    """
    ordered: list[ValidationError] = []

    shallow = best_match(errors, key=relevance)
    if shallow is not None:
        ordered.append(shallow)

    deep = _best_deep_match(errors)
    if deep is not None and deep is not shallow:
        ordered.append(deep)

    ordered.extend(errors)

    seen: set[tuple[str, str, str]] = set()
    out: list[Problem] = []
    for err in ordered:
        problem = Problem(
            origin=origin,
            path=err.json_path,
            keyword=str(err.validator),
            message=err.message,
        )
        key = (problem.path, problem.keyword, problem.message)
        if key in seen:
            continue
        seen.add(key)
        out.append(problem)
    return out


def _best_deep_match(errors: Sequence[ValidationError]) -> ValidationError | None:
    """The deepest error in the flattened tree.

    `check-jsonschema`'s second heuristic. `ValidationError.context` holds the
    sub-errors of an `anyOf` / `oneOf` branch, and the one that addresses the
    most specific location is usually the one the author meant to be told about.
    """
    best: ValidationError | None = None
    for err in _flatten(errors):
        if best is None or len(err.absolute_path) > len(best.absolute_path):
            best = err
    return best


def _flatten(errors: Sequence[ValidationError]) -> list[ValidationError]:
    out: list[ValidationError] = []
    for err in errors:
        out.append(err)
        if err.context:
            out.extend(_flatten(err.context))
    return out
