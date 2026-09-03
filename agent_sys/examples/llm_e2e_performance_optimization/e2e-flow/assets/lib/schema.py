#!/usr/bin/env python3
"""The one schema loader, imported by producers and by validators alike.

Mission rule G2: *"所有结构化的文档，尽量有自己的 json schema, 该 schema 同时暴露
给 producer & validator"*. This module is the "同时暴露" half — one file, one
resolution rule, so a producer and the validator that grades it cannot be
looking at two different documents.

**The framework's `items_schema` does not do this job**, which is why this
exists. `handoff/content.py:184-197` validates a file or tree item by building
``{item_name: <filename string>}`` and checking *that*: the file's contents are
never read. It is an admission check at the seal boundary and it is never
exported to a body. Both facts were measured before this module was written.

Resolution, in one line and with no fallback to a private copy::

    <AGENT_SYS_TASK_PACKAGE | AGENT_SYS_DEMO_PACKAGE>/assets/schemas/<name>.schema.json

**Both variables, always.** A validator's *input* phase gets the GLOBAL
environment row and never ``AGENT_SYS_TASK_PACKAGE``; only the PRODUCER row
exports it. A body that reads one of the two works in testing and fails in a
phase — it has already cost one run.

Used as a library::

    from schema import validate, load
    validate("environment", doc)          # raises SchemaError with every problem

or from a shell body, which is the common case::

    python3 "$PKG/assets/lib/schema.py" --schema environment \\
            --doc "$OUT/items/env/environment.yaml"
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

__all__ = ["SchemaError", "package_root", "schema_path", "load", "validate", "main"]


class SchemaError(Exception):
    """A document did not validate. The message lists **every** problem.

    One problem at a time turns a five-field mistake into five runs. `jsonschema`
    can enumerate, so it does.
    """


def package_root() -> pathlib.Path:
    """The staged copy of this package, from whichever row exported it."""
    for var in ("AGENT_SYS_TASK_PACKAGE", "AGENT_SYS_DEMO_PACKAGE"):
        value = os.environ.get(var)
        if value:
            return pathlib.Path(value)
    # A body run outside a zone — a developer at a shell — still resolves,
    # because this file's own location is the package. This is a convenience for
    # authoring and is never the path a run takes.
    return pathlib.Path(__file__).resolve().parent.parent.parent


def schema_path(name: str) -> pathlib.Path:
    """`environment` -> `<package>/assets/schemas/environment.schema.json`.

    A name rather than a path, because the name is what a step yaml writes in
    `args.schema` and what a task readme quotes. A path would let a producer and
    its validator disagree by one directory and never notice.
    """
    if name.endswith(".schema.json"):
        name = name[: -len(".schema.json")]
    path = package_root() / "assets" / "schemas" / f"{name}.schema.json"
    if not path.is_file():
        have = sorted(p.name for p in path.parent.glob("*.schema.json")) if path.parent.is_dir() else []
        raise SchemaError(f"no schema {name!r} at {path} (have: {have})")
    return path


def _read_doc(path: pathlib.Path):
    """JSON or YAML, decided by suffix.

    YAML is the mission's preference for anything a person edits (M3.7.2 —
    能用 yaml/json 的尽量不用 markdown), and JSON is what a program writes. Both
    are the same document to a JSON Schema, so both are accepted and the
    difference never reaches a validator.
    """
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml  # a dependency of agent_sys already

        return yaml.safe_load(text)
    return json.loads(text)


def load(name: str) -> dict:
    return json.loads(schema_path(name).read_text())


def validate(name: str, doc) -> None:
    """Validate `doc` against the named schema, or raise with every problem.

    The `referencing` registry mirrors `agent_sys/spec_loader/validate.py:56`,
    so one schema may `$ref` another by filename — `environment.schema.json` is
    referenced from several and should be written once.
    """
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

    root = schema_path(name).parent
    # `default_specification=DRAFT202012` and not `None`: a resource whose own
    # contents do not carry `$schema` has nothing to detect from, and `None`
    # raises `AttributeError: 'NoneType' object has no attribute 'detect'` —
    # from inside the registry build, *before* any document is looked at, so it
    # fails identically for a good document and a bad one. Same line as
    # `spec_loader/bundled.py:84`.
    registry = Registry().with_resources(
        (p.name, Resource.from_contents(json.loads(p.read_text()), default_specification=DRAFT202012))
        for p in sorted(root.glob("*.schema.json"))
    )
    validator = Draft202012Validator(load(name), registry=registry)
    problems = sorted(validator.iter_errors(doc), key=lambda e: list(e.absolute_path))
    if problems:
        lines = [f"{name}: {len(problems)} problem(s)"]
        for err in problems:
            where = "$." + ".".join(str(p) for p in err.absolute_path) if err.absolute_path else "$"
            lines.append(f"  {where}: {err.message}")
        raise SchemaError("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate a document against one of this package's schemas.")
    ap.add_argument("--schema", required=True, help="schema name, e.g. `environment`")
    ap.add_argument("--doc", required=True, help="path to the .json/.yaml document")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    try:
        validate(a.schema, _read_doc(pathlib.Path(a.doc)))
    except SchemaError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if not a.quiet:
        print(f"ok: {a.doc} validates against {a.schema}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
