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


def _inline_refs(node, root: pathlib.Path, seen: frozenset[str] = frozenset()):
    """Replace every `{"$ref": "<file>.schema.json"}` with that file's contents.

    **One code path, no optional dependency, and that is the point.**

    The first version of this module used a `referencing` registry, copied from
    `spec_loader/validate.py:56`. That works for a *validator*, which runs under
    the interpreter `AGENT_SYS_DEMO_PYTHON` names, and **not for a task body**:
    `cli/main.py:668` puts that variable in `validation_env` only, and its own
    comment says a task body never reaches it. So a mock body's policy `PATH`
    resolves `python3` to `/usr/bin/python3`, which on this host has `yaml` and
    `jsonschema` and **no `referencing`** — and `env_render.py` validates before
    it writes, so every module's MOCK-MAP (A) rendering died there with a
    `ModuleNotFoundError`. Reported by m1, measured in-zone 2026-09-03.

    The obvious repair — try `referencing`, else validate without a registry —
    was written here and **its stated justification was false**. It claimed no
    schema in `assets/schemas/` uses `$ref`. Measured: three sites do, in
    `kernel_optimization.schema.json` (two) and `workset.schema.json` (one), all
    to `environment.schema.json`. Under the interpreter that lacks
    `referencing`, those two schemas would have validated **without resolving
    the reference** — a silently weaker check, wearing a comment saying it was
    equivalent. That is the exact failure this package is built against, so the
    fallback is gone rather than corrected.

    Inlining is safe *and checked*, not assumed: the only cross-file target,
    `environment.schema.json`, carries no `$defs` and no internal `#/` pointer,
    so it has nothing whose meaning could change by being moved. A target that
    did would need its pointers rewritten, and `seen` makes a cycle an error
    rather than a hang.
    """
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and not ref.startswith("#") and ref.endswith(".schema.json"):
            if ref in seen:
                raise SchemaError(f"$ref cycle through {ref}")
            target = root / ref
            if not target.is_file():
                raise SchemaError(f"$ref {ref!r} names no file in {root}")
            loaded = json.loads(target.read_text())
            inner = json.dumps(loaded)
            if '"#/' in inner:
                raise SchemaError(
                    f"{ref} carries an internal '#/' pointer, so inlining it would "
                    f"change what that pointer resolves against. Rewrite the pointers "
                    f"or restore a $ref registry."
                )
            merged = {k: v for k, v in loaded.items() if k not in ("$schema", "$id")}
            # Sibling keys beside a `$ref` are allowed in 2020-12 and must survive.
            merged.update({k: v for k, v in node.items() if k != "$ref"})
            return _inline_refs(merged, root, seen | {ref})
        return {k: _inline_refs(v, root, seen) for k, v in node.items()}
    if isinstance(node, list):
        return [_inline_refs(v, root, seen) for v in node]
    return node


def _inlined(name: str) -> dict:
    return _inline_refs(load(name), schema_path(name).parent)


def validate(name: str, doc) -> None:
    """Validate `doc` against the named schema, or raise with every problem.

    Cross-file `$ref`s are inlined first (`_inline_refs`), so this needs nothing
    beyond `jsonschema` and behaves identically under the run's interpreter and
    under a bare `/usr/bin/python3`.
    """
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(_inlined(name))
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
