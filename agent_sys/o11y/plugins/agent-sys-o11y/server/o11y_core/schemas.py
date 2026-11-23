"""Protocol schemas: loading, validation and one-line problem translation."""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path
from typing import Any

import jsonschema

_HERE = Path(__file__).resolve()
# <plugin>/protocol is a symlink to agent_sys/o11y/protocol; the install cache's copy dereferences it.
SCHEMA_DIR = next(
    (
        d
        for d in (
            _HERE.parents[2] / "protocol" / "schemas",
            _HERE.parents[4] / "protocol" / "schemas",
        )
        if d.is_dir()
    ),
    _HERE.parents[2] / "protocol" / "schemas",
)


@functools.cache
def load(name: str) -> dict:
    """Load `<name>.schema.json`, e.g. load("runs.v1")."""
    with open(SCHEMA_DIR / f"{name}.schema.json", encoding="utf-8") as f:
        return json.load(f)


@functools.cache
def validator(name: str) -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(load(name), format_checker=jsonschema.FormatChecker())


def _where(err: jsonschema.ValidationError) -> str:
    return "/" + "/".join(str(p) for p in err.absolute_path) if err.absolute_path else "(record)"


def _one_line(err: jsonschema.ValidationError) -> str:
    """Translate one jsonschema error into a line a model can act on (never dumps the instance)."""
    where = _where(err)
    v = err.validator
    if v == "required":
        missing = err.message.split("'")[1] if "'" in err.message else err.message
        return f"{where}: missing required field '{missing}'"
    if v == "anyOf" and all(
        isinstance(s, dict) and set(s) == {"required"} for s in err.validator_value
    ):
        options = " or ".join("+".join(s["required"]) for s in err.validator_value)
        return f"{where}: needs {options}"
    if v in ("anyOf", "oneOf"):
        return f"{where}: does not match any allowed form ({len(err.validator_value)} alternatives)"
    if v == "enum":
        return f"{where}: {err.instance!r} is not one of {err.validator_value}"
    if v == "const":
        return f"{where}: must be {err.validator_value!r}"
    if v == "type":
        return f"{where}: must be of type {err.validator_value}, got {type(err.instance).__name__}"
    if v == "pattern" and err.validator_value.startswith("^\\d{4}"):
        return f"{where}: {err.instance!r} is not an RFC 3339 time (e.g. 2026-11-23T12:00:00Z)"
    if v == "unevaluatedProperties":
        names = re.findall(r"'([^']+)'", err.message.split("(")[-1])
        return f"{where}: unknown field(s) {', '.join(names)} for this kind"
    if v == "additionalProperties":
        return f"{where}: {err.message}"
    msg = err.message
    if len(msg) > 200:
        msg = msg[:200] + "…"
    return f"{where}: {msg}"


def problems(name: str, instance: Any, limit: int = 20) -> list[str]:
    """All validation problems of `instance` against schema `name`, one line each, deduplicated."""
    errs = sorted(
        validator(name).iter_errors(instance),
        key=lambda e: (list(map(str, e.absolute_path)), e.validator),
    )
    # A failing per-kind `then` branch drops its annotations, which makes every field of that kind look
    # "unevaluated"; report unknown fields only once the record is otherwise valid.
    if any(e.validator != "unevaluatedProperties" for e in errs):
        errs = [e for e in errs if e.validator != "unevaluatedProperties"]
    out: list[str] = []
    for e in errs:
        line = _one_line(e)
        if line not in out:
            out.append(line)
    context = ""
    if isinstance(instance, dict) and "kind" in instance:
        context = f"kind={instance.get('kind')}"
        if "status" in instance:
            context += f" status={instance.get('status')}"
        context += ": "
    if name == "runs.v1" and isinstance(instance, dict):
        out = [p + _suggest(instance.get("kind"), p) for p in out]
    return [context + p for p in out[:limit]]


def _nested_paths(node: dict, defs: dict, prefix: str, depth: int = 0) -> list[str]:
    """Dotted paths of nested fields of a kind, e.g. machines[].node (for 'did you mean' hints)."""
    if depth > 3 or not isinstance(node, dict):
        return []
    if "$ref" in node:
        node = defs.get(node["$ref"].split("/")[-1], {})
    out = []
    items = node.get("items")
    if isinstance(items, dict):
        return _nested_paths(items, defs, prefix + "[]", depth + 1)
    for k, sub in (node.get("properties") or {}).items():
        path = f"{prefix}.{k}" if prefix else k
        out.append(path)
        out += _nested_paths(sub, defs, path, depth + 1)
    return out


KIND_DEF = {"machine": "machine_event"}


def _suggest(kind, problem: str) -> str:
    if "unknown field(s)" not in problem or not isinstance(kind, str):
        return ""
    defs = load("runs.v1")["$defs"]
    kdef = defs.get(KIND_DEF.get(kind, kind))
    if not kdef:
        return ""
    names = problem.split("unknown field(s) ")[1].split(" for this kind")[0].split(", ")
    paths = [p for p in _nested_paths(kdef, defs, "") if "." in p or "[]" in p]
    hints = [f"{n} -> {p}" for n in names for p in paths if p.endswith("." + n)]
    return (
        f"; did you mean {', '.join(hints[:3])}?"
        if hints
        else "; see ledger_schema(kind) for the fields"
    )


def defaults(name: str) -> dict:
    """Default document for a config schema: every property's `default`, recursively for objects."""

    def fill(node: dict) -> Any:
        if "default" in node:
            return node["default"]
        if node.get("type") == "object" and "properties" in node:
            return {
                k: fill(p)
                for k, p in node["properties"].items()
                if "default" in p or p.get("type") == "object"
            }
        return None

    doc = fill({"type": "object", "properties": load(name)["properties"]})
    doc["v"] = 1
    return doc
