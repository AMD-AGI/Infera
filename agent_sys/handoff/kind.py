"""`HandoffKind` — a handoff spec in the shape the store and the checks need.

Spec §8 declares a kind in a task package as jsonnet, rendered and validated
against the handoff JSON Schema. **That schema is not in this package**: main
design §2.2 puts all five in `spec_loader/schemas/`, read through
`importlib.resources`, because a bare directory of `.json` at the `agent_sys/`
top level is not installable. So the declarative pass is `spec_loader`'s, and
what is here is the kind's **own** load-time checks — the ones §4.3 of that
document says a declarative layer expresses badly.

Spec §8 lists five checks. Their homes:

| # | Check | Runs |
|---|---|---|
| 1 | the name is unique | `SpecRegistry.add` — the base class's |
| 2 | `items_schema` is itself a valid schema | here, `check_schema` as a named step |
| 3 | every named validator resolves | the **closure pass** — the validator registry may not be loaded yet |
| 4 | at least one validator, or the flag | here |
| 5 | `reproducible` + `script`/`command` ⇒ `env` | here |
"""

from __future__ import annotations

from collections.abc import Mapping

from handoff.content import check_items_schema, content_type
from handoff.errors import Malformed
from handoff.protocols import HandoffKind, Scope

__all__ = ["REQUIRED_KEYS", "from_spec", "permits_exec_without_env"]

#: The keys a kind must carry for a `HandoffKind` to exist at all — this is the
#: constructor refusing to build a value it cannot represent, which is why it
#: names the key it wanted.
#:
#: **The set is the schema's, minus `description`.** `spec_loader/schemas/
#: handoff.schema.json` requires `name`, `description`, `content_type` and
#: `scope`; `description` is for a human and is not a field of `HandoffKind`,
#: so nothing here can check it and the schema is its only writer.
#:
#: `items_schema` and `validators` are deliberately **not** here even though a
#: kind needs both. An absent `validators` must reach check 4 and be *reported*
#: — that is the whole of spec §5.3's escape hatch — and a construction error
#: would turn a reportable finding into a crash one layer too early.
REQUIRED_KEYS = ("name", "content_type", "scope")


def from_spec(spec: Mapping[str, object], *, origin: str) -> HandoffKind:
    """Build a `HandoffKind`, or raise `Malformed` naming the origin and key.

    `version` is **maintenance metadata only** — nothing at runtime reads it
    (closure spec §1.2) — and it is carried so that a human diffing two kinds
    can see which revision an artefact was produced against.
    """
    for key in REQUIRED_KEYS:
        if key not in spec:
            raise Malformed(f"{origin}: $.{key} is a required property")

    name = spec["name"]
    ctype = spec["content_type"]
    if not isinstance(name, str) or not name:
        raise Malformed(f"{origin}: $.name must be a non-empty string")
    if not isinstance(ctype, str):
        raise Malformed(f"{origin}: $.content_type must be a string")
    content_type(ctype)  # raises Malformed naming the four that exist

    try:
        scope = Scope(spec["scope"])
    except ValueError:
        raise Malformed(
            f"{origin}: $.scope {spec['scope']!r} is not one of {[s.value for s in Scope]}"
        ) from None

    # Absent is empty, and empty is what the checks are for: an empty schema
    # permits anything, so check 5 fires for a `reproducible` kind, and an
    # empty validator list is exactly what check 4 reports.
    items_schema = spec.get("items_schema", {})
    if not isinstance(items_schema, Mapping):
        raise Malformed(f"{origin}: $.items_schema must be an object")

    validators = spec.get("validators", ())
    if not isinstance(validators, (list, tuple)):
        raise Malformed(f"{origin}: $.validators must be an array")

    return HandoffKind(
        name=name,
        content_type=ctype,
        items_schema=dict(items_schema),
        validators=tuple(str(v) for v in validators),
        scope=scope,
        version=spec.get("version"),  # type: ignore[arg-type]
    )


def permits_exec_without_env(kind: HandoffKind) -> bool:
    """Check 5, and it is subtle enough to be worth a name.

    It is **not** "does this content have `env`" — no content exists at
    kind-admission time. It is *"could a document satisfying this
    `items_schema` carry `script` or `command` without `env`"*, which is a
    question about the author's schema. The naive reading of criterion 4 puts
    the check in the wrong phase entirely.

    Answered by reading the schema rather than by probing it with a synthetic
    document: a probe needs a value for every *other* required key, and there
    is no way to mint one for an arbitrary subschema. The two mechanisms by
    which a document can carry `script` are a declared property and an open
    `additionalProperties`, and both are read here.
    """
    schema = kind.items_schema
    properties = schema.get("properties") or {}
    declares_exec = bool({"script", "command"} & set(properties))
    open_ended = schema.get("additionalProperties", True) is not False
    if not (declares_exec or open_ended):
        return False
    return "env" not in set(schema.get("required") or ())


def check(kind: HandoffKind, *, origin: str, allow_no_validator: bool = False) -> list[str]:
    """Checks 2, 4 and 5, as messages. Empty means admitted.

    A list rather than a raise, because `load_package` collects failures: one
    broken spec must not hide the other nine.
    """
    problems: list[str] = []
    schema_ok = True
    try:
        check_items_schema(kind.items_schema, origin=origin)
    except Malformed as exc:
        problems.append(str(exc))
        schema_ok = False

    if not kind.validators and not allow_no_validator:
        problems.append(
            f"{origin}: $.validators is empty. A kind with no validator cannot "
            f"be admitted (spec §5.3) — 'checkable by construction' is the "
            f"whole point. The bring-up flag permits it and reports the name"
        )

    # A layering gate, and Kubernetes CRD validation is the precedent: CEL
    # rules are skipped when the schema itself failed, because "your schema
    # permits script without env" on top of "your schema is not a schema" is
    # noise the author cannot act on.
    if schema_ok and kind.content_type == "reproducible" and permits_exec_without_env(kind):
        problems.append(
            f"{origin}: $.items_schema permits a document carrying 'script' or "
            f"'command' with no 'env'. An executable with no declared "
            f"environment is not reproducible, whatever it claims (spec §3.2). "
            f"Add 'env' to $.items_schema.required"
        )
    return problems
