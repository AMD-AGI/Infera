"""Content: a README plus a typed dictionary, and the four content types.

Spec §3.1 fixes the content as a README plus `items`. On disk:

```
content/
├── README.md
└── items/
    ├── <key>          a file, or a directory of files
    └── items.json     the typed values that are not files
```

**The dict is split in two because its values are of two kinds** (`design.md`
D3). A `reproducible` handoff's `logs` is a file, possibly a large one, and
forcing it through JSON is a decision nobody could undo later; a
`structured_text` handoff's `schema` is data. The spec's model is preserved and
only its on-disk realisation is two files.

The four types are a **table in one module** so the sets are visible together
and a fifth type is a row rather than a code change.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import jsonschema

from handoff import readme as readme_mod
from handoff.digest import canonical, tree_digest
from handoff.errors import Malformed
from handoff.protocols import Content, ContentType, Item

__all__ = [
    "CONTENT_TYPES",
    "ITEMS_DIR",
    "ITEMS_JSON",
    "check_items",
    "check_items_schema",
    "content_type",
    "digest_of",
    "load",
    "required_sections",
    "write_items_json",
]

ITEMS_DIR = "items"
ITEMS_JSON = "items.json"

#: The four types of spec §3.2. `readme_sections` is a **tuple** here and a set
#: at the check: ordered so `readme.template()` can be generated from it,
#: checked as membership because markdownlint #394 is a required-headings
#: matcher that accepted every document — it failed *open*.
#:
#: Which sections each type requires is not fixed by any document in the set;
#: these are this module's choice, derived from spec §3.2's description of what
#: each type carries. A change here is a change to what an existing handoff
#: must contain, so it is a table edit somebody reviews.
CONTENT_TYPES: Mapping[str, ContentType] = {
    "reproducible": ContentType(
        name="reproducible",
        required_items=frozenset({"result", "env"}),
        optional_items=frozenset({"script", "command", "code", "logs", "watchout"}),
        readme_sections=("Purpose", "How to run", "Result", "Environment", "Watch out"),
    ),
    "code": ContentType(
        name="code",
        required_items=frozenset({"codes"}),
        optional_items=frozenset({"logs", "watchout"}),
        readme_sections=("Purpose", "Interface", "Boundary"),
    ),
    "structured_text": ContentType(
        name="structured_text",
        required_items=frozenset(),
        optional_items=frozenset({"text.json", "text.yaml", "text.xml", "schema"}),
        readme_sections=("Purpose", "Schema"),
    ),
    "text": ContentType(
        name="text",
        required_items=frozenset({"content"}),
        optional_items=frozenset(),
        readme_sections=("Purpose",),
    ),
}

#: "One of these" requirements, which the frozen `ContentType` has no field for.
#: A group is satisfied when at least one of its keys is present.
_ALTERNATIVES: Mapping[str, tuple[frozenset[str], ...]] = {
    "reproducible": (frozenset({"script", "command"}),),
    "structured_text": (frozenset({"text.json", "text.yaml", "text.xml"}),),
}


def content_type(name: str) -> ContentType:
    """The `ContentType` row, or `Malformed` naming the four that exist."""
    try:
        return CONTENT_TYPES[name]
    except KeyError:
        raise Malformed(f"unknown content_type {name!r} (have: {sorted(CONTENT_TYPES)})") from None


def required_sections(ctype: ContentType, extra: Sequence[str] = ()) -> tuple[str, ...]:
    """The type's sections plus the kind's `readme_sections`, order preserved."""
    out = list(ctype.readme_sections)
    out += [s for s in extra if s not in out]
    return tuple(out)


def load(root: Path) -> Content:
    """Read a `content/` directory into a `Content`. Does no type checking.

    A directory under `items/` is a `tree`, a file is a `file`, and every key
    of `items.json` is `data`. A key present as both is `Malformed` — two
    sources for one item is the one shape the digest cannot represent twice.
    """
    root = Path(root)
    items: dict[str, Item] = {}
    items_dir = root / ITEMS_DIR

    if items_dir.is_dir():
        for entry in sorted(items_dir.iterdir()):
            if entry.name == ITEMS_JSON:
                continue
            kind = "tree" if entry.is_dir() and not entry.is_symlink() else "file"
            items[entry.name] = Item(key=entry.name, kind=kind, path=entry)

        blob = items_dir / ITEMS_JSON
        if blob.is_file():
            try:
                data = json.loads(blob.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise Malformed(f"{blob}: not valid JSON: {exc}") from exc
            if not isinstance(data, dict):
                raise Malformed(f"{blob}: must be a JSON object of key -> value")
            for key, value in data.items():
                if key in items:
                    raise Malformed(
                        f"{blob}: {key!r} is also a file under {ITEMS_DIR}/. "
                        f"An item has one source, or the digest records it twice"
                    )
                items[key] = Item(key=key, kind="data", value=value)

    return Content(root=root, readme=root / readme_mod.README_NAME, items=items)


def check_items(content: Content, ctype: ContentType, items_schema: Mapping | None = None) -> None:
    """Criterion 3 and criterion 5, in that order.

    The **type** decides which keys it defines; the **kind's `items_schema`**
    decides whether a key it never declared may appear at all. That split is
    what makes a runtime-generated key expressible with no second mechanism and
    no `allow_runtime_keys` flag: `additionalProperties: {...}` permits one,
    `additionalProperties: false` does not, and it is the same `jsonschema`
    pass every other field gets (`design.md` §3.3).
    """
    known = ctype.required_items | ctype.optional_items
    for group in _ALTERNATIVES.get(ctype.name, ()):
        known |= group

    missing = sorted(ctype.required_items - set(content.items))
    if missing:
        raise Malformed(
            f"{content.root}: content_type {ctype.name!r} requires {missing} "
            f"and the handoff carries {sorted(content.items)}"
        )
    for group in _ALTERNATIVES.get(ctype.name, ()):
        if not group & set(content.items):
            raise Malformed(
                f"{content.root}: content_type {ctype.name!r} requires one of "
                f"{sorted(group)} and the handoff carries none of them"
            )

    declared = set(items_schema.get("properties", {})) if items_schema else set()
    unknown = sorted(set(content.items) - known - declared)
    if unknown and items_schema is None:
        raise Malformed(
            f"{content.root}: items {unknown} are not defined by content_type "
            f"{ctype.name!r} (defines: {sorted(known)})"
        )

    if items_schema is not None:
        instance = {
            k: (item.value if item.kind == "data" else str(item.path.name))
            for k, item in content.items.items()
        }
        try:
            jsonschema.Draft202012Validator(dict(items_schema)).validate(instance)
        except jsonschema.ValidationError as exc:
            where = exc.json_path if exc.absolute_path else "$"
            raise Malformed(
                f"{content.root}: items {where}: {exc.message} "
                f"(the kind's items_schema; a key it never declared needs "
                f"additionalProperties to permit it)"
            ) from exc


def check_items_schema(items_schema: Mapping, *, origin: str) -> None:
    """`Draft202012Validator.check_schema`, **as a named step, never `$ref`**.

    Measured (`docs/design.md` §3.5): `$ref`-ing the 2020-12 metaschema turns
    `{"type": "nonsense"}` into **8 identical errors**, one per failing `anyOf`
    branch, and a package author reading eight copies of
    `is not of type 'object', 'boolean'` learns nothing. `check_schema` gives
    one actionable line.

    **This is the expensive step in admitting a kind, and it is not cached.**
    It runs once per *occurrence* — `SpecRegistry._validate` fires for every
    admission attempt, so a kind vendored in two packages pays twice. Measured
    on this machine:

    | `items_schema` | `check_schema` |
    |---|---|
    | `{"type": "object"}` | 122 µs |
    | a realistic `reproducible` kind — six keys, `required`, closed | 1.05 ms |
    | 50 properties, pathological | 6.6 ms |

    So ~104 ms of load for 100 realistic kinds, against ~300 ms for the
    parallel render of as many sources. Same order, not dominant.

    **The obvious optimisation does not exist**: the cost is jsonschema
    validating the document against the metaschema, not building the validator.
    Hoisting a prebuilt `Draft202012Validator(META_SCHEMA)` out of the call
    saves **1.5%** (6565 µs → 6465 µs, measured), so there is nothing to hoist.

    The lever that would work, if load time ever becomes the complaint, is an
    `lru_cache` **here**, keyed on `json.dumps(items_schema, sort_keys=True)` —
    per distinct *schema*, which is where the cost is. Not on
    `BaseSpecRegistry._canonical`: that key is computed *after* `_validate`
    returns, and it covers the whole spec, so two kinds differing only in
    `description` would miss it. It is deliberately not taken today, for the
    reason `design.md` §4.9 gives for not caching the tree digest — recomputing
    is cheap enough that a cache would only add a class of staleness bug.
    """
    # `isinstance` rather than `dict(items_schema)`, and the coercion is the
    # bug it replaces. `dict("notaschema")` — main design §3.5's own example —
    # raises **`ValueError`**, which the old `except (TypeError, AttributeError)`
    # did not catch, so it escaped this function as a bare `ValueError` past
    # every caller catching `Malformed`. And `dict([("a", 1)])` *succeeds*, so a
    # list of pairs was silently accepted as a schema object.
    #
    # Found by asking `closure`'s question of the branch — *has anything ever
    # taken this?* — and driving it when the answer was no. A guard nobody has
    # watched fire is a guard that may be catching the wrong thing.
    if not isinstance(items_schema, Mapping):
        raise Malformed(
            f"{origin}: $.items_schema must be an object, not {type(items_schema).__name__}"
        )
    try:
        jsonschema.Draft202012Validator.check_schema(dict(items_schema))
    except jsonschema.SchemaError as exc:
        raise Malformed(f"{origin}: $.items_schema is not a valid schema: {exc.message}") from exc


def digest_of(content_root: Path) -> bytes:
    """The tree digest of a `content/` directory.

    `items.json` is digested as the bytes on disk; `canonical()` is what a
    *writer* puts there, so that a value from a jsonnet default and the same
    value from an agent produce identical bytes (`design.md` §4.6).
    """
    return tree_digest(os.fsencode(content_root))


def write_items_json(items_dir: Path, values: Mapping[str, object]) -> Path:
    """Write `items.json` in canonical form. The one writer of that file."""
    items_dir = Path(items_dir)
    items_dir.mkdir(parents=True, exist_ok=True)
    blob = items_dir / ITEMS_JSON
    blob.write_bytes(canonical(dict(values)))
    return blob
