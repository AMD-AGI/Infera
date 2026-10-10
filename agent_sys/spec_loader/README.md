# `spec_loader` — the loader, the schemas, and the shared vocabulary

Turns a directory of YAML into validated spec documents, and holds the
vocabulary every other package names. **It is the leaf: it imports nothing from
this repository, ever.**

| | |
|---|---|
| Specified by | [`../docs/spec.md`](../docs/spec.md) §4.3–§4.5 · [`../docs/design.md`](../docs/design.md) §3–§5 — this package has no `docs/` of its own |
| Seam | [`../docs/interfaces.md`](../docs/interfaces.md) §3, §4.1 |
| Contract | [`protocols.py`](protocols.py) + `protocols.pyi` |
| Schemas | [`schemas/`](schemas/) — one JSON Schema per object. **The only enforcement point** |
| Tests | `../tests/spec_loader/` |

## Who uses it

All eight of the other packages. Four of them own a spec registry built on
`BaseSpecRegistry`; every one of them names `Problem`. `cli` calls
`load_package`.

**Nothing may reverse that edge.** The moment this package imports `handoff` to
understand a handoff spec, *"the loader does not interpret a package's content"*
stops being structural and becomes a promise.
`tests/interfaces/test_import_rules.py` is the enforcement.

## The pipeline

```
   YamlPackage            validate.py              a SpecRegistry
  ┌──────────────┐ docs  ┌──────────────┐  doc    ┌───────────────┐
  │ scan, parse, ├──────►│ schema check ├────────►│ own the kind  │
  │ substitute   │       │ -> [Problem] │         │ it was given  │
  └──────────────┘       └──────────────┘         └───────────────┘
```

`validate` takes a **parsed document and no path**, which is the type boundary
that makes the claim above checkable rather than conventional: there is no field
through which a path could arrive.

## The interface

```python
from spec_loader import (
    TaskPackage, YamlPackage, load_package, PackageContents,   # reading a package
    SpecDocument, MODULE_KEY, KINDS,                           # what a document is
    validate, schema_for, Problem, report, format_problems,    # checking it
    SpecRegistry, BaseSpecRegistry, Registries,                # where specs live
    Body, body_of, task_of, subgraph_of, validator_agent_of,   # accessors
    AssetIndex, ASSETS_DIRNAME, ASSETS_VAR, ENTRY_FILENAME,    # bodies found by convention
    LoadReport, failed_names, rejected,
    SpecNotFound, SpecInvalid, SpecInconsistent,
)
```

## Variable substitution — two forms, plus an escape

| | |
|---|---|
| `${NAME}` | required. No value supplied is a load error naming the variable and the file |
| `${NAME:-default}` | optional, with a fallback |
| `$$` | a literal dollar |

**There is no third form, and the missing one fails quietly.** `${NAME:?message}`
is *not* supported: it matches no pattern, so it is passed through unsubstituted
and produces **no problem at all** — a package that writes it gets a literal
`${NAME:?message}` in its environment and no error. Use `${NAME}`, which is
already the required form.

**A reference may not nest.** That is deliberate: a nested reference would make
this a grammar, and the alternative to a grammar is a parser nobody asked for.

## What is inside

| | |
|---|---|
| `protocols.py` / `.pyi` | the vocabulary: `Problem`, `SpecDocument`, the protocols |
| `yaml_source.py` | text → a tree that still knows its line positions |
| `variables.py` | `${…}` expansion over that tree |
| `package.py` | `YamlPackage`: scan, discriminate by `module:`, emit documents |
| `assets.py` | `AssetIndex` — a body found by filename convention |
| `validate.py` | document → `[Problem]`. Takes no path and no bytes |
| `bundled.py` | the schemas, as a `referencing` registry |
| `registry.py` | `BaseSpecRegistry`, the shared base of the four |
| `access.py`, `report.py`, `errors.py` | accessors, a problem rendered for a human, the errors |

## What a package must contain

Two names, and only two. `main.yaml` — present means the package is a **run
entry**, absent means it is a **library**. And `assets/`, because every document
may write an unqualified path and something has to resolve it.
