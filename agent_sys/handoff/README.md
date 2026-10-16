# `handoff` — the content and lifecycle layer

A handoff is a module's input or output, and **the only thing that crosses
between tasks** — which makes it the only place where quality can be enforced
system-wide.

| | |
|---|---|
| Specification | [`docs/spec.md`](docs/spec.md) — 17 acceptance criteria |
| Design | [`docs/design.md`](docs/design.md) |
| Seam | [`../docs/interfaces.md`](../docs/interfaces.md) §4.2 |
| Contract | [`protocols.py`](protocols.py) + `protocols.pyi`, declarations only |
| Tests | `../tests/handoff/` |

## Who uses it

`validator` binds a check to a kind and catches this package's errors.
`task_graph` owns the runtime **slot** and holds a `Handoff.type` that names one
of this package's kinds. `agent` seals through it. `cli` publishes through it.

**The slot is not here.** `Handoff`, `HandoffVersion`, `open_next()` and
`seal()` belong to `task_graph`. The two layers meet at exactly one point —
`Handoff.type` names a **kind**, and a kind is what this package specifies.

## The interface

```python
from handoff import (
    HandoffKind, HandoffSpecRegistry,     # what a kind is, and where kinds live
    Content, ContentType, CONTENT_TYPES,  # what a handoff carries
    Manifest, Item, Scope,                # its shape on disk
    HandoffStore, FilesystemStore,        # where versions are stored
    Verdict,                              # persisted here, because this module reads it back
    tree_digest, canonical, ALGORITHM,    # identity
    resolve, check_contained,             # addressing into content, and the containment rule
)
```

Every error a caller may catch is exported too: `Malformed`, `NotSealable`,
`NotContained`, `DigestMismatch`, `BindingConflict`, `PointerInvalid`,
`PointerMiss`. **They are declared once, in `protocols.py`, and re-exported** —
a `validator` catching `handoff.Malformed` must catch the one this package
raises, and a second class of the same name would make that silently false.

## What is inside

| | |
|---|---|
| `protocols.py` / `.pyi` | the vocabulary, declarations only. The importable half of the seam |
| `kind.py` | a handoff kind: what it carries, what checks it, what scope it has |
| `content.py` | the content types and their shape |
| `digest.py` | identity — a git-shaped sha256 over the subtree, canonicalised by RFC 8785 |
| `pointer.py` | RFC 6901 addressing into content, and the three outcomes a caller must tell apart |
| `containment.py` | the rule that a handoff's items stay inside it |
| `locality.py` | whether an artefact would still mean the same thing on another host |
| `readme.py` | the README every handoff carries |
| `store.py` | the on-disk store, and `version_dir` |
| `verdict.py` | a verdict, persisted so `validator` can read it back |
| `registry.py`, `errors.py` | the kind registry, and the errors above |

## The one rule worth knowing before changing anything

**`Verdict` lives here and is `validator`'s payload.** The module that persists a
record is the module that has to keep it readable, so the type is declared here
even though nothing in this package produces one. Moving it would split a
record's writer from its reader.
