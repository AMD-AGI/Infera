"""handoff — the content and lifecycle layer.

A handoff is a module's input or output, and the only thing that crosses
between tasks — which makes it the only place where quality can be enforced
system-wide.

The runtime *slot* — `Handoff`, `HandoffVersion`, `open_next()`, `seal()` — is
`task_graph`'s and is not here. The two layers meet at one point,
`Handoff.type`, which names a **kind**, and a kind is what this package
specifies.

The types are declared once, in `protocols.py`, and re-exported here. Nothing
in this package declares a second class of a name that file already holds: a
`validator` catching `handoff.Malformed` must catch the one this package
raises, and a duplicate class would still pass every test in this directory.

See `docs/spec.md` (rev. 5, 17 criteria) and `docs/design.md`.
"""

from handoff.containment import check_contained
from handoff.content import CONTENT_TYPES
from handoff.digest import ALGORITHM, canonical, tree_digest
from handoff.errors import (
    BindingConflict,
    DigestMismatch,
    Malformed,
    NotContained,
    NotSealable,
    PointerInvalid,
    PointerMiss,
)
from handoff.pointer import resolve
from handoff.protocols import (
    Content,
    ContentType,
    HandoffKind,
    HandoffLoadReport,
    HandoffStore,
    Item,
    Manifest,
    Scope,
    Verdict,
)
from handoff.registry import HandoffSpecRegistry
from handoff.store import FilesystemStore, store_name_for, version_dir

__all__ = [
    "ALGORITHM",
    "CONTENT_TYPES",
    "BindingConflict",
    "Content",
    "ContentType",
    "DigestMismatch",
    "FilesystemStore",
    "HandoffKind",
    "HandoffLoadReport",
    "HandoffSpecRegistry",
    "HandoffStore",
    "Item",
    "Malformed",
    "NotSealable",
    "Manifest",
    "NotContained",
    "PointerInvalid",
    "PointerMiss",
    "Scope",
    "Verdict",
    "canonical",
    "check_contained",
    "resolve",
    "store_name_for",
    "tree_digest",
    "version_dir",
]
