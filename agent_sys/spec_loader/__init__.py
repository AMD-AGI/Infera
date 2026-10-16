"""The loader, the five schemas, and the vocabulary every other package shares.

`spec_loader` **imports nothing from this repository, ever.** It is the leaf and
must stay one: the moment it imports `handoff` to understand a handoff spec, main
spec §4.4's *"the loader does not interpret a package's content"* stops being
structural and becomes a promise. `tests/interfaces/test_import_rules.py` is the
enforcement.

The pipeline is main spec §4.4's, and **the seam moved at rev. 10** — the render
step is gone and what crosses is parsed documents::

      YamlPackage                validate.py                 registry.py
     ┌───────────────┐  docs   ┌───────────────┐   doc     ┌──────────────┐
     │ *.yaml -> obj │────────►│ against the   │──────────►│ name -> spec │
     │ (the format)  │         │ JSON Schema   │           └──────────────┘
     └───────────────┘         └───────────────┘
       the only side              never sees a path, and now
       that opens a file          never sees a byte either

Everything about the *format* is on the package's side. `load_package` is
validate-and-admit, and a second source format would be a second `TaskPackage`
rather than a change to it — which is main spec criterion 4 as a type boundary
instead of an ordering convention.

`docs/interfaces.md` §4.1 fixes what leaves this package: §3's whole table, plus
`render`, `validate`, `load_package` and `report`. **`render` no longer exists**
and §4.1 needs the row removed; the names below it are proposed additions, each
because something the frozen contract already names would otherwise have nowhere
to come from — see `README.md` §"What this package exports that §4.1 does not
list".
"""

from __future__ import annotations

from .access import body_of, subgraph_of, task_of, validator_agent_of
from .assets import ASSETS_DIRNAME, AssetIndex
from .bundled import KINDS, schema_for
from .package import ENTRY_FILENAME, MODULE_KEY, YamlPackage, load_package
from .protocols import (
    Body,
    ClosureDoc,
    LoadReport,
    PackageContents,
    Problem,
    Registries,
    SpecDocument,
    SpecInconsistent,
    SpecInvalid,
    SpecNotFound,
    SpecRegistry,
    TaskPackage,
    TaskSpec,
)
from .registry import BaseSpecRegistry
from .report import failed_names, format_problems, rejected, report
from .validate import validate
from .variables import ASSETS_VAR

__all__ = [
    # docs/interfaces.md §3 — the shared vocabulary. `SpecSource` and
    # `ImportResolver` are gone: one file / one object / kind-by-directory is
    # the shape the user-interface stage breaks, and there is nothing to
    # resolve an import for.
    "Body",
    "ClosureDoc",
    "LoadReport",
    "PackageContents",
    "Problem",
    "Registries",
    "SpecDocument",
    "SpecInconsistent",
    "SpecInvalid",
    "SpecNotFound",
    "SpecRegistry",
    "TaskPackage",
    "TaskSpec",
    # docs/interfaces.md §4.1 — the verbs, plus the accessors over the
    # vocabulary this package already declares. `Body` was declared three times
    # in Python over one `$defs.body`; `subgraph`'s key twice.
    "body_of",
    "load_package",
    "report",
    "subgraph_of",
    "task_of",
    "validate",
    "validator_agent_of",
    # docs/interfaces.md §2 step 5 — the composition root's derivations over
    # `Problem` and `LoadReport`. They live here because the module that owns a
    # type owns the operations over it (engineer_principle.md §3). §2's fourth,
    # `merged`, is NOT here: it folds `handoff.HandoffLoadReport`, whose shape is
    # `handoff`'s to define and which this package may not name.
    "failed_names",
    "format_problems",
    "rejected",
    # Proposed additions to §4.1. README.md says why each is not optional.
    "ASSETS_DIRNAME",
    "ASSETS_VAR",
    "AssetIndex",
    "BaseSpecRegistry",
    "ENTRY_FILENAME",
    "KINDS",
    "MODULE_KEY",
    "YamlPackage",
    "schema_for",
]
