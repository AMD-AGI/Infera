"""The five JSON Schemas, read from inside the installed package.

`docs/design.md` D1: the README projected `agent_sys/schemas/` at the top level
and **that is not installable**. A bare directory of `.json` files has no
`__init__.py`, so `find_packages` cannot see it and setuptools will not ship it;
anything reading it by relative path works from a git checkout and fails from a
wheel. So they live here and are read through `importlib.resources`, which
behaves identically from a checkout, a wheel, and a zipimport.

`_common.schema.json` is not a kind. It holds the shapes more than one schema
names — today just `body`, which `closure` spec §2.6 and `validator` spec §6.1
describe as deliberately the same thing. One declaration, `$ref`-ed twice.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from functools import cache, lru_cache
from importlib.resources import files
from typing import Any

from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from .protocols import SpecNotFound

__all__ = ["KINDS", "bundled_registry", "schema_for"]

#: The five spec kinds. `task` is here and is **not** discoverable as a file:
#: a task spec is declared inside its closure, as the `task` key (closure spec
#: §2), so `closure.schema.json` `$ref`s this one rather than a package shipping
#: it alone. `docs/design.md` §2.3 states the same thing from the package side.
KINDS: tuple[str, ...] = ("handoff", "validator", "task", "agent", "closure")

_DIR = "schemas"


@cache
def _load(filename: str) -> Mapping[str, Any]:
    # The anchor is the already-imported module OBJECT, not the string
    # `"spec_loader"`. `closure/schema.py` measured why that matters: under
    # `pytest tests/<pkg>`, `files("<pkg>")` was observed resolving to
    # `tests/<pkg>/`, because pytest puts a test directory holding an
    # `__init__.py` on `sys.path` and it is then importable under this package's
    # own name. Resolving the name again is what loses; the module object in
    # `sys.modules` is the one that is actually executing.
    #
    # This keeps `docs/design.md` D1's argument intact — `importlib.resources`
    # behaves identically from a checkout, a wheel, and a zipimport — while
    # closing the hazard that made `closure` reach for `__file__` instead.
    return json.loads((files(sys.modules[__package__]) / _DIR / filename).read_text())


def schema_for(kind: str) -> Mapping[str, Any]:
    """The JSON Schema for one of the five spec kinds.

    Exported because the five schemas live in this package and four other
    modules own what is *in* them. Without one accessor each of those four would
    hand-roll an `importlib.resources` read, which is D1's failure mode
    reintroduced four times.
    """
    if kind not in KINDS:
        raise SpecNotFound(f"no schema for spec kind {kind!r} (have: {', '.join(KINDS)})")
    return _load(f"{kind}.schema.json")


@lru_cache(maxsize=1)
def bundled_registry() -> Registry:
    """A `referencing` registry over every bundled schema, keyed by `$id`.

    This is what makes `{"$ref": "task.schema.json"}` resolve inside
    `closure.schema.json` without a network fetch and without inlining one
    schema into another — two declarations of one shape being the duplication
    `engineer_principle.md` §1 forbids.

    `referencing` ships with `jsonschema` >= 4.18, so this adds no dependency.
    """
    resources = []
    for name in (*KINDS, "_common"):
        filename = f"{name}.schema.json"
        resources.append(
            (filename, Resource.from_contents(_load(filename), default_specification=DRAFT202012))
        )
    return Registry().with_resources(resources)
