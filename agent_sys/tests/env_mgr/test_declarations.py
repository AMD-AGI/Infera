# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Every declaration-pair in this package, not just the one that broke.

`protocols.py` declares the shapes; six of them are implemented a second time
elsewhere in `env_mgr`. Each pair is the duplication `engineer_principle.md` §1
names, admissible because a `NamedTuple` cannot carry a real method and stay a
declaration — **and the price of each is a test that compares them.**

Two of those tests existed. `test_prepared_matches_the_declared_surface` covers
`Prepared`, and part of `EnvManager`'s signature is checked beside it. The other
four had nothing, and it was not because they were judged safe: nobody looked.

**This file exists because of the order in which today went.** A defect was
found in `Prepared`'s annotations and fixed there; four hundred lines away
`EnvManager` had the identical hole and was not examined, so `place_zone`
returned `Any` against a declared `Zone` for as long as it took someone to trip
over it separately. `main` made the rule out of it: *when a defect is found in
one declaration-pair, sweep the file for the others before closing it.* This is
that sweep, made permanent so it does not depend on someone remembering.

**The sweep found one live divergence in a pair nobody was checking**:
`isolation.Policy.granted` defaulted to `()` where the declaration has no
default. Fixed on the implementation side rather than pinned — see the field's
comment for why an empty granted set is the one thing that module must not make
easy.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from env_mgr import protocols
from env_mgr.fs.domain import DomainRegistry as DomainRegistryImpl
from env_mgr.fs.zone import Zone as ZoneImpl
from env_mgr.isolation.policy import Policy as PolicyImpl
from env_mgr.prepare import EnvManager as EnvManagerImpl
from env_mgr.prepare import Prepared as PreparedImpl
from env_mgr.prepare import ValidationZone as ValidationZoneImpl

#: Every shape `protocols.py` declares that is implemented a second time.
#:
#: `Confinement`, `SyncReport`, `Granted`, `Mode`, `Domain`, `DomainKind`,
#: `Tier` and `Context` are **imported** from `protocols` at their point of use
#: rather than redeclared, so they are one definition and cannot drift. They are
#: named here so that "why is X missing" has an answer in the file.
NAMED_TUPLE_PAIRS = [
    ("Zone", ZoneImpl, protocols.Zone),
    ("Policy", PolicyImpl, protocols.Policy),
    ("ValidationZone", ValidationZoneImpl, protocols.ValidationZone),
    ("Prepared", PreparedImpl, protocols.Prepared),
]

PROTOCOL_PAIRS = [
    ("DomainRegistry", DomainRegistryImpl, protocols.DomainRegistry),
    ("EnvManager", EnvManagerImpl, protocols.EnvManager),
]

#: `(pair, field)` the implementation may annotate as `Any`, **and only these.**
#:
#: The rule is `main`'s, ruled as a class: *an `Any` on a cross-module surface
#: that the implementation could name is a defect, not a convention.* So an
#: entry here is a claim that the implementing module **cannot** name the type
#: without an import the seam forbids — checked, not assumed:
#:
#: | entry | the type, and who owns it |
#: |---|---|
#: | `Zone.task_id` | `TaskId` is `task_graph`'s; `fs/zone.py` imports only `fs.path` |
#: | `ValidationZone.materials` | `HandoffId` is `task_graph`'s; `prepare.py` imports nothing from it |
#: | `Prepared.output_paths` | the same `HandoffId`, in the same module |
#:
#: Two entries have left this list in opposite directions, which is the argument
#: for a list over a rule: `Prepared.confinement` by being **fixed** (the
#: declaration forbade a value four consumers branch on) and `Prepared.zone` by
#: being **narrowed** (`Any` only because it sat beside something forced).
#: **An exemption list earns its keep only if every entry is forced** — the
#: moment a fixable thing rests in one because of its neighbours, the list stops
#: meaning *cannot* and starts meaning *did not*.
FORCED_ANY = {
    ("Zone", "task_id"),
    ("ValidationZone", "materials"),
    ("Prepared", "output_paths"),
}


@pytest.mark.parametrize(("name", "impl", "declared"), NAMED_TUPLE_PAIRS, ids=lambda v: v)
def test_a_declared_shape_and_its_implementation_agree(name: str, impl: Any, declared: Any) -> None:
    """Names, defaults and annotations — the three things a `NamedTuple` pair
    can drift in. Checking one of them is what let `Prepared` drift twice."""
    assert impl._fields == declared._fields, (
        f"{name}: implementation has {impl._fields}, the declaration {declared._fields}"
    )

    # Defaults by value **and** by type. An empty `dict` where the other side has
    # a `MappingProxyType` is an equal value and a different mutability contract,
    # which `==` alone reports as agreement.
    assert impl._field_defaults == declared._field_defaults, (
        f"{name}: implementation defaults {impl._field_defaults}, "
        f"the declaration {declared._field_defaults}. A field optional on one side "
        f"and required on the other is a different constructor, not a nicety"
    )
    for field, value in declared._field_defaults.items():
        assert type(impl._field_defaults[field]) is type(value), (
            f"{name}.{field}: {type(impl._field_defaults[field])} against {type(value)}"
        )

    # As source text: `from __future__ import annotations` makes both sides
    # strings, and resolving them would need the implementing module to import
    # the very names it deliberately does not.
    for field in declared._fields:
        if (name, field) in FORCED_ANY:
            continue
        assert str(impl.__annotations__[field]) == str(declared.__annotations__[field]), (
            f"{name}.{field}: implementation says {impl.__annotations__[field]}, "
            f"declaration says {declared.__annotations__[field]}. If the "
            f"implementation cannot name the declared type, add it to FORCED_ANY "
            f"with the reason — but check that it is forced first, because both "
            f"`confinement` and `zone` looked forced and neither was"
        )


@pytest.mark.parametrize(("name", "impl", "declared"), PROTOCOL_PAIRS, ids=lambda v: v)
def test_a_declared_protocol_and_its_implementation_agree(
    name: str, impl: Any, declared: Any
) -> None:
    """Parameters, **and return annotations.**

    The return half is why this exists. `EnvManager`'s existing check compared
    `list(inspect.signature(...).parameters)` — parameter *names* only — so
    `place_zone` could return `Any` against a declared `Zone` with a test
    sitting right next to it whose job was catching exactly that.
    """
    for method in [m for m in dir(declared) if not m.startswith("_")]:
        if not callable(getattr(declared, method, None)):
            continue
        assert hasattr(impl, method), f"{name}: the implementation has no {method!r}"
        implemented = inspect.signature(getattr(impl, method))
        spec = inspect.signature(getattr(declared, method))

        assert str(implemented.return_annotation) == str(spec.return_annotation), (
            f"{name}.{method}: returns {implemented.return_annotation}, declared as "
            f"{spec.return_annotation}"
        )

        # The implementation may **add** trailing parameters — `prepare` grew
        # `agent_spec` with a default, which is design rev. 4 §11.5 and leaves
        # both call shapes working. It may not drop or reorder what is declared.
        spec_names = list(spec.parameters)
        assert list(implemented.parameters)[: len(spec_names)] == spec_names, (
            f"{name}.{method}: implementation takes {list(implemented.parameters)}, "
            f"the declaration {spec_names}"
        )


def test_every_redeclared_shape_is_in_this_files_pair_list() -> None:
    """The sweep must not be able to fall behind the package.

    A list of pairs typed by hand is the same instrument as the wall test's
    hand-typed `ABOVE` — it is silent about whatever nobody added, and the
    failure is always the unlisted one. So the list is checked against the
    package rather than trusted: anything `protocols.py` declares and some other
    module declares again must be under test here.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "env_mgr"
    declared = {
        node.name
        for node in ast.parse((root / "protocols.py").read_text()).body
        if isinstance(node, ast.ClassDef)
    }

    redeclared: set[str] = set()
    for path in root.rglob("*.py"):
        if path.name in {"protocols.py", "protocols.pyi"}:
            continue
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.ClassDef) and node.name in declared:
                redeclared.add(node.name)

    covered = {name for name, _, _ in NAMED_TUPLE_PAIRS + PROTOCOL_PAIRS}
    assert redeclared == covered, (
        f"declaration-pairs in the package: {sorted(redeclared)}; under test here: "
        f"{sorted(covered)}. A pair nobody compares is a pair that drifts — which "
        f"is how `place_zone` returned `Any` against a declared `Zone`."
    )
