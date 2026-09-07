"""The admission model — what makes a validator spec real.

`protocols.Validator` is a static type and **not** the gate. Measured
(`scratch/impl-2026-08/validator/p1_pydantic_shapes.py`, and design §3.2 before
it): `issubclass` raises outright on a Protocol with non-method members, and
`isinstance` is presence-only, so `strength=None` passes it. Worse,
`inputs="trace"` passes and then iterates as five characters — one declared kind
silently becoming five nonexistent ones.

So the gate is a pydantic model over the spec *record*. Re-measured here on
pydantic 2 rather than taken from the design: a bare string is a `tuple_type`
error, `extra="forbid"` names the key in `loc`, and a `list` is coerced to a
`tuple`, which is what jsonnet's JSON array output needs.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from spec_loader.protocols import Problem
from validator.protocols import Body, Dimension, Strength, ValidatorInvalid

__all__ = [
    "Cost",
    "LogicSource",
    "Tags",
    "ValidatorSpec",
    "admit",
    "check_binding_symlinks",
    "check_body_resolves",
]


class LogicSource(str, Enum):
    """Spec §5.2's taxonomy. Carried so the claim is written down and reviewable;
    **nothing verifies it** — the graph shape is what makes `external_dynamic`
    true, and the registry cannot see the graph (spec open question 3)."""

    EXTERNAL_STATIC = "external_static"
    EXTERNAL_DYNAMIC = "external_dynamic"
    AGENT_WRITTEN = "agent_written"


class Cost(str, Enum):
    """An order of magnitude, not a number. The one tag the system reads: §5.3
    orders a phase's validators cheap-first from it.

    Ordering by a declared cost tag has no prior art in anything surveyed, so the
    failure mode is owed rather than a citation: **the tag can be wrong and
    nothing here detects it.**
    """

    SECONDS = "seconds"
    MINUTES = "minutes"
    GPU_HOURS = "gpu_hours"

    @property
    def rank(self) -> int:
        return _COST_ORDER[self]


_COST_ORDER = {Cost.SECONDS: 0, Cost.MINUTES: 1, Cost.GPU_HOURS: 2}


class Tags(BaseModel, extra="allow"):
    """Spec §9.2's dictionary. `extra="allow"` here and `extra="forbid"` on
    `ValidatorSpec`, deliberately: a tag dictionary is where a site adds its own
    key, while a stray *top-level* field is a typo.

    `logic_source` lives here and **only** here. Design §3.2 also listed it as a
    top-level field while §3.5 said the tag dictionary carries everything that is
    not `dimension` or `strength`; two writers of one fact is
    `engineer_principle.md` §1's failure, so the field is a read-only delegation
    (`ValidatorSpec.logic_source`) and the tag is the writer.
    """

    logic_source: LogicSource
    cost: Cost
    domain: tuple[str, ...] = ()


class ValidatorSpec(BaseModel, extra="forbid"):
    """One registry entry. **One spec per instance** — two validators differing
    only in a threshold are two records naming one body folder (§10.6, D5).

    Spec §3.5's two structural constraints have no field at all, and that is the
    strongest form they can take: a validator spec has no `subtasks` and no
    validation phase of its own, so `extra="forbid"` makes naming one a load
    error. The violation is unrepresentable rather than checked, which is exactly
    the kind of guarantee a later field addition silently removes — hence
    `test_neither_field_exists_on_the_model`.
    """

    name: str
    brief: str  # no default. Criterion 1
    inputs: tuple[str, ...]
    dimension: Dimension  # no default
    strength: Strength  # no default
    tags: Tags
    #: Required for a leaf, absent for a composite — enforced in `admit`, because
    #: "exactly one of these two" is not a schema keyword and an `if`/`then`
    #: produces a message nobody can act on. `spec_loader`'s schema leaves `body`
    #: optional for the same reason and names the check instead.
    #:
    #: **Absent is `{}`, one spelling across the seam.** `spec_loader.body_of`
    #: returns `{}` for none, and a second spelling here would put the two-absents
    #: problem right back at the boundary. Safe because a body that exists is
    #: never `{}`: `_common.schema.json` makes `readme` required and non-empty. An
    #: author cannot write `{}` either — measured, pydantic rejects an explicit
    #: one as a missing `readme` while leaving the default alone.
    body: Body = {}  # noqa: RUF012 - pydantic copies per-instance
    #: The agent spec whose `env` is §8.2 row 1's environment. A **name**, not an
    #: inline block: `agent.schema.json` already has `env`, and a second copy here
    #: would be two writers of one fact. **Absent is legal** and takes the global
    #: row — which is why it is not required — and present-but-unresolvable is
    #: fatal, checked by `closure`'s pass because the agent registry may not be
    #: loaded when this document is.
    agent: str | None = None
    description: str | None = None  # inert, from `_common`; all five kinds carry it
    version: str | None = None  # maintenance metadata; nothing at runtime reads it
    members: tuple[str, ...] = ()  # non-empty iff this is a composite
    reduce: str | None = None
    args: Mapping[str, Any] = {}

    @property
    def logic_source(self) -> LogicSource:
        return self.tags.logic_source

    @property
    def cost(self) -> Cost:
        return self.tags.cost

    @property
    def is_composite(self) -> bool:
        return bool(self.members)


def admit(record: Mapping[str, Any], *, origin: str) -> ValidatorSpec:
    """The admission gate. Raises `ValidatorInvalid` naming `origin` and the field.

    `origin` is the label the loader was given — printed, never opened. Go's
    `ImportErrorf` panics when the offending path is absent from the message; the
    discipline is worth copying even where the mechanism is not.
    """
    try:
        spec = ValidatorSpec(**dict(record))
    except ValidationError as exc:
        faults = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        )
        raise ValidatorInvalid(f"{origin}: {faults}") from exc
    if spec.is_composite:
        if spec.reduce is None:
            raise ValidatorInvalid(f"{origin}: a composite must name a reducer")
        if spec.body:
            raise ValidatorInvalid(
                f"{origin}: a composite's implementation is its members; "
                f"{spec.name} also declares a body"
            )
    else:
        if spec.reduce is not None:
            raise ValidatorInvalid(
                f"{origin}: reduce is a composite's field; {spec.name} has no members"
            )
        if not spec.body:
            raise ValidatorInvalid(f"{origin}: {spec.name} declares neither a body nor members")
        _refuse_empty_paths(spec, origin=origin)
    return spec


def _refuse_empty_paths(spec: ValidatorSpec, *, origin: str) -> None:
    """An empty path string is a fault, not an absence.

    `entry: ""` is the one that bites: it is falsy, so `runner_for` would read it
    as *no entry* and run the validator as agent-bodied — a programmatic check
    silently becoming an agent's opinion, with no error anywhere. A value that is
    wrong but not **type**-wrong, which is `interfaces.md` §4.11's family and the
    third instance of it in this package today.

    `_common.schema.json#/$defs/body` gives every one of these `minLength: 1`, so
    this is also the two-gates rule applied to ourselves: a document the schema
    rejects must not be one the model accepts.
    """
    body = spec.body
    if not body:  # pragma: no cover - the caller has already refused this
        return
    empties = [
        label
        for label, value in (("readme", body.get("readme")), ("entry", body.get("entry")))
        if value == ""
    ]
    empties += [f"materials[{i}]" for i, m in enumerate(body.get("materials", ())) if m == ""]
    if empties:
        raise ValidatorInvalid(
            f"{origin}: {spec.name} declares an empty path for {', '.join(empties)}; "
            f"an empty path is a fault, not an absence — omit the key instead"
        )


def check_body_resolves(spec: ValidatorSpec, package_root: Path) -> None:
    """Every path a body names exists. Existence only — §10.3 check 1b.

    A dangling one is a load error **naming the path**, the same rule spec §9.1
    gives a binding symlink. What is lost with the callable is that a dotted path
    to a Python function was checkable by import; a path to a script is checkable
    only for existence (§10.2, D6).
    """
    if not spec.body:
        return  # a composite's implementation is its members
    for label, rel in _body_paths(spec.body):
        target = package_root / rel
        if not target.exists():
            raise ValidatorInvalid(f"{spec.name}: {label} does not resolve: {target}")


def check_binding_symlinks(spec: ValidatorSpec, folder: Path) -> list[Problem]:
    """The folder's binding symlinks, against the field that is the source of truth.

    Spec §9.1: a validator's folder *"carries relative symlinks to the handoff
    kinds it binds to, so the binding is visible in a directory listing and not
    only in the registry."* That makes the symlinks a **second, redundant
    statement of the binding, and redundant statements disagree.** They are not
    the source of truth — `inputs` is — so the loader reads the field and the
    symlinks are for a human with `ls`.

    Two faults, and they are opposite directions of the same primitive:

    * **A dangling symlink is a load error naming the path.** Spec §9.1 says so
      outright — *"not a puzzle for the loader to solve"*.
    * **A symlink naming a kind `inputs` does not** is a disagreement, and
      §10.3's resolution check catches only the reverse.

    Collected rather than raised, in the style everything else at load time uses:
    one broken spec must not hide the other nine.
    """
    problems: list[Problem] = []
    if not folder.is_dir():
        return problems
    for entry in sorted(folder.iterdir()):
        if not entry.is_symlink():
            continue
        if not entry.exists():
            problems.append(
                Problem(
                    origin=spec.name,
                    path="$.inputs",
                    keyword="binding",
                    message=f"dangling binding symlink: {entry} -> {os.readlink(entry)}",
                )
            )
            continue
        kind = entry.name.split(".")[0]
        if kind not in spec.inputs:
            problems.append(
                Problem(
                    origin=spec.name,
                    path="$.inputs",
                    keyword="binding",
                    message=(
                        f"{entry} names handoff kind {kind!r}, which is not in "
                        f"inputs {sorted(spec.inputs)}"
                    ),
                )
            )
    return problems


def _body_paths(body: Body) -> list[tuple[str, str]]:
    paths = [("readme", body["readme"])]
    if body.get("entry") is not None:
        paths.append(("entry", body["entry"]))
    paths += [("material", m) for m in body.get("materials", ())]
    return paths
