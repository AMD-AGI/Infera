"""The vocabulary every other package shares.

`spec_loader` imports nothing from this repository and must stay that way
(`docs/design.md` §2.3): the moment it imports `handoff` to understand a handoff
spec, "the loader does not interpret a package's content" stops being structural.

Declarations only. See `docs/interfaces.md` §3.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeAlias, TypedDict

__all__ = [
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
    "body_of",
    "load_package",
    "subgraph_of",
    "task_of",
    "validate",
    "validator_agent_of",
]

# --------------------------------------------------------------------------- #
# A spec is a plain dict throughout — docs/design.md §4.1 measured why a typed
# model is not a substitute for the schema. These aliases buy the name that
# `task_graph.check_graph` and `closure.check` both use, without a second
# declaration of the shape.

TaskSpec: TypeAlias = Mapping[str, Any]
ClosureDoc: TypeAlias = Mapping[str, Any]


class _BodyRequired(TypedDict):
    readme: str


class Body(_BodyRequired, total=False):
    """What a thing *is*: `readme.md` always, `entry.sh` iff programmatic, plus
    its own materials.

    One declaration, mirroring `schemas/_common.schema.json`'s single
    `$defs.body` — which `task.schema.json` and `validator.schema.json` both
    `$ref`, because `closure` spec §2.6 and `validator` spec §6.1 say a task's
    body and a validator's are deliberately the same thing. It was declared three
    times in Python before this moved here.

    **A `TypedDict` and not a dataclass**, which is the same argument §4.1 makes
    against pydantic spec models: a spec is a plain `dict` throughout, and typing
    is the only thing wanted here. A dataclass has to *construct*, which means
    coercing — and a constructor that turns a missing `readme` into `""` reports
    a body that is present and empty where the document had none. The two
    dataclass versions this replaces both did exactly that, and one of them was
    truthy for a task with no body at all.

    Split across two classes rather than written with `NotRequired`, which is
    3.11 and would cost a `typing_extensions` dependency on the 3.10 floor.
    """

    entry: str
    materials: list[str]


# --------------------------------------------------------------------------- #
# Errors. Three classes, not one: JPMS separates "not found" from "found, but
# inconsistent", and the distinction is load-bearing — a missing validator is a
# typo, while a two-way mismatch means one of two records is lying.


class SpecNotFound(LookupError):
    """A name does not resolve. The message enumerates the candidates."""


class SpecInvalid(ValueError):
    """A spec failed its schema or one of its own load-time checks."""


class SpecInconsistent(ValueError):
    """Two specs that both loaded disagree with each other."""


# --------------------------------------------------------------------------- #
# Problems


@dataclass(frozen=True)
class Problem:
    """One load-time fault, from any of the five spec kinds or the two passes.

    `origin` is the label the loader was given — never opened, only printed.
    Since rev. 10 it is a path **plus an optional JSON pointer** into the file,
    because one file may hold many objects: `steps/collect.yaml#/2`. See
    `SpecDocument.origin`, which is where the string is built and where the
    format is specified.

    `path` is a JSONPath into the document (`$`, `$.items_schema`).

    `fatal` is False for a report-severity finding. There are two producers:
    `closure/check.py`'s check 3, for a handoff kind admitted under the
    escape-hatch flag, and the assets resolver, for a body path bound by hand
    where the convention would have found it (`assets.py`).

    `line` and `column` are **1-based** and locate the *document* inside its
    file, not the offending field: the schema reports a field as `path`, and
    joining that back onto a source position would need the parse tree, which
    `validate` deliberately cannot see. They are `None` when the package did not
    report one, never guessed — the discipline the deleted `RenderError`
    carried, moved onto the type that survives it. Two fields with defaults
    rather than a widened `origin`, so that every existing construction of a
    `Problem` in five other packages stays valid and unchanged.
    """

    origin: str
    path: str
    keyword: str
    message: str
    fatal: bool = True
    line: int | None = None
    column: int | None = None


@dataclass(frozen=True)
class LoadReport:
    """One task package's load.

    Not to be confused with `handoff.HandoffLoadReport`, which is
    `(admitted, without_validator)`. Both are in scope in `closure/check.py`;
    docs/interfaces.md §3.1 records why they stayed two types.
    """

    admitted: Sequence[str]
    problems: Sequence[Problem]


# --------------------------------------------------------------------------- #
# The registry base


class SpecRegistry(Protocol):
    """A name table over admitted specs. Four subclasses, one policy.

    Deliberately the opposite of `task_graph.Registry`, which overwrites so a
    test can swap a component after wiring. Here a duplicate is an error: two
    specs claiming one name is a fault, and a validator registered under two
    names would run twice and record two verdicts against one handoff version.
    """

    kind: str

    def add(self, name: str, spec: Mapping[str, Any], *, origin: str) -> None:
        """Admit a spec.

        Raises `SpecInconsistent` if `name` is held by a *different* spec; a
        byte-identical re-registration is a no-op.
        """
        ...

    def get(self, name: str) -> Mapping[str, Any]:
        """Raises `SpecNotFound`, naming the kind, the name, and the candidates."""
        ...

    def names(self) -> list[str]:
        """Every admitted name, sorted."""
        ...

    def origin_of(self, name: str) -> str:
        """Where `name` was loaded from — the label, never opened.

        On the Protocol because a whole-catalogue pass **calls it unguarded**.
        `docs/design.md` §6.2's report rule is *name both sides*, and a
        cross-registry check holds neither side's file path otherwise.

        It was on the shared base and not here, so `closure` guarded it with a
        `getattr` fallback to the spec's *name* — and when `build_registry`
        began taking `registries=` from a caller, that fallback became reachable
        and labelled every `Problem` with a name where a path belongs.
        Indistinguishable from a real origin in a message. They removed the
        guard and called it an obligation; this is the obligation being written
        down, so the next implementer is told rather than left to infer it from
        a base class they may not subclass.
        """
        ...

    def __contains__(self, name: str) -> bool: ...


class Registries(Protocol):
    """A read-only view over the five spec registries.

    A Protocol rather than a class so a test supplies five dicts and no
    composition root. Nothing on it mutates.

    It deliberately carries no handoff load report: that type lives in
    `handoff`, and naming it here would make the leaf import a module package.
    `check_closures` takes it as a separate argument instead.
    """

    handoff_specs: SpecRegistry
    validator_specs: SpecRegistry
    task_specs: SpecRegistry
    agent_specs: SpecRegistry
    closures: SpecRegistry

    def for_kind(self, kind: str) -> SpecRegistry:
        """The registry for one of the five spec kinds. Raises `KeyError`."""
        ...


# --------------------------------------------------------------------------- #
# Packages
#
# **The seam moved at rev. 10 and this is the whole of the change.** It was
# `SpecSource(path, kind)` — one file, one object, kind claimed by the directory
# it sat in — and three of the user-interface stage's five requirements break
# that shape at once: several objects per file, kind claimed by a `module:` key
# rather than by location, and inline definitions that have no file of their
# own. What crosses now is *parsed documents*.


@dataclass(frozen=True)
class SpecDocument:
    """One object a package produced, ready to validate.

    `kind` is one of `KINDS` — the **schema** kind, which is not always the word
    the author typed. A package author writes `module: task` and never
    `closure` (`closure` spec §2), and the document that arrives here is the
    closure with the task spec nested inside it.

    `doc` is parsed and fully substituted. **No variable and no source syntax
    survives into it**, which is what makes main spec criterion 4 a type
    boundary rather than an ordering convention: this dataclass has no field
    through which a path, a template, or a byte of source could reach the
    loader.

    `origin` is *path*, optionally followed by `#` and a **JSON pointer** to the
    object inside the file — `steps/collect.yaml#/2`, or
    `steps/collect.yaml#/0/task/subgraph/1/closure` for one written inline. The
    pointer is **omitted when the file holds a single object at its root**, so
    the common case reads exactly as it did before rev. 10 and so does every
    message quoting it. That is unambiguous rather than merely tidy: within one
    file the root is either a mapping (one object, no pointer) or a sequence
    (indexed), never both.

    **The format is load-bearing beyond messages.** `task_graph/bootstrap.py`'s
    `_names_for` bridges `Problem.origin` to a spec *name* by exact string
    equality against `SpecRegistry.origin_of`, so the string `load_package`
    hands to `validate` and the string it hands to `registry.add` must be the
    same one. `SpecRegistry.origin_of`'s docstring records what a *plausible but
    wrong* origin cost the last time the two drifted.

    `line` / `column` are **1-based** — `ruamel.yaml` reports 0-based and the
    package adds one — and locate the object's first key. `None` when the
    package did not report one; never guessed.
    """

    kind: str
    doc: Mapping[str, Any]
    origin: str
    line: int | None = None
    column: int | None = None


@dataclass(frozen=True)
class PackageContents:
    """What a package hands across the seam: its documents, and its faults.

    **Two fields and not one, and that is a departure from the approved sketch**
    (`docs/ui-stage.md` §2 wrote `documents() -> Sequence[SpecDocument]`).
    Reported rather than done quietly, because a seam has two sides.

    A package can fail in ways that must not raise. One is fatal and structural
    — no `main.yaml`, no `assets/` — and one is deliberately *not* fatal: an
    explicit body binding where the convention would have found the file warns
    at compile time and the package still loads (`refine.task_package.define.md`
    §2.3.5). A `Sequence[SpecDocument]` can express neither. The alternatives
    were a second method, which makes a caller remember to call two things in
    the right order (`engineer_principle.md` §1), or raising, which cannot carry
    a warning at all.

    `Problem` is already this package's fault vocabulary (`interfaces.md` §3),
    so a package-side fault is reported in it rather than in a second shape.
    """

    documents: Sequence[SpecDocument]
    problems: Sequence[Problem]


class TaskPackage(Protocol):
    """A task package: whatever an author wrote, turned into documents.

    **The loader knows nothing about how these were produced** — not the file
    format, not the layout, not how many objects came out of one file. Main spec
    §4.4 says the loader "does not read, audit, or constrain" a package's source
    and that "only the result is checked"; before rev. 10 that was an ordering
    convention inside `load_package`, and this Protocol is what makes it
    structural.
    """

    root: Path

    def documents(self) -> PackageContents:
        """Every object this package declares, in definition order.

        One call, complete answer: a caller never has to follow it with a second
        one to find out whether it worked (`engineer_principle.md` §1). A
        package that cannot be read reports it here as a fatal `Problem` rather
        than raising, so that one broken file does not hide the other nine —
        which is `load_package`'s first stated property.

        **Definition order is the package's own and nothing else can see it**,
        which is why the forward-reference check lives on this side of the seam.
        An object defined inline is emitted *before* the object that references
        it.
        """
        ...


def validate(doc: Any, schema: Mapping[str, Any], *, origin: str) -> list[Problem]:
    """Check an already-parsed document against `schema`.

    **This signature is the enforcement of main spec §4.4.** There is no
    parameter through which a path could reach this function, and therefore no
    way for it to read a package's source. `origin` is an opaque label used only
    in messages. `test_validate_takes_no_path` guards it.

    **It took `data: bytes` and parsed, and it no longer does** — the parse
    moved to the package with the rest of the source format (rev. 10). Path-free
    is what criterion 4 rests on and is unchanged; `bytes` was never the point,
    and a document is one step *further* from a path than bytes were.

    Keeping `bytes` would have meant `load_package` serialising a parsed
    document back so this function could parse it again, and the two parsers do
    not agree: `ruamel.yaml` is YAML 1.2 and PyYAML's `safe_load` is 1.1, so
    `12:30` is the string on one side and the integer 750 on the other
    (`scratch/ui-yaml-2026-08/w3/probe_ruamel_semantics.py`). One document, two
    readings, decided by which side of a needless round trip you look from.

    The return is a bare problem list. The pair existed to carry `None` for "did
    not parse", and nothing here parses.
    """
    ...


def body_of(spec: Mapping[str, Any]) -> Body:
    """A task's or a validator's declared body. `{}` when it declares none."""
    ...


def subgraph_of(task: TaskSpec) -> tuple[Mapping[str, Any], ...]:
    """The declared expansion, **as written**. `()` for a leaf.

    Entries are returned unnormalised: no mark is defaulted, because
    `is_start` / `is_end` mean something only once an entry is a
    `task_graph.SubgraphEntry`, and that type is not this package's to name.
    `task_graph` normalises on top of this.

    **The key and the entry shape are named by no specification.** `task_graph`
    chose `task.subgraph` as `[{closure, is_start?, is_end?}]`; this accessor
    moved here so the *key* has one reader, and moving it does not promote the
    convention to a rule.
    """
    ...


def task_of(doc: ClosureDoc) -> TaskSpec:
    """The task spec inside a closure document — `closure` spec §2, key `task`.

    Returns the nested object **itself**, never a copy: both callers read
    further into what they get back. `{}` when the key is absent or not a
    mapping.
    """
    ...


def validator_agent_of(spec: Mapping[str, Any]) -> str | None:
    """The agent spec a **validator** names, or `None`.

    Not `agent_of`: `closure.agent_of` reads a closure document and returns
    `str`, this reads a validator spec and returns `str | None`, and both take a
    `Mapping` without raising on the wrong one.
    """
    ...


def load_package(pkg: TaskPackage, registries: Registries) -> LoadReport:
    """Discover, render in parallel, validate, and admit one package.

    Collects failures rather than raising: one broken spec must not hide the
    other nine. **Runs no cross-registry check** — neither the closure pass nor
    the two-way binding check nor the separation check, because each needs a
    registry this call may not have filled yet. Those run once, at the
    composition root (`docs/interfaces.md` §2).
    """
    ...
