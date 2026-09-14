"""What leaves `handoff/`.

The content and lifecycle layer. The runtime *slot* — `Handoff`,
`HandoffVersion`, `open_next()`, `seal()` — is `task_graph`'s and is not here;
the two layers meet at one point, `Handoff.type`, which names a kind this module
defines.

Declarations only. See `docs/interfaces.md` §4.2.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Literal, Protocol

from task_graph.ids import AgentId, HandoffId, TaskId

__all__ = [
    "Content",
    "ContentType",
    "DigestMismatch",
    "HandoffKind",
    "HandoffLoadReport",
    "HandoffStore",
    "Item",
    "Malformed",
    "Manifest",
    "NotContained",
    "PointerInvalid",
    "PointerMiss",
    "Scope",
    "Verdict",
    "check_contained",
    "resolve",
    "tree_digest",
]


# --------------------------------------------------------------------------- #
# Errors


class Malformed(ValueError):
    """Content that does not satisfy its kind: no README, a missing section, a
    locality-dependent path, an uncanonicalisable value."""


class DigestMismatch(ValueError):
    """A recomputed tree digest differs from the manifest's. Names both."""


class NotContained(ValueError):
    """A path does not resolve inside the zone it was checked against."""


class PointerInvalid(ValueError):
    """A malformed RFC 6901 pointer — the binding author's typo."""


class PointerMiss(LookupError):
    """A well-formed pointer that addresses nothing.

    Distinct from `PointerInvalid` on purpose, and this distinction is the whole
    reason spec §5.1 rev. 5 says Pointer rather than jsonpath: RFC 9535 §2.5.1.2
    forbids a valid JSONPath query from erroring, so no JSONPath implementation
    can tell a wrong path from an absent value.
    """


# --------------------------------------------------------------------------- #
# Content


class Scope(str, Enum):
    """One tag, four values, consumed exactly once — at publish, to pick a store.

    Storage, permission and retention all follow from *where the artefact lands*,
    never from reading the tag back at use time.
    """

    FIXED_REQUIRED = "fixed.required"
    FIXED_OPTIONAL = "fixed.optional"
    ADDONS_TEMP = "addons.temp"
    ADDONS_KNOWLEDGE = "addons.knowledge"


@dataclass(frozen=True)
class Item:
    """One entry of a handoff's `items`.

    Three-valued rather than two: a `code` handoff's `codes` is a directory and a
    `reproducible` handoff's `script` is one file, and the digest walks them
    differently.
    """

    key: str
    kind: Literal["file", "tree", "data"]
    path: Path | None = None
    value: Any | None = None


@dataclass(frozen=True)
class ContentType:
    """One of the four content types, as a row rather than a code branch."""

    name: str
    required_items: frozenset[str]
    optional_items: frozenset[str]
    readme_sections: tuple[str, ...]


@dataclass(frozen=True)
class Content:
    """A handoff version's content: the README plus the typed dictionary."""

    root: Path
    readme: Path
    items: Mapping[str, Item]


@dataclass(frozen=True)
class Manifest:
    """`manifest.yaml` — beside the content subtree, and outside the digest.

    `digest` is a **map**, not a string: in-toto's algorithm agility, the only
    cheap migration path, and it costs one nesting level today. `algorithm`
    namespaces the walk rule, so a change to it is a `v2` and never a silent
    redefinition.
    """

    digest: Mapping[str, str]
    algorithm: str
    kind: str
    producer: TaskId
    created_at: datetime


@dataclass(frozen=True)
class Verdict:
    """One validation result, persisted in `validation.yaml`.

    **This module owns the type**, because the layer that persists a record is
    the layer that has to keep it readable. `validator` re-exports the name and
    keeps `VerdictRecord` as its own view of one.

    The file is a *sibling* of `content/` and is excluded from the digest, so
    recording a verdict does not change the artefact's identity. It is created
    empty at publication rather than on first verdict: an empty `verdicts:` list
    says "nothing has checked this yet", a missing file says something is wrong.

    **`agent_id` is optional, and that is the whole of the decision.** A
    programmatic validator — a script body — has no agent, and the field was
    required, so `validator` had to fall back to *the producing agent's* id with
    `attributed: False` recorded beside it in `environment`. A record whose own
    field says the producer validated the artefact is the claim `validator` spec
    §8.1 forbids, annotated rather than avoided, in the one artefact criterion 8
    asserts over.

    The two alternatives were worse. A **sentinel** `AgentId` is a UUID: a reader
    who does not know it takes it for a real agent, and one who looks it up in
    `agent_mgr` finds nothing — a plausible value flowing on undetected, which is
    the failure class this system spent a day cataloguing. **Documenting** the
    fallback leaves the record stating a falsehood with the correction in a side
    channel.

    So `None` means *no agent ran*, and the reader's cost was measured before it
    was accepted: nothing in the tree read this field. `None` here is an
    **answer**, not a default — the producer's id in this slot was the default,
    and it is the most dangerous instance available, because attribution is the
    entire purpose of the field.

    This does not weaken criterion 8. The history must name the versioned agent,
    and it does so by recording the truth about one; a field filled with a false
    id defeats the criterion rather than satisfying it. Widening also costs no
    stored record: every `validation.yaml` written so far still reads.
    """

    validator: str
    result: bool
    strength: str
    dimension: str
    task_id: TaskId
    agent_id: AgentId | None
    environment: Mapping[str, Any]
    at: datetime


@dataclass(frozen=True)
class HandoffKind:
    """An admitted handoff spec, in the shape the store and the checks need."""

    name: str
    content_type: str
    items_schema: Mapping[str, Any]
    validators: tuple[str, ...]
    scope: Scope
    version: str | None = None


@dataclass(frozen=True)
class HandoffLoadReport:
    """The escape-hatch report — a value, not a log line.

    Criterion 12 asserts that a kind admitted without a validator appears by name
    in the startup report *and* the run record, and an assertion over a log
    capture is a test of the logging configuration.

    Distinct from `spec_loader.LoadReport`, which is `(admitted, problems)`.
    """

    admitted: Sequence[str]
    without_validator: Sequence[str]


# --------------------------------------------------------------------------- #
# Storage


class HandoffStore(Protocol):
    """Eleven operations, designed against the seven places a filesystem leaks.

    Two instances, one implementation: `handoff_store` and `knowledge_store`
    differ only in root. The differences the spec lists — lifetime, breadth of
    readership — are permission and retention, which are `env_mgr`'s and the
    root's, not behaviours of a store.

    **`allocate` / `seal` are `interfaces.md` §4.14's two halves**, and they
    split what `put` does into the two moments the ruling needs: a version
    directory that exists and can be granted *before* the body runs, and a
    commit that happens *after* it, over bytes the agent wrote itself.
    """

    def list_versions(self, hid: HandoffId) -> list[int]:
        """**Published versions only**, ascending. An allocated-but-unsealed
        directory is invisible here, so a failed attempt leaves a gap in the
        numbering rather than a version that cannot be read."""
        ...

    def latest(self, hid: HandoffId) -> int | None:
        """The highest published version, or `None` if nothing is published."""
        ...

    def get_manifest(self, hid: HandoffId, version: int) -> Manifest: ...

    def open_item(self, hid: HandoffId, version: int, key: str) -> BinaryIO: ...

    def copy_out(self, hid: HandoffId, version: int, dst: Path) -> Content:
        """Copy this version's `content/` into `dst`; return a `Content` over the
        copy. Verifies the digest first, raising `DigestMismatch`.

        **`dst` is mandatory and there is no `get_local_path`.** MLflow's
        equivalent returns the store's own path, so an agent handed the return
        value edits the store in place and nothing in the type says so. The
        guarantee is the signature, which is why the signature is tested.
        """
        ...

    def allocate(self, hid: HandoffId) -> int:
        """Reserve the next version, create its directory, return the number.

        **The dispatch half of §4.14.** The directory is what `env_mgr`'s
        kind-named grant resolves to, so it must exist before the body runs.
        It holds no manifest and is therefore not a published version.
        """
        ...

    def seal(self, hid: HandoffId, version: int, *, producer: TaskId) -> str | None:
        """Publish a version whose `content/` is already written in place.

        **The close half of §4.14**, and the same admission checks as `put`.

        Returns `None` when it published, otherwise **the reason the artefact
        was not publishable** — a string for the record, not a code to branch
        on. A refusal leaves the version unsealed, which is a hole and the
        honest record of an attempt that did not deliver.

        **Raises `NotSealable`, and only that**: no such version, or already
        published. Those say nothing about the content and are wiring bugs, so
        they escape rather than becoming a return value. The prospective
        caller is `agent`'s runner, which `interfaces.md` §4 forbids from
        importing this package — it cannot name an exception of mine to catch
        one, and `except Exception` would swallow the wiring bug. A return
        value crosses that boundary; an exception type does not.
        """
        ...

    def put(self, hid: HandoffId, content_dir: Path, *, producer: TaskId) -> int:
        """Publish `content_dir` as the next version; return its number.

        Runs the README check and the locality check **before** anything is
        created, raising `Malformed`; a handoff that reached storage malformed
        would need retracting, and nobody anywhere has solved retraction.

        `put` is the commit token, not `rename` — if rename were the interface,
        an object-store backend would have nothing to implement.
        """
        ...

    def exists(self, hid: HandoffId, version: int | None = None) -> bool: ...

    def record_verdict(self, hid: HandoffId, version: int, verdict: Verdict) -> None:
        """Append to `validation.yaml`. Does not move the digest."""
        ...

    def read_verdicts(self, hid: HandoffId, version: int) -> list[Verdict]: ...


# --------------------------------------------------------------------------- #
# Pure functions, importable without a store


def tree_digest(root: bytes) -> bytes:
    """sha256 over the `content/` subtree, git-shaped and specified exactly.

    Takes and walks **bytes**: `pathlib.Path` rejects bytes paths, and mixing the
    two is how a name comparison silently becomes `False`. Sorts by
    `os.fsencode(name)` in plain byte order — sorting the `str` puts a
    surrogate-escaped byte in the wrong place, and agrees with byte order for all
    valid UTF-8, so the bug would survive every ordinary test.
    """
    ...


def resolve(doc: Any, pointer: str) -> Any:
    """Resolve an RFC 6901 pointer into a handoff's content.

    Raises `PointerInvalid` for a malformed pointer and `PointerMiss` for one
    that addresses nothing. A JSON `null` is returned as `None` and is **not** a
    miss — three outcomes, three answers.
    """
    ...


def check_contained(candidate: Path, zone: Path) -> None:
    """Raise `NotContained` unless `candidate` resolves inside `zone`.

    Rejects `..` by policy *before* resolving, so a rejected path is reported as
    written rather than as resolved.

    **Fails closed: unresolvable means denied.** `validator`'s separation check
    needs the opposite direction — there, unresolvable must be treated as
    *inside*, because containment means reject — so importing this and negating
    at the call site would accept a dangling validator symlink. Two uses of one
    idea, two failure directions, and they are not one function.
    """
    ...
