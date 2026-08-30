"""`FilesystemStore` — v1 storage, designed against the seven places a filesystem leaks.

Spec §6.1 fixes v1 as a filesystem tree and asks for a clear interface. The risk
is not that a filesystem is wrong; it is that **the filesystem becomes the
interface** and a second backend is then impossible. Prior art names exactly
where that happens, and each is answered by an absence:

| Leak | What this interface does |
|---|---|
| atomic directory rename | **`put` is the commit token, not `rename`** — an object store would otherwise have nothing to implement |
| directory listing vs prefix listing | no `list_items`; the manifest enumerates, so a store never answers "is this a directory" |
| empty directories | the digest records them, so a backend recreates rather than discovers them |
| append | none. A version is written once |
| `stat`, mtime, permissions | not in the interface; §4.4 excludes all of it from the digest anyway |
| locking | §6.3's allocator needs none |
| delete | **not in the Protocol at all** — `design.md` O3 |

Two instances, one implementation: `handoff_store` and `knowledge_store` differ
only in root. The differences spec §6.2 lists — outlives every run, broadly
readable — are lifetime and permission, which are `env_mgr`'s and the root's,
not behaviours of a store. The day a `KnowledgeStore` subclass is needed is the
day this interface was wrong.
"""

from __future__ import annotations

import io
import itertools
import os
import shutil
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Protocol

import yaml

from handoff import content as content_mod
from handoff import locality, readme
from handoff import verdict as verdict_mod
from handoff.digest import ALGORITHM, canonical, tree_digest
from handoff.errors import DigestMismatch, Malformed, NotSealable
from handoff.protocols import Content, HandoffKind, Manifest, Scope, Verdict
from task_graph.ids import HandoffId, TaskId

__all__ = [
    "CLAIM_DIR",
    "CONTENT_DIR",
    "MANIFEST_FILE",
    "STAGING_PREFIX",
    "FilesystemStore",
    "KindSource",
    "store_name_for",
    "version_dir",
]

CONTENT_DIR = "content"
MANIFEST_FILE = "manifest.yaml"

#: A sibling of `content/`, granted to the producing agent alongside it, and
#: **outside the digest** — `validation.yaml`'s precedent, in the other
#: direction. `content/` is the artefact; this is what the producer says *about*
#: the attempt, starting with `monitor` §4.1.2's `done_by_self_check`.
#:
#: A directory rather than a file for two reasons: a grant is on a directory,
#: so a file has no representation on `env_mgr`'s side at all; and a second
#: claim then needs no second ruling. `env_mgr/grants.py:137` spells it again
#: rather than importing it — they do not import this package, and an import
#: edge is permanent where a duplicated constant is one grep and one test.
CLAIM_DIR = "claim"

#: Reserved, and the lister filters it — MLflow's local repository does the
#: same. A half-written version must not be visible as a version.
STAGING_PREFIX = ".staging-v"

#: Where each scope tag lands. **The tag is consumed exactly once**, here, at
#: publish — storage, permission and retention all follow from *where the
#: artefact is*, never from reading the tag back at use time. That is why
#: criterion 14 is asserted by the path and not by a round-tripped string.
_STORE_FOR_SCOPE: Mapping[Scope, str] = {
    Scope.FIXED_REQUIRED: "handoff_store",
    Scope.FIXED_OPTIONAL: "handoff_store",
    Scope.ADDONS_TEMP: "playground",
    Scope.ADDONS_KNOWLEDGE: "knowledge_store",
}


def store_name_for(scope: Scope) -> str:
    """The registered name of the store a kind with this scope publishes to."""
    return _STORE_FOR_SCOPE[scope]


def version_dir(root: Path, hid: HandoffId, version: int) -> Path:
    """**The one function that computes a path**, and the on-disk shape is private.

    Bazel #23576 is the lesson: a path-shape change survived only because
    consumers use `file.path` rather than composing strings. Every other module
    asks for a path; none builds one.
    """
    return Path(root) / str(hid) / f"v{version}"


class KindSource(Protocol):
    """How a store learns which kind an id holds.

    `put(hid, content_dir, *, producer)` carries no kind, and it needs one for
    three things: the README's required sections, the `items` check against the
    kind's `items_schema`, and `Manifest.kind`. The slot's `Handoff.type` is
    `task_graph`'s and this package does not import its models, so the mapping
    is injected at composition instead.

    **A store without one reads but does not publish.** `put` raises rather
    than running a weakened check — see its docstring for what that cost before
    it raised.

    Satisfying it needs both halves of the mapping, which is why it is a
    Protocol and not a concrete type: `hid -> Handoff.type` lives on
    `task_graph`'s `HandoffMgr`, and `type -> HandoffKind` on
    `HandoffSpecRegistry`. Only the composition root holds both, and this
    package resolves nothing (`docs/interfaces.md` §4.2).
    """

    def kind_for(self, hid: HandoffId) -> HandoffKind | None: ...


class FilesystemStore:
    """`<root>/<hid>/v<N>/{content/,validation.yaml,manifest.yaml}`."""

    def __init__(
        self,
        root: Path | str | None,
        *,
        kinds: KindSource | None = None,
        oracles: locality.Oracles | None = None,
    ) -> None:
        if root is None:
            raise Malformed("a store needs a root; None is not one")
        self._root = Path(root)
        self._kinds = kinds
        self._oracles = oracles or locality.Oracles(store_root=self._root)

    @property
    def root(self) -> Path:
        """Read-only. A caller that wants a path inside asks `version_dir`."""
        return self._root

    # ---- reads ----

    def list_versions(self, hid: HandoffId) -> list[int]:
        """**Published versions only, and the gaps are the point.**

        Since `interfaces.md` §4.14 a version directory is *allocated at
        dispatch*, before the body runs, so `<hid>/v3/` can exist while the
        attempt that owns it is still running — or has failed and left it
        empty. **`MANIFEST_FILE` is what makes a version published**, because
        the manifest is written last and by the seal.

        A failed attempt therefore leaves a **hole**: v3 is absent from this
        list forever and v4 is allocated next. Holes are skipped, never
        compacted — renumbering would move an artefact a digest already names.

        Measured before it was filtered: an unpublished `v3` on top made
        `agent/gate.py:90` (`versions[-1]`, then `get_manifest`) raise a bare
        `FileNotFoundError`, and `cli/main.py:761`'s verdict loop raise
        `Malformed`. Filtering here fixes both readers without either of them
        learning that allocation exists.
        """
        return [n for n, path in self._version_dirs(hid) if (path / MANIFEST_FILE).is_file()]

    def latest(self, hid: HandoffId) -> int | None:
        """The highest **published** version, or `None` if nothing is published.

        A question, not raw material — `engineer_principle.md` §4.2. Every
        caller that writes `list_versions(hid)[-1]` has to invent an answer for
        the empty list, and each invents a different one: the gate reports an
        absence, the demo skips the slot, a validator would raise. The empty
        case has exactly one right answer per caller and none of them is
        `IndexError`.

        **It answers "highest published *at the moment you ask*", and it has no
        production caller yet.** Measured across the tree: the three live
        `.latest(` calls are all `task_graph`'s `handoff_mgr.latest`, which is
        the *slot* manager and a different object. This one is exercised only by
        `tests/handoff/conformance.py` and `tests/interfaces/test_composition.py`.

        So the first real caller gets to shape the contract, and one thing is
        worth knowing before it does: **a caller that wants the version some
        earlier decision was about must carry that number itself.** The store
        cannot reconstruct it — a producer re-run publishes a new version and
        this returns the new one, with a different digest.

        That is *not* a general hazard, and the narrowing is the honest form of
        a claim I first made too widely. `task_graph`'s dispatch re-asks its
        gate immediately before pinning, in the same lock (`scheduler.py:245`),
        so it has no stale window — and reading the newer version there is
        required rather than wrong, by its criterion 17. The hazard is specific
        to a caller that evaluates a gate **once** and pins **later**. Whether
        one exists is open.
        """
        published = self.list_versions(hid)
        return published[-1] if published else None

    def exists(self, hid: HandoffId, version: int | None = None) -> bool:
        """Existence means **published**, matching `list_versions`.

        An allocated-but-unsealed directory is not a version that exists. That
        keeps `agent/gate.py`'s two branches coherent: after a failed attempt
        `exists(hid)` is False and the gate reports `OUTPUT_ABSENT`, which is
        the truth about the attempt.
        """
        if version is None:
            return bool(self.list_versions(hid))
        return (version_dir(self._root, hid, version) / MANIFEST_FILE).is_file()

    def _version_dirs(self, hid: HandoffId) -> list[tuple[int, Path]]:
        """Every `v<N>/` on disk, published or not. **Internal**: allocation
        must see the unpublished ones or it would hand out a number in use."""
        base = self._root / str(hid)
        if not base.is_dir():
            return []
        out = [
            (int(entry.name[1:]), entry)
            for entry in base.iterdir()
            if entry.is_dir() and entry.name.startswith("v") and entry.name[1:].isdigit()
        ]
        return sorted(out)

    def get_manifest(self, hid: HandoffId, version: int) -> Manifest:
        path = self._require(hid, version, wanted="the manifest") / MANIFEST_FILE
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        try:
            return Manifest(
                digest=dict(doc["digest"]),
                algorithm=str(doc["algorithm"]),
                kind=str(doc["kind"]),
                producer=TaskId(str(doc["producer"])),
                created_at=datetime.fromisoformat(str(doc["created_at"])),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise Malformed(f"{path}: unreadable manifest: {exc}") from exc

    def open_item(self, hid: HandoffId, version: int, key: str) -> BinaryIO:
        """A binary stream over one item. A `tree` item is not a stream."""
        loaded = content_mod.load(self._require(hid, version, wanted="an item") / CONTENT_DIR)
        item = loaded.items.get(key)
        if item is None:
            raise Malformed(f"{hid} v{version}: no item {key!r} (have: {sorted(loaded.items)})")
        if item.kind == "tree":
            raise Malformed(f"{hid} v{version}: item {key!r} is a directory, not a stream")
        if item.kind == "data":
            return io.BytesIO(canonical(item.value))
        return open(item.path, "rb")  # noqa: SIM115 - the caller owns the handle

    def copy_out(self, hid: HandoffId, version: int, dst: Path) -> Content:
        """Copy this version's `content/` into `dst`; return a `Content` over the copy.

        **`dst` is mandatory and there is no `get_local_path`.** MLflow's
        equivalent returns the store's own path, so an agent handed the return
        value edits the store in place and nothing in the type says so. Making
        the parameter mandatory makes that failure unrepresentable — the
        guarantee is the signature, which is why the signature is tested.

        **The copy mode is named, not left to the caller.** Measured:
        `shutil.copytree`'s default dereferences symlinks into regular files at
        mode 0644, and since §4.2 hashes a symlink over its target text, an
        agent copying with the defaults would produce a different digest and no
        obvious cause.

        Verifies before returning, on **every** consumption. An unenforced
        integrity sidecar is worse than none — PEP 815 is removing `RECORD.jws`
        because neither pip nor uv validate it — and recomputation is
        I/O-bound at ~1000 MB/s, so there is no cache to invalidate.
        """
        src = self._require(hid, version, wanted="the content") / CONTENT_DIR
        dst = Path(dst)
        if dst.exists():
            raise Malformed(f"{dst} already exists; copy_out creates its destination")
        shutil.copytree(src, dst, symlinks=True)

        manifest = self.get_manifest(hid, version)
        got = tree_digest(os.fsencode(dst)).hex()
        want = manifest.digest.get("sha256")
        if got != want:
            raise DigestMismatch(
                f"{version_dir(self._root, hid, version)}: manifest records "
                f"sha256={want}, the copy at {dst} recomputes to sha256={got} "
                f"(algorithm {manifest.algorithm})"
            )
        return content_mod.load(dst)

    # ---- the write ----

    def allocate(self, hid: HandoffId) -> int:
        """Reserve the next version and create its directory; return the number.

        **`interfaces.md` §4.14, the dispatch half.** An output's version used
        to be pinned only when the attempt closed, so `env_mgr`'s kind-named
        grant had no `N` to resolve against and raised `UnresolvedGrant` for
        the whole of the attempt that was supposed to fill it. This is the call
        that pins it early: the caller allocates before the body runs, records
        `N` on the `Execution`, and `env_mgr/grants.py:96` resolves the grant to
        this very directory.

        **What is created is `v<N>/` and its `content/`, and no manifest.**
        Without the manifest it is not a published version, so no reader of
        `list_versions`, `latest` or `exists` can see it; `content/` is where
        the agent writes, and **it must exist here because it is the granted
        path.**

        That last part was a defect, and `env_mgr` measured it rather than
        hitting it in production. This used to create `v<N>/` *and nothing
        else*, on the premise that `v<N>/` was the agent's grant. `0c2df28`
        narrowed the grant to `v<N>/content/` — an agent that can write
        `manifest.yaml` now publishes its own unsealed version, since the
        manifest became the seal. A granted path must exist when
        `env_mgr.prepare` builds the ruleset, and neither outcome was
        survivable:

        | | |
        |---|---|
        | granted non-optional | `FileNotFoundError`, so **every output-producing dispatch dies in `prepare`** — no isolation, no start |
        | granted optional | the rule is silently dropped and the agent has nowhere to write, found out later. `env_mgr` design §5.4's evaporating allow-list |

        And the agent cannot create it itself: `mkdir` inside `v<N>/` needs
        write on `v<N>/`, which is exactly what the narrowing removes. **So the
        allocator creates every directory it expects to be granted** — that
        keeps `env_mgr` from becoming a second writer of this package's layout,
        which is what `env_mgr` design §1.2 defers here in the first place.

        **A second granted sibling for the producer's `done_by_self_check`
        claim is ruled and lands here**, next to `content/`, once its name is
        agreed. It is deliberately absent rather than guessed at.

        **Racing allocators cannot collide.** The same `os.mkdir` token `put`
        uses: `FileExistsError` means someone else took the number, so try the
        next. A read-max-then-add-one allocator hands v3 to two writers, which
        is the MLflow `FileStore` bug this avoids for free. `content/` is made
        only after the version directory is won, so a loser creates nothing.

        **No `KindSource` is needed and none is asked for.** Allocation makes a
        directory; publication is what needs a kind to check against, and `put`
        and `seal` are where that refusal lives. Requiring it here would block
        dispatch on a resolver only the seal can use.
        """
        base = self._root / str(hid)
        base.mkdir(parents=True, exist_ok=True)
        for n in itertools.count(self._next_guess(base)):
            target = version_dir(self._root, hid, n)
            try:
                os.mkdir(target)
            except FileExistsError:
                continue
            os.mkdir(target / CONTENT_DIR)
            os.mkdir(target / CLAIM_DIR)
            return n
        raise AssertionError("unreachable: itertools.count does not end")

    def seal(self, hid: HandoffId, version: int, *, producer: TaskId) -> str | None:
        """Publish a version the agent has already written into. **The other
        half of §4.14: `put` copies in, `seal` commits in place.**

        Returns **`None` when it published**, and otherwise **the reason the
        artefact was not publishable** — a string for the record, not a code to
        branch on. The only caller in prospect is `agent`'s runner, and
        `tests/interfaces/test_import_rules.py:42` lets it import
        `spec_loader`, `task_graph` and `monitor` — **not this package**. So it
        cannot name an exception of mine to catch one, and `except Exception`
        would swallow exactly the wiring bug below. A return value crosses that
        boundary and an exception type does not.

        **`NotSealable` is therefore the only thing this raises**, and it means
        the call could not have succeeded: no such version, or already
        published. A caller writes no `try` at all, and anything escaping
        `seal` is unambiguously a wiring bug.

        **An unpublishable attempt is not exceptional.** An agent that wrote
        nothing, or wrote something that is not a handoff, is an ordinary
        outcome the monitor decides about — `monitor` design §8, the runner
        reports and decides nothing. Raising for it would have made the normal
        path the exceptional one.

        The content is expected at `<root>/<hid>/v<N>/content/`, where the
        agent wrote it inside its own grant. Nothing is copied and nothing
        moves — the digest is taken over the bytes as they lie, so what a
        consumer verifies is what the producer wrote.

        **The same two admission checks as `put`, for the same reason**: a
        malformed handoff that reached storage would need retracting. A refusal
        leaves the directory unsealed, which is exactly a hole — the attempt
        failed, and the failure is recorded by the absence of a version rather
        than by a version that lies.

        **The manifest is written last, and atomically.** It is the token
        `list_versions` reads, so a crash between the two writes must leave the
        version unpublished rather than published-and-half-written.
        """
        # **Not `NotSealable`.** A store built without a `KindSource` is a
        # composition error, not a per-attempt one — it breaks `put` everywhere
        # too, and it is uniform and immediate rather than silent on retries.
        # It stays the `Malformed` that `put` has always raised for it.
        kind = self._kinds.kind_for(hid) if self._kinds is not None else None
        if kind is None:
            raise Malformed(
                f"cannot seal {hid}: this store has no kind for it, so the "
                f"README's required sections and the items check have nothing "
                f"to check against. Construct it as "
                f"FilesystemStore(root, kinds=<KindSource>) — a store without "
                f"one reads but does not write"
            )

        # `NotSealable`, and it raises rather than returning a reason: neither
        # of these says anything about the content, so neither is an outcome of
        # the attempt. See `errors.NotSealable` for the re-run case.
        target = version_dir(self._root, hid, version)
        if not target.is_dir():
            raise NotSealable(
                f"cannot seal {hid} v{version}: {target} does not exist. A version "
                f"is sealed where it was allocated; call allocate() first "
                f"(have: {[n for n, _ in self._version_dirs(hid)] or 'none'})"
            )
        if (target / MANIFEST_FILE).is_file():
            raise NotSealable(
                f"cannot seal {hid} v{version}: already published. A version is "
                f"written once; allocate a new one"
            )

        content_dir = target / CONTENT_DIR
        # **Empty, not absent.** `allocate` creates `content/` because it is the
        # granted path, so its existence says nothing about whether the agent
        # wrote. Emptiness is the honest test, and it is worth its own message:
        # *the agent wrote nothing* and *the agent wrote something malformed*
        # are different facts, and `monitor` criterion 5 — refused versus never
        # attempted — is exactly that distinction.
        if not content_dir.is_dir() or not any(content_dir.iterdir()):
            return (
                f"nothing was written to {content_dir}. That directory is the "
                f"agent's grant and it is empty, so this attempt produced no "
                f"content at all"
            )

        # The admission checks stay exceptions **inside** — they are how the
        # rest of this package reports bad content, and `put` needs them to
        # raise. Only the boundary changes shape.
        ctype = content_mod.content_type(kind.content_type)
        try:
            readme.check(content_dir, content_mod.required_sections(ctype))
            locality.check(content_dir, oracles=self._oracles)
            content_mod.check_items(content_mod.load(content_dir), ctype, kind.items_schema)
        except Malformed as exc:
            return str(exc)

        digest = tree_digest(os.fsencode(content_dir))
        verdict_mod.create_empty(target / verdict_mod.VERDICT_FILE)
        pending = target / f".{MANIFEST_FILE}.pending"
        self._write_manifest(pending, digest=digest, kind=kind.name, producer=producer)
        os.rename(pending, target / MANIFEST_FILE)
        return None

    def put(self, hid: HandoffId, content_dir: Path, *, producer: TaskId) -> int:
        """Publish `content_dir` as the next version; return its number.

        `content_dir` is the content directory itself — the one holding
        `README.md` and `items/` — and it is copied to `<v>/content/`.

        **The two admission checks run before anything is created.** A
        malformed handoff that reached storage would need retracting, and
        nobody anywhere has solved retraction; refusing at the door is the
        cheap half of a problem whose expensive half is unsolved everywhere.

        **A store with no `KindSource` cannot publish, and says so.** Both
        checks need the kind — the README's required sections come from the
        content type, and `items` is checked against the kind's
        `items_schema` — so a store without one could only ever publish a
        half-checked artefact. It used to do exactly that: criteria 2 and 3
        went unenforced and the manifest recorded `kind: ""`, while every test
        in this package passed because they all inject a resolver.

        That was a fallback deciding the absent case was normal. It is not:
        reads work without a resolver and publication does not.
        """
        content_dir = Path(content_dir)
        kind = self._kinds.kind_for(hid) if self._kinds is not None else None
        if kind is None:
            raise Malformed(
                f"cannot publish {hid}: this store has no kind for it, so the "
                f"README's required sections and the items check have nothing "
                f"to check against. Construct it as "
                f"FilesystemStore(root, kinds=<KindSource>) — a store without "
                f"one reads but does not write"
            )

        ctype = content_mod.content_type(kind.content_type)
        readme.check(content_dir, content_mod.required_sections(ctype))
        locality.check(content_dir, oracles=self._oracles)
        content_mod.check_items(content_mod.load(content_dir), ctype, kind.items_schema)

        base = self._root / str(hid)
        base.mkdir(parents=True, exist_ok=True)
        for n in itertools.count(self._next_guess(base)):
            stage = base / f"{STAGING_PREFIX}{n}"  # a SIBLING of the destination
            try:
                # An atomic allocator, free. MLflow's FileStore is read-max+1
                # and two writers both allocate v3; FileExistsError costs
                # nothing and cannot be ignored.
                os.mkdir(stage)
            except FileExistsError:
                continue
            try:
                shutil.copytree(content_dir, stage / CONTENT_DIR, symlinks=True)
                digest = tree_digest(os.fsencode(stage / CONTENT_DIR))
                self._write_manifest(
                    stage / MANIFEST_FILE,
                    digest=digest,
                    kind=kind.name,
                    producer=producer,
                )
                verdict_mod.create_empty(stage / verdict_mod.VERDICT_FILE)
                # ENOTEMPTY if v{n} exists: never overwrite, and fail loudly.
                os.rename(stage, base / f"v{n}")
                return n
            except BaseException:
                shutil.rmtree(stage, ignore_errors=True)
                raise
        raise AssertionError("unreachable: itertools.count does not end")

    def _next_guess(self, base: Path) -> int:
        """**Counts holes.** `list_versions` hides an allocated-but-unsealed
        directory from every reader; this one must not, or the next allocation
        would hand out a number an attempt is still writing into."""
        existing = [
            int(e.name[1:])
            for e in base.iterdir()
            if e.is_dir() and e.name.startswith("v") and e.name[1:].isdigit()
        ]
        return max(existing, default=-1) + 1

    @staticmethod
    def _write_manifest(path: Path, *, digest: bytes, kind: str, producer: TaskId) -> None:
        """The digest is a **map**, and the algorithm is namespaced.

        in-toto: *"multiple entries MAY be used for algorithm agility"*. It is
        the only cheap migration path and it costs one nesting level today.
        W&B prefixes its manifest digest; DVC did not, and its md5 migration
        cost a permanent second cache and a `dvc cache migrate` command.
        """
        path.write_text(
            yaml.safe_dump(
                {
                    "digest": {"sha256": digest.hex()},
                    "algorithm": ALGORITHM,
                    "kind": kind,
                    "producer": str(producer),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    # ---- verdicts ----

    def record_verdict(self, hid: HandoffId, version: int, verdict: Verdict) -> None:
        """Append to `validation.yaml`. **Does not move the digest.**"""
        verdict_mod.append(
            self._require(hid, version, wanted="verdicts") / verdict_mod.VERDICT_FILE, verdict
        )

    def read_verdicts(self, hid: HandoffId, version: int) -> list[Verdict]:
        return verdict_mod.read(
            self._require(hid, version, wanted="verdicts") / verdict_mod.VERDICT_FILE
        )

    # ---- internal ----

    def _require(self, hid: HandoffId, version: int, *, wanted: str) -> Path:
        """A read reaches **published** versions only.

        Measured before this required the manifest: reading an allocated-but-
        unsealed `v3` got past `is_dir()` and then raised a bare
        `FileNotFoundError` out of `get_manifest`'s `read_text` — an error from
        outside `handoff.errors`, which no caller can be catching on purpose.
        Refusing here names what was looked for instead.

        **`wanted` exists because this gate serves two questions and used to
        answer both in the manifest's vocabulary.** `validator` found it:
        `record_verdict` and `read_verdicts` come through here too, so a
        verdict call got *"has no published version 3"* — a manifest-flavoured
        answer to a question about `validation.yaml`, which is a sibling of
        `content/` and outside the digest.

        **The gate itself is unchanged, and deliberately.** Since §4.14 a
        verdict can only be recorded against a sealed version, which is a real
        coupling I introduced — `b1b356e` changed this from `path.is_dir()`
        without my noticing that these two call sites shared it. It is **not
        live**: `agent/runner.py:636-637` seals before the gate and
        `OUTPUT_VALIDATING` is later, so a version is always sealed by the time
        a verdict is written. `validator` verified that themselves and declined
        the lifecycle change that would decouple them, on the grounds that
        *what a refusal should be recorded as* is `monitor`'s open question and
        a second home for that fact would pre-empt it. **It returns the day
        F-D1 moves the seal after output validation, and not before.**
        """
        path = version_dir(self._root, hid, version)
        if not (path / MANIFEST_FILE).is_file():
            have = self.list_versions(hid)
            allocated = path.is_dir()
            raise Malformed(
                f"cannot read {wanted} of {hid} v{version}: it is not published "
                f"(published: {have or 'none'})"
                + (
                    "; the directory is allocated and unsealed — an attempt is "
                    "writing it, or one failed and left a hole"
                    if allocated
                    else ""
                )
            )
        return path
