"""Criteria 13 and 14, plus the store properties a refactor could quietly lose.

Two of these test a claim that is **structural** rather than behavioural —
`test_copy_out_refuses_to_return_store_path` and `test_staging_is_a_sibling` —
and they exist because the design argues that the structure *is* the guarantee.
A claim of that kind should fail loudly when someone adds a convenience
overload.
"""

from __future__ import annotations

import inspect
import os
import shutil
import threading
from pathlib import Path

import pytest

from handoff import FilesystemStore, Scope, store_name_for, version_dir
from handoff.errors import Malformed
from handoff.protocols import HandoffStore
from handoff.store import STAGING_PREFIX, handoff_dir
from task_graph.ids import HandoffId, TaskId
from tests.handoff.conftest import FixedKind, make_content, make_kind, open_kind


def test_filesystem_store_satisfies_the_protocol() -> None:
    """Every operation the Protocol declares, with the same signature.

    **This compares types, not names** — `docs/interfaces.md` §8.7, where five
    packages wrote guards over field *names* and a name survives exactly the
    change that breaks a caller. `Signature.__eq__` includes annotations, so a
    `dst: Path` quietly becoming `dst: str` fails here with the name unchanged.
    Driven against a deliberately drifted subclass to confirm it, not assumed.

    One honest limit: `protocols.py` and `store.py` both carry
    `from __future__ import annotations`, so the annotations compared are
    **source text**, not resolved objects. `Path` and `pathlib.Path` are the
    same type and would not compare equal here. That direction is a false
    positive, which is the safe one.

    **The floor is not decoration.** This loop is a *discovery* loop: it checks
    whatever `getmembers` returns, so if that ever returns nothing it checks
    nothing and passes. Measured — the same body over an empty `Protocol`
    passes — and it is one bad import away from being the object it iterates.
    `agent`'s `9528305` is the same hole in a different instrument: their AST
    scan for `store.<attr>` went vacuous when the local was renamed, so a
    one-word edit silenced the guard. **A discovery loop needs a floor, and the
    floor has to be the thing that fails.**
    """
    declared_names = {
        n for n, _ in inspect.getmembers(HandoffStore, inspect.isfunction) if not n.startswith("_")
    }
    assert declared_names >= {
        "allocate",
        "copy_out",
        "exists",
        "get_manifest",
        "latest",
        "list_versions",
        "open_item",
        "put",
        "read_verdicts",
        "record_verdict",
        "seal",
    }, f"the Protocol lost members, so the loop below would check less: {sorted(declared_names)}"

    for name in sorted(declared_names):
        declared = getattr(HandoffStore, name)
        got = getattr(FilesystemStore, name, None)
        assert got is not None, f"FilesystemStore is missing {name}"
        assert inspect.signature(got) == inspect.signature(declared), name


def test_the_protocol_guard_can_actually_fail() -> None:
    """A guard nobody has watched fail is a guard nobody should trust.

    §8.7 asks for this explicitly. Drifts the *type* of one parameter while
    keeping every name identical — the case a name-comparing guard misses — and
    asserts the check rejects it.
    """

    class TypeDrifted(FilesystemStore):
        def copy_out(self, hid: HandoffId, version: int, dst: str) -> object:  # type: ignore[override]
            ...

    declared = inspect.signature(HandoffStore.copy_out)
    assert inspect.signature(FilesystemStore.copy_out) == declared, "the real one agrees"
    assert inspect.signature(TypeDrifted.copy_out) != declared, (
        "a parameter whose type changed while its name did not must be caught; "
        "a guard comparing names would pass this"
    )


def test_copy_out_refuses_to_return_store_path() -> None:
    """§5.1. MLflow's `download_artifacts(dst_path=None)` returns the store's
    own path, so an agent handed the return value edits the store in place and
    nothing in the type says so. **The guarantee is the signature**, so the
    signature is what is tested."""
    sig = inspect.signature(FilesystemStore.copy_out)
    assert sig.parameters["dst"].default is inspect.Parameter.empty
    assert not hasattr(FilesystemStore, "get_local_path")


def test_copy_out_returns_a_copy_the_agent_may_edit(kinded_store, tmp_path: Path) -> None:
    """Spec §6.3: the agent works on the copy, so a re-run is comparable to the
    run before it — the input is the same bytes both times."""
    store, hid = kinded_store
    version = store.put(hid, make_content(tmp_path / "c"), producer=TaskId.new())
    content = store.copy_out(hid, version, tmp_path / "play")

    (content.root / "items" / "result").write_text("edited\n")
    stored = store.copy_out(hid, version, tmp_path / "play2")
    assert (stored.root / "items" / "result").read_text() == "42\n"


def test_copy_out_will_not_overwrite_its_destination(kinded_store, tmp_path: Path) -> None:
    store, hid = kinded_store
    store.put(hid, make_content(tmp_path / "c"), producer=TaskId.new())
    (tmp_path / "occupied").mkdir()
    with pytest.raises(Malformed, match="already exists"):
        store.copy_out(hid, 0, tmp_path / "occupied")


def test_copy_out_preserves_symlinks(tmp_path: Path) -> None:
    """`shutil.copytree`'s **default dereferences**, and since a symlink is
    hashed over its target text, an agent copying with the defaults would
    produce a different digest with no obvious cause. The copy mode is named
    here so a caller cannot get it wrong."""
    hid = HandoffId.new()
    store = FilesystemStore(tmp_path / "s", kinds=FixedKind({hid: open_kind()}))
    src = make_content(tmp_path / "c")
    os.symlink("result", src / "items" / "alias")
    version = store.put(hid, src, producer=TaskId.new())

    out = store.copy_out(hid, version, tmp_path / "out")
    assert (out.root / "items" / "alias").is_symlink()


def test_put_is_the_commit_token_and_rename_is_not_the_interface() -> None:
    """If rename were the interface, an object-store backend would have nothing
    to implement — Arrow's S3 refuses directory moves outright, and Hadoop S3A
    documents that callers cannot rely on atomic renames in a commit algorithm."""
    names = {n for n, _ in inspect.getmembers(HandoffStore, inspect.isfunction)}
    assert "put" in names
    assert {"rename", "append", "delete_version", "list_items"} & names == set()


def test_staging_is_a_sibling(kinded_store, tmp_path: Path, monkeypatch) -> None:
    """§6.3. `/tmp` and the working tree are different filesystems on this
    machine, so a `/tmp` stage cannot be renamed into the store — the rename
    falls back to a copy or fails, and atomicity ends up decided by mount
    layout. Asserted so nobody "tidies" the staging directory elsewhere."""
    store, hid = kinded_store
    seen: list[Path] = []
    real_copytree = shutil.copytree

    def spy(src, dst, *a, **kw):
        seen.append(Path(dst))
        return real_copytree(src, dst, *a, **kw)

    monkeypatch.setattr("handoff.store.shutil.copytree", spy)
    version = store.put(hid, make_content(tmp_path / "c"), producer=TaskId.new())

    staged = seen[0]
    assert staged.parent.parent == version_dir(store.root, hid, version).parent
    assert staged.parent.name.startswith(STAGING_PREFIX)


def test_a_failed_put_leaves_no_staging_directory(tmp_path: Path) -> None:
    """The store is **kinded** here on purpose. With no `KindSource`, `put` now
    refuses before it looks at the content at all — so this test would still
    have raised `Malformed`, and would have stopped testing the README refusal
    and the staging cleanup it is named for."""
    hid = HandoffId.new()
    store = FilesystemStore(tmp_path / "s", kinds=FixedKind({hid: make_kind()}))
    bad = tmp_path / "bad"
    bad.mkdir()  # no README

    with pytest.raises(Malformed, match="README.md"):
        store.put(hid, bad, producer=TaskId.new())
    assert store.list_versions(hid) == []
    assert list(handoff_dir(store.root, hid).glob(f"{STAGING_PREFIX}*")) == []


def test_a_store_with_no_kind_source_reads_but_does_not_publish(tmp_path: Path) -> None:
    """The fallback that used to sit here decided the absent case was normal.

    It is not. As `interfaces.md` §2 builds it — `FilesystemStore(handoff_root)`,
    no resolver — `put` used to publish a handoff whose README was missing four
    of its five required sections, carrying an item no content type defines,
    with `kind: ""` in the manifest. **Criteria 2 and 3 were unenforced on the
    production path** while all 135 tests here passed, because every one of them
    injects a resolver.

    Reads need no kind and still work; publication refuses and names the wiring.
    """
    hid = HandoffId.new()
    unkinded = FilesystemStore(tmp_path / "s")  # exactly the composition root's call
    content = make_content(tmp_path / "c")

    with pytest.raises(Malformed, match="has no kind for it"):
        unkinded.put(hid, content, producer=TaskId.new())
    assert unkinded.list_versions(hid) == [], "and nothing was created"

    # The same store reads a version another one published.
    writer = FilesystemStore(tmp_path / "s", kinds=FixedKind({hid: make_kind()}))
    version = writer.put(hid, content, producer=TaskId.new())
    assert unkinded.get_manifest(hid, version).kind == "trace"
    assert unkinded.copy_out(hid, version, tmp_path / "out").root.is_dir()


def test_an_unknown_id_is_refused_even_with_a_kind_source(tmp_path: Path) -> None:
    """A resolver that answers `None` for this id is the same situation as no
    resolver, and must not be a quieter version of it."""
    known, stranger = HandoffId.new(), HandoffId.new()
    store = FilesystemStore(tmp_path / "s", kinds=FixedKind({known: make_kind()}))
    with pytest.raises(Malformed, match="has no kind for it"):
        store.put(stranger, make_content(tmp_path / "c"), producer=TaskId.new())


def test_concurrent_put_allocates_distinct_versions(tmp_path: Path) -> None:
    """MLflow's FileStore is read-max+1 and its own `overwrite=False` guard
    ignores the argument, so two writers both allocate v3 and the second wins.
    `os.mkdir` is a free atomic allocator: `FileExistsError` cannot be ignored."""
    hid = HandoffId.new()
    store = FilesystemStore(tmp_path / "s", kinds=FixedKind({hid: make_kind()}))
    sources = [make_content(tmp_path / f"c{i}") for i in range(8)]

    got: list[int] = []
    lock = threading.Lock()

    def publish(src: Path) -> None:
        n = store.put(hid, src, producer=TaskId.new())
        with lock:
            got.append(n)

    threads = [threading.Thread(target=publish, args=(s,)) for s in sources]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(got) == list(range(8)), "a version was lost or reused"
    assert store.list_versions(hid) == list(range(8))


def test_versions_are_contiguous_integers_and_the_digest_is_the_identity(
    kinded_store, tmp_path: Path
) -> None:
    store, hid = kinded_store
    first = store.put(hid, make_content(tmp_path / "a"), producer=TaskId.new())
    second = store.put(
        hid, make_content(tmp_path / "b", data={"env": {"gpu": "X"}}), producer=TaskId.new()
    )
    assert (first, second) == (0, 1)
    assert store.get_manifest(hid, 0).digest != store.get_manifest(hid, 1).digest
    assert store.get_manifest(hid, 0).algorithm == "agent_sys.handoff.tree.v1"


def test_the_digest_is_a_map_so_a_second_algorithm_is_a_row(kinded_store, tmp_path: Path) -> None:
    """in-toto: *"multiple entries MAY be used for algorithm agility"*. DVC did
    not namespace and its md5 migration cost a permanent second cache."""
    store, hid = kinded_store
    store.put(hid, make_content(tmp_path / "c"), producer=TaskId.new())
    digest = store.get_manifest(hid, 0).digest
    assert isinstance(digest, dict) and set(digest) == {"sha256"}


def test_scope_tags_land_where_declared() -> None:
    """Criterion 14, asserted by **where the artefact lands**, not by reading
    the tag back — a test that read the tag would be testing that a string
    round-trips. The tag is consumed exactly once, at publish."""
    assert store_name_for(Scope.FIXED_REQUIRED) == "handoff_store"
    assert store_name_for(Scope.FIXED_OPTIONAL) == "handoff_store"
    assert store_name_for(Scope.ADDONS_KNOWLEDGE) == "knowledge_store"
    assert store_name_for(Scope.ADDONS_TEMP) == "playground"
    assert len({store_name_for(s) for s in Scope}) == 3


def test_knowledge_instance_is_separate(tmp_path: Path) -> None:
    """Criterion 13: two `FilesystemStore` objects with different roots and
    nothing else different. The differences spec §6.2 lists — outlives every
    run, broadly readable — are lifetime and permission, which are `env_mgr`'s
    and the root's, not behaviours of a store. There is no `KnowledgeStore`."""
    hid = HandoffId.new()
    kinds = FixedKind({hid: make_kind(name="cluster_notes", scope=Scope.ADDONS_KNOWLEDGE)})
    handoffs = FilesystemStore(tmp_path / "handoffs", kinds=kinds)
    knowledge = FilesystemStore(tmp_path / "knowledge", kinds=kinds)

    assert type(handoffs) is type(knowledge)
    knowledge.put(hid, make_content(tmp_path / "c"), producer=TaskId.new())

    assert knowledge.exists(hid, 0)
    assert not handoffs.exists(hid)
    assert version_dir(tmp_path / "knowledge", hid, 0).is_dir()
    assert not handoff_dir(tmp_path / "handoffs", hid).exists()


def test_a_store_needs_a_root(tmp_path: Path) -> None:
    with pytest.raises(Malformed, match="needs a root"):
        FilesystemStore(None)


def test_an_absent_version_names_what_exists_and_what_was_wanted(store, tmp_path: Path) -> None:
    """Two facts in the message, and `validator` asked for the second.

    `_require` serves content reads and verdict reads alike, and used to answer
    both in the manifest's vocabulary — so a `read_verdicts` refusal said *"has
    no published version 3"* about a file that is a sibling of `content/` and
    outside the digest. It now names what the caller was after.
    """
    hid = HandoffId.new()
    assert store.list_versions(hid) == []
    assert not store.exists(hid)
    with pytest.raises(Malformed, match="published: none"):
        store.get_manifest(hid, 0)
    with pytest.raises(Malformed, match="the manifest"):
        store.get_manifest(hid, 0)
    with pytest.raises(Malformed, match="verdicts"):
        store.read_verdicts(hid, 0)


def test_open_item_refuses_a_directory_and_names_the_keys(tmp_path: Path) -> None:
    hid = HandoffId.new()
    store = FilesystemStore(tmp_path / "s", kinds=FixedKind({hid: open_kind()}))
    src = make_content(tmp_path / "c")
    (src / "items" / "logs").mkdir()
    store.put(hid, src, producer=TaskId.new())

    with pytest.raises(Malformed, match="is a directory"):
        store.open_item(hid, 0, "logs")
    with pytest.raises(Malformed, match="have: "):
        store.open_item(hid, 0, "absent")
    with store.open_item(hid, 0, "script") as fh:
        assert fh.read() == b"echo hi\n"


def test_put_refuses_a_malformed_handoff_before_anything_is_created(
    kinded_store, tmp_path: Path
) -> None:
    """The two admission checks run **before** publication. A malformed handoff
    that reached storage would need retracting, and nobody anywhere has solved
    retraction — `delete_version` is deliberately absent for the same reason."""
    store, hid = kinded_store
    src = make_content(tmp_path / "c", sections=("Purpose",))
    with pytest.raises(Malformed, match="How to run"):
        store.put(hid, src, producer=TaskId.new())
    assert store.list_versions(hid) == []
