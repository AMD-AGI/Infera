"""`StoreConformance` — a suite any `HandoffStore` backend runs against itself.

Capability goes in a conformance suite, **never a flags dict**. MLflow pushed
backend-specific operations into four ABCs; Arrow put 13 deviation predicates
in its *test* header (`allow_move_dir`, `have_implicit_directories`) and has no
`supports_*` field at runtime. That is `docs/design.md`'s "backends raise; no
capability matrix" arriving from a second direction, and it means v2 finds the
leaks at implementation time rather than in production.

A backend subclasses this and overrides `make_store`. Nothing else.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from handoff.digest import tree_digest
from handoff.errors import Malformed, NotSealable
from handoff.store import CLAIM_DIR
from task_graph.ids import HandoffId, TaskId
from tests.handoff.conftest import FixedKind, make_content, make_kind, open_kind


class StoreConformance:
    """The behaviour every backend owes, independent of how it stores bytes."""

    def make_store(self, tmp_path: Path, kinds: FixedKind):  # pragma: no cover - overridden
        raise NotImplementedError

    # -- helpers ------------------------------------------------------------

    def _store_and_id(self, tmp_path: Path):
        hid = HandoffId.new()
        return self.make_store(tmp_path, FixedKind({hid: make_kind()})), hid

    # -- the suite ----------------------------------------------------------

    def test_a_published_version_is_readable(self, tmp_path: Path) -> None:
        store, hid = self._store_and_id(tmp_path)
        version = store.put(hid, make_content(tmp_path / "c"), producer=TaskId.new())

        assert store.list_versions(hid) == [version]
        assert store.exists(hid, version) and store.exists(hid)
        assert store.get_manifest(hid, version).digest["sha256"]

    def test_versions_are_never_overwritten(self, tmp_path: Path) -> None:
        store, hid = self._store_and_id(tmp_path)
        a = store.put(hid, make_content(tmp_path / "a"), producer=TaskId.new())
        b = store.put(
            hid, make_content(tmp_path / "b", data={"env": {"gpu": "X"}}), producer=TaskId.new()
        )
        assert a != b
        assert store.get_manifest(hid, a).digest != store.get_manifest(hid, b).digest

    def test_copy_out_round_trips_the_digest(self, tmp_path: Path) -> None:
        store, hid = self._store_and_id(tmp_path)
        version = store.put(hid, make_content(tmp_path / "c"), producer=TaskId.new())
        out = store.copy_out(hid, version, tmp_path / "out")
        assert (
            tree_digest(os.fsencode(out.root)).hex()
            == store.get_manifest(hid, version).digest["sha256"]
        )

    def test_an_empty_directory_survives_a_round_trip(self, tmp_path: Path) -> None:
        """S3 fakes empty directories with zero-byte markers; the manifest
        records them here, so a backend recreates rather than discovers them."""
        hid = HandoffId.new()
        store = self.make_store(tmp_path, FixedKind({hid: open_kind()}))
        src = make_content(tmp_path / "c")
        (src / "items" / "logs").mkdir()
        version = store.put(hid, src, producer=TaskId.new())
        out = store.copy_out(hid, version, tmp_path / "out")
        assert (out.root / "items" / "logs").is_dir()

    def test_verdicts_start_empty_and_accumulate(self, tmp_path: Path) -> None:
        from tests.handoff.test_verdict import _verdict

        store, hid = self._store_and_id(tmp_path)
        version = store.put(hid, make_content(tmp_path / "c"), producer=TaskId.new())
        assert store.read_verdicts(hid, version) == []

        store.record_verdict(hid, version, _verdict())
        assert len(store.read_verdicts(hid, version)) == 1

    def test_recording_a_verdict_does_not_move_the_digest(self, tmp_path: Path) -> None:
        from tests.handoff.test_verdict import _verdict

        store, hid = self._store_and_id(tmp_path)
        version = store.put(hid, make_content(tmp_path / "c"), producer=TaskId.new())
        before = store.get_manifest(hid, version).digest
        store.record_verdict(hid, version, _verdict())
        assert store.get_manifest(hid, version).digest == before

    def test_an_absent_handoff_answers_rather_than_crashing(self, tmp_path: Path) -> None:
        store, _ = self._store_and_id(tmp_path)
        stranger = HandoffId.new()
        assert store.list_versions(stranger) == []
        assert not store.exists(stranger)

    def test_a_malformed_handoff_is_refused_and_publishes_nothing(self, tmp_path: Path) -> None:
        store, hid = self._store_and_id(tmp_path)
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(Malformed):
            store.put(hid, empty, producer=TaskId.new())
        assert store.list_versions(hid) == []

    # -- §4.14: allocate at dispatch, seal at close -------------------------

    def test_latest_is_the_highest_published_version(self, tmp_path: Path) -> None:
        store, hid = self._store_and_id(tmp_path)
        assert store.latest(hid) is None, "nothing published answers, it does not raise"
        a = store.put(hid, make_content(tmp_path / "a"), producer=TaskId.new())
        assert store.latest(hid) == a
        b = store.put(
            hid, make_content(tmp_path / "b", data={"env": {"gpu": "X"}}), producer=TaskId.new()
        )
        assert store.latest(hid) == max(a, b)

    def test_an_allocated_version_is_not_a_published_one(self, tmp_path: Path) -> None:
        """The whole reason pre-allocation is safe: the directory exists, and
        no reader of the published set can see it."""
        store, hid = self._store_and_id(tmp_path)
        published = store.put(hid, make_content(tmp_path / "a"), producer=TaskId.new())

        allocated = store.allocate(hid)
        assert allocated != published
        assert store.list_versions(hid) == [published]
        assert store.latest(hid) == published
        assert not store.exists(hid, allocated)
        assert store.exists(hid), "the handoff still exists; this version does not"

    def test_a_read_of_an_allocated_version_is_refused_by_name(self, tmp_path: Path) -> None:
        """Not an `IndexError`, not a `FileNotFoundError` — the store's own
        error, naming what it looked for."""
        store, hid = self._store_and_id(tmp_path)
        allocated = store.allocate(hid)
        for call in (
            lambda: store.get_manifest(hid, allocated),
            lambda: store.read_verdicts(hid, allocated),
            lambda: store.copy_out(hid, allocated, tmp_path / "out"),
        ):
            with pytest.raises(Malformed):
                call()

    def test_a_failed_attempt_leaves_a_hole_and_the_hole_is_skipped(self, tmp_path: Path) -> None:
        """Version numbers are **not** renumbered. v1 is allocated, its attempt
        fails, and v1 stays absent forever while the next allocation is v2."""
        store, hid = self._store_and_id(tmp_path)
        first = store.put(hid, make_content(tmp_path / "a"), producer=TaskId.new())
        hole = store.allocate(hid)  # the attempt fails; nothing seals it
        after = store.put(
            hid, make_content(tmp_path / "b", data={"env": {"gpu": "X"}}), producer=TaskId.new()
        )

        assert hole not in store.list_versions(hid)
        assert store.list_versions(hid) == [first, after]
        assert after > hole, "the hole is skipped, not reused"
        assert store.latest(hid) == after

    def test_seal_publishes_what_the_agent_wrote_in_place(self, tmp_path: Path) -> None:
        """§4.14's whole point: the bytes are never copied, so what a consumer
        verifies is what the producer wrote inside its own grant."""
        store, hid = self._store_and_id(tmp_path)
        version = store.allocate(hid)
        make_content(tmp_path / "written")
        # The agent writes into the granted directory itself. `allocate` already
        # made it, because it is the granted path and a grant on a path that
        # does not exist either raises in `prepare` or evaporates.
        shutil.copytree(
            tmp_path / "written",
            store.root / str(hid) / f"v{version}" / "content",
            dirs_exist_ok=True,
        )

        store.seal(hid, version, producer=TaskId.new())

        assert store.list_versions(hid) == [version]
        assert store.latest(hid) == version
        out = store.copy_out(hid, version, tmp_path / "out")
        assert (
            tree_digest(os.fsencode(out.root)).hex()
            == store.get_manifest(hid, version).digest["sha256"]
        )

    def test_seal_refuses_a_malformed_version_and_leaves_a_hole(self, tmp_path: Path) -> None:
        store, hid = self._store_and_id(tmp_path)
        version = store.allocate(hid)
        # Something was written, and it is not a handoff: no README, no items.
        (store.root / str(hid) / f"v{version}" / "content" / "stray.txt").write_text("x")

        why = store.seal(hid, version, producer=TaskId.new())
        assert why, "a refusal reports its reason rather than raising"
        assert store.list_versions(hid) == [], "a refused seal publishes nothing"

    def test_allocate_creates_the_directory_the_grant_names(self, tmp_path: Path) -> None:
        """**`content/` must exist at allocation, not at first write.**

        `interfaces.md` §4.14 with `0c2df28`'s narrowing grants the agent
        `v<N>/content/`, and `env_mgr` measured both outcomes for a granted
        path that does not exist: non-optional raises `FileNotFoundError` in
        `prepare` — no isolation, no start, on every output-producing
        dispatch — and optional drops the rule silently. The agent cannot
        create it either, since `mkdir` inside `v<N>/` needs write on `v<N>/`,
        which the narrowing removes.
        """
        store, hid = self._store_and_id(tmp_path)
        version = store.allocate(hid)
        vdir = store.root / str(hid) / f"v{version}"
        for granted in ("content", "claim"):
            assert (vdir / granted).is_dir(), f"{granted}/ exists before the body runs"
            assert list((vdir / granted).iterdir()) == [], f"{granted}/ is empty"
        assert store.list_versions(hid) == [], "and still nothing is published"

    def test_the_claim_directory_is_outside_the_digest(self, tmp_path: Path) -> None:
        """`claim/` is what the producer says *about* the attempt, so it must
        not move the artefact's identity. That is the property that ruled out
        every location inside `content/`.

        **Two versions, identical content, one of them claimed** — because the
        obvious single-version assertion cannot fail. Driving the first draft
        against a deliberate break (the claim written *inside* `content/`)
        showed that `copy_out`'s digest still matched the manifest: both are
        computed over the same `content/`, so they agree no matter what is in
        it. Comparing two versions is what makes the claim's absence from the
        digest observable.

        `env-mgr` passed on the general form of that trap after their own
        positive control turned out to be silenceable by `-q`.
        """
        store, hid = self._store_and_id(tmp_path)

        plain = store.allocate(hid)
        make_content(store.root / str(hid) / f"v{plain}" / "content")
        store.seal(hid, plain, producer=TaskId.new())

        claimed = store.allocate(hid)
        vdir = store.root / str(hid) / f"v{claimed}"
        make_content(vdir / "content")
        (vdir / CLAIM_DIR / "self_check.yaml").write_text("done: true\n", encoding="utf-8")
        store.seal(hid, claimed, producer=TaskId.new())

        assert store.get_manifest(hid, claimed).digest == store.get_manifest(hid, plain).digest, (
            "identical content, one claimed: a claim must not move the artefact's identity"
        )

        out = Path(store.copy_out(hid, claimed, tmp_path / "out").root)
        assert not (out / CLAIM_DIR).exists(), "copy_out carries content/, not the claim"
        assert not (out / "self_check.yaml").exists()
        assert (out / "README.md").is_file(), "and it did carry the content"

    def test_seal_says_so_when_the_agent_wrote_nothing(self, tmp_path: Path) -> None:
        """Distinct from malformed, because `monitor` criterion 5 is exactly
        *refused* versus *never attempted* — and `content/` now always exists,
        so its presence can no longer stand in for the agent having written."""
        store, hid = self._store_and_id(tmp_path)
        version = store.allocate(hid)
        assert "nothing was written" in (store.seal(hid, version, producer=TaskId.new()) or "")

    def test_seal_never_republishes(self, tmp_path: Path) -> None:
        store, hid = self._store_and_id(tmp_path)
        version = store.put(hid, make_content(tmp_path / "a"), producer=TaskId.new())
        with pytest.raises(NotSealable):
            store.seal(hid, version, producer=TaskId.new())

    def test_seal_without_an_allocation_is_refused(self, tmp_path: Path) -> None:
        store, hid = self._store_and_id(tmp_path)
        with pytest.raises(NotSealable):
            store.seal(hid, 0, producer=TaskId.new())

    def test_a_bad_artefact_returns_and_a_bad_call_raises(self, tmp_path: Path) -> None:
        """**The distinction a caller acts on, and it crosses a package
        boundary** — so it is a return value on one side and an exception on
        the other, and asserted rather than left to the docstrings.

        `agent`'s runner is the only caller in prospect and
        `tests/interfaces/test_import_rules.py:42` forbids it from importing
        this package. It cannot name `Malformed` to catch it, and
        `except Exception` would swallow the wiring bug — which is the re-run
        case, where a second body writes into a sealed version and the loop
        looks like it worked while the output is discarded. Silent, and only
        on retries.
        """
        store, hid = self._store_and_id(tmp_path)

        assert store.seal(hid, store.allocate(hid), producer=TaskId.new()), (
            "an unpublishable attempt is an ordinary outcome: it reports, it does not raise"
        )

        published = store.put(hid, make_content(tmp_path / "a"), producer=TaskId.new())
        with pytest.raises(NotSealable):
            store.seal(hid, published, producer=TaskId.new())

    def test_a_successful_seal_returns_none(self, tmp_path: Path) -> None:
        """The positive control for the two tests above: a truthy return has
        to mean *refused*, so a successful seal must be falsy — and `None`
        rather than `""`, so nothing reads a reason that is not there."""
        store, hid = self._store_and_id(tmp_path)
        version = store.allocate(hid)
        make_content(store.root / str(hid) / f"v{version}" / "content")
        assert store.seal(hid, version, producer=TaskId.new()) is None
        assert store.latest(hid) == version
