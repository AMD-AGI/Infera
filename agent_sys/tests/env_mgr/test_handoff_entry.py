# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Criteria 19 and 21 — an agent works on a copy, and conventions come from a
knowledge handoff."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from env_mgr import meta
from env_mgr.fs import layout
from env_mgr.fs.domain import DomainKind, DomainRegistry
from task_graph.ids import HandoffId

from .stubs import Execution, Task, context


def _digest(root: str) -> str:
    """A tree digest, computed here rather than imported.

    `handoff` owns digests and this module owns copies (design §1.2), so
    criterion 19's *"its digest still verifies"* is asserted against a digest the
    test computes — which is also what keeps this test runnable before `handoff`
    exists.
    """
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            h.update(os.path.relpath(path, root).encode())
            h.update(Path(path).read_bytes())
    return h.hexdigest()


@pytest.fixture
def stored(tmp_path: Path) -> tuple[str, HandoffId, int]:
    store = tmp_path / "store"
    hid = HandoffId.new()
    version = 3
    # `content/`, because that is what a version directory holds and what
    # `stage` now copies. `handoff.allocate` creates it at dispatch; a fixture
    # that fabricated a bare `v<N>/` was staging a layout the store never
    # produces, and `stage` skipping it is the intended behaviour rather than
    # the bug — a hole is skipped, not half-copied.
    body = store / str(hid) / f"v{version}" / "content"
    body.mkdir(parents=True)
    (body / "readme.md").write_text("# the artefact\n")
    (body / "data.bin").write_bytes(b"\x00\x01\x02")
    return str(store), hid, version


# ---------------------------------------------------------- criterion 19


def test_agent_works_on_a_copy(tmp_path: Path, stored: tuple[str, HandoffId, int]) -> None:
    """Read from storage into the zone, work on the copy, never edit the stored
    artefact. Rule 2 of spec §6.3 is what makes a re-run comparable to the run
    before it."""
    store, hid, version = stored
    reg = DomainRegistry()
    reg.register("store", str(tmp_path / "root"), DomainKind.HANDOFF_STORAGE)
    task = Task(inputs=[hid])
    execution = Execution(attempt=0, input_versions={hid: version})
    zone = layout.create(task, execution, reg)

    staged = layout.stage_handoffs(task, execution, zone, context(domains=reg, store_root=store))[
        hid
    ]
    assert zone.contains(staged)
    assert Path(staged, "readme.md").read_text() == "# the artefact\n"

    # The agent edits its copy.
    Path(staged, "readme.md").write_text("# edited by the agent\n")
    # The stored artefact is the `content/` subtree, which is what was copied.
    stored_readme = Path(store, str(hid), f"v{version}", "content", "readme.md")
    assert stored_readme.read_text() == "# the artefact\n"


def test_stored_artefact_byte_identical(tmp_path: Path, stored: tuple[str, HandoffId, int]) -> None:
    store, hid, version = stored
    # `stage` copies `content/`, so `content/` is what the staged tree must be
    # identical to. Comparing against `v<N>/` would have compared the copy to a
    # tree holding `manifest.yaml` and the producer's `claim/` as well.
    source = Path(layout.handoff_version_dir(store, hid, version), "content")
    before = _digest(source)

    reg = DomainRegistry()
    reg.register("store", str(tmp_path / "root"), DomainKind.HANDOFF_STORAGE)
    task = Task(inputs=[hid])
    execution = Execution(attempt=0, input_versions={hid: version})
    zone = layout.create(task, execution, reg)
    staged = layout.stage_handoffs(task, execution, zone, context(domains=reg, store_root=store))[
        hid
    ]

    assert layout.trees_identical(str(source), staged)
    Path(staged, "data.bin").write_bytes(b"\xff")
    assert _digest(source) == before, "the stored artefact changed"


def test_staging_does_not_hand_a_consumer_the_producers_own_claim(
    tmp_path: Path, stored: tuple[str, HandoffId, int]
) -> None:
    """The exposure, named — otherwise the narrowing is only a fixture detail.

    `stage` copied the whole of ``v<N>/`` until `handoff` measured what that put
    in front of a consumer. The one that matters is ``claim/self_check.yaml``:
    the **producing agent's own assertion that it finished**, readable by the
    independent validator whose job is deciding exactly that. `validator`
    spec:676's `weak` goal validator is an agent making that judgement, and
    anchoring it on the producer's claim is what `validator` spec §8's
    *"the producer cannot"* table exists to prevent, even though no row names
    this direction.

    `manifest.yaml` and `validation.yaml` go with it, and losing them costs
    nothing: `validator` confirmed against their own code that the prior-verdict
    path is `store.read_verdicts`, and a body that needs a manifest should ask
    `store.get_manifest`, which verifies a digest where a staged copy does not.

    Asserted as **exact directory contents**, not as "claim is absent": a test
    naming only the file it fears is one new sibling from being wrong again, and
    a new sibling is precisely how this arose.
    """
    store, hid, version = stored
    version_dir = Path(layout.handoff_version_dir(store, hid, version))
    (version_dir / "manifest.yaml").write_text("digest: {sha256: deadbeef}\n")
    (version_dir / "validation.yaml").write_text("checks: []\n")
    (version_dir / "claim").mkdir()
    (version_dir / "claim" / "self_check.yaml").write_text("done_by_self_check: true\n")

    reg = DomainRegistry()
    reg.register("store", str(tmp_path / "root"), DomainKind.HANDOFF_STORAGE)
    task = Task(inputs=[hid])
    execution = Execution(attempt=0, input_versions={hid: version})
    zone = layout.create(task, execution, reg)
    staged = layout.stage_handoffs(task, execution, zone, context(domains=reg, store_root=store))[
        hid
    ]

    assert sorted(p.name for p in Path(staged).iterdir()) == ["data.bin", "readme.md"]


def test_a_staged_input_is_the_artefact_itself_not_a_content_subdirectory(
    tmp_path: Path, stored: tuple[str, HandoffId, int]
) -> None:
    """**The shape, pinned by name because `validator` records it.**

    They write the mapped path into `materials.json`, which is the declared name
    by which a body finds what it is validating, so whether a body's relative
    paths say ``<path>/result.json`` or ``<path>/content/result.json`` is a
    contract rather than a detail.

    It is the first, and it matches `handoff.copy_out` — ``copytree(<v>/content,
    dst)``, the artefact's own files landing *at* `dst`. Deliberately identical:
    the defect being fixed was two answers to one question, and preserving a
    ``content/`` level here would have kept a third alive while making every
    body quote `handoff`'s directory vocabulary to reach its own inputs.
    """
    store, hid, version = stored
    reg = DomainRegistry()
    reg.register("store", str(tmp_path / "root"), DomainKind.HANDOFF_STORAGE)
    task = Task(inputs=[hid])
    execution = Execution(attempt=0, input_versions={hid: version})
    zone = layout.create(task, execution, reg)
    staged = layout.stage_handoffs(task, execution, zone, context(domains=reg, store_root=store))[
        hid
    ]

    assert Path(staged, "readme.md").is_file(), "the artefact is at the mapped path"
    assert not Path(staged, "content").exists(), "no content/ level survives the copy"


def test_a_version_directory_with_no_content_stages_nothing(tmp_path: Path) -> None:
    """A hole is skipped, not half-copied.

    Every write route creates ``content/`` — `allocate` at dispatch, and `put`
    via ``copytree(src, <stage>/content)`` — so a version directory without one
    is **not a state the store produces**. Falling back to copying the whole of
    ``v<N>/`` would therefore reinstate the wide copy in exactly the situation
    where the layout is unexpected, which is when guessing is least safe.

    That fallback is not hypothetical: the probe that measured this change
    carried one, which is why it reported five green suites for a narrowing the
    real implementation does not perform.
    """
    store = tmp_path / "store"
    hid = HandoffId.new()
    (store / str(hid) / "v0").mkdir(parents=True)
    (store / str(hid) / "v0" / "manifest.yaml").write_text("digest: {}\n")

    reg = DomainRegistry()
    reg.register("store", str(tmp_path / "root"), DomainKind.HANDOFF_STORAGE)
    task = Task(inputs=[hid])
    execution = Execution(attempt=0, input_versions={hid: 0})
    zone = layout.create(task, execution, reg)
    staged = layout.stage_handoffs(
        task, execution, zone, context(domains=reg, store_root=str(store))
    )

    assert staged == {}, "a version with no artefact stages nothing, visibly"


def test_copy_out_refuses_to_copy_onto_itself(stored: tuple[str, HandoffId, int]) -> None:
    """`handoff`'s ``copy_out(hid, version, dst)`` has no default for `dst`
    because an agent handed the store's own path edits the store in place. The
    same reasoning one level down: this never returns the source."""
    store, hid, version = stored
    source = layout.handoff_version_dir(store, hid, version)
    with pytest.raises(ValueError, match="onto itself"):
        layout.copy_out(source, source)


# ---------------------------------------------------------- criterion 21


def test_conventions_come_from_a_knowledge_handoff(tmp_path: Path) -> None:
    """Changing the handoff changes behaviour without a code change.

    **Half the criterion**, and the design says which half is missing: the
    designated system-level task that *produces* such a handoff is unspecified,
    so this builds the artefact rather than receiving one. The consumption route
    is real, and it is version-selected like any other handoff.
    """
    store = tmp_path / "store"
    hid = HandoffId.new()

    def write(version: int, remote_root: str) -> None:
        body = store / str(hid) / f"v{version}"
        body.mkdir(parents=True)
        meta.save(
            meta.Meta(mappings=(meta.RemoteMapping("/local", remote_root, target="gpu-01"),)),
            str(body / meta.CONVENTIONS),
        )

    write(0, "/scratch/old")
    write(1, "/scratch/new")

    assert meta.from_knowledge(str(store), hid, 0).mapping_roots() == {"/local": "/scratch/old"}
    assert meta.from_knowledge(str(store), hid, 1).mapping_roots() == {"/local": "/scratch/new"}


def test_a_missing_knowledge_handoff_is_the_empty_default(tmp_path: Path) -> None:
    """Not an error: a run with no cluster conventions declared has no remote,
    and `prepare` skips the sync rather than refusing."""
    assert meta.from_knowledge(str(tmp_path), HandoffId.new(), 0) == meta.Meta()
