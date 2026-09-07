"""The gate, driven against the real `FilesystemStore` exactly once.

`docs/interfaces.md` §8.7: drift has a trigger, **wrong-on-the-day has none** —
nothing ever happens and the stub is as wrong on day 300 as on day 1. The only
thing that catches it is running the stub's subject rather than the stub.

It caught one here. `tests/agent/conftest.py::StubManifest` carried an `items`
mapping, and `handoff.protocols.Manifest` has five fields — `digest`,
`algorithm`, `kind`, `producer`, `created_at` — and has never had `items`. The
gate's `getattr(manifest, "items", None)` therefore returned `None` in
production for every handoff ever published, `_executable` returned early, and
**`OUTPUT_NOT_EXECUTABLE` was unreachable while its unit tests were green**.

A store on `tmp_path` needs no credentials, no network and no sandbox, so the
cost of this file is that it is a *real* test rather than a stricter stub —
which `handoff` warned is the reason nobody writes it: the expensive part of
the check is the part the code does not use.

Tests are not under §4's import rule (`tests/interfaces/test_handoff_layout.py`
states the terms), so this file may import `handoff`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent.gate import run_gate
from handoff import content as content_mod
from handoff.protocols import HandoffKind
from handoff.store import FilesystemStore
from monitor.protocols import EventKind
from task_graph.ids import HandoffId, TaskId

SECTIONS = ("Purpose", "How to run", "Result", "Environment", "Watch out")


def _kind() -> HandoffKind:
    return HandoffKind(
        name="trace",
        content_type="reproducible",
        items_schema={
            "type": "object",
            "properties": {"script": {}, "result": {}, "env": {}},
            "required": ["env"],
            "additionalProperties": False,
        },
        validators=(),
        scope="fixed.required",
        version=None,
    )


def _content(root: Path, *, executable: bool) -> Path:
    """A publishable `content/` whose `script` item is or is not runnable."""
    items = root / content_mod.ITEMS_DIR
    items.mkdir(parents=True, exist_ok=True)
    body = ["# Trace", ""]
    for name in SECTIONS:
        body += [f"## {name}", "", f"Real prose for {name}.", ""]
    (root / "README.md").write_text("\n".join(body), encoding="utf-8")

    script = items / "script"
    script.write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
    script.chmod(0o755 if executable else 0o644)
    (items / "result").write_text("42\n", encoding="utf-8")
    content_mod.write_items_json(items, {"env": {"gpu": "MI300X"}})
    return root


@pytest.fixture
def published(tmp_path: Path):
    """`(store, publish)` — publish returns the id of a real, readable version."""
    hid = HandoffId.new()
    store = FilesystemStore(tmp_path / "handoffs", kinds=_FixedKind(hid, _kind()))

    def publish(*, executable: bool) -> HandoffId:
        src = _content(tmp_path / f"c-{executable}", executable=executable)
        store.put(hid, src, producer=TaskId.new())
        return hid

    return store, publish


class _FixedKind:
    """Satisfies `handoff.store.KindSource` with one entry."""

    def __init__(self, hid: HandoffId, kind: HandoffKind) -> None:
        self._hid, self._kind = hid, kind

    def kind_for(self, hid: HandoffId) -> HandoffKind | None:
        return self._kind if hid == self._hid else None


def test_a_real_manifest_has_no_items_field(published) -> None:
    """The defect itself, pinned so it cannot come back as a `getattr` default.

    This is the fact the stub contradicted. If `handoff` ever adds `items`, this
    test fails and the gate can go back to the cheap pre-check on purpose.
    """
    store, publish = published
    hid = publish(executable=True)
    manifest = store.get_manifest(hid, store.list_versions(hid)[-1])
    assert not hasattr(manifest, "items")


def test_a_non_executable_script_is_reported(published) -> None:
    """The failure that was unreachable. Mode 644 on a declared `script`."""
    store, publish = published
    hid = publish(executable=False)
    failures = run_gate([hid], {}, store=store, budget=None)
    assert [f.kind for f in failures] == [EventKind.OUTPUT_NOT_EXECUTABLE]
    assert failures[0].handoff_id == hid


def test_an_executable_script_passes(published) -> None:
    """The control. Without it the check above is satisfied by any failure."""
    store, publish = published
    hid = publish(executable=True)
    assert run_gate([hid], {}, store=store, budget=None) == []


def test_the_item_key_is_read_from_the_content_not_the_manifest(published) -> None:
    """Why the fix has to copy out: nothing cheaper knows the keys.

    `Manifest.digest` is a whole-tree digest, not a per-item map, and the
    Protocol exposes no listing call — `copy_out` is the only way to learn that
    a `script` item exists at all.
    """
    store, publish = published
    hid = publish(executable=True)
    manifest = store.get_manifest(hid, store.list_versions(hid)[-1])
    assert "script" not in dict(manifest.digest)


def test_an_undeclared_output_is_absent(published) -> None:
    """`OUTPUT_ABSENT` against the real store, not a stub returning `False`."""
    store, _ = published
    missing = HandoffId.new()
    failures = run_gate([missing], {}, store=store, budget=None)
    assert [f.kind for f in failures] == [EventKind.OUTPUT_ABSENT]


def test_copy_out_leaves_nothing_behind(published, tmp_path: Path) -> None:
    """The gate copies into a `TemporaryDirectory`; a leak would be per-task."""
    store, publish = published
    hid = publish(executable=True)
    before = set(os.listdir("/tmp"))
    run_gate([hid], {}, store=store, budget=None)
    leaked = {n for n in set(os.listdir("/tmp")) - before if n.startswith("agent-gate-")}
    assert not leaked


def test_items_json_is_what_declares_the_key(tmp_path: Path) -> None:
    """Guards the fixture rather than the gate: if `items.json` stopped being
    the declaration, the two tests above would pass for the wrong reason."""
    root = _content(tmp_path / "c", executable=True)
    doc = json.loads((root / content_mod.ITEMS_DIR / content_mod.ITEMS_JSON).read_text())
    assert "env" in doc
