"""Criterion 11: lookup by uuid, a pointer into content, and three-way failure.

The three-way separation is the whole reason spec §5.1 rev. 5 says Pointer
rather than jsonpath. RFC 9535 §2.5.1.2 **forbids** a valid JSONPath query from
erroring, so no JSONPath implementation can distinguish a wrong path from an
absent value — the caller gets `[]` either way, which is the silent pass this
system exists to prevent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from handoff import resolve
from handoff.errors import PointerInvalid, PointerMiss
from handoff.store import FilesystemStore
from task_graph.ids import HandoffId, TaskId
from tests.handoff.conftest import FixedKind, make_content, make_kind

DOC = {"env": {"gpu": "MI300X", "note": None, "ranks": [0, 1]}, "a/b": 1, "a~b": 2}


def test_pointer_three_way() -> None:
    # 1. a value.
    assert resolve(DOC, "/env/gpu") == "MI300X"
    # 2. a JSON null is returned as None and is **not** a miss.
    assert resolve(DOC, "/env/note") is None
    # 3. a well-formed pointer addressing nothing.
    with pytest.raises(PointerMiss):
        resolve(DOC, "/env/absent")
    with pytest.raises(PointerMiss):
        resolve(DOC, "/env/ranks/9")
    # 4. a malformed pointer — the binding author's typo, a different class.
    with pytest.raises(PointerInvalid):
        resolve(DOC, "env/gpu")
    with pytest.raises(PointerInvalid):
        resolve(DOC, "/env/~2")


def test_miss_and_invalid_are_not_one_class() -> None:
    """A validator must be able to treat "I was written wrong" and "the
    artefact is wrong" differently, which `jsonpointer` 3.1.1 cannot: it raises
    the same `JsonPointerException` for both."""
    assert not issubclass(PointerMiss, PointerInvalid)
    assert not issubclass(PointerInvalid, PointerMiss)


def test_escapes_address_a_key_containing_slash_or_tilde() -> None:
    """`items` keys can be agent-generated, and `/` is the only byte POSIX
    forbids in a filename — so `~1` and `~0` are not decoration."""
    assert resolve(DOC, "/a~1b") == 1
    assert resolve(DOC, "/a~0b") == 2


def test_the_empty_pointer_addresses_the_whole_document() -> None:
    assert resolve(DOC, "") == DOC


def test_lookup_by_uuid(tmp_path: Path) -> None:
    """Lookup is by handoff uuid, not by kind."""
    hid = HandoffId.new()
    store = FilesystemStore(tmp_path / "s", kinds=FixedKind({hid: make_kind()}))
    version = store.put(hid, make_content(tmp_path / "c"), producer=TaskId.new())

    assert store.exists(hid, version)
    assert not store.exists(HandoffId.new())
    with store.open_item(hid, version, "env") as fh:
        assert resolve(json.loads(fh.read()), "/gpu") == "MI300X"


def test_two_inputs_same_kind(tmp_path: Path) -> None:
    """A task with two inputs of the same kind is unambiguous, because the key
    is the uuid: keying by kind would collide here."""
    kind = make_kind()
    left, right = HandoffId.new(), HandoffId.new()
    store = FilesystemStore(tmp_path / "s", kinds=FixedKind({left: kind, right: kind}))

    store.put(left, make_content(tmp_path / "l", data={"env": {"gpu": "A"}}), producer=TaskId.new())
    store.put(
        right, make_content(tmp_path / "r", data={"env": {"gpu": "B"}}), producer=TaskId.new()
    )

    assert left != right
    reads = {}
    for hid in (left, right):
        with store.open_item(hid, 0, "env") as fh:
            reads[hid] = resolve(json.loads(fh.read()), "/gpu")
    assert reads == {left: "A", right: "B"}
