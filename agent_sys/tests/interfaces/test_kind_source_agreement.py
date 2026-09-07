"""`handoff`'s `KindSource` double and the real one must answer alike.

**Written because `agent` found the same bug four times in one day**, and the
fourth was the one nobody had a check for: `tests/agent/conftest.py`'s
`StubStore.seal` *raised* a refusal that `FilesystemStore.seal` had started
*returning*. Their conformance rule compared **presence** — both objects had a
`seal`, so it passed — while the drift was in the **contract**. A green suite
measured a world that no longer existed.

`handoff` has exactly one double standing in for something it does not own.
`store.KindSource` is a Protocol precisely because no single component can
satisfy it: `hid -> Handoff.type` is `task_graph.HandoffMgr`'s and
`type -> HandoffKind` is `HandoffSpecRegistry`'s, so only the composition root
holds both (`interfaces.md` §2.6). Every test in `tests/handoff` injects
`FixedKind` instead; the real implementation is
`task_graph/bootstrap.py::_KindSource`, and **nothing compared them.**

**That gap has already cost once, and `store.put`'s docstring records it**:
`put` used to publish without a kind — no README sections checked, `items`
validated against nothing, `kind: ""` in the manifest — and *"all 135 of
`handoff`'s tests were green because every one injects a resolver."* The
double was the reason the hole was invisible, which is this file's whole
subject.

**Behaviour, not signatures.** `_KindSource.kind_for` is annotated
`(hid: Any) -> Any` where the Protocol says `HandoffId -> HandoffKind | None`,
so a signature comparison fails on a difference that is not a contract
difference — `Any` accepts what the Protocol promises. `agent`'s own lesson is
that the part which moves is the contract, so that is what is asserted.
"""

from __future__ import annotations

import pytest

from handoff.registry import HandoffSpecRegistry
from handoff.store import FilesystemStore, KindSource
from task_graph.bootstrap import _KindSource
from task_graph.handoff import HandoffMgr
from task_graph.ids import HandoffId, TaskId
from task_graph.registry import Registry
from tests.handoff.conftest import FixedKind, make_content


def _real() -> _KindSource:
    """The composition root's `KindSource`, over empty but real collaborators."""
    registry = Registry()
    registry.register("handoff_mgr", HandoffMgr(registry))
    registry.register("handoff_specs", HandoffSpecRegistry(allow_no_validator=True))
    return _KindSource(registry)


@pytest.mark.parametrize("source", [_real(), FixedKind()], ids=["real", "double"])
def test_an_unresolvable_id_answers_none(source: KindSource) -> None:
    """**The one answer every `handoff` test depends on**, from both.

    `None` is what makes `put` and `seal` refuse with a message naming the
    wiring. If the real one raised instead — a `KeyError` from an unregistered
    component, say — `handoff`'s suite would stay green and production would
    fail differently. Measured on the way in: `HandoffMgr.type_of` answers `""`
    for an undeclared slot rather than raising, and `_KindSource` converts that
    to `None`, so the agreement holds for a reason rather than by luck.
    """
    assert source.kind_for(HandoffId.new()) is None


def test_the_refusal_a_none_produces_is_the_same_for_both(tmp_path) -> None:
    """A double that answers alike but is *used* differently is still a lie, so
    drive the answer through the code that consumes it."""
    hid = HandoffId.new()
    for name, source in (("real", _real()), ("double", FixedKind())):
        store = FilesystemStore(tmp_path / name, kinds=source)
        with pytest.raises(Exception, match="has no kind for it"):
            store.put(hid, make_content(tmp_path / f"c-{name}"), producer=TaskId.new())
        assert store.list_versions(hid) == [], f"{name}: and nothing was created"
