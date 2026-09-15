"""Criterion 15: an `addons` handoff does not satisfy a `fixed.required` input.

Checked on the **binding**, at load, not on the artefact at run time
(`design.md` §3.4): a graph whose declared interface is satisfiable by
injection is *misdeclared*, not misrun. Spec §4.2 gives the reason — "if it
could, the declared interface would be advisory".

The runtime half of the criterion — the consuming task stays in
`WAITING_HANDOFF` — is `task_graph`'s state machine and is asserted there; what
this package owns is the question that keeps a misdeclared graph from loading.
"""

from __future__ import annotations

import pytest

from handoff import HandoffSpecRegistry, Scope
from handoff.store import store_name_for
from spec_loader.protocols import SpecNotFound


def _kind(name: str, scope: str) -> dict:
    return {
        "name": name,
        "content_type": "text",
        "scope": scope,
        "items_schema": {"type": "object"},
        "validators": ["shape"],
    }


@pytest.fixture
def kinds() -> HandoffSpecRegistry:
    reg = HandoffSpecRegistry()
    for name, scope in (
        ("trace", "fixed.required"),
        ("tuning", "fixed.optional"),
        ("scratch", "addons.temp"),
        ("cluster_notes", "addons.knowledge"),
    ):
        reg.add(name, _kind(name, scope), origin=f"handoff/{name}.jsonnet")
    return reg


def test_addons_does_not_satisfy_required(kinds: HandoffSpecRegistry) -> None:
    assert kinds.can_satisfy_required("trace")
    assert kinds.can_satisfy_required("tuning")
    assert not kinds.can_satisfy_required("scratch")
    assert not kinds.can_satisfy_required("cluster_notes")


def test_it_is_a_question_and_not_a_scope_getter(kinds: HandoffSpecRegistry) -> None:
    """`engineer_principle.md` §3: every caller comparing a tag against the same
    constant is one branch copied N times, and the copies are what stop being
    updated. An unknown kind is `SpecNotFound` — a decision this module can only
    make because the question comes to it."""
    with pytest.raises(SpecNotFound):
        kinds.can_satisfy_required("absent")


def test_the_tag_is_consumed_once_at_publish(kinds: HandoffSpecRegistry) -> None:
    """None of storage, permission or retention is implemented by reading the
    tag at use time: storage is decided once by which store the kind resolves
    to, permission is a property of the zone the artefact lands in, and
    retention is a property of the store's root. So everything downstream sees
    only *where* the artefact is."""
    landed = {name: store_name_for(kinds.kind_of(name).scope) for name in kinds.names()}
    assert landed == {
        "trace": "handoff_store",
        "tuning": "handoff_store",
        "scratch": "playground",
        "cluster_notes": "knowledge_store",
    }


def test_the_vocabulary_is_closed() -> None:
    assert [s.value for s in Scope] == [
        "fixed.required",
        "fixed.optional",
        "addons.temp",
        "addons.knowledge",
    ]
