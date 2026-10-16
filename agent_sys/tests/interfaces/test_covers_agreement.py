"""`Permissions.covers` and `closure.check.covers` are two bodies of one relation.

`Grant`, `Access` and `Permissions` are `task_graph`'s (`docs/interfaces.md`
§4.7), and `Permissions.covers` exists for `closure`'s load check 6 — which is
its only consumer. So the relation is `task_graph`'s and the check is
`closure`'s, and they must agree.

**They are obliged to exist twice, and that is the finding rather than the
defect.** At load time there is no `Task`: `closure` holds a `TaskSpec`, which is
a `Mapping[str, Any]`, and building a `Permissions` from it would mean importing
`task_graph`, which §4.5 forbids. So one relation is expressed once over pydantic
objects and once over raw dicts. That is admissible on exactly the terms §8 sets
for `Pushable`: *"the test is not a nicety attached to the decision; it is the
decision's price."* This file is that price.

The semantics are `task_graph`'s, ruled after the two had already drifted into
opposite answers for one pair. The reason is not seniority — it is that
`env_mgr` turns a grant into a Landlock `Mode`, and **a write grant on a
directory without read and execute is not usable**: you cannot create a file in
a directory you cannot traverse. The implication is what the enforcement layer
performs whatever anyone declares, which is why §3.1 split `Access` from `Mode`
in the first place.
"""

from __future__ import annotations

import pytest

from closure.check import covers as covers_dict
from task_graph.permissions import Access, Grant, Permissions

# (granted access, requested access, does it cover)
TABLE = [
    ("read", "read", True),
    ("read", "write", False),
    ("write", "write", True),
    # The pair the two bodies disagreed on, and the whole reason this file
    # exists. `closure` widened to match; before it did, this row was the one
    # that failed.
    ("write", "read", True),
]


def as_permissions(granted: str) -> Permissions:
    return Permissions(grants=(Grant(kind="trace", access=Access(granted)),))


def as_task_spec(granted: str) -> dict:
    return {"permissions": {"grants": [{"path": "/z", "kind": "trace", "access": granted}]}}


@pytest.mark.parametrize(("granted", "requested", "expected"), TABLE)
def test_the_two_bodies_agree(granted: str, requested: str, expected: bool) -> None:
    """One relation, two representations, one answer for every pair."""
    typed = as_permissions(granted).covers("trace", Access(requested))
    raw = covers_dict(as_task_spec(granted), "trace", requested)
    assert typed == raw == expected


def test_a_kind_nobody_granted_is_covered_by_neither() -> None:
    """The negative case needs no ruling and is the one both got right."""
    assert not as_permissions("write").covers("absent", Access.READ)
    assert not covers_dict(as_task_spec("write"), "absent", "read")


def test_a_task_with_no_permissions_grants_nothing() -> None:
    assert not Permissions().covers("trace", Access.READ)
    assert not covers_dict({}, "trace", "read")


def test_an_omitted_access_defaults_to_read_on_both_sides() -> None:
    """`access` has a default in two places — the schema's and the model's — and
    a grant that omits it must mean one thing.

    Added by `closure`: this is the second pair the two bodies could drift on
    without the table noticing, because the table always writes `access` out.
    """
    typed = Permissions(grants=(Grant(kind="trace"),))
    raw = {"permissions": {"grants": [{"path": "/z", "kind": "trace"}]}}

    assert typed.covers("trace", Access.READ) == covers_dict(raw, "trace", "read") is True
    assert typed.covers("trace", Access.WRITE) == covers_dict(raw, "trace", "write") is False


def test_the_name_grammar_is_exact_on_both_sides() -> None:
    """A `*` is a kind named `*`, on both sides, and covers nothing else.

    Added by `closure`: the access axis is an order on a closed enum and the name
    axis is not, so widening one must not be read as licence to widen the other.
    kubernetes#122154 is what an under-specified name grammar costs — a second
    component gave `example.com/*` a meaning the authorization API never defined,
    the covering check then wrongly rejected a legal delegation, and it was closed
    `/remove-kind bug` rather than fixed.
    """
    typed = Permissions(grants=(Grant(kind="*", access=Access.WRITE),))
    raw = {"permissions": {"grants": [{"path": "/z", "kind": "*", "access": "write"}]}}

    assert typed.covers("trace", Access.READ) == covers_dict(raw, "trace", "read") is False
    assert typed.covers("*", Access.READ) == covers_dict(raw, "*", "read") is True


def test_the_access_vocabulary_is_the_same_two_words() -> None:
    """A third value on either side would make the table above incomplete
    without failing it, so the vocabulary is pinned rather than assumed."""
    assert {a.value for a in Access} == {"read", "write"}


def test_the_relation_really_does_exist_twice() -> None:
    """Stated as an assertion so that removing one body is a visible event.

    If `closure` ever gains a way to reach `Permissions` without importing
    `task_graph`, the right move is to delete its copy and this file with it —
    not to leave two bodies and a test that has stopped being the price of
    anything.
    """
    assert callable(covers_dict)
    assert callable(Permissions.covers)
