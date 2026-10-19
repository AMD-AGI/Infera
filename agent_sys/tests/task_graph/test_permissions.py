"""`Permissions` — carried here, interpreted in `env_mgr`.

Criterion 44's own assertions live in `test_structure_blind.py`; this file pins
the type's two properties: `covers` answers a question about a declared *kind
name*, and nothing here resolves a path.
"""

import pytest
from pydantic import ValidationError

from task_graph.permissions import Access, Grant, Permissions


def test_the_default_covers_nothing():
    assert not Permissions().covers("trace", Access.READ)


def test_a_read_grant_covers_reading_and_not_writing():
    perms = Permissions(grants=(Grant(kind="trace", access=Access.READ),))
    assert perms.covers("trace", Access.READ)
    assert not perms.covers("trace", Access.WRITE)


def test_a_write_grant_implies_read():
    """An author who may replace an artefact may certainly look at it."""
    perms = Permissions(grants=(Grant(kind="trace", access=Access.WRITE),))
    assert perms.covers("trace", Access.WRITE)
    assert perms.covers("trace", Access.READ)


def test_a_grant_for_another_kind_does_not_cover_this_one():
    perms = Permissions(grants=(Grant(kind="profile", access=Access.WRITE),))
    assert not perms.covers("trace", Access.READ)


def test_a_kind_is_a_name_and_not_an_id():
    """The correction rev. 12 makes, and the reason it had to be made.

    A grant is written at declaration time, where no instance exists — so the
    only thing it can name is the kind. Typed `HandoffId`, the declared value
    could not even be loaded.
    """
    from task_graph.ids import HandoffId

    with pytest.raises(ValueError):
        HandoffId("trace")
    assert Grant(kind="trace").kind == "trace"


def test_a_grant_is_frozen():
    """Permissions are versioned *with the task*; mutating one in place would
    make the version meaningless."""
    grant = Grant(kind="trace")
    with pytest.raises(ValidationError):
        grant.kind = "profile"


def test_an_undeclared_field_is_an_error():
    with pytest.raises(ValidationError):
        Grant(kind="trace", handoff="x")


def test_it_round_trips_through_json():
    perms = Permissions(
        grants=(Grant(path="/w", access=Access.WRITE, kind="trace"), Grant(kind="profile"))
    )
    assert Permissions.model_validate(perms.model_dump(mode="json")) == perms


def test_nothing_here_resolves_a_path():
    """`task_graph` carries the value and never interprets it — the same shape
    `Task.resources` already has, one level up. A path is opaque here."""
    perms = Permissions(grants=(Grant(path="/zone/a", access=Access.WRITE, kind="trace"),))
    assert not hasattr(perms, "contains")
    assert not hasattr(perms, "resolve")
    assert perms.grants[0].path == "/zone/a"  # stored verbatim, not normalised
