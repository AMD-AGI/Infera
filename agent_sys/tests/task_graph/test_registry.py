"""Registry and the resume protocol — criterion 15.

Resolve at use time, not construction time: a test must be able to swap an
implementation after the system is wired.
"""

import pytest

from task_graph.registry import RESUME_ORDER, Registry, Resumable, resume_all


class Spy:
    def __init__(self, log: list, name: str) -> None:
        self.log, self.name = log, name

    def resume_system(self) -> None:
        self.log.append(self.name)


class NotResumable:
    """Has a `resume`, but not the one the protocol names."""

    def __init__(self, log: list) -> None:
        self.log = log

    def resume(self) -> None:
        self.log.append("wrong-method")


def test_register_and_get():
    r = Registry()
    r.register("policy", "a-policy")
    assert r.get("policy") == "a-policy"


def test_get_fails_loudly_and_names_the_key():
    r = Registry()
    with pytest.raises(KeyError, match="policy"):
        r.get("policy")


def test_registration_replaces_deliberately():
    """The swap mechanism. Rejecting a duplicate would forbid exactly this."""
    r = Registry()
    r.register("runner", "real")
    r.register("runner", "fake")
    assert r.get("runner") == "fake"


def test_two_registries_are_isolated():
    a, b = Registry(), Registry()
    a.register("runner", "a")
    assert a.get("runner") == "a"
    with pytest.raises(KeyError):
        b.get("runner")


def test_resolve_a_plain_name_returns_one():
    r = Registry()
    r.register("scheduler", "s")
    assert r.resolve("scheduler") == ["s"]


def test_resolve_a_plain_name_that_is_missing_raises():
    with pytest.raises(KeyError):
        Registry().resolve("scheduler")


def test_resolve_a_wildcard_returns_the_prefix_group_in_registration_order():
    r = Registry()
    r.register("resource:gpu", "gpu")
    r.register("task_mgr", "tasks")
    r.register("resource:token", "token")
    assert r.resolve("resource:*") == ["gpu", "token"]


def test_a_wildcard_that_matches_nothing_is_empty_not_an_error():
    assert Registry().resolve("resource:*") == []


def test_a_wildcard_does_not_match_the_bare_prefix():
    r = Registry()
    r.register("resource", "not-a-pool")
    assert r.resolve("resource:*") == []


# ------------------------------------------------------------- resume_all


def test_resume_order_is_the_documented_one():
    assert RESUME_ORDER == [
        "handoff_mgr",
        "agent_mgr",
        "task_mgr",
        "resource:*",
        "scheduler",
    ]


def test_resume_all_follows_resume_order_not_registration_order():
    log = []
    r = Registry()
    for name in ["scheduler", "task_mgr", "handoff_mgr", "agent_mgr"]:
        r.register(name, Spy(log, name))
    r.register("resource:gpu", Spy(log, "resource:gpu"))

    resume_all(r)

    assert log == ["handoff_mgr", "agent_mgr", "task_mgr", "resource:gpu", "scheduler"]


def test_resume_all_skips_a_component_that_is_not_resumable():
    log = []
    r = Registry()
    r.register("handoff_mgr", NotResumable(log))
    r.register("task_mgr", Spy(log, "task_mgr"))

    resume_all(r)

    assert log == ["task_mgr"]
    assert "wrong-method" not in log


def test_resume_all_tolerates_a_missing_component():
    log = []
    r = Registry()
    r.register("task_mgr", Spy(log, "task_mgr"))
    resume_all(r)
    assert log == ["task_mgr"]


def test_resumable_matches_on_the_method_name_alone():
    """Why the scheduler's per-task resume must not be called `resume_system`."""
    assert isinstance(Spy([], "x"), Resumable)
    assert not isinstance(NotResumable([]), Resumable)
