"""Criterion 3 — design §12."""

from __future__ import annotations

import pytest

from agent.backend import Assignment
from agent.backends.program import ProgramExecutor
from agent.selection import BackendUnavailable, select_backend
from agent.spec import AgentSpec

from .conftest import ScriptedBackend, ai_spec, program_spec

#: Every call supplies one now: `select_backend`'s `assignment` lost its
#: default, because the probe is the constructor and a caller that omitted it
#: built an agent with no instruction, no entry point and no zone — one that
#: starts and does nothing (`interfaces.md` §4.11).
ASSIGNMENT = Assignment(entry="/bin/true")

SCRIPTED = "tests.agent.conftest:ScriptedBackend"
BROKEN = "tests.agent.conftest:BrokenBackend"


def _spec(*decls: dict) -> AgentSpec:
    return AgentSpec.of(ai_spec(backends=list(decls)))


def _unusable(key: str) -> dict:
    return {"key": key, "backend_entry": SCRIPTED, "config": {"unavailable": f"{key} is down"}}


def _usable(key: str) -> dict:
    return {"key": key, "backend_entry": SCRIPTED}


@pytest.mark.parametrize("source", ["cli", "spec", "config"])
def test_selection_precedence(source: str) -> None:
    """The three sources, in descending precedence — spec §3.3.

    `source` on the result is what makes this assertable without reading logs,
    and it is what keyring's issue #632 asks for and its `diagnose` command
    still does not give.
    """
    if source == "cli":
        chosen = select_backend(
            _spec(_usable("declared")), override="declared", config_order=(), assignment=ASSIGNMENT
        )
    elif source == "spec":
        chosen = select_backend(
            _spec(_unusable("first"), _usable("second")),
            override=None,
            config_order=(),
            assignment=ASSIGNMENT,
        )
        assert chosen.key == "second"
        assert [r.key for r in chosen.rejected] == ["first"]
    else:
        chosen = select_backend(
            _spec(_unusable("first")),
            override=None,
            config_order=["program"],
            assignment=Assignment(entry="/bin/true"),
        )
        assert chosen.key == "program"
    assert chosen.source == source


def test_first_available_in_declared_order_wins() -> None:
    """`virtualenv`'s `default = next(iter(choices))`, which is criterion 3's
    second mechanism literally."""
    chosen = select_backend(
        _spec(_usable("first"), _usable("second")),
        override=None,
        config_order=(),
        assignment=ASSIGNMENT,
    )
    assert chosen.key == "first"
    assert chosen.rejected == ()


def test_cli_override_does_not_fall_through() -> None:
    """**If it names something unusable that is an error, not a hint.**
    matplotlib says it on the last line of `use()`: *do not helpfully
    fallback*. A pin that is advisory is worthless for reproducing anything."""
    spec = _spec(_unusable("pinned"), _usable("healthy"))
    with pytest.raises(Exception) as caught:
        select_backend(spec, override="pinned", config_order=["program"], assignment=ASSIGNMENT)
    assert "pinned is down" in str(caught.value)


def test_an_override_need_not_be_declared() -> None:
    """Design D6: the case that most needs pinning is a backend the spec's
    author did not foresee."""
    chosen = select_backend(
        _spec(_usable("first")),
        override="program",
        config_order=(),
        assignment=Assignment(entry="/bin/true"),
    )
    assert chosen.key == "program"
    assert isinstance(chosen.backend, ProgramExecutor) or chosen.backend is not None


def test_rejections_carry_reasons_not_only_names() -> None:
    """keyring #316: broadening the catch made every rejection reason vanish.
    You get the reasons or you get a broad catch, never both, unless the probe
    returns a structured result."""
    with pytest.raises(BackendUnavailable) as caught:
        select_backend(
            _spec(_unusable("a"), _unusable("b")),
            override=None,
            config_order=(),
            assignment=ASSIGNMENT,
        )
    assert [r.key for r in caught.value.rejected] == ["a", "b"]
    assert all("is down" in r.reason for r in caught.value.rejected)
    assert "a is down" in str(caught.value)


def test_human_fails_at_selection_not_at_load() -> None:
    spec = AgentSpec.of({**ai_spec(), "kind": "human", "backends": []})
    with pytest.raises(BackendUnavailable) as caught:
        select_backend(spec, override=None, config_order=(), assignment=ASSIGNMENT)
    assert "human" in str(caught.value)


def test_program_kind_selects_the_program_executor() -> None:
    """Design §7.2.1: the kind is **one line of dispatch and not a second
    control flow**."""
    chosen = select_backend(
        AgentSpec.of(program_spec()),
        override=None,
        config_order=(),
        assignment=Assignment(entry="/bin/true"),
    )
    assert isinstance(chosen.backend, ProgramExecutor)
    assert chosen.key == "program"


def test_an_unimplemented_entry_is_named_by_its_own_message() -> None:
    """The one thing a dotted string buys that an entry point cannot — fsspec's
    per-entry `err`."""
    spec = _spec({"key": "x", "backend_entry": "no.such:Thing", "err": "Install the widget extra"})
    with pytest.raises(BackendUnavailable) as caught:
        select_backend(spec, override=None, config_order=(), assignment=ASSIGNMENT)
    assert "Install the widget extra" in str(caught.value)


def test_nothing_is_cached() -> None:
    """Design §6.4. `env_mgr` deploys the environment, so a probe taken before
    deployment is taken at the one moment it is guaranteed to be wrong — every
    dispatch probes again, and two selections are two objects."""
    spec = _spec(_usable("first"))
    one = select_backend(spec, override=None, config_order=(), assignment=ASSIGNMENT)
    two = select_backend(spec, override=None, config_order=(), assignment=ASSIGNMENT)
    assert isinstance(one.backend, ScriptedBackend)
    assert one.backend is not two.backend
