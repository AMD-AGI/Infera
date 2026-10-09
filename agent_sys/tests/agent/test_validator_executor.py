"""`validator_executor` — O6's mechanism, and the property it exists for.

`validator` §8.2 owns the requirement (a phase must be separately attributable);
this module owns the mechanism, and the one thing every test here is really
about is that **the phase's `AgentId` is a different agent from the producer's**,
not a derivation of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from agent.backend import AgentResult, AgentStatus
from agent.validator_executor import ValidatorExecutor, ValidatorExecutorUnconfigured
from task_graph.agent import AgentMgr
from task_graph.models import Agent
from task_graph.store import MemoryStoreMgr

from .conftest import ScriptedBackend, ai_spec, dispatched


@dataclass(frozen=True)
class StubEnv:
    """`validator.environment.ValidationEnvironment`, as much as we touch."""

    zone: Path
    cwd: Path
    env: dict[str, str]
    agent_id: str

    @property
    def verdict_file(self) -> Path:
        return self.zone / "verdict.json"


@dataclass(frozen=True)
class StubBody:
    readme: str = "check that the trace parses"
    entry: str | None = None
    materials: tuple[str, ...] = ()


@dataclass(frozen=True)
class StubValidatorSpec:
    """**`agent` is here because the real `ValidatorSpec` has it**, defaulting
    to `None` exactly as theirs does (`str | None`, optional, `minLength: 1`).

    It was absent for one commit's worth of writing and `_agent_spec` raised
    `AttributeError` immediately — which is the outcome to want. A stub carrying
    the field only when a test needs it would have made the ordinary path work
    by accident.
    """

    name: str = "trace_parses"
    brief: str = "the trace parses"
    body: StubBody = StubBody()
    agent: str | None = None


@pytest.fixture()
def validating(wired, tmp_path):
    """The runner's registry, plus what a validation phase brings."""
    wired.registry.register("store_mgr", MemoryStoreMgr())
    wired.registry.register("agent_mgr", AgentMgr(wired.registry))
    wired.registry.get("agent_mgr").register("checker")
    wired.registry.get("agent_specs").add("checker", ai_spec(name="checker"), origin="tests")
    zone = tmp_path / "vzone"
    zone.mkdir()
    wired.env = StubEnv(zone=zone, cwd=zone, env={"TMPDIR": str(zone)}, agent_id="derived")
    wired.executor = ValidatorExecutor(wired.registry, agent_spec="checker")
    return wired


def test_each_phase_runs_as_a_different_agent_from_the_producer(validating) -> None:
    """**The whole point, and `validator`'s one non-negotiable constraint.**

    Their interim id was `f"{producing_agent_id}:{kind}"` — distinct per phase
    and *not a distinct agent*. Criterion 10 wants a checking context the
    producer cannot reach, and a string suffix does not buy that.
    """
    task, producer = dispatched("writer", "leaf_ai", validating)
    first = validating.executor.run_body(
        StubValidatorSpec(), validating.env, {}, validating.registry
    )
    second = validating.executor.run_body(
        StubValidatorSpec(), validating.env, {}, validating.registry
    )

    assert first != producer.id
    assert first != task.current.agent_id
    assert first != second  # one per phase, not one per run


def test_the_phase_agent_is_unbound(validating) -> None:
    """Unbound is the mechanism, not an accident: a `task_id` would re-create by
    the back door exactly the coupling criterion 10 forbids."""
    agent_id = validating.executor.run_body(
        StubValidatorSpec(), validating.env, {}, validating.registry
    )
    minted = validating.registry.get("agent_mgr").get(agent_id)
    assert isinstance(minted, Agent)
    assert minted.task_id is None
    assert minted.spec == "checker"


def test_the_body_gets_the_rebuilt_environment_and_not_the_inherited_one(validating) -> None:
    """`validator` criterion 21. A fresh directory does not close the channels
    their `CHANNELS` list enumerates, so the block is explicit."""
    validating.executor.run_body(StubValidatorSpec(), validating.env, {}, validating.registry)
    # The executor was constructed with the assignment; selection is per call,
    # so reach it the way the runner would rather than reading a private field.
    from agent.selection import select_backend

    from .conftest import ai_spec as _spec

    chosen = select_backend(
        validating.registry.get("agent_specs").spec("checker"),
        override=None,
        config_order=(),
        assignment=_assignment_of(validating),
    )
    assert chosen.backend.assignment.environment == {"TMPDIR": str(validating.env.zone)}
    assert chosen.backend.assignment.zone == str(validating.env.cwd)
    assert _spec is ai_spec  # the fixture's spec is the one selected against


def test_the_readme_is_the_instruction_and_there_is_no_entry(validating) -> None:
    """`validator`'s `runner_for` picks `ScriptBodyRunner` whenever
    `body.entry` is present, so a programmatic body never reaches here — which
    is why this class has no branch on the body's kind."""
    assignment = _assignment_of(validating)
    assert assignment.readme == "check that the trace parses"
    assert assignment.entry is None


def test_an_unregistered_agent_spec_is_loud(validating) -> None:
    """Loud rather than defaulted, for `validator`'s own reason about their
    missing-component path: a phase that quietly ran under some arbitrary agent
    is worse than one that did not run."""
    executor = ValidatorExecutor(validating.registry, agent_spec="nobody")
    with pytest.raises(ValidatorExecutorUnconfigured) as caught:
        executor.run_body(StubValidatorSpec(), validating.env, {}, validating.registry)
    assert "nobody" in str(caught.value)
    assert "checker" in str(caught.value)


def test_a_body_that_fails_raises_rather_than_writing_a_verdict(validating) -> None:
    """The body writes `verdict_file`; we do not. A body that did not finish
    wrote nothing, and `validator.read_verdict_file` raises on a missing file —
    so failing here rather than returning keeps the two from disagreeing."""
    validating.registry.get("agent_specs")._models["checker"].backends[0].config = {
        "results": [AgentResult(status=AgentStatus.FAILED, detail="the harness died")]
    }
    with pytest.raises(RuntimeError) as caught:
        validating.executor.run_body(StubValidatorSpec(), validating.env, {}, validating.registry)
    assert "the harness died" in str(caught.value)
    assert not validating.env.verdict_file.exists()


def _assignment_of(validating) -> Any:
    executor: Any = validating.executor._executor(StubValidatorSpec(), validating.env, "checker")
    assert isinstance(executor, ScriptedBackend)
    return executor.assignment


# --------------------------------------------------------------------------- #
# Which agent runs this validator — `ValidatorSpec.agent`, step 4 of four


def test_a_validator_that_names_an_agent_runs_as_that_one(validating) -> None:
    """The point of the field. Before it, **every agent-bodied validator in the
    system ran as one spec**, chosen once at the composition root."""
    validating.registry.get("agent_mgr").register("profiler")
    validating.registry.get("agent_specs").add("profiler", ai_spec(name="profiler"), origin="tests")

    agent_id = validating.executor.run_body(
        StubValidatorSpec(agent="profiler"), validating.env, {}, validating.registry
    )
    assert validating.registry.get("agent_mgr").get(agent_id).spec == "profiler"


def test_a_validator_that_names_nobody_runs_as_the_wiring_default(validating) -> None:
    """The ordinary case, and always will be — most validators need no specific
    agent. This is what makes the constructor argument a real default rather
    than dead weight."""
    agent_id = validating.executor.run_body(
        StubValidatorSpec(), validating.env, {}, validating.registry
    )
    assert validating.registry.get("agent_mgr").get(agent_id).spec == "checker"


def test_a_named_agent_that_does_not_resolve_raises_rather_than_falling_back(
    validating,
) -> None:
    """**Absent and unresolvable are different questions** — `closure`'s
    distinction, which `validator` adopted after first conflating them.

    Absent means *take the default*, a declaration. Unresolvable means the
    author named an agent that does not exist, and falling back would hand them
    a **working** run in an environment they did not configure. The `or` cannot
    reach this case: it falls back on a falsy value, and `minLength: 1` means a
    present name is never falsy.
    """
    with pytest.raises(ValidatorExecutorUnconfigured) as caught:
        validating.executor.run_body(
            StubValidatorSpec(agent="profilr"), validating.env, {}, validating.registry
        )
    assert "profilr" in str(caught.value)
    assert "checker" in str(caught.value)  # the candidates, so the typo is visible


def test_the_named_agent_also_chooses_the_backend(validating) -> None:
    """Not only the identity: `_executor` selects against the **named** spec, so
    a validator naming an agent gets that agent's backends and config too.

    Asserted because the two reads are separate lines and only one of them was
    changed at first."""
    validating.registry.get("agent_mgr").register("profiler")
    validating.registry.get("agent_specs").add(
        "profiler",
        ai_spec(
            name="profiler",
            backends=[
                {"key": "scripted-2", "backend_entry": "tests.agent.conftest:ScriptedBackend"}
            ],
        ),
        origin="tests",
    )
    executor = validating.executor._executor(
        StubValidatorSpec(agent="profiler"), validating.env, "profiler"
    )
    assert executor.key == "scripted-2"


def test_the_real_validator_spec_has_the_field_and_defaults_to_none() -> None:
    """Driven against the subject, not the stub — `interfaces.md` §8.7.

    `StubValidatorSpec` above declares `agent` because the real one does; this
    is the assertion that keeps that true. A stub agreeing with a field that
    moved is the whole failure class.
    """
    from validator.spec import ValidatorSpec

    field = ValidatorSpec.model_fields["agent"]
    assert field.default is None
    assert not field.is_required()
    assert "str" in str(field.annotation) and "None" in str(field.annotation)
