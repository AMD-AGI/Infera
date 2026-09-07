"""Running an agent-bodied validator — `validator` design §5, registered as
`validator_executor`.

**This is O6's mechanism, and it is chosen rather than assumed.** `validator`
§8.2 owns the requirement — a phase must be separately attributable or its
criterion 10 is untestable — and this module owns how. Four candidates survived
the design stage (`fork_session`, `resume`, a subagent per phase, a second
client) and the question that settled it was not which SDK feature to use, but
**whose `agent_id` the assertion is about**. `validator` checked their own code
and answered: *ours*, not the SDK's.

So the mechanism is **one fresh executor per phase, bound to its own `Agent`
record**, and it is better than the three SDK-flavoured candidates on three
counts:

- It needs no SDK feature, so it works for any backend.
- It works for a `kind: program` body as well as an AI one.
- **The identity is real rather than derived.** `validator`'s interim id was
  `f"{producing_agent_id}:{kind}"` — distinct per phase and *not a distinct
  agent*, which their own §8.1 says is what criterion 10 wants. They found that
  weakness while answering this question and recorded it in `40764ea`.

**What is still open is not the mechanism.** Whether `fork_session=True` leaves
the main phase's session interruptible is a measurement nobody has taken, and it
matters only if the assertion ever becomes about the SDK's `agent_id`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agent.backend import AgentStatus, Assignment
from agent.selection import select_backend

__all__ = ["ValidatorExecutorUnconfigured", "ValidatorExecutor"]


class ValidatorExecutorUnconfigured(RuntimeError):
    """No agent spec is configured for validator bodies.

    Loud rather than defaulted, for the reason `validator` gives about their own
    missing-component path: a validation phase that quietly ran under some
    arbitrary agent is worse than one that did not run.
    """


class ValidatorExecutor:
    """Satisfies the component `validator/phase.py::AgentBodyRunner` resolves.

    **`agent_spec` is the wiring's default, not the only answer.** A validator
    that names an agent runs as that one; one that names nobody runs as this.
    See `_agent_spec`.

    **Only ever called for an agent-bodied validator.** `validator`'s
    `runner_for` picks `ScriptBodyRunner` whenever `spec.body.entry` is present,
    so a programmatic body never reaches here and this class needs no branch on
    the body's kind.
    """

    def __init__(
        self,
        registry: Any,
        *,
        agent_spec: str,
        override: str | None = None,
        config_order: Sequence[str] = (),
    ) -> None:
        self._r = registry
        self.agent_spec = agent_spec
        self.override = override
        self.config_order = tuple(config_order)

    def run_body(
        self,
        spec: Any,
        env: Any,
        handoffs: Mapping[Any, Any],
        registry: Any,
    ) -> Any:
        """Run the body to completion. It writes `env.verdict_file`; we do not.

        Returns the `AgentId` this phase ran as, which is the fact `validator`
        needs to retire its derived id. **The declared return is `None` and
        returning a value breaks no caller** — theirs ignores it today.

        The environment is `env.env` **and not the inherited one**, and the
        working directory is `env.cwd`. That is `validator` criterion 21: a
        fresh directory does not close the channels their `CHANNELS` list
        enumerates, so the block is explicit.
        """
        name = self._agent_spec(spec)
        agent = self._mint(name)
        executor = self._executor(spec, env, name)
        result = executor.start()
        if result.status is not AgentStatus.FINISHED:
            raise RuntimeError(
                f"the validator body for {getattr(spec, 'name', '?')} ended "
                f"{result.status.value}: {result.detail}"
            )
        return agent.id

    # ---- internals -------------------------------------------------------- #

    def _agent_spec(self, spec: Any) -> str:
        """**Which agent runs this validator** — the spec's, or the wiring's.

        `ValidatorSpec.agent` is optional (`spec-loader` `fe9fd55`,
        `minLength: 1`) and is `None` for the ordinary validator that names
        nobody, which is what makes `or` correct here rather than merely
        convenient: it falls back on *absent*, never on *unresolvable*. A name
        that is present and does not resolve reaches `_mint` and raises, which
        is `closure`'s distinction — falling back there would hand the author a
        **working** run in an environment they did not configure.

        Read directly rather than with a `getattr` default: the field is
        declared, so a default could not fire, and a future removal must be a
        loud `AttributeError` rather than a silent fall-through that reads as
        *nobody declared an agent*. `validator` made that correction to this
        line before it was written.

        **The absent case falls back to a different place than `validator`'s,
        and that is deliberate.** Identity and backend come from
        `self.agent_spec`; the *environment* is `env.env`, which `validator`
        resolved through §8.2's last row — the `validation_env` component — and
        never through this name. Two questions, two declared answers, and the
        symmetry only looks broken if `self.agent_spec` is read as *the
        producing task's agent*. It is not: it is a composition-root choice
        about **what a validator runs as**, and it must not be the producer's,
        or criterion 10's separate attributability goes with it. Raised by
        `validator`, who noted neither docstring said so.
        """
        return spec.agent or self.agent_spec

    def _mint(self, name: str) -> Any:
        """A fresh, **unbound** `Agent` — and unbound is the point.

        `AgentMgr.get(spec_name)` mints one that is not bound to a task, so the
        checking context is not the producing task's. A `task_id` here would
        re-create by the back door exactly the coupling criterion 10 forbids.
        """
        agent_mgr = self._r.get("agent_mgr")
        if not agent_mgr.is_registered(name):
            raise ValidatorExecutorUnconfigured(
                f"validator bodies run as agent spec {name!r}, which "
                f"is not registered with agent_mgr; registered: {sorted(agent_mgr.specs())}"
            )
        return agent_mgr.get(name)

    def _executor(self, spec: Any, env: Any, name: str) -> Any:
        body = getattr(spec, "body", None)
        assignment = Assignment(
            readme=str(getattr(body, "readme", "") or ""),
            entry=None,  # a body with an entry never reaches here
            goal=str(getattr(spec, "brief", "") or ""),
            zone=str(getattr(env, "cwd", "") or ""),
            materials=tuple(str(m) for m in (getattr(body, "materials", None) or ())),
            environment=dict(getattr(env, "env", {}) or {}),
        )
        selection = select_backend(
            self._r.get("agent_specs").spec(name),
            override=self.override,
            config_order=self.config_order,
            assignment=assignment,
        )
        return selection.backend
