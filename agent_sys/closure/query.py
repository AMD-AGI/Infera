"""The read-only query helpers, and the index behind them.

Nobody surveyed maintains a reverse map incrementally, and the two ends of the
spectrum are both instructive. dbt has no invalidation at all: a full O(V+E)
rebuild called at six separate sites, each immediately before a consumer, with
the policy stated in a comment. Sphinx is the one system whose index outlives the
load, and the price is a `clear_doc` *and* a `merge_domaindata` obligation on
every owner, third-party extensions included. Cargo inverts the graph
destructively per invocation and throws it away.

Our world is closed — five registries, all loaded before any query, and a closure
is read to assemble a graph and never again. So the index is built once, at the
end of the load pass, over the closures that passed, and then frozen.

**Membership is derived, not restated.** The sources of every edge are the
accessors in `model.py`, so adding a referential key to the closure schema
without adding it here is a change in one file that fails a test in the other.
An edge whose target is not in the corresponding registry is not silently
dropped: it cannot occur, because the checks rejected the closure and the index
is built only over closures that passed — and if it occurs anyway that is a
programming error, which is the half dbt#14436 got wrong.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from spec_loader.protocols import ClosureDoc, SpecNotFound

from .model import agent_of, named_kinds, phase_validators, task_of

__all__ = ["Index", "build_index", "handoff_kinds"]


def handoff_kinds(doc: ClosureDoc) -> tuple[str, ...]:
    """Every kind this closure touches, inputs and outputs.

    Reads the task rather than the declared list, because the declared list may
    legally name a kind the task does not — a closure may declare a kind its
    subgraph uses internally.
    """
    return named_kinds(task_of(doc))


def _universe(registry: Any) -> frozenset[str]:
    """Every name a registry holds, snapshotted.

    A snapshot rather than a live handle. Returning or retaining the registry
    would make the index a second reader of somebody else's mutable state, and
    the whole point of freezing is that the answer stops moving.
    """
    return frozenset(registry.names())


@dataclass(frozen=True)
class Index:
    """The reverse index, and the four universes its queries validate against.

    The universes are what separate "not found" from "found, used by nothing".
    They are held rather than recomputed because the registries they came from
    outlive the pass and could, in principle, be reloaded; the index's answer
    must not silently start meaning something else.
    """

    by_kind: Mapping[str, tuple[str, ...]]
    by_agent: Mapping[str, tuple[str, ...]]
    by_validator: Mapping[str, tuple[str, ...]]
    validators_by_closure: Mapping[str, tuple[str, ...]]
    known_kinds: frozenset[str]
    known_agents: frozenset[str]
    known_validators: frozenset[str]

    def closures_using_kind(self, kind: str) -> tuple[str, ...]:
        if kind not in self.known_kinds:
            raise SpecNotFound(
                f"no handoff kind named {kind!r} "
                f"(have: {', '.join(sorted(self.known_kinds)) or 'nothing'})"
            )
        return self.by_kind.get(kind, ())

    def closures_using_agent(self, agent: str) -> tuple[str, ...]:
        if agent not in self.known_agents:
            raise SpecNotFound(
                f"no agent spec named {agent!r} "
                f"(have: {', '.join(sorted(self.known_agents)) or 'nothing'})"
            )
        return self.by_agent.get(agent, ())

    def closures_using_validator(self, name: str) -> tuple[str, ...]:
        if name not in self.known_validators:
            raise SpecNotFound(
                f"no validator spec named {name!r} "
                f"(have: {', '.join(sorted(self.known_validators)) or 'nothing'})"
            )
        return self.by_validator.get(name, ())

    def validators_for(self, closure: str, doc: ClosureDoc) -> tuple[str, ...]:
        return self.validators_by_closure.get(closure, tuple(phase_validators(doc)))


def _append(table: dict[str, list[str]], key: str, value: str) -> None:
    bucket = table.setdefault(key, [])
    if value not in bucket:
        bucket.append(value)


def build_index(closures: Mapping[str, ClosureDoc], regs: Any) -> Index:
    """Invert the closure catalogue, once.

    `regs` supplies the four universes and the per-handoff validator join. It is
    read here and not retained.
    """
    by_kind: dict[str, list[str]] = {}
    by_agent: dict[str, list[str]] = {}
    by_validator: dict[str, list[str]] = {}
    validators_by_closure: dict[str, tuple[str, ...]] = {}

    handoff_specs = regs.handoff_specs

    for name in sorted(closures):
        doc = closures[name]
        task = task_of(doc)

        joined: list[str] = list(phase_validators(doc))
        for kind in named_kinds(task):
            _append(by_kind, kind, name)
            if kind in handoff_specs:
                for validator in handoff_specs.get(kind).get("validators", ()) or ():
                    if isinstance(validator, str) and validator not in joined:
                        joined.append(validator)

        agent = agent_of(doc)
        if agent:
            _append(by_agent, agent, name)

        for validator in phase_validators(doc):
            _append(by_validator, validator, name)

        validators_by_closure[name] = tuple(joined)

    return Index(
        by_kind={k: tuple(v) for k, v in by_kind.items()},
        by_agent={k: tuple(v) for k, v in by_agent.items()},
        by_validator={k: tuple(v) for k, v in by_validator.items()},
        validators_by_closure=validators_by_closure,
        known_kinds=_universe(handoff_specs),
        known_agents=_universe(regs.agent_specs),
        known_validators=_universe(regs.validator_specs),
    )
