"""Five dicts and a closure builder.

`docs/interfaces.md` §6: *"The six checks need only `Registries`, which is a
Protocol a test satisfies with five dicts."* This file is that sentence, made to
run. Nothing here imports a sibling package's implementation, and nothing needs a
composition root.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from closure.registry import ClosureRegistry
from closure.task_registry import TaskSpecRegistry
from spec_loader.protocols import SpecInconsistent, SpecNotFound


class DictRegistry:
    """A `SpecRegistry` that is a dict with a collision policy.

    The stub the Protocol promises. It carries `origin_of` because the closure
    pass prints the file the author wrote, and because the real base will.
    """

    def __init__(self, kind: str, specs: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        self.kind = kind
        self._specs: dict[str, Mapping[str, Any]] = dict(specs or {})
        self._origins: dict[str, str] = {n: f"{kind}/{n}.jsonnet" for n in self._specs}
        #: closure name -> the phase validators it named. Only `validator_specs`
        #: is ever asked for this; the other four stubs carry it unused.
        self.phase_edges: dict[str, list[str]] = {}

    def add(self, name: str, spec: Mapping[str, Any], *, origin: str) -> None:
        held = self._specs.get(name)
        if held is not None and held != spec:
            raise SpecInconsistent(f"{self.kind} {name!r} already registered")
        self._specs[name] = spec
        self._origins[name] = origin

    def get(self, name: str) -> Mapping[str, Any]:
        try:
            return self._specs[name]
        except KeyError:
            raise SpecNotFound(f"no {self.kind} named {name!r} (have: {self.names()})") from None

    def names(self) -> list[str]:
        return sorted(self._specs)

    def origin_of(self, name: str) -> str:
        return self._origins[name]

    def bind_phase(self, closure: str, validators: Sequence[str]) -> None:
        """`ValidatorSpecRegistry`'s, and the stub carries it deliberately.

        The closure pass calls this directly rather than through a `getattr`
        guard, so a stub that lacked it would fail loudly here instead of leaving
        the edge silently unrecorded in the assembled system. The obligation is
        visible in the double because it is real in production.
        """
        self.phase_edges.setdefault(closure, []).extend(validators)

    def __contains__(self, name: str) -> bool:
        return name in self._specs


@dataclass
class Regs:
    """A `Registries`. Five registries and `for_kind`, and nothing else."""

    handoff_specs: DictRegistry = field(default_factory=lambda: DictRegistry("handoff"))
    validator_specs: DictRegistry = field(default_factory=lambda: DictRegistry("validator"))
    task_specs: TaskSpecRegistry = field(default_factory=TaskSpecRegistry)
    agent_specs: DictRegistry = field(default_factory=lambda: DictRegistry("agent"))
    closures: ClosureRegistry = field(default_factory=ClosureRegistry)

    def for_kind(self, kind: str) -> Any:
        return {
            "handoff": self.handoff_specs,
            "validator": self.validator_specs,
            "task": self.task_specs,
            "agent": self.agent_specs,
            "closure": self.closures,
        }[kind]

    # -- convenience for building a world, not part of the Protocol --------- #

    def with_kinds(self, *names: str, validators: Sequence[str] = ()) -> Regs:
        for name in names:
            self.handoff_specs.add(
                name,
                {"name": name, "version": "1", "validators": list(validators)},
                origin=f"handoff/{name}.jsonnet",
            )
        return self

    def with_agents(self, *names: str) -> Regs:
        for name in names:
            self.agent_specs.add(
                name, {"name": name, "kind": "program", "version": "1"}, origin=f"agent/{name}.j"
            )
        return self

    def with_validators(self, *names: str) -> Regs:
        for name in names:
            self.validator_specs.add(
                name, {"name": name, "version": "1"}, origin=f"validator/{name}.jsonnet"
            )
        return self

    def with_closure(self, doc: Mapping[str, Any], *, origin: str | None = None) -> Regs:
        name = doc["name"]
        where = origin or f"packages/demo/closures/{name}.jsonnet"
        # Only the closure. `check_closures` is what keys the nested task spec
        # into `task_specs`, and a fixture that did it too would hide the day
        # that stopped being true — which is how it went unnoticed the first
        # time (`task_graph` found `check_graph` walking an empty catalogue).
        self.closures.add(name, doc, origin=where)
        return self


def grant(kind: str, access: str = "read", path: str | None = None) -> dict[str, Any]:
    return {"path": path or f"handoffs/{kind}", "access": access, "kind": kind}


def make_closure(
    name: str = "collect_trace",
    *,
    inputs: Sequence[str] = (),
    outputs: Sequence[str] = (),
    handoffs: Sequence[str] | None = None,
    agent: str | None = "profiler",
    validators: Sequence[str] = (),
    grants: Sequence[Mapping[str, Any]] | None = None,
    body: Mapping[str, Any] | None = None,
    subgraph: Sequence[Mapping[str, Any]] | None = None,
    **task_extra: Any,
) -> dict[str, Any]:
    """A closure document that passes every check unless told otherwise.

    Permissions default to *exactly* covering the declared inputs and outputs, so
    a coverage test says what it removed rather than what it added.
    """
    if grants is None:
        grants = [grant(k, "read") for k in inputs] + [grant(k, "write") for k in outputs]
    task: dict[str, Any] = {
        "goal": f"{name}: do the thing",
        "body": dict(body) if body is not None else {"readme": f"{name}/readme.md"},
        "inputs": list(inputs),
        "outputs": list(outputs),
        "permissions": {"grants": [dict(g) for g in grants]},
        "version": "1",
        **task_extra,
    }
    if subgraph is not None:
        task["subgraph"] = [dict(s) for s in subgraph]
    doc: dict[str, Any] = {
        "name": name,
        "description": f"the {name} step",
        "task": task,
        "handoffs": list(handoffs if handoffs is not None else [*inputs, *outputs]),
        "validators": list(validators),
    }
    if agent is not None:
        doc["agent"] = agent
    return doc


@dataclass(frozen=True)
class Report:
    """`handoff.HandoffLoadReport`, satisfied structurally.

    Every call to `check_closures` supplies one, because the real parameter is
    required and a `None` raises — `docs/interfaces.md` §4.11. An empty
    `without_validator` is "no escape-hatch admissions"; "nobody told me" is not
    representable, which is the whole point of the ruling.
    """

    admitted: list[str] = field(default_factory=list)
    without_validator: list[str] = field(default_factory=list)


NO_ESCAPE_HATCH = Report()


@pytest.fixture
def regs() -> Regs:
    return Regs()


@pytest.fixture
def report() -> Report:
    return Report()
