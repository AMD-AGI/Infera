# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Stand-ins for the `task_graph` shapes this package reads.

`Task.permissions`, `Grant` and `Access` are rev.-12 material that is being
written in another package as this one is written. The implementation-stage plan
says to code against the documented shape and to satisfy the neighbour with a
stub in one's own tests, so these are that stub: the shapes exactly as
`task_graph/docs/design.md` §3.5 and §3.1 declare them.

Real `TaskId` / `HandoffId` are used, because those **are** shipped — a stub for
them would be a stub for the thing the zone name is built out of.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from task_graph.ids import HandoffId, TaskId

__all__ = ["Access", "Execution", "Grant", "Handoff", "Permissions", "Task"]


class Access(str, Enum):
    """`task_graph` design §3.5. What the author **declared**."""

    READ = "read"
    WRITE = "write"


@dataclass
class Grant:
    path: str = ""
    access: Access = Access.READ
    kind: str | None = None  # a handoff KIND NAME, never an id


@dataclass
class Permissions:
    grants: tuple[Grant, ...] = ()


@dataclass
class Handoff:
    id: HandoffId
    type: str = ""  # the kind name


@dataclass
class Execution:
    attempt: int = 0
    input_versions: dict[HandoffId, int] = field(default_factory=dict)
    output_versions: dict[HandoffId, int] = field(default_factory=dict)


@dataclass
class Task:
    id: TaskId = field(default_factory=TaskId.new)
    inputs: list[HandoffId] = field(default_factory=list)
    outputs: list[HandoffId] = field(default_factory=list)
    parent: TaskId | None = None
    closure: str | None = None
    permissions: Permissions = field(default_factory=Permissions)
    kinds: dict[HandoffId, str] = field(default_factory=dict)
    history: list[Execution] = field(default_factory=list)

    def push_execution(self) -> Execution:
        execution = Execution(attempt=len(self.history))
        self.history.append(execution)
        return execution


@dataclass
class AgentSpec:
    """`agent`'s shape, reduced to the four keys design §11.5 routes here."""

    env: dict[str, str] = field(default_factory=dict)
    rules: tuple[str, ...] = ()
    hooks: tuple[str, ...] = ()
    skills: tuple[str, ...] = ()


def context(
    *,
    domains: Any,
    store_root: str,
    main_repo: str = "",
    handoffs: dict[HandoffId, Handoff] | None = None,
    mapping: dict[str, str] | None = None,
    interpreter_grants: tuple[Any, ...] = (),
    tier: Any = None,
    agent_cli: str | None = None,
    repo_locations: dict[str, str] | None = None,
) -> Any:
    """A `Context`, with the one attribute `protocols.Context` has no field for.

    `repo_locations` is design §7.1.1's *"nothing resolves a name to a
    location"* — a declared ``repos`` entry is a key into the run configuration
    and `Context` carries the mapping. `protocols.Context` does not declare it;
    the README reports that, and `prepare` reads it with `getattr` so the frozen
    seven-field shape still works.
    """
    from env_mgr.protocols import Context, Tier

    ctx = Context(
        domains=domains,
        handoffs=handoffs or {},
        store_root=store_root,
        main_repo=main_repo,
        mapping=mapping or {},
        interpreter_grants=interpreter_grants,
        tier=tier or Tier.PRODUCTION,
        agent_cli=agent_cli,
    )
    if repo_locations:
        return _WithRepos(ctx, repo_locations)
    return ctx


class _WithRepos:
    """`Context` is a NamedTuple, so it cannot carry an eighth attribute. This
    is the adapter a composition root would write until §7.1.1's field lands."""

    def __init__(self, ctx: Any, repo_locations: dict[str, str]) -> None:
        self._ctx = ctx
        self.repo_locations = repo_locations

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ctx, name)
