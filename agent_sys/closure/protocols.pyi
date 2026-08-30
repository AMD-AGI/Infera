# Stub for closure. Signature-only view of the same contract.
#
# Generated from protocols.py and kept in step by
# tests/interfaces/test_stub_agreement.py, which fails if the two
# public surfaces diverge. Reasons live in the .py; this file is the
# shape a type checker reads.

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from typing import Any, Protocol

from spec_loader.protocols import ClosureDoc, Problem, Registries, SpecRegistry, TaskSpec

__all__ = [
    "ClosureDoc",
    "ClosureRegistry",
    "TaskSpec",
    "TaskSpecRegistry",
    "agent_of",
    "check_closures",
    "declared_handoffs",
    "named_kinds",
    "permissions_of",
    "phase_validators",
]

def declared_handoffs(doc: ClosureDoc) -> tuple[str, ...]: ...
def named_kinds(task: TaskSpec) -> tuple[str, ...]: ...
def phase_validators(doc: ClosureDoc) -> tuple[str, ...]: ...
def agent_of(doc: ClosureDoc) -> str: ...
def permissions_of(task: TaskSpec) -> Mapping[str, Any]: ...

class TaskSpecRegistry(SpecRegistry, Protocol): ...

class ClosureRegistry(SpecRegistry, Protocol):
    def handoff_kinds(self, closure: str) -> tuple[str, ...]: ...
    def validators_for(self, closure: str) -> tuple[str, ...]: ...
    def closures_using_kind(self, kind: str) -> tuple[str, ...]: ...
    def closures_using_agent(self, agent: str) -> tuple[str, ...]: ...
    def closures_using_validator(self, name: str) -> tuple[str, ...]: ...
    def agent_of(self, closure: str) -> str: ...
    def freeze(self) -> None: ...

def check_closures(
    regs: Registries, handoff_report: Any, *, skip: AbstractSet[str] = frozenset()
) -> list[Problem]: ...
