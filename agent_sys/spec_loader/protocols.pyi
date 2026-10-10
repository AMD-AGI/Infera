# Stub for spec_loader. Signature-only view of the same contract.
#
# Generated from protocols.py and kept in step by
# tests/interfaces/test_stub_agreement.py, which fails if the two
# public surfaces diverge. Reasons live in the .py; this file is the
# shape a type checker reads.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeAlias, TypedDict

__all__ = [
    "ClosureDoc",
    "LoadReport",
    "PackageContents",
    "Problem",
    "Registries",
    "SpecDocument",
    "SpecInconsistent",
    "SpecInvalid",
    "SpecNotFound",
    "SpecRegistry",
    "TaskPackage",
    "TaskSpec",
    "load_package",
    "validate",
]
TaskSpec: TypeAlias = Mapping[str, Any]
ClosureDoc: TypeAlias = Mapping[str, Any]

class _BodyRequired(TypedDict):
    readme: str

class Body(_BodyRequired, total=False):
    entry: str
    materials: list[str]

class SpecNotFound(LookupError): ...
class SpecInvalid(ValueError): ...
class SpecInconsistent(ValueError): ...

@dataclass(frozen=True)
class Problem:
    origin: str
    path: str
    keyword: str
    message: str
    fatal: bool = True
    line: int | None = None
    column: int | None = None

@dataclass(frozen=True)
class LoadReport:
    admitted: Sequence[str]
    problems: Sequence[Problem]

class SpecRegistry(Protocol):
    kind: str

    def add(self, name: str, spec: Mapping[str, Any], *, origin: str) -> None: ...
    def get(self, name: str) -> Mapping[str, Any]: ...
    def names(self) -> list[str]: ...
    def origin_of(self, name: str) -> str: ...
    def __contains__(self, name: str) -> bool: ...

class Registries(Protocol):
    handoff_specs: SpecRegistry
    validator_specs: SpecRegistry
    task_specs: SpecRegistry
    agent_specs: SpecRegistry
    closures: SpecRegistry

    def for_kind(self, kind: str) -> SpecRegistry: ...

@dataclass(frozen=True)
class SpecDocument:
    kind: str
    doc: Mapping[str, Any]
    origin: str
    line: int | None = None
    column: int | None = None

@dataclass(frozen=True)
class PackageContents:
    documents: Sequence[SpecDocument]
    problems: Sequence[Problem]

class TaskPackage(Protocol):
    root: Path

    def documents(self) -> PackageContents: ...

def validate(doc: Any, schema: Mapping[str, Any], *, origin: str) -> list[Problem]: ...
def body_of(spec: Mapping[str, Any]) -> Body: ...
def subgraph_of(task: TaskSpec) -> tuple[Mapping[str, Any], ...]: ...
def task_of(doc: ClosureDoc) -> TaskSpec: ...
def validator_agent_of(spec: Mapping[str, Any]) -> str | None: ...
def load_package(pkg: TaskPackage, registries: Registries) -> LoadReport: ...
