# Stub for env_mgr. Signature-only view of the same contract.
#
# Generated from protocols.py and kept in step by
# tests/interfaces/test_stub_agreement.py, which fails if the two
# public surfaces diverge. Reasons live in the .py; this file is the
# shape a type checker reads.

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from enum import Enum, Flag, auto
from types import MappingProxyType
from typing import Any, Literal, NamedTuple, Protocol

from task_graph.ids import HandoffId, TaskId

__all__ = [
    "Confinement",
    "Context",
    "Domain",
    "DomainKind",
    "DomainRegistry",
    "EnvManager",
    "Granted",
    "Mode",
    "NoConfinement",
    "Policy",
    "Prepared",
    "PrepareRefused",
    "SyncReport",
    "Tier",
    "UnresolvedGrant",
    "Zone",
    "contained",
]

class NoConfinement(RuntimeError): ...
class PrepareRefused(RuntimeError): ...
class UnresolvedGrant(ValueError): ...

def contained(path: str, zone: str) -> bool: ...

class DomainKind(str, Enum):
    HANDOFF_STORAGE = "handoff_storage"
    PLAYGROUND = "playground"
    WORKSPACE = "workspace"

class Domain(NamedTuple):
    name: str
    root: str
    kind: DomainKind

class DomainRegistry(Protocol):
    def register(self, name: str, root: str, kind: DomainKind) -> Domain: ...
    def get(self, name: str) -> Domain: ...
    def __iter__(self) -> Iterator[Domain]: ...

class Zone(NamedTuple):
    task_id: TaskId
    attempt: int
    root: str

    def contains(self, path: str) -> bool: ...

class Mode(Flag):
    READ_EXEC = auto()
    READ_WRITE = auto()

class Granted(NamedTuple):
    path: str
    mode: Mode
    optional: bool = False

class Policy(NamedTuple):
    granted: tuple[Granted, ...]

    def with_(self, *more: Granted) -> Policy: ...

class Confinement(NamedTuple):
    mechanism: Literal["bwrap", "landlock"]
    filesystem: bool
    network: bool
    pid: bool
    abi: int | None

class Tier(str, Enum):
    PRODUCTION = "production"
    STRICT = "strict"

class SyncReport(NamedTuple):
    sent: int
    received: int
    conflicts: tuple[str, ...]

class Context(NamedTuple):
    domains: DomainRegistry
    handoffs: Mapping[HandoffId, Any]
    store_root: str
    main_repo: str
    mapping: Mapping[str, str]
    interpreter_grants: tuple[Granted, ...]
    tier: Tier
    agent_cli: str | None = None
    package: str | None = None
    package_stage: tuple[str, ...] | None = None
    transports: Mapping[str, Any] = ...
    far_roots: Mapping[str, str] = ...

class Prepared(NamedTuple):
    zone: Zone
    workspace: Any
    policy: Policy
    confinement: Confinement | None
    sync: SyncReport
    environment: Mapping[str, str] = MappingProxyType({})
    agent_cli: str | None = None
    permissions_enforced: bool = False
    output_paths: Mapping[HandoffId, str] = MappingProxyType({})
    staged_package: str | None = None
    tools: tuple[Any, ...] = ...
    def spawn(self, argv: Sequence[str], **popen_kwargs: Any) -> Any: ...
    def wrap_argv(self, argv: Sequence[str]) -> list[str]: ...

class ValidationZone(NamedTuple):
    root: str
    phase: str
    materials: Mapping[HandoffId, str]

class EnvManager(Protocol):
    def prepare(self, task: Any, execution: Any, agent_spec: Any = None) -> Prepared: ...
    def prepare_validation(self, task: Any, execution: Any, phase: Any) -> ValidationZone: ...
    def place_zone(self, task: Any, execution: Any) -> Zone: ...
