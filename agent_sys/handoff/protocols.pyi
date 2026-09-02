# Stub for handoff. Signature-only view of the same contract.
#
# Generated from protocols.py and kept in step by
# tests/interfaces/test_stub_agreement.py, which fails if the two
# public surfaces diverge. Reasons live in the .py; this file is the
# shape a type checker reads.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Literal, Protocol

from task_graph.ids import AgentId, HandoffId, TaskId

__all__ = [
    "Content",
    "ContentType",
    "DigestMismatch",
    "HandoffKind",
    "HandoffLoadReport",
    "HandoffStore",
    "Item",
    "Malformed",
    "Manifest",
    "NotContained",
    "PointerInvalid",
    "PointerMiss",
    "Scope",
    "Verdict",
    "check_contained",
    "resolve",
    "tree_digest",
]

class Malformed(ValueError): ...
class DigestMismatch(ValueError): ...
class NotContained(ValueError): ...
class PointerInvalid(ValueError): ...
class PointerMiss(LookupError): ...

class Scope(str, Enum):
    FIXED_REQUIRED = "fixed.required"
    FIXED_OPTIONAL = "fixed.optional"
    ADDONS_TEMP = "addons.temp"
    ADDONS_KNOWLEDGE = "addons.knowledge"

@dataclass(frozen=True)
class Item:
    key: str
    kind: Literal["file", "tree", "data"]
    path: Path | None = None
    value: Any | None = None

@dataclass(frozen=True)
class ContentType:
    name: str
    required_items: frozenset[str]
    optional_items: frozenset[str]
    readme_sections: tuple[str, ...]

@dataclass(frozen=True)
class Content:
    root: Path
    readme: Path
    items: Mapping[str, Item]

@dataclass(frozen=True)
class Manifest:
    digest: Mapping[str, str]
    algorithm: str
    kind: str
    producer: TaskId
    created_at: datetime

@dataclass(frozen=True)
class Verdict:
    validator: str
    result: bool
    strength: str
    dimension: str
    task_id: TaskId
    agent_id: AgentId | None
    environment: Mapping[str, Any]
    at: datetime

@dataclass(frozen=True)
class HandoffKind:
    name: str
    content_type: str
    items_schema: Mapping[str, Any]
    validators: tuple[str, ...]
    scope: Scope
    version: str | None = None

@dataclass(frozen=True)
class HandoffLoadReport:
    admitted: Sequence[str]
    without_validator: Sequence[str]

class HandoffStore(Protocol):
    def list_versions(self, hid: HandoffId) -> list[int]: ...
    def latest(self, hid: HandoffId) -> int | None: ...
    def get_manifest(self, hid: HandoffId, version: int) -> Manifest: ...
    def open_item(self, hid: HandoffId, version: int, key: str) -> BinaryIO: ...
    def copy_out(self, hid: HandoffId, version: int, dst: Path) -> Content: ...
    def allocate(self, hid: HandoffId) -> int: ...
    def seal(self, hid: HandoffId, version: int, *, producer: TaskId) -> str | None: ...
    def put(self, hid: HandoffId, content_dir: Path, *, producer: TaskId) -> int: ...
    def exists(self, hid: HandoffId, version: int | None = None) -> bool: ...
    def record_verdict(self, hid: HandoffId, version: int, verdict: Verdict) -> None: ...
    def read_verdicts(self, hid: HandoffId, version: int) -> list[Verdict]: ...

def tree_digest(root: bytes) -> bytes: ...
def resolve(doc: Any, pointer: str) -> Any: ...
def check_contained(candidate: Path, zone: Path) -> None: ...
