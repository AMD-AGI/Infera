# Stub for validator. Signature-only view of the same contract.
#
# Generated from protocols.py and kept in step by
# tests/interfaces/test_stub_agreement.py, which fails if the two
# public surfaces diverge. Reasons live in the .py; this file is the
# shape a type checker reads.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from handoff.protocols import Verdict
from spec_loader.protocols import Body, SpecInvalid
from task_graph.ids import HandoffId

__all__ = [
    "Body",
    "Dimension",
    "NestedComposite",
    "PhaseKind",
    "PhaseOutcome",
    "PhaseRunner",
    "Reducer",
    "SeparationViolation",
    "SkipRecord",
    "Strength",
    "StrictLevel",
    "Validator",
    "ValidatorInvalid",
    "Verdict",
    "VerdictRecord",
]

class ValidatorInvalid(SpecInvalid): ...
class SeparationViolation(ValueError): ...
class NestedComposite(ValidatorInvalid): ...

class Dimension(str, Enum):
    COMPLETENESS = "completeness"
    USABILITY = "usability"
    TRUSTWORTHINESS = "trustworthiness"

class Strength(str, Enum):
    STRONG = "strong"
    LONG_TERM_STRONG = "long_term_strong"
    WEAK = "weak"

class PhaseKind(str, Enum):
    INPUT = "input_validation"
    OUTPUT = "output_validation"

class StrictLevel(str, Enum):
    NONE = "none"
    DEFAULT = "default"
    STRICT = "strict"

class Validator(Protocol):
    brief: str
    inputs: tuple[str, ...]
    dimension: Dimension
    strength: Strength

    def __call__(self, handoffs: Mapping[HandoffId, Any]) -> dict[HandoffId, bool]: ...

class Reducer(Protocol):
    name: str

    def __call__(self, verdicts: Sequence[bool]) -> bool: ...

@dataclass(frozen=True)
class VerdictRecord:
    verdict: Verdict
    handoff_id: HandoffId
    version: int

@dataclass(frozen=True)
class SkipRecord:
    validator: str
    reason: str
    reused: VerdictRecord | None

@dataclass(frozen=True)
class PhaseOutcome:
    kind: PhaseKind
    ran: Sequence[VerdictRecord]
    reused: Sequence[VerdictRecord]
    skipped: Sequence[SkipRecord]
    empty: bool
    verdicts_expected: bool

    @property
    def passed(self) -> bool: ...

class PhaseRunner(Protocol):
    def run_phase(self, kind: PhaseKind, task: Any, registry: Any) -> PhaseOutcome: ...
