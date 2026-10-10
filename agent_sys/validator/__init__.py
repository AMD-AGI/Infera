"""validator — a handoff is only a contract if something checks it.

What leaves this package is `docs/interfaces.md` §4.3's list. The vocabulary
itself lives in `protocols.py`, which is the frozen, importable half of that
contract, and is re-exported here rather than re-declared: two records of one
fact is `engineer_principle.md` §1's failure.

`Verdict` is **`handoff`'s** — the module that persists a record is the module
that has to keep it readable — and `VerdictRecord` is this module's view of one.
"""

from validator.composite import Composite
from validator.phase import PhaseRunner
from validator.protocols import (
    Body,
    Dimension,
    NestedComposite,
    PhaseKind,
    Reducer,
    SeparationViolation,
    SkipRecord,
    Strength,
    StrictLevel,
    Validator,
    ValidatorInvalid,
    Verdict,
    VerdictRecord,
)
from validator.reducers import REDUCERS, get_reducer
from validator.registry import RunRecord, RunState, ValidatorSpecRegistry
from validator.report import Evidence, PhaseOutcome
from validator.separation import check_separation
from validator.spec import Cost, LogicSource, Tags, ValidatorSpec

__all__ = [
    "REDUCERS",
    "Body",
    "Composite",
    "Cost",
    "Dimension",
    "Evidence",
    "LogicSource",
    "NestedComposite",
    "PhaseKind",
    "PhaseOutcome",
    "PhaseRunner",
    "Reducer",
    "RunRecord",
    "RunState",
    "SeparationViolation",
    "SkipRecord",
    "Strength",
    "StrictLevel",
    "Tags",
    "Validator",
    "ValidatorInvalid",
    "ValidatorSpec",
    "ValidatorSpecRegistry",
    "Verdict",
    "VerdictRecord",
    "check_separation",
    "get_reducer",
]
