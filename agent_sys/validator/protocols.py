"""What leaves `validator/`.

A validator is a phase inside `TaskRunner`, not a graph node: the scheduler
dispatches one task and gets one completion, and what happened in between is the
runner's business.

Declarations only. See `docs/interfaces.md` §4.3.
"""

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


# --------------------------------------------------------------------------- #
# Errors


class ValidatorInvalid(SpecInvalid):
    """A validator spec, or a composite, that cannot be admitted.

    **A `SpecInvalid`, and that relationship is deliberate rather than
    decorative.** `handoff` and `agent` raise `SpecInvalid` itself out of their
    registries; this package raises its own type, and for a while that type was a
    bare `ValueError` — so `load_package`, which caught `SpecInvalid` and
    `SpecInconsistent`, did not catch it. One package's choice of exception
    aborted the whole multi-package load, against `load_package`'s first stated
    property: *one broken spec must not hide the other nine.*

    `spec_loader` fixed it properly by catching `ValueError` (80e2f42), so this
    is belt-and-braces rather than the repair. It is worth having anyway: the
    kinship is now stated in the type rather than resting on both happening to
    descend from `ValueError`, and a reader of either side can see it.
    """


class SeparationViolation(ValueError):
    """Criterion 11: a validator whose logic lives where the producing task's
    declared permissions reach. The producer may execute in its own zone; it may
    not write the rule it will be judged by."""


class NestedComposite(ValidatorInvalid):
    """A composite naming a composite. Rejected by the schema; this is the guard
    for the path that bypasses the loader."""


# --------------------------------------------------------------------------- #
# The implementation
#
# `Body` is **re-exported from `spec_loader`, not declared here.** It stays in
# this package's exports exactly as `docs/interfaces.md` §4.3 lists it — the name
# still leaves `validator` — but there is one declaration of the shape instead of
# three. `closure` spec §2.6 and `validator` spec §6.1 both say a task's body and
# a validator's are deliberately the same thing, and `_common.schema.json` has
# one `$defs.body` that both schemas `$ref`; three Python copies of it was
# `engineer_principle.md` §1's failure, and the copies had **disagreed** —
# `closure`'s returned `{}` for an absent body and `agent`'s returned a truthy
# `Body(readme='')`.
#
# A `TypedDict`, so absent is `{}` and falsy. That is the **designed** absent and
# is not the `entry: ""` trap one level down: a body that exists always has a
# non-empty `readme`, because `_common` makes it required with `minLength: 1`, so
# `{}` cannot be a body that exists. `test_an_existing_body_is_never_empty` pins
# that reasoning, because it is the whole safety argument and it is invisible in
# the code.

# --------------------------------------------------------------------------- #
# Vocabulary


class Dimension(str, Enum):
    COMPLETENESS = "completeness"
    USABILITY = "usability"
    TRUSTWORTHINESS = "trustworthiness"


class Strength(str, Enum):
    """Three, not two. `long_term_strong` is **not** a weaker `strong` — the
    rigour is the same and the *timing* differs, so it folds as `strong` and
    renders as itself."""

    STRONG = "strong"
    LONG_TERM_STRONG = "long_term_strong"
    WEAK = "weak"


class PhaseKind(str, Enum):
    INPUT = "input_validation"
    OUTPUT = "output_validation"


class StrictLevel(str, Enum):
    """`--validation-strict-level`. Decides which phases *run*, never which
    verdicts *bind* — and that is structural, not maintained by care: the knob
    reaches the run set only, and the fold is over `ran | reused`."""

    NONE = "none"
    DEFAULT = "default"
    STRICT = "strict"


# --------------------------------------------------------------------------- #
# The two protocols


class Validator(Protocol):
    """One verdict per input handoff.

    **This Protocol is a static type and is not the admission gate.** Measured:
    `issubclass` raises outright on a Protocol with non-method members, and
    `isinstance` is presence-only — `strength=None` passes it. Worse,
    `inputs="trace"` passes, because a bare string is iterable, and one declared
    kind silently becomes five characters. The gate is the pydantic model over
    the spec record.
    """

    brief: str
    inputs: tuple[str, ...]
    dimension: Dimension
    strength: Strength

    def __call__(self, handoffs: Mapping[HandoffId, Any]) -> dict[HandoffId, bool]:
        """Every key of `handoffs` appears in the result. A member that omits one
        raises rather than yielding `None`, because `None` folded as falsy is
        indistinguishable from a genuine `False`."""
        ...


class Reducer(Protocol):
    """Folds several members' verdicts on ONE handoff into one verdict.

    Injected rather than chosen here; the alpha registers `all` and nothing else.
    A composite reduces **per handoff**, which keeps its result type identical to
    a leaf's — so a composite is type-substitutable and `run_phase` has no place
    to apply a reducer even if someone wanted one.
    """

    name: str

    def __call__(self, verdicts: Sequence[bool]) -> bool: ...


# --------------------------------------------------------------------------- #
# Records and outcomes


@dataclass(frozen=True)
class VerdictRecord:
    """This module's view of one persisted `handoff.Verdict`.

    What `top()` returns and `may_skip()` reads. The persisted shape is
    `handoff`'s; this is not a second record of it.
    """

    verdict: Verdict
    handoff_id: HandoffId
    version: int


@dataclass(frozen=True)
class SkipRecord:
    """A skipped validator, as a value rather than a log line.

    Criterion 7 asserts the skip is reported, and an assertion over a log capture
    is a test of the logging configuration.
    """

    validator: str
    reason: str
    reused: VerdictRecord | None


@dataclass(frozen=True)
class PhaseOutcome:
    """What one validation phase produced.

    **No field defaults to success, and an empty phase is not a pass.** Four
    systems reached that independently and none of them spells the third state
    "pass" — pytest exits 5, Bazel's `NO_STATUS = 0` precedes `PASSED = 1`,
    GitHub leaves a skipped workflow Pending and blocking, SARIF has
    `notApplicable`. JUnit XML is the counter-example to design against: pass is
    its structural default, so a producer that forgets `<skipped/>` emits a pass.

    The aggregate **carries the qualification**, not only the per-item line. That
    is where pytest's XPASS lost it: distinguishable rendering was not enough,
    because the exit code stayed 0 and the bar stayed green.

    `verdicts_expected` is what makes two kinds of `empty` distinguishable, and
    it is `interfaces.md` §4.15's ruling: under `StrictLevel.NONE` an empty phase
    means *no validation was asked for*, and under every other level an empty
    **output** phase means *nothing checked what this task produced* — a fault
    that blocks. It is set by the constructor rather than folded, so the level
    never reaches `fold`.
    """

    kind: PhaseKind
    ran: Sequence[VerdictRecord]
    reused: Sequence[VerdictRecord]
    skipped: Sequence[SkipRecord]
    empty: bool
    verdicts_expected: bool

    @property
    def passed(self) -> bool:
        """Folded over `ran | reused` — **never over `ran` alone**. That is the
        only reading under which a recorded failure still fails the phase, and it
        is what makes the strict level structurally unable to reach a verdict."""
        ...


# --------------------------------------------------------------------------- #
# The seam `agent.Runner` calls


class PhaseRunner(Protocol):
    """The two validation phases. Registered once as `phase_runner`; called twice
    per dispatch, by `agent.Runner`, around the main phase.

    `strict_level` is bound at construction because it is a run-wide policy;
    `registry` is per call because that is how the phase reaches `handoff_mgr`
    and `agent_mgr` at the moment it needs them.
    """

    def run_phase(self, kind: PhaseKind, task: Any, registry: Any) -> PhaseOutcome:
        """Select the bound validators cheap-first, run or reuse each, fold.

        Builds a **fresh** environment per validator, inside the loop — rebuilt,
        never inherited. There is no cost argument against it: a full private
        read-only namespace was measured at 13.8 ms against 61–66 ms for the
        Python interpreter that runs inside it.
        """
        ...
