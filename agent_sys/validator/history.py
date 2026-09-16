"""Re-validation is decided from the verdict record, not from a cache key.

The natural instinct at this seam is a content-addressed verdict cache keyed on
the handoff digest plus something identifying the checker. Five such key schemes
were measured against six things that can change the answer, and **"the
validator's implementation changed" is a stale hit under every one of them**,
because implementation source appears in no spec file. The record answers the
question the cache was being built to answer — *did this exact validator run
against this exact version, and what did it say* — without a key at all, because
the record **is** the answer rather than an index into one.

What it does not do, stated rather than left to be discovered: it does not detect
that the validator's code changed since. Nix names the same distinction and says
the second half is unavailable — *"there is no way to audit a build trace entry
except for by performing the build again from scratch"*, and *"the decision of
whether to trust a counterparty's build trace is a fundamentally subjective
policy choice."* That is the honest description of `--validation-strict-level`:
**a trust policy over recorded verdicts, not a correctness mechanism.**

Imports nothing from this package except `protocols`, so it is testable over a
list of records with no registry, runner or store of ours.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from task_graph.ids import HandoffId
from validator.protocols import StrictLevel, VerdictRecord

__all__ = ["Target", "may_skip", "phase_is_switched_off", "priors", "top"]


@dataclass(frozen=True)
class Target:
    """One handoff version a phase checks. Keyed by uuid, never by kind: a task
    with two inputs of the same kind would be ambiguous under a kind key."""

    handoff_id: HandoffId
    version: int


def top(store: Any, target: Target, validator: str) -> VerdictRecord | None:
    """The most recent execution of `validator` against this handoff version.

    The history is per validator and each execution appends, so the decision
    reads the top and everything beneath it is retained as the record. `None`
    means it has never run against this version.
    """
    for verdict in reversed(list(store.read_verdicts(target.handoff_id, target.version))):
        if verdict.validator == validator:
            return VerdictRecord(
                verdict=verdict, handoff_id=target.handoff_id, version=target.version
            )
    return None


def priors(
    store: Any, targets: Sequence[Target], validator: str
) -> tuple[VerdictRecord, ...] | None:
    """Every target's top record, or `None` if any target has none.

    All-or-nothing on purpose. A validator declaring three kinds that has run
    against two of them has not validated this phase, and a partial reuse would
    be a phase that reports on less than it was asked to check — which is the
    silent-narrowing failure spec §1 exists to prevent.
    """
    if not targets:
        return None
    found = [top(store, t, validator) for t in targets]
    if any(record is None for record in found):
        return None
    return tuple(record for record in found if record is not None)


def may_skip(prior: Sequence[VerdictRecord] | None, level: StrictLevel) -> bool:
    """True iff a prior verdict exists for this exact version and the level
    permits reusing it. **Never consults the verdict's value.**

    The last clause is load-bearing. A `may_skip` that reused a pass and re-ran a
    failure would be the level deciding a verdict by the back door. Bazel does
    make that asymmetry deliberately — `--cache_test_results=auto` reuses a
    cached pass and re-runs a cached failure — and it is a coherent policy, but it
    is a *policy* and ours is not it: spec §5.5 has one rule, and a recorded
    failure must bind exactly as a fresh one does.
    """
    if prior is None or not prior:
        return False
    return level is not StrictLevel.STRICT


def phase_is_switched_off(level: StrictLevel) -> bool:
    """Spec §3.3's other skip: a phase switched off wholesale.

    Separated from `may_skip` because the two answer different questions and
    `may_skip`'s docstring has to stay true. Note what the separation buys: a
    switched-off phase runs nothing, so it folds to an outcome that is `empty`,
    and an empty phase is **not** a pass. The knob therefore cannot turn a
    failing phase green — which is criterion 20, structurally.
    """
    return level is StrictLevel.NONE
