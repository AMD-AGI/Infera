"""Criteria 7, 8 and 20 — the skip, the knob, and what the knob cannot reach.

Criterion 20 is the one with teeth: *"the strict level changes which phases run,
and never which verdicts bind"*. ESLint's `--quiet` erased an error-severity
finding and flipped the exit code to 0 (#14202), and the fix was a **variable
split**, not an enumeration of interactions — which is shown by what happened
next: the optimisation came back one layer down under an explicit claim of
verdict-neutrality, and #19625 found the missed case 22 months later. So the test
here is structural, not a walk over level × verdict combinations.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from handoff.protocols import Verdict
from task_graph.ids import AgentId, HandoffId, TaskId
from tests.validator.conftest import MemoryHandoffStore
from validator.history import Target, may_skip, phase_is_switched_off, priors, top
from validator.protocols import PhaseKind, SkipRecord, StrictLevel
from validator.report import PhaseOutcome


def verdict(name: str, result: bool, strength: str = "strong") -> Verdict:
    return Verdict(
        validator=name,
        result=result,
        strength=strength,
        dimension="completeness",
        task_id=TaskId.new(),
        agent_id=AgentId.new(),
        environment={},
        at=datetime.now(timezone.utc),
    )


@pytest.fixture
def store() -> MemoryHandoffStore:
    return MemoryHandoffStore()


def test_top_is_the_most_recent_execution(store: MemoryHandoffStore) -> None:
    """The history is per validator and each execution appends; the decision reads
    the top and everything beneath it is retained as the record."""
    target = Target(HandoffId.new(), 0)
    store.record_verdict(target.handoff_id, 0, verdict("shape", False))
    store.record_verdict(target.handoff_id, 0, verdict("other", True))
    store.record_verdict(target.handoff_id, 0, verdict("shape", True))

    assert top(store, target, "shape").verdict.result is True
    assert top(store, target, "never") is None


def test_top_is_per_version(store: MemoryHandoffStore) -> None:
    """A verdict is against *this exact version*. A rerun that produced v1 has not
    been validated by v0's record."""
    hid = HandoffId.new()
    store.record_verdict(hid, 0, verdict("shape", True))
    assert top(store, Target(hid, 0), "shape") is not None
    assert top(store, Target(hid, 1), "shape") is None


def test_priors_are_all_or_nothing(store: MemoryHandoffStore) -> None:
    """A validator declaring three kinds that has run against two of them has not
    validated this phase. A partial reuse would be a phase reporting on less than
    it was asked to check."""
    a, b = Target(HandoffId.new(), 0), Target(HandoffId.new(), 0)
    store.record_verdict(a.handoff_id, 0, verdict("shape", True))
    assert priors(store, [a, b], "shape") is None
    store.record_verdict(b.handoff_id, 0, verdict("shape", True))
    assert len(priors(store, [a, b], "shape")) == 2


def test_skip_by_prior_verdict_is_reported(store: MemoryHandoffStore) -> None:
    """Criterion 7. A skip is a **value** on the outcome, not a log line: an
    assertion over a log capture is a test of the logging configuration."""
    target = Target(HandoffId.new(), 0)
    store.record_verdict(target.handoff_id, 0, verdict("shape", True))
    prior = priors(store, [target], "shape")
    assert may_skip(prior, StrictLevel.DEFAULT) is True

    outcome = PhaseOutcome.fold(
        PhaseKind.OUTPUT,
        reused=prior,
        skipped=[SkipRecord("shape", "already validated against this version", prior[0])],
    )
    assert outcome.skipped[0].validator == "shape"
    assert "already validated" in outcome.render()
    assert outcome.passed is True  # the reused verdict is what makes it a pass


def test_skip_by_config_is_reported() -> None:
    """Criterion 7's other half — a phase switched off wholesale.

    Reported unconditionally rather than behind a verbosity flag: pytest's `-r`
    defaults to `fE`, so a skip's *reason* needs `-rs` to appear at all, and
    codeql-action#3156 is what that costs — `upload-sarif` *"silently stopped
    uploading"*, the job stayed green, and the failure was visible only as the
    absence of alerts.
    """
    assert phase_is_switched_off(StrictLevel.NONE) is True
    assert phase_is_switched_off(StrictLevel.DEFAULT) is False
    outcome = PhaseOutcome.fold(
        PhaseKind.INPUT,
        skipped=[SkipRecord("shape", "phase switched off by --validation-strict-level", None)],
    )
    assert outcome.empty is True and outcome.passed is False
    assert "switched off" in outcome.render()


@pytest.mark.parametrize(
    ("level", "prior_skip", "config_skip"),
    [
        (StrictLevel.STRICT, False, False),
        (StrictLevel.DEFAULT, True, False),
        (StrictLevel.NONE, True, True),
    ],
)
def test_strict_level_governs_skips(
    store: MemoryHandoffStore, level: StrictLevel, prior_skip: bool, config_skip: bool
) -> None:
    """Criterion 8 — the level changes **which skips are permitted**, and the
    three levels are three distinct answers rather than a spectrum with a gap."""
    target = Target(HandoffId.new(), 0)
    store.record_verdict(target.handoff_id, 0, verdict("shape", True))
    assert may_skip(priors(store, [target], "shape"), level) is prior_skip
    assert phase_is_switched_off(level) is config_skip


def test_may_skip_never_consults_the_verdict(store: MemoryHandoffStore) -> None:
    """The load-bearing clause. A `may_skip` that reused a pass and re-ran a
    failure would be the level deciding a verdict by the back door. Bazel makes
    exactly that asymmetry — `--cache_test_results=auto` — and it is a coherent
    *policy*; ours is not it, because spec §5.5 has one rule."""
    passing, failing = Target(HandoffId.new(), 0), Target(HandoffId.new(), 0)
    store.record_verdict(passing.handoff_id, 0, verdict("shape", True))
    store.record_verdict(failing.handoff_id, 0, verdict("shape", False))
    for target in (passing, failing):
        assert may_skip(priors(store, [target], "shape"), StrictLevel.DEFAULT) is True


def test_reused_failure_still_fails() -> None:
    """Criterion 20. The fold is over `ran | reused`, so a recorded failure binds
    exactly as a fresh one does."""
    from validator.protocols import VerdictRecord

    hid = HandoffId.new()
    record = VerdictRecord(verdict=verdict("shape", False), handoff_id=hid, version=0)
    outcome = PhaseOutcome.fold(
        PhaseKind.OUTPUT,
        reused=[record],
        skipped=[SkipRecord("shape", "already validated against this version", record)],
    )
    assert outcome.passed is False


def test_strict_level_cannot_reach_the_fold() -> None:
    """Criterion 20, structurally. `fold` takes no level, so there is no argument
    through which the knob could reach a verdict — the ESLint variable split, as a
    signature rather than as care."""
    import inspect

    params = set(inspect.signature(PhaseOutcome.fold).parameters)
    assert params == {"kind", "ran", "reused", "skipped"}
    assert not any("level" in p or "strict" in p for p in params)

    passed = set(inspect.signature(may_skip).parameters)
    assert "level" in passed  # the knob lives here, and only here
