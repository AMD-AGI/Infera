"""Criteria 18 and 19 — a failure binds at every strength, a pass is qualified.

Criterion 19 is close to unprecedented and the one precedent is a cautionary
tale. No surveyed system qualifies a *pass* in its aggregate view: Dagster's
`AssetCheckResult(passed=True, severity=WARN)` is constructible and then ignored
at every consumption site, and SARIF chose a different axis entirely — you *type*
the pass, you do not grade it. The precedent that does exist is pytest's XPASS,
and it fails in practice: `xpassed` has its own count, its own progress character
and its own summary prefix, and the exit code is 0 and the bar is green.

So the aggregate carries the qualification, not only the per-item line.
"""

from __future__ import annotations

from datetime import datetime, timezone

from handoff.protocols import Verdict
from task_graph.ids import AgentId, HandoffId, TaskId
from validator.protocols import PhaseKind, VerdictRecord
from validator.report import Evidence, PhaseOutcome


def record(name: str, result: bool, strength: str) -> VerdictRecord:
    return VerdictRecord(
        verdict=Verdict(
            validator=name,
            result=result,
            strength=strength,
            dimension="trustworthiness",
            task_id=TaskId.new(),
            agent_id=AgentId.new(),
            environment={},
            at=datetime.now(timezone.utc),
        ),
        handoff_id=HandoffId.new(),
        version=0,
    )


def test_weak_failure_binds_like_strong() -> None:
    """Criterion 18 — asserted by running the same phase with one of each and
    observing the same outcome.

    The asymmetry has a plain justification: a check that found something wrong
    found something wrong, whatever its rigour. It is the *absence* of a finding
    that is worth less when the method is weak.
    """
    weak = PhaseOutcome.fold(PhaseKind.OUTPUT, ran=[record("w", False, "weak")])
    strong = PhaseOutcome.fold(PhaseKind.OUTPUT, ran=[record("s", False, "strong")])
    assert weak.passed is strong.passed is False
    assert weak.evidence is strong.evidence is Evidence.FAILED


def test_one_weak_failure_fails_a_phase_of_strong_passes() -> None:
    """Spec §5.5: one fails, all fail. No weighting, no quorum."""
    outcome = PhaseOutcome.fold(
        PhaseKind.OUTPUT,
        ran=[record("a", True, "strong"), record("b", True, "strong"), record("c", False, "weak")],
    )
    assert outcome.passed is False
    assert [r.verdict.validator for r in outcome.failures] == ["c"]


def test_weak_pass_is_qualified_in_the_aggregate() -> None:
    """Criterion 19. **The aggregate** carries the qualification, because that is
    where pytest lost it: distinguishable rendering was demonstrably not enough
    when the exit code stayed 0."""
    weak = PhaseOutcome.fold(PhaseKind.OUTPUT, ran=[record("w", True, "weak")])
    strong = PhaseOutcome.fold(PhaseKind.OUTPUT, ran=[record("s", True, "strong")])

    assert weak.passed is strong.passed is True
    assert weak.evidence is Evidence.LOW_CONFIDENCE
    assert strong.evidence is Evidence.ESTABLISHED
    assert weak.evidence is not strong.evidence  # distinguishable in the output
    assert "low_confidence" in weak.render()
    assert "weak" in weak.render()


def test_one_strong_pass_establishes_a_mixed_phase() -> None:
    """§15 O9 asks whether *"every validator here was weak"* deserves its own
    treatment, distinct from "some weak, some strong". It gets one: the mixed
    phase is `established`, the all-weak phase is not."""
    mixed = PhaseOutcome.fold(
        PhaseKind.OUTPUT, ran=[record("w", True, "weak"), record("s", True, "strong")]
    )
    assert mixed.evidence is Evidence.ESTABLISHED


def test_long_term_strong_renders_as_itself() -> None:
    """§11.1. Three strengths, not two.

    It is *"not a weaker `strong`. The rigour is the same; the **timing** is what
    differs"*, so it **folds** as strong — nothing in the spec makes timing change
    what binds — and **renders** as itself, because a pass that is "evidence once
    the loop closes" is not the same claim as evidence now.
    """
    outcome = PhaseOutcome.fold(PhaseKind.OUTPUT, ran=[record("g", True, "long_term_strong")])
    assert outcome.evidence is Evidence.ESTABLISHED  # folds as strong
    assert "long_term_strong" in outcome.render()  # renders as itself
    assert "(strong," not in outcome.render()


def test_an_empty_phase_does_not_block_the_task() -> None:
    """`demo` F-D9: a task whose phase was empty never advanced past it.

    `agent.Runner` had only `passed` to ask, and `passed` answers *what the phase
    found* — the same question as *may the task proceed?* only when the phase ran
    something. So the third state fell into the failure arm and a graph sat in
    `INPUT_VALIDATING` for 300 s on the ordinary case.

    **Empty not blocking is derived, not chosen.** `StrictLevel.NONE` switches a
    phase off, which folds to `empty`; if empty blocked, the knob would decide
    task outcomes, and criterion 20 says the level changes which phases run and
    never which verdicts bind.
    """
    empty = PhaseOutcome.fold(PhaseKind.INPUT)
    assert empty.blocks_the_task is False
    # And none of the reporting softens — the phase still establishes nothing.
    assert empty.passed is False
    assert empty.evidence is Evidence.NOTHING_RAN


def test_only_a_real_failure_blocks() -> None:
    """The other three states, so the property is a partition rather than a
    special case for `empty`."""
    failed = PhaseOutcome.fold(PhaseKind.OUTPUT, ran=[record("a", False, "weak")])
    weak = PhaseOutcome.fold(PhaseKind.OUTPUT, ran=[record("a", True, "weak")])
    strong = PhaseOutcome.fold(PhaseKind.OUTPUT, ran=[record("a", True, "strong")])

    assert failed.blocks_the_task is True
    assert weak.blocks_the_task is False  # a low-confidence pass still proceeds
    assert strong.blocks_the_task is False

    # A reused failure blocks exactly as a fresh one does — the fold is over both.
    reused = PhaseOutcome.fold(PhaseKind.OUTPUT, reused=[record("a", False, "strong")])
    assert reused.blocks_the_task is True


def test_empty_phase_is_not_a_pass() -> None:
    """§11.2, and four systems reached it independently — pytest exits 5, Bazel's
    `NO_STATUS = 0` precedes `PASSED = 1`, GitHub leaves a skipped workflow
    Pending and blocking, SARIF has `notApplicable`. **None spells it "pass".**"""
    outcome = PhaseOutcome.fold(PhaseKind.INPUT)
    assert outcome.empty is True
    assert outcome.passed is False
    assert outcome.evidence is Evidence.NOTHING_RAN
    assert outcome.evidence.value != "pass"


def test_the_third_state_is_not_a_failure_either() -> None:
    """It is its own outcome. Reporting it as a failure would be as wrong as
    reporting it as a pass, and would make a phase with no bound validators look
    like a broken artefact."""
    assert Evidence.NOTHING_RAN not in (Evidence.FAILED, Evidence.ESTABLISHED)


def test_the_fold_is_over_ran_and_reused() -> None:
    """§7.3. Never over `ran` alone — the only reading under which "one fails, all
    fail" survives a skip."""
    outcome = PhaseOutcome.fold(
        PhaseKind.OUTPUT,
        ran=[record("fresh", True, "strong")],
        reused=[record("recorded", False, "strong")],
    )
    assert len(outcome.verdicts) == 2
    assert outcome.passed is False


def test_render_names_every_skip_and_every_verdict() -> None:
    outcome = PhaseOutcome.fold(
        PhaseKind.OUTPUT,
        ran=[record("a", True, "strong")],
        reused=[record("b", True, "weak")],
    )
    text = outcome.render()
    assert "ran a: pass" in text
    assert "reused b: pass" in text
    assert text.startswith("output_validation: established")
