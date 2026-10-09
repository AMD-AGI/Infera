"""What one validation phase produced, and how a qualified pass renders.

Two properties carry the whole section, and both are shapes rather than rules
somebody has to remember.

**No field defaults to success.** JUnit XML is the counter-example to design
against: a `testcase` with no child element *is* a pass, so a producer that
forgets to emit `<skipped/>` emits a pass and nothing detects it. Four systems
reached the opposite independently and none of them spells the third state
"pass" — pytest exits 5 for no-tests-collected, Bazel's `NO_STATUS = 0` is
ordered before `PASSED = 1`, GitHub leaves a skipped workflow's checks Pending
and blocking, SARIF has `notApplicable`.

**The aggregate carries the qualification, not only the per-item line.** That is
where pytest's XPASS lost it: `xpassed` has its own count, its own progress
character and its own summary prefix — and the exit code is 0 and the bar is
green. Issue #11467, opened by a pytest core developer: *"it was missed that a
test was fixed. That's not an acceptable default."* Distinguishable rendering was
demonstrably not enough.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from validator.protocols import PhaseKind, SkipRecord, Strength, VerdictRecord

__all__ = ["Evidence", "PhaseOutcome"]

#: A pass from one of these is evidence now, or evidence once the loop closes.
#: `long_term_strong` is **not** a weaker `strong` — the rigour is the same and
#: the timing is what differs — so it folds as `strong` here and renders as
#: itself below.
_ESTABLISHING = frozenset({Strength.STRONG, Strength.LONG_TERM_STRONG})


class Evidence(str, Enum):
    """What a phase's *pass* is worth. Never consulted to decide `passed`."""

    NOTHING_RAN = "nothing_ran"
    UNCHECKED = "unchecked"  # nothing checked what this task produced — §4.15
    FAILED = "failed"
    LOW_CONFIDENCE = "low_confidence"  # passed, and no establishing pass among them
    ESTABLISHED = "established"


@dataclass(frozen=True)
class PhaseOutcome:
    """One validation phase's result.

    The field set is `validator/protocols.py`'s, and
    `test_outcome_matches_the_declared_shape` is what keeps the two in step.

    `fold` is the only constructor, and it is what makes "an empty phase is not a
    pass" structural: `empty` is computed from the folded sets rather than
    defaulted, so there is no argument list that produces a green phase from
    nothing.

    **`protocols.py` declares the field `empty: bool` while `docs/interfaces.md`
    §4.3 and design §5.2 call `PhaseOutcome.empty()` a constructor.** A dataclass
    cannot have both. The field wins, because it is the importable contract the
    `.pyi` and `tests/interfaces/test_stub_agreement.py` guard; `fold` with no
    arguments is the constructor. Reported, not resolved quietly.
    """

    # **No defaults, and that is the point rather than an oversight.**
    # `protocols.py` declares all five without one, and they had drifted to
    # `()`, `()`, `()`, `True` here — including a default on `empty`, the single
    # field whose whole job is that nothing about this object defaults. A
    # constructor that lets `empty` go unsaid is one that lets a caller build an
    # outcome without deciding what it is.
    #
    # `fold` supplies all five explicitly, so removing them costs nothing and
    # buys the property: there is no way to construct a `PhaseOutcome` without
    # stating every part of it. Found by
    # `tests/interfaces/test_declaration_conformance.py`, which compares
    # **defaults** and not only names — the default is the drift.
    kind: PhaseKind
    ran: Sequence[VerdictRecord]
    reused: Sequence[VerdictRecord]
    skipped: Sequence[SkipRecord]
    empty: bool
    #: **Were any verdicts asked of this phase?** `interfaces.md` §4.15's ruling
    #: needs two kinds of `empty` to be distinguishable, and this is the fact
    #: that separates them: `False` means nothing was asked — the level is
    #: `NONE`, or the task has no handoff in this position at all — and `True`
    #: means verdicts were expected, so an empty **output** phase is a fault.
    #:
    #: **It is set by the choice of constructor, never folded.** That is what
    #: keeps criterion 20 structural rather than argued: `fold`'s parameters are
    #: unchanged, so there is still no argument through which the knob could
    #: reach a verdict, and what the level decides is which constructor
    #: `run_phase` reaches — *which phases run*, squarely its business.
    verdicts_expected: bool

    @classmethod
    def nothing_expected(
        cls,
        kind: PhaseKind,
        *,
        skipped: Sequence[SkipRecord] = (),
    ) -> PhaseOutcome:
        """No verdict was asked of this phase. **Empty here is expected, not a fault.**

        Two call sites and one sentence, which is why they share a constructor:

        - the level is `StrictLevel.NONE`, so *no validation was asked for* —
          §4.15's first row, and the reason the fault rule cannot be read as the
          knob deciding an outcome;
        - the task has **no handoff in this position**, so there was nothing to
          check. `main` and `consume` in `examples/demo/closures/` are both this:
          `outputs: []`. §4.15's sentence is *nothing checked what this task
          **produced***, and a task that produced nothing has nothing unchecked —
          the narrow reading, taken deliberately and reported, because the wide
          one blocks the demo's **root** task and ends the run before it starts.

        `skipped` is still carried, because criterion 7 reports a skip whatever
        caused it.
        """
        return cls(
            kind=kind,
            ran=(),
            reused=(),
            skipped=tuple(skipped),
            empty=True,
            verdicts_expected=False,
        )

    @classmethod
    def fold(
        cls,
        kind: PhaseKind,
        *,
        ran: Sequence[VerdictRecord] = (),
        reused: Sequence[VerdictRecord] = (),
        skipped: Sequence[SkipRecord] = (),
    ) -> PhaseOutcome:
        """Fold over `ran | reused` — **never over `ran` alone**.

        That is the only reading under which spec §5.5's "one fails, all fail"
        survives a skip, and it is what makes criterion 20 structural rather than
        a property maintained by care. The shape is ESLint's, borrowed with its
        history: `--quiet` once erased an *error*-severity finding and flipped the
        exit code to 0 (#14202), and the fix was not an enumeration of
        interactions but a **variable split** — *"errors and warnings from the
        original unfiltered results should determine the exit code."* The
        optimisation was then reintroduced one layer down by an RFC that
        explicitly claimed verdict-neutrality, and #19625 found the missed case 22
        months later. ESLint enumerated and was still wrong.

        So here the strict level decides membership of the **run set** only, and
        the verdict folds over a set the knob cannot reach.
        """
        ran, reused, skipped = tuple(ran), tuple(reused), tuple(skipped)
        return cls(
            kind=kind,
            ran=ran,
            reused=reused,
            skipped=skipped,
            empty=not (ran or reused),
            # A fold is what a phase that **was asked** produces. The two sites
            # where nothing was asked do not fold; they call `nothing_expected`.
            verdicts_expected=True,
        )

    # ------------------------------------------------------------------ reads

    @property
    def verdicts(self) -> tuple[VerdictRecord, ...]:
        """Everything that binds: produced this run, or recorded and reused."""
        return tuple(self.ran) + tuple(self.reused)

    @property
    def passed(self) -> bool:
        """An empty phase is not a pass, and a failure binds at every strength."""
        return not self.empty and all(r.verdict.result for r in self.verdicts)

    @property
    def unchecked(self) -> bool:
        """**Nothing checked what this task produced.** `interfaces.md` §4.15.

        Ruled by the user, and it overturns what this class used to say — the
        docstring below claimed the answer *cannot* be "it blocks", because
        `NONE` folds every phase to `empty` and criterion 20 forbids the level
        deciding outcomes. What that missed is that the two `empty`s are
        different claims, and `verdicts_expected` is what makes them
        distinguishable:

        | | `empty` means |
        |---|---|
        | nothing was asked | **no validation was asked for.** Expected |
        | verdicts expected | **nothing checked this output.** A fault |

        **Criterion 20 survives, checked rather than assumed.** Its wording is
        *"the level changes which phases run, and never which verdicts bind"*
        (spec §11, criterion 20). `NONE` not running a phase is *which phases
        run*; the fault rule is **identical at every other level**, so the level
        changes no verdict — and there is no verdict here to change, because the
        fault is precisely the case where none exists.

        **The kind asymmetry is the whole point and is not a tidy-up.** An empty
        **input** phase means *this task consumes nothing.* An empty **output**
        phase means *nothing checked what this task produced.* Different claims,
        and only the second is a candidate fault.
        """
        return self.kind is PhaseKind.OUTPUT and self.verdicts_expected and self.empty

    @property
    def blocks_the_task(self) -> bool:
        """**May the task proceed?** — the runner's question, answered here.

        `passed` answers *what the phase found*, and those are the same question
        only when the phase ran something. `agent.Runner` had only `passed` to
        ask, so the third state fell into the failure arm and a task whose phase
        was empty **never advanced past it** — `demo` measured a graph sitting in
        `INPUT_VALIDATING` for 300 s on the ordinary case, a non-leaf with no
        validators bound. `engineer_principle.md` §4.4: when a caller seems to
        need your properties, work out what it intends to compute and offer that
        computation instead.

        **Two arms, and they are different claims about the same phase.** A real
        failure blocks; and, since §5.16 was ruled in §4.15, so does `unchecked`.
        An empty phase that nobody asked anything of still does **not** block —
        `demo` criterion 3, *"empty is the normal case and must be shown to be
        normal, not degenerate"*, and F-D9's 300 s deadlock is what that costs
        when it is got wrong.

        None of this softens "an empty phase is not a pass". `passed` still
        returns `False` for every empty phase, whether or not it was expected to
        produce something; what changed is only that one of the two empties is
        now also a fault. Two facts, two questions, and conflating them is what
        produced the deadlock in the first place.
        """
        return self.unchecked or (not self.empty and not self.passed)

    @property
    def failures(self) -> tuple[VerdictRecord, ...]:
        return tuple(r for r in self.verdicts if not r.verdict.result)

    @property
    def pass_strengths(self) -> tuple[Strength, ...]:
        return tuple(Strength(r.verdict.strength) for r in self.verdicts if r.verdict.result)

    @property
    def evidence(self) -> Evidence:
        """The aggregate qualification. Criterion 19.

        Machine-readable, because that is where pytest lost the distinction: its
        JUnit XML makes a non-strict xfail *"appear as a passing test"*, and the
        qualification is erased in exactly the artefact a dashboard reads.
        """
        # Ordered before `NOTHING_RAN`, because a phase that blocked must not
        # report the word for the case that does not. `agent._evidence` puts this
        # value straight onto the `VALIDATION_FAILED` record, so "nothing_ran" as
        # the stated reason a task was blocked would be the misleading half of a
        # true sentence — codeql-action#3156's shape, one field along.
        if self.unchecked:
            return Evidence.UNCHECKED
        if self.empty:
            return Evidence.NOTHING_RAN
        if not self.passed:
            return Evidence.FAILED
        if any(s in _ESTABLISHING for s in self.pass_strengths):
            return Evidence.ESTABLISHED
        return Evidence.LOW_CONFIDENCE

    def render(self) -> str:
        """A human line per item, then the aggregate.

        Every skip is reported unconditionally rather than behind a verbosity
        flag: pytest's `-r` defaults to `fE`, so a skip's *reason* needs `-rs` to
        appear at all, and our skip counts are small enough that there is nothing
        to save. The incident that makes it worth a sentence is
        codeql-action#3156 — `upload-sarif` *"silently stopped uploading"*, the
        job stayed green, and the failure was visible only as the absence of
        alerts.
        """
        lines = [f"{self.kind.value}: {self.evidence.value}"]
        for record in self.ran:
            lines.append(_verdict_line(record, "ran"))
        for record in self.reused:
            lines.append(_verdict_line(record, "reused"))
        for skip in self.skipped:
            reused = "" if skip.reused is None else f", reusing {skip.reused.verdict.validator}"
            lines.append(f"  skipped {skip.validator}: {skip.reason}{reused}")
        return "\n".join(lines)


def _verdict_line(record: VerdictRecord, how: str) -> str:
    verdict = record.verdict
    result = "pass" if verdict.result else "FAIL"
    # The strength is printed beside every result, and `long_term_strong` prints
    # as itself: a pass that is "evidence once the loop closes" is not the same
    # claim as evidence now, so folding it into `strong` here would lose the one
    # thing the label records.
    return f"  {how} {verdict.validator}: {result} ({verdict.strength}, {verdict.dimension})"
