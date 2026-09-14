"""What a task package promises will go wrong, keyed by package directory name.

**This file is a compromise and says so.** `cli/` is the program's single entry
point over *any* task package, and everything else in it — `build.py`,
`environment.py`, `package.py`, `render/` — names no closure, no handoff kind
and no validator. This table names three of `examples/demo`'s: the closure
`consume`, the validator `check_grounded` and the kind `summary`. **A package
name appearing in the CLI is a leak**, and enumerating it here does not stop
being one just because it is now in a file of its own; what it buys is that the
leak is in one place, is declared, and has a default that new packages get for
free.

**The alternative, and why it is not taken here.** The honest home for "this
package promises `check_grounded` will fail" is the package: a `promises:` block
in `main.yaml`, loaded by `spec_loader` and read by whoever does the accounting.
That is a new schema, a new loader field and a change on both sides of a module
seam — `interfaces.md` §1.1 — for an entry point that today serves two packages,
one of which promises nothing. Out of scope, and recorded rather than done.

A future fix looks like: `main.yaml` declares `promises: [{name, description,
observed_when, judged_when}]` where the two conditions are expressed in terms
the loader already has (a closure name and a task status; a validator name, a
handoff kind and a result). `for_package` then becomes a read of the loaded
registry instead of a dict lookup, this module keeps only `EMPTY` and the two
NamedTuples, and the `demo` entry below moves into `examples/demo/main.yaml`.

**The default is `EMPTY`, and an empty set is a statement rather than a gap.**
It says *this package promises nothing will fail* — which for `examples/demo2`
is the whole claim its run makes. It does **not** say every promise was kept:
see `main._strict`, which reports the two differently, because "no promise was
tested" and "every promise held" are the same green and different facts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

from task_graph import TaskStatus

__all__ = ["EMPTY", "Expectation", "ExpectationSet", "for_package"]


class Expectation(NamedTuple):
    """A promised failure, and how to tell whether the run ever got to test it.

    **`was_judged` exists because "did not happen" and "never got the chance"
    are different facts**, and this accounting had one answer for both. Found by
    `main` in a run where `produce` failed before `describe` could produce a
    summary: `UNEXPECTED_SUCCESS` was true about what was observed and false
    about what it meant — it claimed a safety property had stopped holding when
    the property had simply never been exercised.

    That is F-D9's shape — three outcomes, two branches — in this package's own
    code, one day after reporting it to `agent` and `validator`. Worth the
    embarrassment of the comment: the pattern is not other people's.

    **And the field was added and then implemented wrongly, which is the second
    half of the same lesson.** `reachable` was answered by *does the subject
    exist* rather than *did the judgement happen*, so a `check_grounded` that
    crashed with a `KeyError` was reported as a property that stopped holding.
    **The three-term vocabulary was right and the predicate behind it was not** —
    a distinction is only as good as the question that routes into it.
    """

    description: str
    #: **`was_judged`, and the name is the fix.** It was `reachable`, and that
    #: word invited the reading that sank it: *is there something here to
    #: judge?* rather than *did the judgement happen?* A predicate answering the
    #: first will call a promise tested whenever its subject exists, which is
    #: how a crashed validator came to be reported as a property that stopped
    #: holding. `validator` named the rename as the part that stops it
    #: recurring, and they are right — the wrong implementation now looks wrong.
    was_judged: Callable[[Any], bool]


class ExpectationSet(NamedTuple):
    """One package's promises, its deliberate drops, and how a run observes them.

    The two observation callables return **the expectation name** a fact
    satisfies, or `None`. One place each, because the `expected` field on the
    emitted event and the `observed` accounting that decides the exit code must
    agree **by construction** rather than by both being edited: they were two
    copies of the same three-term condition, and a change to one would have been
    a stream that contradicted its own exit code.
    """

    #: name -> the promise. Empty means the package promises nothing will fail.
    promises: Mapping[str, Expectation]
    #: name -> why it was deliberately not run. See `main.report_dropped`.
    dropped: Mapping[str, str]
    #: A task in its final state -> the promise it observes, if any.
    observed_by_task: Callable[[Any], str | None]
    #: A recorded verdict and its handoff -> the promise it observes, if any.
    observed_by_verdict: Callable[[Any, Any], str | None]


def _nothing_from_task(task: Any) -> str | None:
    return None


def _nothing_from_verdict(verdict: Any, handoff: Any) -> str | None:
    return None


#: The default, and what a new package gets without editing this file.
EMPTY = ExpectationSet({}, {}, _nothing_from_task, _nothing_from_verdict)


# --------------------------------------------------------------------------- #
# examples/demo — the first reference package


def _grounded_verdict_exists(registry: Any) -> bool:
    """Did `check_grounded` actually record a verdict on a **summary**?

    **Not "does a summary exist".** That was the old predicate and it was wrong
    in a way that only got worse: since `interfaces.md` §4.14 a store version
    becomes visible the moment `_seal_outputs` publishes it — `agent/runner.py`,
    **before** the gate and well before `OUTPUT_VALIDATING`. So it flipped true
    at a point where `check_grounded` provably had not run. `validator` traced
    that; it would have been defensible when `put` happened at close, and the
    seal moving is what made it false.

    **Measured consequence, on a real run.** `check_grounded` crashed with a
    `KeyError` and recorded nothing; a summary existed; the run reported
    *"did NOT happen … a property stopped holding"* and exited 3. **That is
    `UNTESTED` wearing `UNEXPECTED`'s label**, produced by the artefact whose
    whole purpose is telling those two apart, on the loudest line it prints.

    **The property this buys:** `UNEXPECTED_SUCCESS` now requires **positive
    evidence of a pass**. A claim that a safety property stopped holding must
    rest on a recorded verdict saying so, never on an absence — and every route
    to an absence (a crashed body, a validator never selected, a phase never
    reached, `StrictLevel.NONE`) lands in `EXPECTATION_UNREACHED`, which is
    where they belong.

    **Keyed to the kind as well as the name**, on `validator`'s advice. It costs
    nothing today, because `check_grounded`'s spec declares `inputs:
    ['summary']` and it can only be recorded against one. It forecloses the
    case where it is not.
    """
    store, handoff_mgr = registry.get("handoff_store"), registry.get("handoff_mgr")
    return any(
        verdict.validator == "check_grounded"
        for hid in handoff_mgr.all_ids()
        if handoff_mgr.get(hid).type == "summary"
        for version in store.list_versions(hid)
        for verdict in store.read_verdicts(hid, version)
    )


def _consumer_exists(registry: Any) -> bool:
    """`consume` is only expected to wait if the graph ever grew it.

    **Weaker than `_grounded_verdict_exists`, and knowingly so.** This asks
    whether the subject exists, which is the reading that sank the other
    predicate — and the same criticism applies: `consume` sitting in
    `WAITING_HANDOFF` because `describe` never produced anything is reported as
    the promise *kept*, when the description's own words are *"because of it"*
    and the causal half is unverified. Observed on several runs today.

    **Not fixed here, and the reason is scope rather than doubt.** The honest
    predicate is *did `consume` get a valid `summary` to consume*, which is a
    claim about the other promise; wiring one expectation's judgement to the
    other's is a change to what the pair means, not a bug fix. Reported.
    """
    return any(task.closure == "consume" for task in registry.get("task_mgr").all())


def _demo_task(task: Any) -> str | None:
    """`consume` ending the run in `WAITING_HANDOFF` is the promise, observed."""
    if task.status is TaskStatus.WAITING_HANDOFF and task.closure == "consume":
        return "consumer_waits"
    return None


def _demo_verdict(verdict: Any, handoff: Any) -> str | None:
    """Is this the verdict the demo promises will be recorded, and be `False`?

    Keyed to the **kind** as well as the validator name, matching
    `_grounded_verdict_exists`.
    """
    if verdict.validator == "check_grounded" and handoff.type == "summary" and not verdict.result:
        return "grounded_verdict_fails"
    return None


#: What `examples/demo` promises will go wrong, and therefore what it FAILS for
#: if it does not. `demo` design §7.5, and pytest's vocabulary adopted directly:
#: these are `xfail(strict=True)`, so an expected failure that passes is a
#: failure.
#:
#: A demo that prints "all good" because the sandbox stopped blocking, or
#: because the validator stopped failing, is the single worst outcome available
#: to this artefact — it would assert, in the most visible place in the
#: repository, that a safety property holds when it does not.
_DEMO_PROMISES = {
    "grounded_verdict_fails": Expectation(
        "check_grounded records a failing verdict on the summary",
        _grounded_verdict_exists,
    ),
    "consumer_waits": Expectation(
        "consume ends the run still in WAITING_HANDOFF, because of it",
        _consumer_exists,
    ),
}

#: Validations `examples/demo` deliberately did **not** perform, and why.
#:
#: `docs/interfaces.md` §4.17 — *a green run that is itself the corruption*. The
#: vocabulary already separated three outcomes: observed, **did not happen**
#: (`UNEXPECTED_SUCCESS`), and **never reached** (`EXPECTATION_UNREACHED`). It
#: had no way to say **deliberately not run**, and a dropped check that simply
#: vanishes from the output is indistinguishable from one that was never
#: declared. Every entry here is emitted as a `VALIDATION_DROPPED` event and
#: counted in `RUN_COMPLETE`, so the run states the size of its own claim.
#:
#: **It is empty, and that is a measurement rather than a placeholder.** The
#: instruction that produced this machinery was *"remove the validations that
#: cannot pass, so we get an e2e result"*. Driven end to end, the un-passable
#: set is empty and `validator` reached the same answer independently:
#:
#: - `check_facts` **passes** — a real `PASS / completeness / strong` is on disk
#:   from it, written by the script body in a validation zone.
#: - `check_grounded` has never run, which is `UNTESTED`. A promise the run did
#:   not reach is not a promise that cannot be kept.
#:
#: And the decisive half: **what blocks the run is a body, not a verdict.**
#: Dropping every validation in the package would not move it one step further,
#: because the failures are a validator body that cannot find the store and an
#: AI backend that refuses to start unconfined. So there is nothing here to
#: remove, and an empty drop list reported as empty is the honest output.
#:
#: An entry is `name -> why`, and the *why* is not optional: a dropped check
#: whose reason is not in the run's own output is a decision nobody can review.
_DEMO_DROPPED: dict[str, str] = {}

DEMO = ExpectationSet(_DEMO_PROMISES, _DEMO_DROPPED, _demo_task, _demo_verdict)


#: Package directory name -> what that package promises. **Absent means `EMPTY`**,
#: so `examples/demo2` needs no entry: it promises that nothing will fail, which
#: is a complete statement and the one its run makes.
_BY_PACKAGE: dict[str, ExpectationSet] = {"demo": DEMO}


def for_package(root: Path) -> ExpectationSet:
    """This package's expectation set, or `EMPTY`.

    Keyed on the directory name rather than the path, so a checkout, a copy
    under `tmp_path` and a `--package` pointing at either answer the same.
    """
    return _BY_PACKAGE.get(Path(root).name, EMPTY)
