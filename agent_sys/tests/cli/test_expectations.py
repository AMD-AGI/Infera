"""`cli/expectations.py` — the table, and what an **empty** set must exit.

The accounting these tests drive has been wrong twice, in the same way both
times: it reported *a safety property stopped holding* about a promise the run
had merely never exercised. `cli/main.py`'s docstrings record both incidents.
Making the set per-package adds a third route to the same mistake — a package
with **no** promises — and the whole of this file is about that route.

**Nothing was promised** and **everything promised held** are both green, and
they are not the same fact. The exit code cannot tell them apart, so the report
must.
"""

from __future__ import annotations

import pathlib
from typing import Any

from cli import expectations
from cli import main as cli_main
from cli.events import EventKind
from cli.stream import Stream


def _layout(tmp_path: pathlib.Path) -> Any:
    from cli.environment import layout_for

    return layout_for(tmp_path).create()


def _fields(stream: Stream, kind: EventKind) -> list[dict[str, Any]]:
    return [event.fields for event in stream.events if event.kind is kind]


# --------------------------------------------------------------------------- #
# The table


def test_the_default_is_empty() -> None:
    """A package the CLI has never heard of promises nothing, and gets there
    without an entry. That default is what makes the table a leak rather than a
    coupling: a new package needs no edit here."""
    assert expectations.for_package(pathlib.Path("/tmp/whatever")) is expectations.EMPTY
    assert expectations.EMPTY.promises == {}
    assert expectations.EMPTY.dropped == {}


def test_demo_keeps_exactly_its_two_promises() -> None:
    """The move out of `cli/main.py` changed nothing about what `examples/demo`
    promises. Named here because a silently shortened list is the failure this
    whole file exists to prevent."""
    demo = expectations.for_package(pathlib.Path("agent_sys/examples/demo"))
    assert demo is expectations.DEMO
    assert set(demo.promises) == {"grounded_verdict_fails", "consumer_waits"}
    assert demo.dropped == {}


def test_demo2_promises_nothing() -> None:
    """**Not an omission.** `examples/demo2` is a package whose run is meant to
    complete with every task succeeded, so *nothing will fail* is its claim and
    an empty set states it exactly."""
    assert expectations.for_package(pathlib.Path("agent_sys/examples/demo2")) is expectations.EMPTY


def test_the_key_is_the_directory_name_not_the_path() -> None:
    """So a checkout, a copy under `tmp_path` and a `--package` pointing at
    either answer the same."""
    assert expectations.for_package(pathlib.Path("/tmp/x/demo")) is expectations.DEMO


def test_the_empty_set_observes_nothing() -> None:
    """Both callables return `None` for anything, which is what keeps
    `_report`'s `expected=` field honest for a package with no promises."""
    assert expectations.EMPTY.observed_by_task(object()) is None
    assert expectations.EMPTY.observed_by_verdict(object(), object()) is None


# --------------------------------------------------------------------------- #
# The empty set through `_strict` — the part that had to be got right


def test_an_empty_expectation_set_exits_ok(tmp_path: pathlib.Path) -> None:
    """**Exit `OK`, and specifically not `UNEXPECTED_SUCCESS`.**

    With no promises there is nothing to miss and nothing to leave unreached, so
    a graph that completed and a validator set that recorded only passes is a
    green run. The failure mode this guards is the arithmetic reading *zero
    expected failures were observed* as *an expected failure did not happen* —
    which is exit 3, and which would make every package but `examples/demo`
    fail for promising nothing.
    """
    stream = Stream()
    code = cli_main._strict(
        stream, set(), unreachable=set(), layout=_layout(tmp_path), promises=expectations.EMPTY
    )
    assert code == cli_main.OK == 0
    assert stream.count(EventKind.UNEXPECTED_SUCCESS) == 0
    assert stream.count(EventKind.EXPECTATION_UNREACHED) == 0
    assert stream.count(EventKind.EXPECTED_FAILURE) == 0
    done = _fields(stream, EventKind.RUN_COMPLETE)[-1]
    assert done["ok"] is True and done["unobserved"] == [] and done["unreached"] == []


def test_an_empty_set_says_it_tested_nothing_rather_than_that_all_held(
    tmp_path: pathlib.Path,
) -> None:
    """**The distinction the exit code cannot carry.**

    `ok: true` is the same byte for *no promise was made* and *every promise was
    kept*, and the second is a much larger claim. A reader given only
    *"0 of 0 expected failures observed"* supplies the larger one themselves.
    So the human line says which it is, and `expectations` carries the count for
    a machine reader — this is the same three-term discipline
    `EXPECTATION_UNREACHED` exists for, applied to the case where the set itself
    is empty.
    """
    stream = Stream()
    cli_main._strict(stream, set(), set(), _layout(tmp_path), expectations.EMPTY)
    done = _fields(stream, EventKind.RUN_COMPLETE)[-1]
    assert done["expectations"] == 0
    message = [e.message for e in stream.events if e.kind is EventKind.RUN_COMPLETE][-1]
    assert "promises no failure" in message
    assert "expected failures observed" not in message


def test_a_non_empty_set_still_counts_the_way_it_did(tmp_path: pathlib.Path) -> None:
    """The other side of the branch above: `examples/demo`'s wording is
    unchanged, because a difference in its output is a regression rather than an
    improvement."""
    stream = Stream()
    cli_main._strict(
        stream,
        set(expectations.DEMO.promises),
        set(),
        _layout(tmp_path),
        expectations.DEMO,
    )
    message = [e.message for e in stream.events if e.kind is EventKind.RUN_COMPLETE][-1]
    assert "2 of 2 expected failures observed, 0 never reached" in message
    assert _fields(stream, EventKind.RUN_COMPLETE)[-1]["expectations"] == 2


def test_an_empty_set_drops_nothing_and_says_so(tmp_path: pathlib.Path) -> None:
    """`report_dropped` emits nothing, and the summary still carries the count —
    which is the arrangement `main.report_dropped` argues for: silence on the
    human side is not an absence, because the last line states `0`."""
    stream = Stream()
    cli_main.report_dropped(stream, expectations.EMPTY)
    assert stream.count(EventKind.VALIDATION_DROPPED) == 0
    cli_main._strict(stream, set(), set(), _layout(tmp_path), expectations.EMPTY)
    assert (
        "0 validation(s) dropped"
        in [e.message for e in stream.events if e.kind is EventKind.RUN_COMPLETE][-1]
    )


def test_an_empty_set_cannot_produce_an_unreached(tmp_path: pathlib.Path) -> None:
    """A promise that does not exist cannot be untested, so exit 4 is out of
    reach too. Stated because `EXPECTATION_UNREACHED` is the *other* thing a
    zero could be mistaken for, and the two mistakes have opposite exit codes.
    """
    stream = Stream()
    code = cli_main._strict(stream, set(), set(), _layout(tmp_path), expectations.EMPTY)
    assert code != cli_main.UNEXPECTED_FAILURE
    assert code != cli_main.UNEXPECTED_SUCCESS


# --------------------------------------------------------------------------- #
# The empty set does not also mean "the run succeeded"


def test_an_empty_set_whose_graph_did_not_finish_does_not_exit_zero(
    tmp_path: pathlib.Path,
) -> None:
    """**The test that stops this class of green run.** `interfaces.md` §4.17.

    Measured by `main` on a bring-up package: one task ended in
    `output_validating`, another in `running`, the only handoff was `invalid`,
    and the run exited **0** — because a package with no expectations misses no
    promise and leaves none unreached, so every row of the strict table fell
    through to green.

    *There is no promised failure to test* and *the run succeeded* are two
    facts, and this accounting had one answer for both. That is the shape of
    both `was_judged` incidents, one level up: at the level of the set rather
    than of a member.
    """
    stream = Stream()
    code = cli_main._strict(
        stream,
        set(),
        set(),
        _layout(tmp_path),
        expectations.EMPTY,
        ["directions: output_validating", "handoff directions: invalid", "main: running"],
    )
    assert code == cli_main.INCOMPLETE == 5
    assert code != cli_main.OK
    done = _fields(stream, EventKind.RUN_COMPLETE)[-1]
    assert done["ok"] is False and done["exit_code"] == 5
    assert done["unfinished"] == [
        "directions: output_validating",
        "handoff directions: invalid",
        "main: running",
    ]


def test_the_incomplete_line_names_what_was_left_unfinished(tmp_path: pathlib.Path) -> None:
    """A non-zero exit that does not say what is unfinished sends a reviewer to
    read the whole transcript, which is the cost this stream exists to avoid."""
    stream = Stream()
    cli_main._strict(stream, set(), set(), _layout(tmp_path), expectations.EMPTY, ["main: running"])
    message = [e.message for e in stream.events if e.kind is EventKind.RUN_COMPLETE][-1]
    assert "did NOT finish" in message and "main: running" in message


def test_incomplete_is_neither_of_the_two_promise_codes() -> None:
    """**Its own number, and the reason is what the other two words mean.**

    3 is *a promise stopped being kept*, which is a claim about the system under
    test and is the cry-wolf this file has twice been wrong about. 4 is *the
    promises are untested because something else broke*, which needs promises to
    be untested. A package that promised nothing and a graph that did not finish
    is neither.
    """
    assert cli_main.INCOMPLETE not in {
        cli_main.OK,
        cli_main.LOAD_ERROR,
        cli_main.PRECONDITION,
        cli_main.UNEXPECTED_SUCCESS,
        cli_main.UNEXPECTED_FAILURE,
    }


def test_a_package_with_promises_is_still_judged_by_its_promises(tmp_path: pathlib.Path) -> None:
    """**The gap, asserted so it is a decision rather than a surprise.**

    `examples/demo`'s specified ending is `consume` left in `WAITING_HANDOFF`
    and a handoff never made valid — every one of which `_completion_gaps`
    names. So the completion rule is applied to the empty set only, and a
    package with promises that hands over the same gaps still exits `OK`.
    Recorded in `_strict`'s docstring with what would close it.
    """
    stream = Stream()
    code = cli_main._strict(
        stream,
        set(expectations.DEMO.promises),
        set(),
        _layout(tmp_path),
        expectations.DEMO,
        ["consume: waiting_handoff", "handoff summary: invalid"],
    )
    assert code == cli_main.OK
    assert _fields(stream, EventKind.RUN_COMPLETE)[-1]["unfinished"] == []


def test_completion_gaps_reads_the_managers(registry: Any, graph: Any) -> None:
    """`_completion_gaps` over a real, built, undispatched graph: nothing has
    run, so every task is a gap and the evidence is absent rather than negative.

    The point is that it reads `task_mgr` and `handoff_mgr` rather than the
    events already emitted — a report that audits its own output can only agree
    with itself.
    """
    gaps = cli_main._completion_gaps(registry)
    assert gaps, "an undispatched graph has not completed"
    assert all(": " in gap for gap in gaps), "each gap names a subject and its state"
    assert not any(gap.endswith("succeeded") for gap in gaps)
