"""`report` — `check-jsonschema`'s format, over a flat `Problem` list.

`docs/design.md` §3.5 adopts the format whole rather than inventing one, because
the project shipped a **second** relevance heuristic after finding stock
`best_match` insufficient. Two guesses plus an escape hatch is the state of the
art; a third guess would not be an improvement.
"""

from __future__ import annotations

from spec_loader import Problem, report

FIRST = Problem(origin="a.jsonnet", path="$", keyword="required", message="'name' is required")
DEEP = Problem(origin="a.jsonnet", path="$.items.env", keyword="type", message="not of type object")
THIRD = Problem(origin="a.jsonnet", path="$.scope", keyword="enum", message="not one of [...]")
OTHER = Problem(origin="b.jsonnet", path="$", keyword="parse", message="could not parse")


def test_no_problems_is_the_empty_string() -> None:
    """So a caller can print it unconditionally without printing a blank line."""
    assert report([]) == ""


def test_one_problem_is_one_line() -> None:
    assert report([FIRST]) == "a.jsonnet::$: 'name' is required"


def test_the_second_problem_is_labelled_the_deep_match() -> None:
    """`validate` puts them in this order, and this renders that order.

    The headline already *is* the best match, so repeating it under a "Best
    Match:" label — which is what `check-jsonschema` prints — would print one
    error twice here. That is the one adaptation, and `README.md` records it.
    """
    assert report([FIRST, DEEP]).splitlines() == [
        "a.jsonnet::$: 'name' is required",
        "  Best Deep Match: $.items.env: not of type object",
    ]


def test_the_tail_is_counted_not_printed() -> None:
    lines = report([FIRST, DEEP, THIRD]).splitlines()

    assert lines[-1] == "  1 other errors were produced. Pass verbose=True to see all errors."


def test_verbose_prints_every_problem() -> None:
    """The escape hatch is a keyword argument rather than a CLI flag, because
    this package has no CLI. `check-jsonschema` says "Use --verbose"."""
    lines = report([FIRST, DEEP, THIRD], verbose=True).splitlines()

    assert lines == [
        "a.jsonnet::$: 'name' is required",
        "  $.items.env: not of type object",
        "  $.scope: not one of [...]",
    ]


def test_problems_are_grouped_by_origin_in_arrival_order() -> None:
    """A whole-catalogue pass hands over problems from many files at once
    (`docs/interfaces.md` §2 step 5), and a reader fixes them one file at a
    time. Arrival order, not sorted: `validate` already ordered within a file
    and the composition root already ordered the files."""
    lines = report([FIRST, OTHER, DEEP]).splitlines()

    assert lines == [
        "a.jsonnet::$: 'name' is required",
        "  Best Deep Match: $.items.env: not of type object",
        "b.jsonnet::$: could not parse",
    ]


def test_the_origin_is_printed_and_never_opened() -> None:
    """`origin` is an opaque label — `docs/design.md` §3.1.

    cdk8s solves the same problem the same way, joining provenance to violations
    after the plugin returns and degrading to `'N/A'` when it is absent. A label
    that is not a path must still render.
    """
    assert report([Problem(origin="<stdin>", path="$", keyword="type", message="no")]).startswith(
        "<stdin>::"
    )


# --------------------------------------------------------------------------- #
# The composition root's derivations — `docs/interfaces.md` §2 step 5.


def test_format_problems_is_report_under_the_other_normative_name() -> None:
    """§2 writes `format_problems`, §4.1 lists `report`, and both are normative.

    One is expressed in terms of the other rather than reimplemented — main spec
    §3.1 principle 10. If these two ever disagree, someone gave the alias a body.
    """
    from spec_loader.report import format_problems

    problems = [FIRST, DEEP, THIRD]
    assert format_problems(problems) == report(problems)
    assert format_problems(problems, verbose=True) == report(problems, verbose=True)


def test_failed_names_is_the_origins_that_failed() -> None:
    from spec_loader import LoadReport
    from spec_loader.report import failed_names

    good = LoadReport(admitted=("trace",), problems=())
    bad = LoadReport(admitted=(), problems=(FIRST, DEEP, OTHER))

    assert failed_names([good, bad]) == frozenset({"a.jsonnet", "b.jsonnet"})
    assert failed_names([good]) == frozenset()
    assert failed_names([]) == frozenset()


def test_a_non_fatal_problem_is_not_a_failure() -> None:
    """`Problem.fatal` is `False` for exactly one thing today, and it is live.

    `closure/check.py`'s check 3 is its only producer in the system: a closure
    assembled from a handoff kind admitted under the escape-hatch flag.

    That kind **is** admitted, so skipping the closures that use it would
    suppress `closure` criterion 6 — which requires such a closure to load *and
    report that it did*. A gate that swallowed the report would turn the escape
    hatch silent, which is the one property §5.3 forbids it.
    """
    from spec_loader import LoadReport, Problem
    from spec_loader.report import failed_names, rejected

    hatch = Problem(
        origin="handoffs/trace.jsonnet",
        path="$.validators",
        keyword="escape_hatch",
        message="admitted with no validator",
        fatal=False,
    )

    assert failed_names([LoadReport(admitted=("trace",), problems=(hatch,))]) == frozenset()
    assert rejected([hatch]) == frozenset()
    assert rejected([hatch, FIRST]) == frozenset({"a.jsonnet"})


def test_rejected_returns_origins_which_is_the_reported_gap() -> None:
    """Pinning the mismatch rather than papering over it.

    `closure.check_closures` filters with `if name in skip`, where `name` is a
    closure name; a `Problem` carries `origin`, which is a file path. So the
    gate does not close today, and neither side is wrong on its own. This test
    documents which of the two this function returns, so that whoever bridges it
    changes a test that says why rather than discovering the shape by running it.
    """
    from spec_loader.report import rejected

    assert rejected([FIRST, DEEP, OTHER]) == frozenset({"a.jsonnet", "b.jsonnet"})
    assert all("/" in o or o.endswith(".jsonnet") for o in rejected([FIRST]))
