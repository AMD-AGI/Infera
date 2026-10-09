"""Problems to a line a human can act on, and the derivations over them.

Separate from `validate.py` on purpose (`docs/design.md` §3.5): a
machine-readable emitter is then a second function over the same `Problem` list,
rather than a flag threaded through the validator.

The format is `check-jsonschema`'s, adopted rather than invented — see
`_render_origin` for the one place it is adapted and why.

`format_problems`, `failed_names` and `rejected` are the composition root's
(`docs/interfaces.md` §2 step 5). They live here rather than in `bootstrap`
because they are operations over `Problem` and `LoadReport`, and the module that
owns a type owns the operations over it — `engineer_principle.md` §3. §2's fourth
function, `merged`, is **not** here; `merged` is documented below.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .protocols import LoadReport, Problem

__all__ = ["failed_names", "format_problems", "rejected", "report"]


def report(problems: Sequence[Problem], *, verbose: bool = False) -> str:
    """One block per origin, best match first.

    Problems arrive in the order `validate` produced them — best match, then the
    deep match when it differs, then the rest — and that order is what this
    renders. Grouping is by `origin` because a whole-catalogue pass hands over
    problems from many files at once (`docs/interfaces.md` §2 step 5).

    `verbose` prints every problem instead of counting the tail. It is a
    parameter rather than a global because the composition root formats once and
    a test formats often.
    """
    if not problems:
        return ""

    order: list[str] = []
    grouped: dict[str, list[Problem]] = {}
    for problem in problems:
        if problem.origin not in grouped:
            grouped[problem.origin] = []
            order.append(problem.origin)
        grouped[problem.origin].append(problem)

    return "\n".join(_render_origin(origin, grouped[origin], verbose) for origin in order)


def _render_origin(origin: str, problems: Sequence[Problem], verbose: bool) -> str:
    """`check-jsonschema`'s four parts, over a flat `Problem` list.

    Its own shape is::

        <origin>::<path>: <message>
        Best Match: <path>: <message>
        Best Deep Match: <path>: <message>          (only when it differs)
        N other errors were produced. Use --verbose to see all errors.

    Two adaptations, both recorded in `README.md`. The headline already *is* the
    best match here — `validate` sorted it there — so repeating it under a "Best
    Match:" label would print one error twice. And the escape hatch is a
    keyword argument rather than a CLI flag, because this package has no CLI.
    """
    head, rest = problems[0], problems[1:]
    lines = [f"{_where(origin, head)}::{head.path}: {head.message}"]

    if verbose:
        lines += [f"  {p.path}: {p.message}" for p in rest]
        return "\n".join(lines)

    if rest:
        lines.append(f"  Best Deep Match: {rest[0].path}: {rest[0].message}")
    if len(rest) > 1:
        lines.append(
            f"  {len(rest) - 1} other errors were produced. Pass verbose=True to see all errors."
        )
    return "\n".join(lines)


def _where(origin: str, problem: Problem) -> str:
    """`<origin>`, or `<origin>:<line>` / `<origin>:<line>:<column>`.

    The same three shapes the deleted `RenderError._format` produced, for the
    same reason: a position is appended only when the parser reported one, so a
    reader can tell "line 12" from "somewhere in this file" rather than being
    told a guess. Grouping is still by `origin` alone — the position varies
    per problem within one file, and a group per line would scatter the report
    exactly where it is meant to gather.
    """
    if problem.line is None:
        return origin
    if problem.column is None:
        return f"{origin}:{problem.line}"
    return f"{origin}:{problem.line}:{problem.column}"


# --------------------------------------------------------------------------- #
# The composition root's derivations. `docs/interfaces.md` §2 step 5:
#
#     failed    = failed_names(reports)
#     problems  = list(chain.from_iterable(rep.problems for rep in reports))
#     problems += check_closures(views, merged(reports), skip=failed)
#     problems += check_graph(views.task_specs, skip=failed | rejected(problems))
#     if any(p.fatal for p in problems):
#         raise SpecInvalid(format_problems(problems))


def format_problems(problems: Sequence[Problem], *, verbose: bool = False) -> str:
    """`docs/interfaces.md` §2's name for §4.1's `report`.

    One line, delegating, because the two are one operation under two names and
    main spec §3.1 principle 10 is explicit about which shape that takes: *"where
    two operations mean the same thing, one is expressed in terms of the other"*.

    Both names exist because both documents are normative and they disagree — §2
    writes `format_problems(problems)` into the composition root, §4.1 lists the
    export as `report`. Collapsing them to one name is the better end state and
    is a `docs/interfaces.md` edit, not this package's call.
    """
    return report(problems, verbose=verbose)


def failed_names(reports: Iterable[LoadReport]) -> frozenset[str]:
    """Every spec that did not get admitted, by **origin**.

    The layering gate's input (`docs/design.md` §6.2): a spec that already failed
    its own checks is not checked again, because *"your task's handoff kind does
    not resolve"* on top of *"your schema is broken"* is noise. Kubernetes CRD
    validation does exactly this and gives the reason — CEL validation errors
    that are not actionable.

    **Non-fatal problems do not count as failure.** `Problem.fatal` is `False`
    for a report-severity finding, of which there is exactly one today, and it is
    live rather than hypothetical: `closure/check.py`'s check 3 emits it for a
    closure assembled from a handoff kind admitted under the escape-hatch flag.
    That kind *is* admitted, and gating on it would skip the very closure whose
    reporting `closure` criterion 6 requires — *"loads, **and** reports that it
    did"*.

    **These are origins, not spec names**, and that is a gap rather than a
    choice — see `spec_loader/README.md` and the note on `rejected`.
    """
    return frozenset(
        problem.origin for load in reports for problem in load.problems if problem.fatal
    )


def rejected(problems: Iterable[Problem]) -> frozenset[str]:
    """Every origin a whole-catalogue pass rejected.

    `check_graph` takes `skip=failed | rejected(problems)`, so this is the second
    half of the same gate: a task spec whose closure the closure pass already
    rejected should not also be told its subgraph is malformed.

    Fatal only, for `failed_names`' reason.

    **The name/origin mismatch, stated because it is load-bearing.** A `Problem`
    identifies a *file* (`origin`), and `closure.check_closures` filters by
    *closure name* (`if name in skip`). So the value this returns does not match
    what the only consumer compares against, and neither side is wrong on its
    own: `Problem` has carried `origin` since `protocols.py` was frozen, and a
    name is what a registry is keyed by. Bridging it needs the origin-to-name
    map, which only the registries hold — so the composition root can do it in
    one line, or `Problem` grows a field. Reported; not decided here, because it
    is a change to a frozen type or to another module's signature.
    """
    return frozenset(problem.origin for problem in problems if problem.fatal)
