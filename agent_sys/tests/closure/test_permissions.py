"""Check 6 — criterion 5. The check only a closure can perform.

It needs the task's handoffs and its permissions together, and neither registry
sees both.
"""

from __future__ import annotations

from closure.check import READ, WRITE, check_closures, covers

from .conftest import NO_ESCAPE_HATCH, Regs, grant, make_closure


def coverage(problems) -> list:
    return [p for p in problems if p.keyword == "covers"]


def test_permissions_cover_handoffs(regs: Regs) -> None:
    """A closure whose task permissions do not cover its handoffs is rejected at
    load, naming both the handoff and the missing permission — rather than
    deadlocking at dispatch.
    """
    regs.with_kinds("trace", "kernel_ir").with_agents("profiler")
    regs.with_closure(
        make_closure(
            inputs=["trace"],
            outputs=["kernel_ir"],
            grants=[grant("trace", READ)],  # the write grant is missing
        )
    )

    (problem,) = coverage(check_closures(regs, NO_ESCAPE_HATCH))
    assert "kernel_ir" in problem.message
    assert "grant no write" in problem.message
    assert "grants held: trace(read)" in problem.message
    assert "hint: add a write grant for 'kernel_ir'" in problem.message
    assert problem.path == "$.task.permissions.grants"
    assert problem.fatal


def test_an_input_needs_a_read_grant(regs: Regs) -> None:
    regs.with_kinds("trace").with_agents("profiler")
    regs.with_closure(make_closure(inputs=["trace"], grants=[]))

    (problem,) = coverage(check_closures(regs, NO_ESCAPE_HATCH))
    assert "consumes handoff 'trace'" in problem.message
    assert "grant no read" in problem.message
    assert "grants held: none" in problem.message


def test_enumerating_every_other_kind_does_not_cover_one_it_does_not_name(regs: Regs) -> None:
    """The fail-closed direction, which is the one that matters.

    Kubernetes' `TestCoversEnumerationNotCoveringVerbStar` is the model: an
    exhaustive enumeration on the grant side does **not** cover a requirement it
    cannot justify. Ours: a grant list naming every other kind in the catalogue
    still does not cover the one it omits.
    """
    regs.with_kinds("trace", "kernel_ir", "summary").with_agents("profiler")
    regs.with_closure(
        make_closure(
            inputs=["summary"],
            handoffs=["summary"],
            grants=[grant("trace", READ), grant("kernel_ir", READ)],
        )
    )
    assert len(coverage(check_closures(regs, NO_ESCAPE_HATCH))) == 1


def test_a_write_grant_covers_a_read_requirement(regs: Regs) -> None:
    """`WRITE` implies `READ`, and the reverse does not hold.

    Rev. 1 of this module matched access exactly, and `task_graph`'s shipped
    `Permissions.covers` — whose docstring names *this* check as the reason it
    exists — already implemented the implication. Two bodies of one relation,
    opposite answers, both suites green, because each exercised only its own
    side. `tests/interfaces/test_covers_agreement.py` is what stops it recurring.

    The implication is not a policy preference: `env_mgr` turns a grant into a
    Landlock `Mode`, and a write grant on a directory without read and execute is
    unusable, because a file cannot be created in a directory that cannot be
    traversed. The failure the exact version produced was over-rejection — load
    refusing a spec that would have run correctly.
    """
    regs.with_kinds("trace").with_agents("profiler")
    regs.with_closure(make_closure(inputs=["trace"], grants=[grant("trace", WRITE)]))
    assert coverage(check_closures(regs, NO_ESCAPE_HATCH)) == []


def test_a_read_grant_does_not_cover_a_write(regs: Regs) -> None:
    """The half that must fail closed."""
    regs.with_kinds("trace").with_agents("profiler")
    regs.with_closure(make_closure(outputs=["trace"], grants=[grant("trace", READ)]))
    assert len(coverage(check_closures(regs, NO_ESCAPE_HATCH))) == 1


def test_the_name_grammar_is_exact_equality_with_no_wildcards(regs: Regs) -> None:
    """The *name* axis stays exact, and widening the access axis is not licence
    to widen this one.

    Access is a two-element order on a closed enum; a kind name is an open string
    that invites a second interpreter — a trailing slash, a `*`, a symlink, a
    `..`. If a wildcard is ever wanted, adding it is a change to `covers` and to
    the schema that admits its syntax, in that order, and to nothing else.
    """
    task = make_closure(inputs=["trace"], grants=[grant("*", READ), grant("trace_v2", READ)])[
        "task"
    ]
    assert not covers(task, "trace", READ)
    assert covers(task, "trace_v2", READ)


def test_coverage_still_reports_when_the_kind_does_not_resolve(regs: Regs) -> None:
    """Existence and coverage are separate questions and fail separately.

    A closure whose kind does not resolve gets one message about the kind, and
    check 6 still runs — the grant is about a *name* the author wrote and can fix
    whether or not the kind exists. Check 6 reads no registry at all, which is
    why it can.
    """
    regs.with_agents("profiler")
    regs.with_closure(make_closure(inputs=["ghost"], handoffs=["ghost"], grants=[]))

    problems = check_closures(regs, NO_ESCAPE_HATCH)
    assert {p.keyword for p in problems} == {"resolves", "covers"}
