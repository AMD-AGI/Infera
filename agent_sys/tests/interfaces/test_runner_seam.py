"""`monitor`'s `Attempt` / `AttemptRunner` and `agent`'s `TaskAttempt` / `Runner`.

The second half of the same duplication `test_pushable.py` guards, and it exists
because the first half's lesson was learned the expensive way.

`monitor` may not import `agent`, so the *backend* shape it needs was declared
locally as `Pushable` and guarded here from the start. **The `runner` shape it
needs was not declared at all** — four members across two objects, checked by
nothing — and `interfaces.md` §4.9 says `monitor` resolves `runner` without
saying what `runner` must provide.

That gap produced a real defect. `monitor`'s `_advance` branched on
`attempt_of(tid) is None` and called it "the non-leaf case: no live thread"; an
attempt **survives its thread**, so the branch never fired, `wake()` set an Event
no thread was waiting on, and a non-leaf parent stalled in `OUTPUT_VALIDATING`
silently. Nothing anywhere stated what the runner guarantees about an attempt's
lifetime, which is precisely what a declared seam is for.

So the declaration is the fix for the class, and this file is its price — the
same bargain `interfaces.md` §8 names for `Pushable`: *"the test is not a nicety
attached to the decision; it is the decision's price."*
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from agent.protocols import Runner, TaskAttempt
from monitor.protocols import Attempt, AttemptRunner

ROOT = Path(__file__).resolve().parents[2]

#: Which local Protocol stands in for which declaration in `agent`.
PAIRS = {
    "Attempt": "TaskAttempt",
    "AttemptRunner": "Runner",
}


def _members(pkg: str, cls: str) -> dict[str, str]:
    """Public members of one class in one `protocols.py`, as text.

    Read from the source rather than the objects, for `test_pushable.py`'s
    reason: a `Protocol`'s bare annotations do not survive into `__dict__` in a
    form worth comparing, and the signature text is what has to agree. A
    `@property` is recorded as the method it is written as, so a property on one
    side and a plain method on the other still compare equal — the caller writes
    `attempt.is_running` either way.
    """
    tree = ast.parse((ROOT / pkg / "protocols.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    out: dict[str, str] = {}
    for sub in node.body:
        if isinstance(sub, ast.FunctionDef) and not sub.name.startswith("_"):
            ret = ast.unparse(sub.returns) if sub.returns else "None"
            out[sub.name] = f"({ast.unparse(sub.args)}) -> {ret}"
        elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
            if not sub.target.id.startswith("_"):
                out[sub.target.id] = ast.unparse(sub.annotation)
    return out


@pytest.mark.parametrize("mine,theirs", sorted(PAIRS.items()))
def test_agent_declares_everything_the_monitor_needs(mine: str, theirs: str) -> None:
    """Every member the monitor requires exists on the `agent` side."""
    required = _members("monitor", mine)
    declared = _members("agent", theirs)

    missing = sorted(set(required) - set(declared))
    assert not missing, (
        f"monitor.{mine} requires {missing}, which agent.{theirs} does not declare. "
        f"One of the two moved; the monitor's copy is the derived one."
    )


@pytest.mark.parametrize("mine,theirs", sorted(PAIRS.items()))
def test_the_monitors_copy_stays_narrow(mine: str, theirs: str) -> None:
    """It is the part the monitor uses, not a copy of the whole object.

    A local Protocol that grew to the full surface would be the import the
    structural declaration exists to avoid, written out by hand. `TaskAttempt`
    has `task`, `agent` and `release()` that the monitor never touches, and
    `Runner` has `start` and `stop`, which are the *scheduler's* verbs — a
    monitor that could call `start` would be starting tasks.
    """
    required = set(_members("monitor", mine))
    declared = set(_members("agent", theirs))

    assert required < declared, f"monitor.{mine} is not narrower than agent.{theirs}"
    if theirs == "Runner":
        assert not required & {"start", "stop"}, "the monitor must not hold the scheduler's verbs"


def test_the_two_shapes_are_what_advance_actually_calls() -> None:
    """The declaration matches the call sites, not merely `agent`.

    A Protocol that agreed with the neighbour but not with `_advance` would be
    decoration. This walks `monitor/base.py` and `monitor/pusher.py` for the
    attribute names taken off a runner or an attempt, and asserts each one is
    declared.
    """
    #: Both spellings, and the second is not optional: `_advance` reaches the
    #: runner as `self._runner`, so a matcher that only understood a bare `Name`
    #: would never look at `resume` — it would pass because it had not checked,
    #: which is `interfaces.md` §4.11's rule applied to a test.
    holders = {"runner", "attempt", "_runner"}

    used: set[str] = set()
    for name in ("base.py", "pusher.py"):
        tree = ast.parse((ROOT / "monitor" / name).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            inner = node.value
            holder = (
                inner.id
                if isinstance(inner, ast.Name)
                else inner.attr
                if isinstance(inner, ast.Attribute)
                else None
            )
            if holder in holders:
                used.add(node.attr)

    declared = set(_members("monitor", "Attempt")) | set(_members("monitor", "AttemptRunner"))
    undeclared = sorted(used - declared)
    assert not undeclared, (
        f"monitor calls {undeclared} on the runner seam and declares none of them. "
        f"That is the hole this file exists to close, reopened."
    )
    #: Named rather than counted: a matcher that silently stopped matching would
    #: leave `used` small and still pass the emptiness check.
    assert used >= {"attempt_of", "carry_on", "executor"}, (
        f"the walk found only {sorted(used)}; it has stopped seeing the call sites"
    )


def test_the_monitor_declares_nothing_it_does_not_call() -> None:
    """The mirror of the test above, and it is `interfaces.md` §4.12's own family.

    That section names two shapes: **a seam called and declared nowhere** — which
    is what these Protocols were added to close — and **a capability built and
    reachable by nobody**. A local Protocol declaring a member its owner has
    stopped calling is the second one, in miniature, and it is easy to acquire:
    `is_running` and `wake` were both declared here and both became unused the
    hour `carry_on` replaced the branch that read them.

    So the seam is pinned from both sides. It may not be smaller than the call
    sites, and it may not be larger.
    """
    used: set[str] = set()
    for name in ("base.py", "pusher.py"):
        tree = ast.parse((ROOT / "monitor" / name).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            inner = node.value
            holder = (
                inner.id
                if isinstance(inner, ast.Name)
                else inner.attr
                if isinstance(inner, ast.Attribute)
                else None
            )
            if holder in {"runner", "attempt", "_runner"}:
                used.add(node.attr)

    declared = set(_members("monitor", "Attempt")) | set(_members("monitor", "AttemptRunner"))
    unused = sorted(declared - used)
    assert not unused, (
        f"monitor declares {unused} on the runner seam and calls none of them. "
        f"A requirement nobody makes is a promise the other side has to keep for "
        f"no one — interfaces.md §4.12."
    )


def test_both_are_runtime_checkable() -> None:
    """So a real attempt can be checked against them once one is cheap to build.

    The static comparison above is what works today, for `test_pushable.py`'s
    reason: constructing a real `TaskAttempt` needs a runner, a task and a
    closure. This keeps the `isinstance` door open without anyone having to
    remember the decorator.
    """
    assert getattr(Attempt, "_is_runtime_protocol", False)
    assert getattr(AttemptRunner, "_is_runtime_protocol", False)
    assert TaskAttempt is not None and Runner is not None  # imported, not merely named
