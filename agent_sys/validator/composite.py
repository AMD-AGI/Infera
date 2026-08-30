"""The composite — one level deep, reducing per handoff.

The reduce axis is the biggest trap in the module. A composite of *m* validators
over *n* handoffs is an *m×n* grid that has to collapse to an *n*-entry dict, and
three shapes are available:

    A. per-key    {'h1': True,  'h2': True,  'h3': False}   <- this one
    B. scalar     False                                     loses attribution
    C. broadcast  {'h1': False, 'h2': False, 'h3': False}   a lie

C is the trap, and it is what a naive implementation reaches for, because it is
the only shape that both keeps the declared return type *and* lets the reducer
reduce over validators. Under it `h1` passed every single member and is recorded
`False`.

A is Inspect AI's shape, arrived at independently. What is **not** borrowable is
its guard: Inspect rejects mismatched keys outright, because its keys are epochs
of one sample. Ours are handoffs and our members legitimately declare different
input kinds (spec §4.1), so that rejection would forbid a composite the spec
permits. §6.4's two refusals are the answer instead.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from task_graph.ids import HandoffId
from validator.protocols import (
    Dimension,
    NestedComposite,
    Reducer,
    Strength,
    Validator,
    ValidatorInvalid,
)

__all__ = ["Composite", "check_coverage", "kind_of"]


def kind_of(handoff: Any) -> str | None:
    """The handoff kind a runtime slot carries. `task_graph.Handoff.type`."""
    return getattr(handoff, "type", None)


def check_coverage(inputs_per_member: Iterable[Sequence[str]], kinds: Iterable[str]) -> None:
    """Refuse a composite that would fold a handoff from an empty list.

    Checked at **admission**, not per call: membership and declared inputs are
    both static, so the fault is an author's and belongs in the load report. The
    runtime guard in `__call__` remains because a composite can be built directly.
    """
    covered: set[str] = set()
    for declared in inputs_per_member:
        covered |= set(declared)
    missing = sorted({k for k in kinds if k not in covered})
    if missing:
        raise ValidatorInvalid(f"composite covers no validator for {missing}")


class Composite:
    """One level deep. Members are leaf validators; the reducer folds their
    verdicts **per handoff**.

    That keeps the result type `dict[HandoffId, bool]`, so a composite is
    type-substitutable for a leaf. Type-substitutability is what keeps the
    reducer out of the phase: `run_phase` cannot tell a composite from a leaf and
    therefore has no place to apply a reducer even if someone wanted to, so spec
    §5.5's one rule stays the phase's only rule. A composite whose reducer is
    `any` expresses "either of these will do" *inside* one check, and is never a
    way to soften the phase.
    """

    def __init__(
        self,
        name: str,
        *,
        brief: str,
        dimension: Dimension,
        strength: Strength,
        members: Sequence[Validator],
        reduce: Reducer,
    ) -> None:
        nested = [getattr(m, "name", repr(m)) for m in members if isinstance(m, Composite)]
        if nested:
            raise NestedComposite(
                f"{name}: a composite may not contain a composite; nested: {sorted(nested)}"
            )
        self.name = name
        self.brief = brief
        self.dimension = dimension
        self.strength = strength
        self.members: tuple[Validator, ...] = tuple(members)
        self.reduce = reduce
        self.inputs: tuple[str, ...] = tuple(
            sorted({kind for m in self.members for kind in m.inputs})
        )

    def __call__(self, handoffs: Mapping[HandoffId, Any]) -> dict[HandoffId, bool]:
        grid: list[tuple[Validator, dict[HandoffId, bool]]] = []
        for member in self.members:
            share = {h: v for h, v in handoffs.items() if kind_of(v) in set(member.inputs)}
            grid.append((member, dict(member(share)) if share else {}))
            self._refuse_omission(member, share, grid[-1][1])

        check_coverage(
            (m.inputs for m in self.members), (kind_of(v) or "" for v in handoffs.values())
        )
        return {h: self.reduce([v[h] for _, v in grid if h in v]) for h in handoffs}

    def _refuse_omission(
        self,
        member: Validator,
        given: Mapping[HandoffId, Any],
        got: Mapping[HandoffId, bool],
    ) -> None:
        """A member that returns no entry for a handoff it *declared* raises.

        `dict.get` would yield `None`, and `None` folded as falsy is
        indistinguishable from a genuine `False`. DeepEval demonstrates the cost:
        its unreached DAG node leaves `score` as `None`, `is_successful` catches
        the resulting `TypeError` and sets `success = False`, so an unreached
        terminal and a real zero report identically.
        """
        omitted = sorted(str(h) for h in given if h not in got)
        if omitted:
            raise ValidatorInvalid(
                f"{getattr(member, 'name', member)!r} declared "
                f"{sorted(member.inputs)} and returned no verdict for {omitted}"
            )
