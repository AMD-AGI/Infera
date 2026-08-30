"""The reducer table. One entry in the alpha.

Spec §6.2 writes `reduce="all" | "any" | "at_least(k)"`. Those three names are
**ours**, not adopted: Inspect AI's registered set is `collect, at_least,
pass_at, pass_k, max, mean, median, mode`, and `multi_scorer([...], "all")`
raises `LookupError` there. Injecting the reducer behind a Protocol is what makes
the other two additions rather than changes.

Imports nothing from this package except `protocols`, which is what makes it
testable with no registry, no runner and no store.
"""

from __future__ import annotations

from collections.abc import Sequence

from validator.protocols import Reducer, ValidatorInvalid

__all__ = ["REDUCERS", "AllReducer", "get_reducer"]


class AllReducer:
    """`and` semantics over one handoff's member verdicts.

    **`all([])` is `True`**, and that is the vacuous pass spec §1 exists to
    prevent — a handoff no member declares would be folded from an empty list and
    pass, having been checked by nothing. This reducer is not where that is
    caught, because a reducer cannot tell an empty fold from a deliberate one;
    `Composite` refuses the uncovered handoff at admission instead (§6.4).
    """

    name = "all"

    def __call__(self, verdicts: Sequence[bool]) -> bool:
        return all(verdicts)


REDUCERS: dict[str, Reducer] = {"all": AllReducer()}


def get_reducer(name: str) -> Reducer:
    """Raise `ValidatorInvalid` enumerating the table.

    A failed lookup names its candidates — `docs/design.md` §5.2's rule, from
    pytest and dbt. The lists are short today, which is what makes that the right
    default (§15 O11).
    """
    try:
        return REDUCERS[name]
    except KeyError:
        raise ValidatorInvalid(
            f"no reducer named {name!r}; registered: {sorted(REDUCERS)}"
        ) from None
