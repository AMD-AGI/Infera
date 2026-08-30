"""The pool's collection type.

A `dict` has preserved insertion order since 3.7 and gives O(1) membership and
deletion — the two operations `Scheduler._move` does most. `OrderedDict` adds a
doubly-linked list and `move_to_end`, neither of which is wanted; a `list` makes
`discard` O(n); `sortedcontainers` sorts by a key, which is precisely what the
ordered pool exists to stop anyone doing.

Separate from `scheduler.py` because the scheduler and its tests both need it,
and because a collection type living in the scheduler invites the scheduler to
grow collection behaviour.
"""

from collections.abc import Iterable, Iterator

from task_graph.ids import TaskId

__all__ = ["OrderedIdSet"]


class OrderedIdSet:
    """Insertion-ordered set of `TaskId`. Insertion order *is* promotion order."""

    __slots__ = ("_items",)

    def __init__(self, items: Iterable[TaskId] = ()) -> None:
        self._items: dict[TaskId, None] = {tid: None for tid in items}

    def add(self, tid: TaskId) -> None:
        """Append. A no-op if already present, so a re-add cannot reorder."""
        self._items.setdefault(tid, None)

    def discard(self, tid: TaskId) -> None:
        self._items.pop(tid, None)

    def __contains__(self, tid: object) -> bool:
        return tid in self._items

    def __iter__(self) -> Iterator[TaskId]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __or__(self, other: "OrderedIdSet") -> "OrderedIdSet":
        """Union, this one's order first. Step 1 of a dispatch pass unions the
        two waiting pools and iterates the result."""
        return OrderedIdSet([*self, *other])

    def __eq__(self, other: object) -> bool:
        if isinstance(other, OrderedIdSet):
            return list(self) == list(other)
        if isinstance(other, (set, frozenset)):
            return set(self._items) == other
        return NotImplemented

    def __hash__(self) -> int:  # pragma: no cover - a mutable collection
        raise TypeError("OrderedIdSet is unhashable")

    def __repr__(self) -> str:
        return f"OrderedIdSet({list(self._items)!r})"
