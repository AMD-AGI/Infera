"""`OrderedIdSet` — the pool's collection type.

No criterion of its own. It is here because criterion 43 rests on it entirely:
depth-first is a property of *where a task was placed*, and a collection that
loses placement makes the whole policy silently arbitrary.
"""

import pytest

from task_graph.ids import TaskId
from task_graph.ordered import OrderedIdSet


def ids(n: int) -> list[TaskId]:
    return [TaskId.new() for _ in range(n)]


def test_iteration_is_insertion_order():
    a, b, c = ids(3)
    pool = OrderedIdSet()
    for tid in (c, a, b):
        pool.add(tid)
    assert list(pool) == [c, a, b]


def test_re_adding_does_not_move_a_member():
    """The property `Scheduler._move`'s early return exists to protect."""
    a, b = ids(2)
    pool = OrderedIdSet([a, b])
    pool.add(a)
    assert list(pool) == [a, b]


def test_discard_then_add_moves_to_the_end():
    """Stated as the thing to avoid, not as a feature."""
    a, b = ids(2)
    pool = OrderedIdSet([a, b])
    pool.discard(a)
    pool.add(a)
    assert list(pool) == [b, a]


def test_discarding_an_absent_member_is_a_no_op():
    a, b = ids(2)
    pool = OrderedIdSet([a])
    pool.discard(b)
    assert list(pool) == [a]


def test_membership_and_length():
    a, b = ids(2)
    pool = OrderedIdSet([a])
    assert a in pool and b not in pool
    assert len(pool) == 1


def test_union_keeps_the_left_order_first():
    a, b, c = ids(3)
    assert list(OrderedIdSet([a, b]) | OrderedIdSet([c, a])) == [a, b, c]


def test_it_compares_equal_to_a_plain_set():
    """So the shipped tests that assert `pools[X] == {tid}` keep meaning what
    they meant when the pools were sets."""
    a, b = ids(2)
    assert OrderedIdSet([a, b]) == {b, a}
    assert OrderedIdSet([a]) != {b}


def test_order_matters_between_two_ordered_sets():
    a, b = ids(2)
    assert OrderedIdSet([a, b]) != OrderedIdSet([b, a])


def test_it_is_unhashable():
    with pytest.raises(TypeError):
        hash(OrderedIdSet())
