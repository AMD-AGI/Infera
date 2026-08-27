"""Resource pools — criteria 4, 32, 33.

Renewable and consumable differ in three behaviours — release, persistence,
recovery — which is why the split is at the abstract-base level rather than a
boolean flag.
"""

import pytest

from task_graph.registry import Registry
from task_graph.resource import ConsumableMgr, GpuMgr, RenewableMgr, ResourceMgr, TokenMgr
from task_graph.store import MemoryStoreMgr


def pool(cls, capacity: float, *, store=None, **kw):
    registry = Registry()
    registry.register("store_mgr", store or MemoryStoreMgr())
    return cls(registry=registry, capacity=capacity, **kw)


# ------------------------------------------------------------------- shared


@pytest.fixture(params=[RenewableMgr, ConsumableMgr])
def kind(request):
    return request.param


def test_a_fresh_pool_is_full(kind):
    p = pool(kind, 8, name="thing")
    assert p.name == "thing"
    assert p.capacity == 8
    assert p.available == 8


def test_can_afford_is_inclusive_of_the_whole_pool(kind):
    p = pool(kind, 8, name="thing")
    assert p.can_afford(8)
    assert not p.can_afford(8.5)


def test_take_reduces_availability(kind):
    p = pool(kind, 8, name="thing")
    p.take(3)
    assert p.available == 5
    assert p.can_afford(5)
    assert not p.can_afford(6)


def test_take_more_than_available_raises(kind):
    p = pool(kind, 8, name="thing")
    p.take(6)
    with pytest.raises(ValueError):
        p.take(3)
    assert p.available == 2  # and nothing was taken


def test_a_negative_amount_is_rejected(kind):
    p = pool(kind, 8, name="thing")
    with pytest.raises(ValueError):
        p.take(-1)


def test_a_negative_amount_is_not_affordable(kind):
    """Answering True would let a caller past its own guard and into `take`,
    which then raises after earlier pools in the same acquisition were taken."""
    p = pool(kind, 8, name="thing")
    assert not p.can_afford(-1)
    assert p.can_afford(0)


def test_resource_mgr_cannot_be_instantiated():
    with pytest.raises(TypeError):
        ResourceMgr(registry=Registry(), name="x", capacity=1)


# --------------------------------------------------------------- renewable


def test_a_renewable_returns_the_full_amount():
    p = pool(RenewableMgr, 8, name="gpu")
    p.take(3)
    p.give_back(3)
    assert p.available == 8


def test_a_renewable_ignores_actual():
    """`actual` is meaningless for a GPU: you held it or you did not."""
    p = pool(RenewableMgr, 8, name="gpu")
    p.take(3)
    p.give_back(3, actual=1)
    assert p.available == 8


def test_a_renewable_persists_nothing():
    store = MemoryStoreMgr()
    p = pool(RenewableMgr, 8, name="gpu", store=store)
    p.take(3)
    p.give_back(3)
    assert store.read_all("resource") == []


def test_a_renewable_recovers_to_full():
    """No lease survives a restart, so nothing is held — criterion 32."""
    store = MemoryStoreMgr()
    p = pool(RenewableMgr, 8, name="gpu", store=store)
    p.take(5)

    fresh = pool(RenewableMgr, 8, name="gpu", store=store)
    fresh.resume_system()
    assert fresh.available == 8


# -------------------------------------------------------------- consumable


def test_a_consumable_settles_at_actual():
    p = pool(ConsumableMgr, 1000, name="token")
    p.take(400)
    p.give_back(400, actual=250)
    assert p.available == 750


def test_a_consumable_with_no_figure_charges_the_whole_reservation():
    """The conservative reading for a budget; `on_stopped` relies on it."""
    p = pool(ConsumableMgr, 1000, name="token")
    p.take(400)
    p.give_back(400)
    assert p.available == 600


def test_a_consumable_clamps_actual_to_the_reservation():
    p = pool(ConsumableMgr, 1000, name="token")
    p.take(400)
    p.give_back(400, actual=900)
    assert p.available == 600


def test_a_consumable_treats_a_negative_actual_as_zero():
    p = pool(ConsumableMgr, 1000, name="token")
    p.take(400)
    p.give_back(400, actual=-5)
    assert p.available == 1000


def test_a_consumable_persists_only_on_settlement():
    """A reservation is a lease; a settlement is spend. Only spend is durable."""
    store = MemoryStoreMgr()
    p = pool(ConsumableMgr, 1000, name="token", store=store)

    p.take(400)
    assert store.read("resource", "token") is None

    p.give_back(400, actual=300)
    record = store.read("resource", "token")
    assert record["name"] == "token"
    assert record["spent"] == 300.0  # `spent` is the record; `available` is a comment


def test_a_consumable_balance_survives_a_rebuild():
    """Criterion 32."""
    store = MemoryStoreMgr()
    p = pool(ConsumableMgr, 1000, name="token", store=store)
    p.take(400)
    p.give_back(400, actual=300)

    fresh = pool(ConsumableMgr, 1000, name="token", store=store)
    fresh.resume_system()
    assert fresh.available == 700


def test_an_unsettled_reservation_is_not_charged():
    """Criterion 33: a crash mid-run costs the reservation, which is correct —
    the task holding it is not running any more."""
    store = MemoryStoreMgr()
    p = pool(ConsumableMgr, 1000, name="token", store=store)
    p.take(400)  # and then the process dies

    fresh = pool(ConsumableMgr, 1000, name="token", store=store)
    fresh.resume_system()
    assert fresh.available == 1000


def test_a_consumable_with_no_record_recovers_to_capacity():
    p = pool(ConsumableMgr, 1000, name="token")
    p.resume_system()
    assert p.available == 1000


def test_repeated_settlement_accumulates_durably():
    store = MemoryStoreMgr()
    p = pool(ConsumableMgr, 1000, name="token", store=store)
    for _ in range(3):
        p.take(100)
        p.give_back(100, actual=100)

    fresh = pool(ConsumableMgr, 1000, name="token", store=store)
    fresh.resume_system()
    assert fresh.available == 700


# ------------------------------------------------------------ the two named


def test_gpu_is_renewable_and_token_is_consumable():
    """The task definition names both; they add a default name and nothing else yet."""
    gpu = pool(GpuMgr, 8)
    token = pool(TokenMgr, 1_000_000)

    assert (gpu.name, token.name) == ("gpu", "token")
    assert isinstance(gpu, RenewableMgr)
    assert isinstance(token, ConsumableMgr)

    gpu.take(2)
    gpu.give_back(2, actual=1)
    assert gpu.available == 8

    token.take(100)
    token.give_back(100, actual=40)
    assert token.available == 1_000_000 - 40


def test_an_outstanding_reservation_is_not_baked_into_the_record():
    """Criterion 33, the case that matters. `available` at settlement time is
    net of every *other* live reservation, so persisting it would make those
    leases durable — and they are supposed to die with the process."""
    store = MemoryStoreMgr()
    p = pool(ConsumableMgr, 1000, name="token", store=store)

    p.take(400)  # A: still running, never settles
    p.take(500)  # B: settles at 300
    p.give_back(500, actual=300)

    assert p.available == 300  # live: 1000 - 400 (A's lease) - 300 (B's spend)
    assert store.read("resource", "token")["spent"] == 300.0

    fresh = pool(ConsumableMgr, 1000, name="token", store=store)
    fresh.resume_system()
    assert fresh.available == 700  # only B's spend survived, not A's lease
    assert fresh.spent == 300


def test_the_overcharge_does_not_compound_across_restarts():
    """Each restart must charge for spend only. If a lease leaked into the
    record, the same reservation would be paid for twice — once as a phantom
    and once for real when the demoted task re-runs."""
    store = MemoryStoreMgr()
    for _ in range(3):
        p = pool(ConsumableMgr, 10_000, name="token", store=store)
        p.resume_system()
        p.take(400)  # interrupted every time, never settled
        p.take(100)
        p.give_back(100, actual=100)

    fresh = pool(ConsumableMgr, 10_000, name="token", store=store)
    fresh.resume_system()
    assert fresh.spent == 300  # three settled hundreds
    assert fresh.available == 9_700  # and not a token less


def test_spent_and_available_are_complements_after_a_resume():
    store = MemoryStoreMgr()
    p = pool(ConsumableMgr, 1000, name="token", store=store)
    p.take(250)
    p.give_back(250, actual=250)

    fresh = pool(ConsumableMgr, 1000, name="token", store=store)
    fresh.resume_system()
    assert fresh.spent + fresh.available == fresh.capacity


def test_a_capacity_lowered_below_what_is_spent_does_not_go_negative():
    """An operator may cut a budget after some of it is gone. Unclamped,
    `available` goes negative and `can_afford(0)` is then False — every task
    naming the pool queues forever with no diagnostic. Sibling of O7."""
    store = MemoryStoreMgr()
    p = pool(ConsumableMgr, 1000, name="token", store=store)
    p.take(300)
    p.give_back(300, actual=300)

    shrunk = pool(ConsumableMgr, 100, name="token", store=store)
    shrunk.resume_system()

    assert shrunk.available == 0.0
    assert shrunk.can_afford(0)  # a task asking for nothing still fits
    assert not shrunk.can_afford(1)
