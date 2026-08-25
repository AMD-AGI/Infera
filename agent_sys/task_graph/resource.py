"""Resource pools.

The one place inheritance appears. Renewable and consumable differ in *three*
behaviours — release, persistence, recovery — so a boolean flag would mean three
conditionals kept in agreement by hand.
"""

import logging
from abc import ABC, abstractmethod

from task_graph.registry import Registry

__all__ = ["ResourceMgr", "RenewableMgr", "ConsumableMgr", "GpuMgr", "TokenMgr"]

KIND = "resource"

log = logging.getLogger(__name__)


class ResourceMgr(ABC):
    """A quantity, not a collection — the acknowledged exception to §6's rule.

    The name is kept because the task definition uses it.
    """

    def __init__(self, registry: Registry, name: str, capacity: float) -> None:
        self._r = registry
        self.name = name
        self.capacity = float(capacity)
        self.available = float(capacity)

    def can_afford(self, amount: float) -> bool:
        # A negative amount is not "affordable": answering True would let a
        # caller past its own guard and into `take`, which then raises after
        # earlier pools in the same acquisition were already taken.
        return 0 <= amount <= self.available

    def take(self, amount: float) -> None:
        """Reserve. Raises if it does not fit — the caller must check first."""
        self._check(amount)
        if not self.can_afford(amount):
            raise ValueError(f"{self.name}: cannot take {amount}, only {self.available} available")
        self.available -= amount

    @abstractmethod
    def give_back(self, amount: float, actual: float | None = None) -> None:
        """Release a reservation of `amount`, of which `actual` was consumed."""

    @abstractmethod
    def resume_system(self) -> None:
        """The two classes genuinely differ here."""

    @staticmethod
    def _check(amount: float) -> None:
        if amount < 0:
            raise ValueError(f"amount must not be negative, got {amount}")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name}: {self.available}/{self.capacity})"


class RenewableMgr(ResourceMgr):
    """A GPU: held, then handed back whole."""

    def give_back(self, amount: float, actual: float | None = None) -> None:
        """Return the full amount. `actual` is meaningless for a renewable."""
        self._check(amount)
        self.available = min(self.capacity, self.available + amount)

    def resume_system(self) -> None:
        """No lease survives a restart, so nothing is held.

        Persists nothing: there is nothing to remember.
        """
        self.available = self.capacity


class ConsumableMgr(ResourceMgr):
    """A token budget: reserved, then settled at what was actually spent.

    Two quantities, and the distinction is the whole design. ``available`` is
    live and includes every outstanding *reservation*; ``spent`` is the running
    total of what was actually consumed. Only ``spent`` is durable — a
    reservation is a lease that dies with its process, so persisting
    ``available`` would bake every in-flight lease into the record and charge
    for it again on the next run.
    """

    def __init__(self, registry: Registry, name: str, capacity: float) -> None:
        super().__init__(registry, name, capacity)
        self.spent = 0.0

    def give_back(self, amount: float, actual: float | None = None) -> None:
        self._check(amount)
        spent = amount if actual is None else min(max(actual, 0.0), amount)
        self.available = min(self.capacity, self.available + (amount - spent))
        self.spent = min(self.capacity, self.spent + spent)
        # Only here, never in `take`: a settlement is spend and must not be
        # un-spent, while a reservation must never become durable.
        self._persist()

    def resume_system(self) -> None:
        record = self._r.get("store_mgr").read(KIND, self.name)
        self.spent = float(record["spent"]) if record is not None else 0.0
        # Nothing is reserved after a restart, so available is exactly what has
        # not been spent — but an operator may have *lowered* capacity below
        # what is already spent. Unclamped that makes `available` negative, and
        # `can_afford` then refuses even a request for zero: every task naming
        # this pool queues forever with no diagnostic. Sibling of O7.
        if self.spent > self.capacity:
            log.warning(
                "%s: %s already spent exceeds the capacity of %s; the budget is exhausted",
                self.name,
                self.spent,
                self.capacity,
            )
        self.available = max(0.0, self.capacity - self.spent)

    def _persist(self) -> None:
        store = self._r.get("store_mgr")
        # `spent` is the record. `available` is written for a human reading the
        # file with `cat` and is NOT read back — at this instant it still nets
        # out every in-flight reservation.
        record = {"name": self.name, "spent": self.spent, "available": self.available}
        if store.exists(KIND, self.name):
            store.update(KIND, self.name, record)
        else:
            store.create(KIND, self.name, record)


class GpuMgr(RenewableMgr):
    def __init__(self, registry: Registry, capacity: float, name: str = "gpu") -> None:
        super().__init__(registry, name, capacity)


class TokenMgr(ConsumableMgr):
    def __init__(self, registry: Registry, capacity: float, name: str = "token") -> None:
        super().__init__(registry, name, capacity)
