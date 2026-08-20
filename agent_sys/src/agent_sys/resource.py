"""Resource pools.

The one place inheritance appears. Renewable and consumable differ in *three*
behaviours — release, persistence, recovery — so a boolean flag would mean three
conditionals kept in agreement by hand.
"""

from abc import ABC, abstractmethod

from agent_sys.registry import Registry

__all__ = ["ResourceMgr", "RenewableMgr", "ConsumableMgr", "GpuMgr", "TokenMgr"]

KIND = "resource"


class ResourceMgr(ABC):
    """A quantity, not a collection — the acknowledged exception to §6's rule.

    The name is kept because `mission.md` uses it.
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
    """A token budget: reserved, then settled at what was actually spent."""

    def give_back(self, amount: float, actual: float | None = None) -> None:
        self._check(amount)
        spent = amount if actual is None else min(max(actual, 0.0), amount)
        self.available = min(self.capacity, self.available + (amount - spent))
        # Only here, never in `take`: a reservation is a lease and dies with its
        # process; a settlement is spend and must not be un-spent.
        self._persist()

    def resume_system(self) -> None:
        record = self._r.get("store_mgr").read(KIND, self.name)
        if record is not None:
            self.available = float(record["available"])
        else:
            self.available = self.capacity  # first run

    def _persist(self) -> None:
        store = self._r.get("store_mgr")
        record = {"name": self.name, "available": self.available}
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
