"""The handoff collection.

The mgr decides nothing. Version bookkeeping belongs to `Handoff` and the
transition belongs to `HandoffVersion`; what is left here is add / get / query
and durability.
"""

from collections.abc import Iterable

from task_graph.ids import HandoffId, TaskId
from task_graph.models import Handoff, HandoffVersion
from task_graph.registry import Registry

__all__ = ["HandoffMgr"]

KIND = "handoff"


class HandoffMgr:
    def __init__(self, registry: Registry) -> None:
        self._r = registry
        self._handoffs: dict[HandoffId, Handoff] = {}

    # ---- scheduler-facing: read only ----

    def declare(
        self,
        ids: Iterable[HandoffId],
        producer_task_id: TaskId,
        types: dict[HandoffId, str] | None = None,
    ) -> None:
        """Create each handoff with a single CREATED v0.

        Idempotent: an id already present is skipped, never overwritten.
        `update_task` re-declares, and overwriting would delete versions an
        agent had already written.
        """
        types = types or {}
        for hid in ids:
            if hid in self._handoffs:
                continue
            self._handoffs[hid] = Handoff(
                id=hid,
                type=types.get(hid, ""),
                versions=[HandoffVersion(version=0, producer_task_id=producer_task_id)],
            )
            self._store.create(KIND, str(hid), self._handoffs[hid].model_dump(mode="json"))

    def check_if_latest_valid(self, hid: HandoffId) -> bool:
        """False for an unknown id rather than an error.

        A consumer may be submitted before its producer declares the slot;
        "not ready" is the right answer and keeps submission order free.
        """
        handoff = self._handoffs.get(hid)
        return handoff is not None and handoff.is_latest_valid

    def latest(self, hid: HandoffId) -> HandoffVersion | None:
        handoff = self._handoffs.get(hid)
        return handoff.latest if handoff is not None else None

    def get(self, hid: HandoffId) -> Handoff:
        try:
            return self._handoffs[hid]
        except KeyError:
            raise KeyError(f"no handoff {hid}") from None

    def get_many(self, ids: Iterable[HandoffId]) -> list[Handoff]:
        return [self.get(hid) for hid in ids]

    def all_ids(self) -> list[HandoffId]:
        return list(self._handoffs)

    def produced_by_task(self, tid: TaskId) -> list[HandoffId]:
        return [h.id for h in self._handoffs.values() if h.latest.producer_task_id == tid]

    # ---- agent-facing: write ----

    def persist(self, hid: HandoffId) -> None:
        """Write the handoff back after `open_next()` or `seal()` mutated it.

        One write verb: every change is "this handoff changed, store it".
        """
        self._store.update(KIND, str(hid), self.get(hid).model_dump(mode="json"))

    def resume_system(self) -> None:
        """Reload from the store. No regrouping: one record per handoff."""
        self._handoffs = {
            handoff.id: handoff
            for handoff in (Handoff.model_validate(r) for r in self._store.read_all(KIND))
        }

    @property
    def _store(self):
        return self._r.get("store_mgr")
