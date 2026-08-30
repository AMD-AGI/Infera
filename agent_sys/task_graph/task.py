"""The task collection.

Durability, lookup, and the two derived indexes. Transitions are the `Task`'s
own, so there is no `set_status` here — a caller does `task.status = X` through a
transition and then the mgr persists it.
"""

from task_graph.ids import HandoffId, TaskId
from task_graph.models import Task, TaskStatus
from task_graph.registry import Registry

__all__ = ["TaskMgr"]

KIND = "task"


class TaskMgr:
    def __init__(self, registry: Registry) -> None:
        self._r = registry
        self._tasks: dict[TaskId, Task] = {}
        # Two forward edges derived from a stored back edge. `Task` stores
        # `parent`, not `children`; `inputs`, not consumers. Both are asked
        # often enough to matter — leaf-ness at every acquisition point,
        # consumers at every cascade level — and every surveyed system with a
        # reverse edge maintains it eagerly rather than scanning on demand.
        self._children: dict[TaskId, list[TaskId]] = {}
        self._consumers: dict[HandoffId, list[TaskId]] = {}

    def add(self, task: Task) -> None:
        """Register a new task.

        Rejects an id that exists and is not CANCELLED. Reviving a cancelled id
        is forced by `update_task`, which is `remove_queued` + `submit` under
        the same id; it replaces the record, fresh history included.

        This is also where the registry reference is supplied and where the two
        indexes are maintained: `add` and `remove` are the only places a task
        enters or leaves the collection, so covering them covers `submit`,
        `update_task` and `unfold` by construction.
        """
        existing = self._tasks.get(task.id)
        if existing is not None and existing.status is not TaskStatus.CANCELLED:
            raise KeyError(f"task {task.id} already exists ({existing.status.value})")
        if existing is not None:
            self._unindex(existing)
        task._registry = self._r
        self._tasks[task.id] = task
        self._index(task)
        record = task.model_dump(mode="json")
        if existing is not None:
            self._store.update(KIND, str(task.id), record)
        else:
            self._store.create(KIND, str(task.id), record)

    def get(self, tid: TaskId) -> Task:
        try:
            return self._tasks[tid]
        except KeyError:
            raise KeyError(f"no task {tid}") from None

    def all(self) -> list[Task]:
        return list(self._tasks.values())

    def by_status(self, status: TaskStatus) -> list[Task]:
        return [t for t in self._tasks.values() if t.status is status]

    def children(self, tid: TaskId) -> list[Task]:
        """Tasks whose `parent` is `tid`. Empty means leaf.

        A dangling edge is not an error: an unknown id returns empty, matching
        `check_if_latest_valid`'s answer for an unknown handoff and keeping
        submission order unconstrained.
        """
        return [self._tasks[c] for c in self._children.get(tid, ()) if c in self._tasks]

    def consumers(self, hid: HandoffId) -> list[Task]:
        """Tasks naming `hid` in `inputs` — the downstream direction."""
        return [self._tasks[c] for c in self._consumers.get(hid, ()) if c in self._tasks]

    def consumers_of_outputs(self, tid: TaskId) -> list[Task]:
        """Every task consuming anything this one produces, deduplicated.

        The index finds candidates; it never authorises the destructive act. A
        cascade re-reads each task's status before cancelling it rather than
        trusting what this said a moment ago.
        """
        seen: dict[TaskId, Task] = {}
        for hid in self.get(tid).outputs:
            for task in self.consumers(hid):
                if task.id != tid:
                    seen.setdefault(task.id, task)
        return list(seen.values())

    def remove(self, tid: TaskId) -> None:
        """Forget a task entirely. Refuses one the scheduler still indexes.

        Cancellation is `Scheduler.remove_queued`; this is the harder delete,
        for a record an operator wants gone. Without the guard it leaves an id
        in a pool with no task behind it, and every subsequent dispatch pass
        raises `KeyError` at the eligibility re-check — permanently. The
        scheduler is resolved at use time and its absence tolerated, so this
        stays a lookup rather than a dependency.
        """
        task = self.get(tid)
        scheduler = self._r.get("scheduler") if "scheduler" in self._r else None
        if scheduler is not None and any(tid in pool for pool in scheduler.pools.values()):
            raise ValueError(
                f"cannot remove {tid}: the scheduler still indexes it; "
                f"cancel or let it finish first"
            )
        self._unindex(task)
        del self._tasks[tid]
        self._store.delete(KIND, str(tid))

    def persist(self, tid: TaskId) -> None:
        """Write the task back after a caller mutated it through its own methods."""
        self._store.update(KIND, str(tid), self.get(tid).model_dump(mode="json"))

    def resume_system(self) -> None:
        """Reload; close any dangling stack top as SUSPENDED.

        The restart cut the attempt short, it was not judged. Closing it is not
        tidiness: `push_execution` refuses to stack on an open attempt, so
        leaving it open would make the first `resume_task` raise. Where the
        *task* lands is a separate decision the scheduler makes a moment later.

        `model_validate` returns a task with no registry reference under every
        candidate mechanism, so re-supplying it here is not a choice.
        """
        self._tasks = {}
        self._children = {}
        self._consumers = {}
        for record in self._store.read_all(KIND):
            task = Task.model_validate(record)
            task._registry = self._r
            self._tasks[task.id] = task
            self._index(task)
            if task.is_running:
                task.close_execution(TaskStatus.SUSPENDED, detail="interrupted by restart")
                self.persist(task.id)

    # ------------------------------------------------------------- internals

    def _index(self, task: Task) -> None:
        if task.parent is not None:
            bucket = self._children.setdefault(task.parent, [])
            if task.id not in bucket:
                bucket.append(task.id)
        for hid in task.inputs:
            bucket = self._consumers.setdefault(hid, [])
            if task.id not in bucket:
                bucket.append(task.id)

    def _unindex(self, task: Task) -> None:
        if task.parent is not None and task.parent in self._children:
            self._drop(self._children, task.parent, task.id)
        for hid in task.inputs:
            if hid in self._consumers:
                self._drop(self._consumers, hid, task.id)

    @staticmethod
    def _drop(index: dict, key, tid: TaskId) -> None:
        bucket = index[key]
        if tid in bucket:
            bucket.remove(tid)
        if not bucket:
            del index[key]

    @property
    def _store(self):
        return self._r.get("store_mgr")
