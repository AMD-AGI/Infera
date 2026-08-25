"""The task collection.

Durability and lookup. Transitions are the `Task`'s own, so there is no
`set_status` here — a caller does `task.status = X` and then `mgr.persist(tid)`.
"""

from task_graph.ids import TaskId
from task_graph.models import Task, TaskStatus
from task_graph.registry import Registry

__all__ = ["TaskMgr"]

KIND = "task"


class TaskMgr:
    def __init__(self, registry: Registry) -> None:
        self._r = registry
        self._tasks: dict[TaskId, Task] = {}

    def add(self, task: Task) -> None:
        """Register a new task.

        Rejects an id that exists and is not CANCELLED. Reviving a cancelled id
        is forced by `update_task`, which is `remove_queued` + `submit` under
        the same id; it replaces the record, fresh history included.
        """
        existing = self._tasks.get(task.id)
        if existing is not None and existing.status is not TaskStatus.CANCELLED:
            raise KeyError(f"task {task.id} already exists ({existing.status.value})")
        self._tasks[task.id] = task
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

    def remove(self, tid: TaskId) -> None:
        """Forget a task entirely. Refuses one the scheduler still indexes.

        Cancellation is `Scheduler.remove_queued`; this is the harder delete,
        for a record an operator wants gone. Without the guard it leaves an id
        in a pool with no task behind it, and every subsequent dispatch pass
        raises `KeyError` at the eligibility re-check — permanently. The
        scheduler is resolved at use time and its absence tolerated, so this
        stays a lookup rather than a dependency.
        """
        self.get(tid)
        scheduler = self._r.get("scheduler") if "scheduler" in self._r else None
        if scheduler is not None and any(tid in pool for pool in scheduler.pools.values()):
            raise ValueError(
                f"cannot remove {tid}: the scheduler still indexes it; "
                f"cancel or let it finish first"
            )
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
        """
        self._tasks = {}
        for record in self._store.read_all(KIND):
            task = Task.model_validate(record)
            self._tasks[task.id] = task
            if task.is_running:
                task.close_execution({}, TaskStatus.SUSPENDED, detail="interrupted by restart")
                self.persist(task.id)

    @property
    def _store(self):
        return self._r.get("store_mgr")
