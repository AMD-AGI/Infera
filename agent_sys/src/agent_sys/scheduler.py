"""The scheduler.

Decides *when* a task runs and nothing else: it never inspects what a task does
and never writes handoff state. Eligibility is a query re-asked at each decision
point, not a counter maintained across events.
"""

import logging
import threading
from math import isfinite

from agent_sys.ids import TaskId
from agent_sys.models import RESUMABLE, WAITING, Task, TaskStatus
from agent_sys.registry import Registry

__all__ = ["Scheduler"]

log = logging.getLogger(__name__)


class Scheduler:
    def __init__(self, registry: Registry) -> None:
        self._r = registry
        # One bucket per status, so a task is in exactly one pool. Only three
        # are load-bearing; the rest make "which tasks are suspended" a lookup.
        self.pools: dict[TaskStatus, set[TaskId]] = {s: set() for s in TaskStatus}
        self._lock = threading.RLock()
        self._in_dispatch = False
        self._dispatch_again = False

    # ------------------------------------------------------------------ API

    def submit(self, task: Task) -> None:
        """Accept a task, declare its outputs, and try to place it."""
        agent_mgr, task_mgr = self._r.get("agent_mgr"), self._r.get("task_mgr")
        with self._lock:
            if not agent_mgr.is_registered(task.agent_spec):
                raise KeyError(
                    f"unknown agent spec {task.agent_spec!r}; registered: "
                    f"{sorted(agent_mgr.specs())}"
                )
            for name, amount in task.resources.items():
                if f"resource:{name}" not in self._r:
                    raise ValueError(f"task {task.id}: no resource pool named {name!r}")
                # A negative or non-finite amount passes `can_afford` and then
                # raises inside `take` — after earlier pools in the same
                # acquisition have already been taken, which is exactly the
                # partial reservation all-or-nothing exists to prevent.
                if not isfinite(amount) or amount < 0:
                    raise ValueError(
                        f"task {task.id}: {name!r} amount must be a non-negative "
                        f"finite number, got {amount}"
                    )

            task_mgr.add(task)
            self._r.get("handoff_mgr").declare(task.outputs, producer_task_id=task.id)
            self._warn_depends_on(task)
            self._move(task.id, self._landing(task))
            self.try_dispatch()

    def expedite(self, task: Task) -> None:
        """Submit a handoff-complete closure straight to the front of the queue."""
        handoff_mgr = self._r.get("handoff_mgr")
        with self._lock:
            unmet = [h for h in task.inputs if not handoff_mgr.check_if_latest_valid(h)]
            if unmet:
                raise ValueError(
                    f"cannot expedite {task.id}: inputs not valid: {[str(h) for h in unmet]}"
                )
            # Set the flag only once submit has accepted: otherwise a rejection
            # for some *other* reason — a duplicate id, an unknown spec — still
            # leaves the caller's object mutated.
            was = task.expedited
            task.expedited = True
            try:
                self.submit(task)
            except Exception:
                task.expedited = was
                raise

    def remove_queued(self, tid: TaskId) -> None:
        with self._lock:
            self._require(tid, WAITING, "remove")
            self._move(tid, TaskStatus.CANCELLED)

    def stop(self, tid: TaskId) -> None:
        """Ask the runner to stop. The task is SUSPENDED when it acknowledges."""
        with self._lock:
            self._require(tid, {TaskStatus.RUNNING}, "stop")
            self._move(tid, TaskStatus.STOPPING)
            self._r.get("runner").stop(tid, self.on_stopped)

    def resume_task(self, tid: TaskId) -> None:
        """Requeue a stopped or failed task. A new attempt is pushed on top.

        Not named `resume`: `Resumable` is a runtime-checkable Protocol matching
        on method name alone, and `resume_all` would then call this with no
        argument.
        """
        with self._lock:
            self._require(tid, RESUMABLE, "resume")
            task = self._r.get("task_mgr").get(tid)
            self._move(tid, self._landing(task))
            self.try_dispatch()

    def update_task(self, tid: TaskId, **fields) -> Task:
        """Replace a queued task's definition, keeping its id and its place.

        Literally cancel-then-submit, so "an update behaves exactly like a
        resubmission" holds by construction rather than by assertion.
        """
        with self._lock:
            old = self._r.get("task_mgr").get(tid).model_copy(deep=True)
            self.remove_queued(tid)
            # model_copy preserves created_at: an update does not cost a task
            # its place in FIFO order.
            replacement = old.model_copy(
                update={**fields, "status": TaskStatus.WAITING_HANDOFF, "history": []},
                deep=True,
            )
            try:
                self.submit(replacement)
            except Exception:
                # The cancel already happened. Without this the task is simply
                # gone — a rejected update would silently destroy the thing it
                # was asked to change.
                old.status = TaskStatus.WAITING_HANDOFF
                self.submit(old)
                raise
            return replacement

    # --------------------------------------------------------- runner callbacks

    def on_task_done(self, tid: TaskId, status: TaskStatus, usage: dict[str, float]) -> None:
        """The runner finished. Release, record what was written, move on."""
        with self._lock:
            self._require(tid, {TaskStatus.RUNNING}, "complete")
            task = self._r.get("task_mgr").get(tid)
            self._release(task, usage)
            handoff_mgr = self._r.get("handoff_mgr")
            # The scheduler reads output versions for itself, exactly as it read
            # the input versions at dispatch.
            output_versions = {
                hid: version.version
                for hid in task.outputs
                if (version := handoff_mgr.latest(hid)) is not None
            }
            task.close_execution(output_versions, status)
            self._move(tid, status)
            self.try_dispatch()

    def on_stopped(self, tid: TaskId) -> None:
        """The runner acknowledged a stop."""
        with self._lock:
            self._require(tid, {TaskStatus.STOPPING}, "acknowledge a stop for")
            task = self._r.get("task_mgr").get(tid)
            # No usage figures here, so a consumable settles at the full
            # reservation — the safe direction for a budget.
            self._release(task, usage=None)
            task.close_execution({}, TaskStatus.SUSPENDED, detail="stopped on request")
            self._move(tid, TaskStatus.SUSPENDED)
            self.try_dispatch()

    # ------------------------------------------------------------- dispatch

    def try_dispatch(self) -> None:
        with self._lock:
            if self._in_dispatch:  # re-entered from a synchronous runner
                self._dispatch_again = True
                return
            self._in_dispatch = True
            try:
                while True:
                    self._dispatch_pass()
                    if not self._dispatch_again:
                        return
                    self._dispatch_again = False
            finally:
                self._in_dispatch = False

    def _dispatch_pass(self) -> None:
        handoff_mgr, task_mgr = self._r.get("handoff_mgr"), self._r.get("task_mgr")

        # 1. re-check eligibility. Snapshot: _move mutates the sets being scanned.
        for tid in list(
            self.pools[TaskStatus.WAITING_HANDOFF] | self.pools[TaskStatus.WAITING_RESOURCE]
        ):
            self._move(tid, self._landing(task_mgr.get(tid)))

        # 2. order the eligible set — the one scheduling decision in the system
        eligible = [task_mgr.get(t) for t in self.pools[TaskStatus.WAITING_RESOURCE]]
        for tid in self._r.get("policy").select(eligible, self._snapshot()):
            task = task_mgr.get(tid)
            if task.status is not TaskStatus.WAITING_RESOURCE:
                continue  # moved since selection — a synchronous runner re-entered

            # 3. all-or-nothing: verify the FULL set before mutating anything
            pools = {name: self._r.get(f"resource:{name}") for name in task.resources}
            if not all(pools[n].can_afford(amount) for n, amount in task.resources.items()):
                continue  # take nothing; stay queued
            for name, amount in task.resources.items():
                pools[name].take(amount)

            # 4. bind an agent and launch. Everything from here on can fail —
            # an unknown spec, an agent factory that is down, a runner whose
            # harness is unreachable — and by then the lease is already taken.
            # Releasing it is what stops one bad task from permanently
            # shrinking a pool; not re-raising is what stops it from aborting
            # the pass for every other queued task.
            try:
                # PUSH a record; the stack top is the binding
                agent = self._r.get("agent_mgr").instantiate(task.agent_spec, tid)
                task.push_execution(
                    agent_id=agent.id,
                    input_versions={
                        hid: version.version
                        for hid in task.inputs
                        if (version := handoff_mgr.latest(hid)) is not None
                    },
                )  # instantiate() bound agent.task_id; the agent fills agent.handoffs
                self._move(tid, TaskStatus.RUNNING)  # _move persists both
                self._r.get("runner").start(task, agent, on_done=self.on_task_done)
            except Exception:
                self._abort_launch(tid, task, pools)

    # -------------------------------------------------------------- recovery

    def resume_system(self) -> None:
        """Rebuild the index and demote runs the restart interrupted."""
        with self._lock:
            self.pools = {s: set() for s in TaskStatus}
            for task in self._r.get("task_mgr").all():
                status = {
                    TaskStatus.RUNNING: TaskStatus.WAITING_RESOURCE,  # the lease is gone
                    TaskStatus.STOPPING: TaskStatus.SUSPENDED,  # the runner is gone
                }.get(task.status, task.status)
                self._move(task.id, status)
            # Eligibility is not restored, it is recomputed — which is why
            # HandoffMgr must have resumed first.
            self.try_dispatch()

    # ------------------------------------------------------------- internals

    def _move(self, tid: TaskId, status: TaskStatus) -> None:
        """The single writer. Nothing else assigns task.status or writes pools."""
        task_mgr = self._r.get("task_mgr")
        # Discarding from every pool rather than from the recorded status makes
        # this idempotent and self-healing.
        for pool in self.pools.values():
            pool.discard(tid)
        self.pools[status].add(tid)
        task = task_mgr.get(tid)
        if task.status is not status:
            task.status = status
            task_mgr.persist(tid)

    def _abort_launch(self, tid: TaskId, task: Task, pools: dict) -> None:
        """Undo a dispatch that raised between `take` and a live runner.

        Returns the whole reservation — nothing ran, so a consumable spent
        nothing — closes the half-open attempt if one was pushed, and parks the
        task in FAILED. FAILED rather than back in the queue: the pass would
        pick it up again on the next iteration and fail identically, forever.
        An operator resumes it once the cause is fixed.
        """
        for name, amount in task.resources.items():
            pools[name].give_back(amount, actual=0.0)
        if task.is_running:
            task.close_execution({}, TaskStatus.FAILED, detail="failed to launch")
        log.exception("%s: failed to launch; the reservation was released", tid)
        self._move(tid, TaskStatus.FAILED)

    def _landing(self, task: Task) -> TaskStatus:
        return TaskStatus.WAITING_RESOURCE if self._ready(task) else TaskStatus.WAITING_HANDOFF

    def _ready(self, task: Task) -> bool:
        """The query that replaces a dependency counter."""
        handoff_mgr = self._r.get("handoff_mgr")
        return all(handoff_mgr.check_if_latest_valid(hid) for hid in task.inputs)

    def _release(self, task: Task, usage: dict[str, float] | None) -> None:
        for name, amount in task.resources.items():
            actual = None if usage is None else usage.get(name)
            self._r.get(f"resource:{name}").give_back(amount, actual)

    def _snapshot(self) -> dict[str, float]:
        return {pool.name: pool.available for pool in self._r.resolve("resource:*")}

    def _require(self, tid: TaskId, allowed, verb: str) -> Task:
        task = self._r.get("task_mgr").get(tid)
        if task.status not in allowed:
            raise ValueError(
                f"cannot {verb} {tid}: it is {task.status.value}, expected one of "
                f"{sorted(s.value for s in allowed)}"
            )
        return task

    def _warn_depends_on(self, task: Task) -> None:
        """Warn when `depends_on` omits the producer of one of the inputs.

        Warn, not reject: rejecting would make declaration order matter, and
        repairing would make `depends_on` derived and unable to express a
        dependency that shares no handoff.
        """
        handoff_mgr = self._r.get("handoff_mgr")
        for hid in task.inputs:
            version = handoff_mgr.latest(hid)
            if version is None or version.producer_task_id is None:
                continue
            if version.producer_task_id not in task.depends_on:
                log.warning(
                    "%s: depends_on omits %s, which produces %s",
                    task.id,
                    version.producer_task_id,
                    hid,
                )
