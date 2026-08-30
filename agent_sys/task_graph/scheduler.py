"""The scheduler.

Decides *when* a task runs and nothing else: it never inspects what a task does
and never writes handoff state. Eligibility is a query re-asked at each decision
point, not a counter maintained across events.
"""

import logging
import threading
from collections import deque
from math import isfinite

from task_graph.ids import TaskId
from task_graph.models import (
    PHASES,
    RESUMABLE,
    WAITING,
    CascadeReport,
    Task,
    TaskStatus,
)
from task_graph.ordered import OrderedIdSet
from task_graph.registry import Registry

__all__ = ["Scheduler"]

log = logging.getLogger(__name__)


class Scheduler:
    def __init__(self, registry: Registry) -> None:
        self._r = registry
        # One bucket per status, so a task is in exactly one pool. Ten buckets:
        # the two phase statuses create two more by construction. Four are
        # load-bearing — the two waits and the three-member phase group a task
        # holding a lease sits in — and the rest exist so that "which tasks are
        # suspended" is a lookup rather than a scan.
        self.pools: dict[TaskStatus, OrderedIdSet] = {s: OrderedIdSet() for s in TaskStatus}
        self._lock = threading.RLock()
        self._in_dispatch = False
        self._dispatch_again = False
        # Separate from `_dispatch_again`, and the separation is the design.
        # Coalescing N requests into one is right for "re-check eligibility",
        # which is idempotent; it loses work for "cancel these seven tasks".
        self._cascade: deque[tuple[TaskId, str]] = deque()

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
            # `types=` has been on `declare` since spec rev. 4 and nothing ever
            # passed it, because `Task` had no field to pass. Without it every
            # `Handoff.type` is "" and a permission grant naming a kind matches
            # no handoff at all.
            self._r.get("handoff_mgr").declare(
                task.outputs, producer_task_id=task.id, types=task.kinds
            )
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
        """Ask the runner to stop. The task is SUSPENDED when it acknowledges.

        Accepted in all three phase states: all three are a running task from
        the outside.
        """
        with self._lock:
            self._require(tid, PHASES, "stop")
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
            # `model_copy(update=...)` writes straight to __dict__: it honours
            # neither `extra="forbid"` nor `validate_assignment`. Unchecked, a
            # misspelled field would be accepted and silently do nothing, and
            # `id=` would leave the original cancelled and create a second task.
            unknown = set(fields) - set(Task.model_fields)
            if unknown:
                raise ValueError(f"unknown task field(s): {sorted(unknown)}")
            managed = set(fields) & {"id", "status", "history", "created_at"}
            if managed:
                raise ValueError(f"{sorted(managed)} cannot be set by an update")

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

    def on_task_done(
        self,
        tid: TaskId,
        status: TaskStatus,
        usage: dict[str, float],
        *,
        detail: str = "",
    ) -> None:
        """The runner finished. Release, record what was written, move on."""
        with self._lock:
            self._require(tid, PHASES, "complete")
            task = self._r.get("task_mgr").get(tid)
            self._release(task, usage)
            # No output versions are derived here. Since `interfaces.md` §4.14
            # they are pinned at dispatch, by `_pin_outputs`, and re-deriving
            # them at close would overwrite the numbers the grant was resolved
            # from with `HandoffMgr`'s slot numbers — a different quantity that
            # agrees only until the first retry.
            task.close_execution(status, detail=detail)
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
            task.close_execution(TaskStatus.SUSPENDED, detail="stopped on request")
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
                    # A cancel is structurally a system message: it changes what
                    # work exists, and dispatching against a set that is about
                    # to shrink is wasted at best. Drained unconditionally, not
                    # on "something changed" — that condition is how a second
                    # request ends up doing nothing at all.
                    self._drain_cascade()
                    self._dispatch_pass()
                    if not self._dispatch_again:
                        return
                    self._dispatch_again = False
            finally:
                self._in_dispatch = False

    def _dispatch_pass(self) -> None:
        # `handoff_mgr` was resolved here for `input_versions`, which now asks
        # `handoff_store` instead — see `_pin_inputs`. Nothing else in this pass
        # wanted the slot manager, so the name goes rather than lingering as a
        # resolve nobody reads.
        task_mgr = self._r.get("task_mgr")

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

            # An input can have gone stale since step 1: an earlier task in this
            # same pass may have opened one of them for writing. Pinning a
            # GENERATING version would record an input whose content does not
            # exist yet. Re-asked here, before any lease is taken.
            if not self._ready(task):
                self._move(tid, TaskStatus.WAITING_HANDOFF)
                continue

            # 3. all-or-nothing: verify the FULL set before mutating anything.
            # Resolving the pools is inside the guard, not before it: a pool an
            # operator removed between restarts raises here, and outside the
            # guard that KeyError would escape the whole pass — taking every
            # healthy task's recovery down with it.
            pools: dict = {}
            try:
                pools = {name: self._r.get(f"resource:{name}") for name in task.resources}
                if not all(pools[n].can_afford(amount) for n, amount in task.resources.items()):
                    continue  # take nothing; stay queued
                for name, amount in task.resources.items():
                    pools[name].take(amount)

                # 4. bind an agent and launch. Everything from here on can fail
                # — an unknown spec, an agent factory that is down, a runner
                # whose harness is unreachable — and by then the lease is
                # already taken. Releasing it is what stops one bad task from
                # permanently shrinking a pool; not re-raising is what stops it
                # from aborting the pass for every other queued task.
                # PUSH a record; the stack top is the binding
                agent = self._r.get("agent_mgr").instantiate(task.agent_spec, tid)
                task.push_execution(
                    agent_id=agent.id,
                    input_versions=self._pin_inputs(task),
                    output_versions=self._pin_outputs(task),
                )  # instantiate() bound agent.task_id; the agent fills agent.handoffs
                # INPUT_VALIDATING, not RUNNING: the task reaches RUNNING when
                # the runner calls `enter_phase`, which is the point of the two
                # statuses.
                self._move(tid, TaskStatus.INPUT_VALIDATING)  # _move persists both
                self._watch(task)
                self._r.get("runner").start(task, agent, on_done=self.on_task_done)
            except Exception:
                self._abort_launch(tid, task, pools)

    def _pin_inputs(self, task: Task) -> dict:
        """Record which **store** version of each input this attempt consumes.

        **The counterpart of `_pin_outputs`, and it has to be in the same
        currency**, because the two fields are read as one. `env_mgr.grants`
        merges them —

            versions = dict(execution.input_versions)
            versions.update(execution.output_versions)

        — and resolves every entry under ``<store>/<hid>/v<N>/``
        (`env_mgr/grants.py::_versions`, `:60`). `output_versions` is a store
        directory version by construction, so `input_versions` had to be one
        too, and it was not: this read `handoff_mgr.latest(hid).version`, which
        is `HandoffMgr`'s **slot** version. One field, two currencies, merged
        into one dictionary.

        **Measured, on the first full `examples/demo2` run that reached the end**
        (`scratch/demo2-2026-08/runs/full3.log`). Handoff `52c75d0a`, kind
        `scores`:

        | | |
        |---|---|
        | store `v0` | a hole — no manifest |
        | store `v1` | published |
        | `grade`, a non-leaf | pinned store v0 and never wrote it |
        | `score`, its end entry | wrote store v1 |
        | `optimise` | `input_versions: 0` — **the slot number** |

        `validator.PhaseRunner._targets` handed that 0 to `store.read_verdicts`
        and the input phase died on *"cannot read verdicts of 52c75d0a v0: it is
        not published (published: [1])"*, with thirteen of fourteen tasks already
        succeeded. `interfaces.md` §5.12 names the two counters and says the
        reference between them has no owner; this is the third time in one stage
        that a caller has spent one as the other.

        They agree whenever every handoff is dispatched exactly once, which is
        every graph this repository had before a non-leaf declared an output —
        so `examples/demo/` cannot show it and neither can any fixture built to
        its shape.

        **`store.latest`, not `list_versions()[-1]`**: it filters on the
        manifest, so a hole an earlier attempt left is invisible and a concurrent
        retry's unsealed directory cannot be selected. An input with nothing
        published contributes **no entry** rather than a zero — the same silence
        `handoff_mgr.latest(hid) is None` produced before, and for the same
        reason: a task whose input does not exist is not ready, and inventing a
        version here would say it is.
        """
        if "handoff_store" not in self._r:
            # **No store, no store versions.** `interfaces.md` §2.4 permits a
            # system assembled without one, and every `tests/task_graph` fixture
            # is: 164 of them failed on `no component registered as
            # 'handoff_store'` before this guard. In that configuration nothing
            # downstream resolves a `<store>/<hid>/v<N>/` path either, so the
            # slot number is both the only answer available and a harmless one.
            # Stated rather than silently defaulted, because the two branches
            # return different currencies and a reader has to know which.
            handoff_mgr = self._r.get("handoff_mgr")
            return {
                hid: version.version
                for hid in task.inputs
                if (version := handoff_mgr.latest(hid)) is not None
            }
        store = self._r.get("handoff_store")
        pinned: dict = {}
        for hid in task.inputs:
            version = store.latest(hid)
            if version is not None:
                pinned[hid] = version
        return pinned

    def _pin_outputs(self, task: Task) -> dict:
        """Reserve a store version per declared output — `interfaces.md` §4.14.

        `env_mgr`'s kind-named write grant takes ``N`` off the `Execution` and
        resolves it **under** `<store>/<hid>/v<N>/` — `content/` and, for a
        write, `claim/`, but deliberately not the version directory itself
        (`env_mgr/grants.py::_version_paths`): the manifest is the seal, so an
        agent granted `v<N>/` could publish its own unsealed version.

        What this method owes that seam is only the number. Allocating it at
        dispatch is what makes the directory exist and be granted **before the
        body runs**, so the agent writes its output from inside its own grant
        and the store's `seal` becomes a commit rather than the write.

        **Not `handoff_mgr`, and the distinction is the whole of §5.12.** There
        are two version numbers for one artefact: `HandoffMgr`'s *slot* version
        and the store's *directory* version. The grant path is built from the
        second, so the second is what belongs in this field. Deriving it from
        `HandoffMgr.latest` — which is what close used to do — would grant a
        retry the version the previous attempt already wrote, overwriting an
        artefact criterion 16 promises is byte-identical forever.

        **The two counters advance on different events, and that is measured**
        (`scratch/impl-2026-08/task_graph/probe_slot_vs_store_version.py`):

        | | advances on |
        |---|---|
        | store version | **every dispatch** — this method runs unconditionally |
        | slot version | **every agent write**, from inside the run |

        (The slot-side verb is deliberately not named here: `test_authority.py`
        greps this module's source for it, docstrings included, and a mention in
        prose is indistinguishable from a call to a substring check.)

        ```
        attempt 0 dispatched   store_pin=0  slot=0     attempt 1 dispatched  store_pin=1  slot=0
        after produce          store_pin=0  slot=0     after produce #2      store_pin=1  slot=1
        attempt 2 dispatched   store_pin=2  slot=1   <- diverged
        ```

        So they diverge at **the first dispatch that does not write**, not at
        the first retry — an earlier revision of this docstring said the latter
        and it was a code read stated as a fact. A dispatch that never produces
        is the ordinary case rather than the exceptional one, because the
        version must exist before `env_mgr.prepare` resolves the grant and a
        refused `prepare` is *"no isolation, no start"* (`env_mgr` §4.6). Every
        refused dispatch therefore leaves a hole, and a hole is inert: `handoff`
        filters `list_versions` on the manifest, and allocation scans **all**
        directories so a hole is never reused.

        **Criterion 14 still holds, and this is why.** It is about `HandoffMgr`:
        no slot transition and no `persist` from a scheduler frame. This
        allocates a directory in `handoff`'s store — the same standing the
        already-sanctioned `declare` has, which likewise creates state the
        scheduler will not judge. Nothing here reads or writes a slot.

        **An absent `handoff_store` pins nothing, and is a supported mode
        rather than a guard.** `bootstrap.py:214` registers the name only when
        a root was supplied — deliberately, so that "an artefact store rooted
        at a default nobody chose" cannot happen — and `tests/task_graph` runs
        entirely in that mode against `FakeRunner`. With no store there is no
        directory to grant, so there is no version to pin and an empty map is
        the honest answer.
        """
        if "handoff_store" not in self._r:
            if task.outputs:
                # **A log, and settled as one** — `monitor` ruled it, and not
                # for the reason I proposed. I offered "the record is per
                # attempt and a composition fault is not an attempt fact"; they
                # rejected it as a true-sounding sentence on a false premise.
                # The record is *not* per attempt — `NO_TASK` exists for
                # run-level facts and `THREAD_DIED` / `LOOP_STALLED` both use
                # it. A per-task record is wrong here for a mechanical reason
                # instead: `default_fingerprint` includes the task id, so N
                # tasks give N fingerprints for one cause and grouping becomes
                # a no-op exactly where it is needed.
                #
                # This states the run-level **cause** once. The per-task
                # **consequence** belongs in the record and already has a host.
                #
                # **It cannot raise.** The storeless mode is real —
                # `tests/task_graph` runs entirely in it — so the honest empty
                # map stays. What was wrong is that it was *silent*:
                # `bootstrap.py:216` deliberately leaves the name unregistered
                # so the first resolution is a loud `KeyError`, and this early
                # return is the first of **three** tolerant readers that turn
                # that loudness back into nothing. `agent._seal_outputs` skips
                # an output with no pinned version, and `agent._gate`
                # (`runner.py:1066`) returns `[]` when there is no store.
                #
                # **So the task succeeds.** `_main` is
                # `if not failures: return self._report_planned()`, and with no
                # gate failures there is nothing to report. I had written that
                # the gate reports `OUTPUT_ABSENT` naming the wrong cause;
                # `monitor` measured it and there is no report at all. Three
                # locally-justified skips in series turn a task that published
                # none of its declared outputs into a success.
                log.warning(
                    "task %s declares %d output(s) and no handoff_store is registered, "
                    "so nothing was pinned for this attempt; no output can be published "
                    "and the task will nevertheless succeed, because the gate is skipped "
                    "for want of the same store",
                    task.id,
                    len(task.outputs),
                )
            return {}
        store = self._r.get("handoff_store")
        return {hid: store.allocate(hid) for hid in task.outputs}

    # -------------------------------------------------------------- recovery

    def resume_system(self) -> None:
        """Rebuild the index and demote runs the restart interrupted."""
        with self._lock:
            # `OrderedIdSet`, not `set()`. A plain set after every restart
            # destroys promotion order and `DepthFirstPolicy` silently degrades
            # to whatever iteration order the set happens to give — on exactly
            # the path where nobody is watching.
            self.pools = {s: OrderedIdSet() for s in TaskStatus}
            for task in self._r.get("task_mgr").all():
                # All three lease-holding phase states demote identically: the
                # lease is gone in each.
                status = TaskStatus.WAITING_RESOURCE if task.status in PHASES else task.status
                if status is TaskStatus.STOPPING:
                    status = TaskStatus.SUSPENDED  # the runner it waited on is gone
                self._move(task.id, status)
            # Eligibility is not restored, it is recomputed — which is why
            # HandoffMgr must have resumed first.
            self.try_dispatch()

    # ------------------------------------------------------------- internals

    def _move(self, tid: TaskId, status: TaskStatus) -> None:
        """The single writer. Nothing else assigns task.status or writes pools.

        **A task's position changes only when its pool changes.** With a `set`
        the discard-then-add was genuinely idempotent and position meant
        nothing; with an ordered pool it moves the task to the end — and step 1
        of every dispatch pass calls this on every waiting task, so without the
        early return the promotion order would be destroyed on the pass after it
        was established, silently, with no test failing unless one specifically
        checks order across two passes.
        """
        if tid in self.pools[status]:
            # Still self-healing: every *other* pool is swept, so a stale entry
            # cannot survive. What is skipped is only the re-add, which is what
            # would move the task to the end of its own pool.
            for name, pool in self.pools.items():
                if name is not status:
                    pool.discard(tid)
            return self._sync(tid, status)
        # Discarding from every pool rather than from the recorded status makes
        # this idempotent and self-healing.
        for pool in self.pools.values():
            pool.discard(tid)
        self.pools[status].add(tid)  # appended: promotion order
        self._sync(tid, status)

    def _sync(self, tid: TaskId, status: TaskStatus) -> None:
        """Reconcile the stored status and persist. The pool write is skipped."""
        task_mgr = self._r.get("task_mgr")
        task = task_mgr.get(tid)
        if task.status is not status:
            task.status = status
            task_mgr.persist(tid)

    # --------------------------------------------------------------- cascade

    def cascade_cancel(
        self, tid: TaskId, reason: str = "", *, include_self: bool = True
    ) -> CascadeReport:
        """Enqueue a cancel and drain it. Called by `Task.cancel`, never direct.

        `include_self=False` is `replace_with`'s entrance: it cancels this
        graph's downstream and leaves the task itself alone, because the task
        may be terminal and a terminal task is not cancellable.
        """
        with self._lock:
            task_mgr = self._r.get("task_mgr")
            if include_self:
                self._cascade.append((tid, reason))
            else:
                for consumer in task_mgr.consumers_of_outputs(tid):
                    if self._same_graph(consumer, task_mgr.get(tid)):
                        self._cascade.append((consumer.id, reason))
            report = self._drain_cascade()
            self.try_dispatch()
            return report

    def _drain_cascade(self) -> CascadeReport:
        """Level by level, which is `popleft`.

        Recursion produces depth-first order, which is observably different on
        any graph that is not a chain — and the spec says "level by level", so
        this is not a free choice between two equal options.
        """
        task_mgr = self._r.get("task_mgr")
        seen: set[TaskId] = set()
        reached: list[tuple[TaskId, str]] = []
        refused: list[tuple[TaskId, TaskStatus]] = []
        while self._cascade:
            tid, reason = self._cascade.popleft()
            if tid in seen:
                continue  # a diamond, not an error — and without this it raises
            seen.add(tid)
            try:
                task = task_mgr.get(tid)  # re-read; the index finds candidates only
            except KeyError:
                continue
            if task.status not in WAITING:
                refused.append((tid, task.status))
                continue
            self._move(tid, TaskStatus.CANCELLED)
            reached.append((tid, reason))  # the reason travels in the report
            for consumer in task_mgr.consumers_of_outputs(tid):
                if self._same_graph(consumer, task):
                    self._cascade.append((consumer.id, f"upstream {tid} cancelled"))
        return CascadeReport(reached=tuple(reached), refused=tuple(refused))

    @staticmethod
    def _same_graph(consumer: Task, task: Task) -> bool:
        """Within this graph, and no further — criterion 49.

        Two tasks share a graph when they share a parent. A task in another
        subgraph consuming the same handoff kind is a different graph and is
        untouched; that is the boundary `check_graph`'s check 2 keeps honest.
        """
        return consumer.parent == task.parent

    def _abort_launch(self, tid: TaskId, task: Task, pools: dict) -> None:
        """Undo a dispatch that raised between `take` and a live runner.

        Returns the whole reservation — nothing ran, so a consumable spent
        nothing — closes the half-open attempt if one was pushed, and parks the
        task in FAILED. FAILED rather than back in the queue: the pass would
        pick it up again on the next iteration and fail identically, forever.
        An operator resumes it once the cause is fixed.

        Every step is individually guarded. This is the handler, so an exception
        escaping *here* has nowhere left to go: it would propagate out of
        `submit` or `resume_all` leaving the task RUNNING with a half-open
        attempt, which nothing but a restart can clear. Reaching FAILED matters
        more than any single step succeeding.
        """
        # `pools` is empty when resolution itself failed — nothing was taken.
        for name, amount in task.resources.items():
            pool = pools.get(name)
            if pool is None:
                continue
            try:
                pool.give_back(amount, actual=0.0)
            except Exception:
                log.exception("%s: could not release %s of %r", tid, amount, name)
        try:
            if task.is_running:
                task.close_execution(TaskStatus.FAILED, detail="failed to launch")
        except Exception:
            log.exception("%s: could not close the half-open attempt", tid)
        log.exception("%s: failed to launch; the reservation was released", tid)
        self._move(tid, TaskStatus.FAILED)

    def _watch(self, task: Task) -> None:
        """Tell this task's monitor to watch it — `interfaces.md` §2.1 rev. 5.

        **At dispatch, not at birth**, and the reason is not convenience: this is
        where a task first has a phase to advance. A `WAITING_HANDOFF` task has
        no planned advance to make and no agent to poll for a stall, so watching
        it would put an entry in the monitor's set that its loop can do nothing
        with. Spec §3.5's "every task has a monitor" is satisfied by every task
        that *runs*.

        Measured, and it is what rules out the two birth sites: `TaskMgr.add`
        fires on **none** of the tasks after a restart, because `resume_system`
        rebuilds the collection from the store and cannot go through `add` —
        that is the new-task path and it raises on a duplicate id. Dispatch
        covers recovery for free, because resume re-dispatches.

        `monitor` owns resolving `Task.monitor_spec`, including the default name
        and the message naming an unregistered one, so the resolver is asked for
        by name rather than reimplemented here. A name that will not resolve
        raises **into the launch guard**, which parks the task in `FAILED` — an
        unwatchable task is one that would never advance a phase, so it fails
        loudly at dispatch rather than stalling silently forever.
        """
        if "monitor_for" not in self._r:
            return  # `monitor` is declaration-only; nothing watches anything yet
        self._r.get("monitor_for")(task, self._r).set_task(task.id)

    def _landing(self, task: Task) -> TaskStatus:
        return TaskStatus.WAITING_RESOURCE if self._ready(task) else TaskStatus.WAITING_HANDOFF

    def _ready(self, task: Task) -> bool:
        """The query that replaces a dependency counter."""
        handoff_mgr = self._r.get("handoff_mgr")
        return all(handoff_mgr.check_if_latest_valid(hid) for hid in task.inputs)

    def _release(self, task: Task, usage: dict[str, float] | None) -> None:
        """Give every lease back. One pool failing must not strand the rest.

        The task is finishing either way — the run is over and the scheduler
        cannot un-finish it. Letting an exception escape here would leave the
        task `RUNNING` with an open execution record and its *other* leases
        still held, recoverable only by a restart. Failing loudly per pool and
        continuing is strictly better.
        """
        for name, amount in task.resources.items():
            actual = None if usage is None else usage.get(name)
            try:
                self._r.get(f"resource:{name}").give_back(amount, actual)
            except Exception:
                log.exception("%s: could not release %s of %r", task.id, amount, name)

        # A non-leaf acquires nothing (criterion 53), so its validation phases'
        # spend arrives against a pool it never declared. `give_back` cannot
        # record it — the clamp is `min(actual, amount)` and `amount` is zero —
        # so it is charged instead. A renewable has no equivalent and must not
        # get one: "spend" is not a concept a renewable has.
        for name, spent in (usage or {}).items():
            if name in task.resources:
                continue
            pool = self._r.get(f"resource:{name}") if f"resource:{name}" in self._r else None
            charge = getattr(pool, "charge", None)
            if charge is None:
                log.warning(
                    "%s: usage names %r, which the task did not declare and which "
                    "cannot record unreserved spend; %s is not booked",
                    task.id,
                    name,
                    spent,
                )
                continue
            try:
                charge(spent)
            except Exception:
                log.exception("%s: could not charge %s to %r", task.id, spent, name)

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
