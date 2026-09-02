"""The loop, the two handlers, the scope guard, and liveness — design §5–§7.

Everything both kinds of monitor share. `decide` is the one method they differ
in; `_advance` is the one method **no subclass may replace**, which is how spec
§2.2's "the planned channel is program, always" becomes structural rather than a
convention someone has to keep.
"""

from __future__ import annotations

import threading
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from task_graph.ids import TaskId
from task_graph.registry import Registry

from .buffer import ExceptionBuffer, PlannedQueue, Unit
from .protocols import PLANNED, EventKind, Pushable, ScopeViolation, UserSink
from .record import EventRecord, Recorder, event, rekeyed

__all__ = [
    "DEFAULT_MONITOR_NAME",
    "ESCALATION_TARGET",
    "LAST_PHASE",
    "NO_TASK",
    "PHASE_ORDER",
    "PUSH_MESSAGE",
    "TARGET_USER",
    "BaseMonitor",
    "Decision",
    "Escalate",
    "GiveUp",
    "NoNextPhase",
    "NullUserSink",
    "Push",
    "ReportToUser",
    "RunningMonitors",
    "check_liveness",
    "install_excepthook",
    "monitor_for",
    "next_phase",
    "reached_the_user",
    "start_monitors",
]

#: A task whose `monitor_spec` is `None` is watched by this one. `task_graph`
#: design §3.8: a name, like every other collaborator.
DEFAULT_MONITOR_NAME = "default"

#: The alpha's whole reaction, content-free. Spec §7.
PUSH_MESSAGE = "continue, do it until finished"

#: What a record carries when the event is not about any one task — a thread
#: that died outside a task, a monitor whose loop stalled. Records need a
#: correlation id and these have none; a nil id says so in one readable value.
NO_TASK = TaskId("00000000-0000-0000-0000-000000000000")

#: The phase sequence, by **name** rather than by member. `task_graph` design
#: §3.2's `PHASES` is the same tuple; resolving it by name against the enum the
#: status itself came from is `engineer_principle.md` §1's "depend on names, not
#: on imports" — and it is what lets this function be tested before
#: `TaskStatus` grows the two phase members.
PHASE_ORDER = ("INPUT_VALIDATING", "RUNNING", "OUTPUT_VALIDATING")

#: The phase whose `PHASE_DONE` is a completion rather than an advance. Named
#: rather than spelled `PHASE_ORDER[-1]` at the call site, because the two uses
#: — the last entry of the order, and the end of the channel — are the same
#: fact and should move together if the order ever changes.
LAST_PHASE = PHASE_ORDER[-1]


class NoNextPhase(RuntimeError):
    """A planned advance arrived for a task that is not in a phase."""


def next_phase(status: Any) -> Any:
    """The successor of `status` in the fixed phase order.

    A mapping, not a computation: there is no policy here, no threshold and no
    model. Spec §2.2's "program, always" is this function being boring.
    """
    try:
        index = PHASE_ORDER.index(status.name)
    except ValueError:
        raise NoNextPhase(
            f"{status} is not a phase; the planned channel advances "
            f"{' -> '.join(PHASE_ORDER)} and nothing else"
        ) from None
    if index + 1 == len(PHASE_ORDER):
        raise NoNextPhase(f"{status} is the last phase; completion is not an advance")
    return type(status)[PHASE_ORDER[index + 1]]


def monitor_for(task: Any, registry: Registry) -> Any:
    """The monitor watching `task`, resolved by name (criteria 1 and 2).

    `None` takes the default. **An unregistered name is rejected with the
    offending value named** — the wrapper exists only to add the task, since
    `Registry.get` already names the key it could not find.

    **`task.monitor_spec` is read directly, and that is a fix rather than a
    style.** It was `getattr(task, "monitor_spec", None)`, whose default is
    **unreachable** — `Task` declares the field and sets `extra="forbid"`, so no
    real task can lack it. What the default could do is absorb a *rename*: the
    field goes, every task silently resolves to the global default monitor, and
    nothing anywhere fails. That is `interfaces.md` §4.11 — a check that reports
    nothing being indistinguishable from a check that found nothing — and it is
    the same defect removed from `pusher.live_handle` the same day, left
    standing here because a guard one has defended is harder to see than one
    one has not.

    The `getattr` in the error message below **stays, and is examined rather
    than overlooked**: an exception formatter must be total, and an
    `AttributeError` raised there would replace the `KeyError` and lose the one
    thing worth reporting — which monitor name could not be resolved.
    """
    name = task.monitor_spec or DEFAULT_MONITOR_NAME
    key = f"monitor:{name}"
    try:
        return registry.get(key)
    except KeyError:
        raise KeyError(
            f"task {getattr(task, 'id', '?')} names monitor_spec {name!r}, "
            f"which is not registered as {key!r}"
        ) from None


#: The `attributes` keys this module **writes and another package reads**.
#:
#: Declared because `interfaces.md` §1.2 makes a name frozen the moment another
#: module names it — *"changing it breaks somebody who is not you"* — and `demo`
#: names this one to tell a task resting at the top of an escalation chain from
#: a task that is stuck. It was documented as behaviour in design §7.3 and
#: declared nowhere, which is the same gap `Attempt` and `AttemptRunner` closed
#: from the other direction: that was what this module *requires*, this is what
#: it *emits*.
ESCALATION_TARGET = "target"
TARGET_USER = "user"


def reached_the_user(record: EventRecord) -> bool:
    """Whether this record is an escalation that **ran out of task tree**.

    The question `demo` needs and had to infer. Their stall detector is a
    heuristic over absence of change, and an escalation resting at the root is
    not a stall — the system is doing exactly what spec §11 says it does. Both
    present as *a task in `running` that stopped changing*, so reporting one as
    the other is a check that reports nothing being indistinguishable from a
    check that found nothing.

    **Offered as a question rather than as the two strings it is made of.**
    `demo` was reading `attributes["target"] == "user"` — correct, documented in
    design §7.3, and a rename away from silently going false, which would put
    their stall detector back to calling a resting state a hang. That is
    `engineer_principle.md` §4.4: when a caller seems to need one of your
    properties, offer the computation instead.

    **It does not answer whether the run should end.** That is spec §11's, still
    open, and this is deliberately the narrower fact — *this escalation reached
    the top* — rather than a verdict about the task.
    """
    return (
        record.kind is EventKind.ESCALATED
        and record.attributes.get(ESCALATION_TARGET) == TARGET_USER
    )


# --------------------------------------------------------------------------- #
# Decisions
#
# `decide` and `_apply` are separate so that *deciding* and *doing* are
# separable: the pusher's decision is testable without a live agent, and the
# analysing dispatcher replaces `decide` and nothing else.


@dataclass(frozen=True)
class Push:
    """Tell a live agent to keep going. ~2 s, and it loses nothing."""

    handle: Pushable
    message: str = PUSH_MESSAGE


@dataclass(frozen=True)
class Escalate:
    """Hand it to the monitor of the parent task (spec §3.1)."""

    why: str


@dataclass(frozen=True)
class ReportToUser:
    """Bypass the chain. The decision is a human's."""

    why: str


@dataclass(frozen=True)
class GiveUp:
    """Nothing in the alpha's action set applies. Recorded, never silent."""

    why: str


Decision = Push | Escalate | ReportToUser | GiveUp


class NullUserSink:
    """The top of the escalation chain, with nothing behind it.

    **How a monitor reaches a human is unspecified anywhere in this system**
    (spec §11), and inventing a channel here would be adding a requirement. The
    arrival is recorded by the monitor before the sink is called, so a branch
    that outruns every monitor is visible rather than silent.
    """

    def __init__(self) -> None:
        self.delivered: list[tuple[EventRecord, str]] = []

    def deliver(self, record: EventRecord, why: str) -> None:
        self.delivered.append((record, why))


# --------------------------------------------------------------------------- #
# Liveness — spec §5.4


def install_excepthook(
    recorder: Recorder,
    sink: UserSink,
    *,
    chain: bool = True,
) -> Callable[[Any], None]:
    """Turn an escaped thread exception into a record and surface it.

    **Measured, not assumed** (`scratch/design/probes-monitor/p4_thread_death.py`,
    re-run on 3.13.13): a `threading.Thread` whose target raises prints a
    traceback to stderr and dies; the process keeps running, the exit code does
    not change, and producers see no error — further reports are accepted and
    queue up behind a consumer that no longer exists.

    **Process-global, and therefore the composition root's, not a constructor's.**
    A module installing this on construction would be a library mutating
    interpreter state, and two monitors would fight over it. Installed once, for
    every thread — any thread that dies of an exception nobody caught is a fact
    the user must be able to see.

    Returns the hook it replaced, so a caller can restore it.
    """
    previous = threading.excepthook

    def hook(args: Any) -> None:
        exc = args.exc_value
        record = event(
            EventKind.THREAD_DIED,
            # Whoever spawned the thread may attribute it by setting `task_id`
            # on the Thread object; nothing is inferred when they did not.
            getattr(args.thread, "task_id", NO_TASK),
            reported_by=getattr(args.thread, "name", "?"),
            exception_type=getattr(args.exc_type, "__name__", str(args.exc_type)),
            exception_message=str(exc),
            exception_stacktrace="".join(
                traceback.format_exception(args.exc_type, exc, args.exc_traceback)
            ),
            severity=17,  # OTel ERROR: nothing recovered from this
        )
        try:
            recorder.write(record)
        finally:
            sink.deliver(record, "an exception escaped a thread")
            if chain:
                previous(args)

    threading.excepthook = hook
    return previous


class RunningMonitors:
    """The threads `start_monitors` took, and the one verb that gives them back.

    Returned rather than left implicit so that stopping is *possible* to get
    right: `stop()` closes every queue, waits for every loop to return, and
    reports which did not. A caller that forgets it leaks daemon threads; a
    caller that half-remembers it used to have to know that `stop` and `join`
    are two steps and which order they go in.
    """

    def __init__(self, monitors: Sequence[Any], threads: Sequence[threading.Thread]) -> None:
        self.monitors = tuple(monitors)
        self._threads = tuple(threads)

    def stop(self, timeout: float = 5.0) -> list[str]:
        """Stop every loop and wait. Returns the names that did not return."""
        for monitor in self.monitors:
            monitor.stop()
        for thread in self._threads:
            thread.join(timeout)
        return [m.name for m, t in zip(self.monitors, self._threads) if t.is_alive()]

    def __len__(self) -> int:
        return len(self.monitors)


def start_monitors(registry: Registry) -> RunningMonitors:
    """Give every registered monitor a thread. **Call this once, from the entry
    point**, and `stop()` the result when the run ends.

    **Not from `build_registry`, and not from a constructor.** A thread is a
    process-level decision and a library that spawns one has taken it on the
    owner's behalf — the same argument that puts `install_excepthook` in the
    entry point rather than in `BaseMonitor.__init__`, and `interfaces.md` §5.9's
    shape.

    **But the assembly is this module's, not the caller's.** Resolving
    `monitor:*`, spawning a daemon thread each, and remembering that stopping is
    `stop()` *then* `join()` is four steps an entry point would otherwise get
    right or wrong on its own — and `demo` found what wrong looks like: the loop
    is never started, `report()` still accepts, the queue still fills, and **the
    task never advances a phase.** That is the failure `interfaces.md` §2.1 rev. 4
    already names for an unresolvable monitor name, reached from the other
    direction.

    **A monitor that was never started is detectable and nothing detects it.**
    `last_beat` is stamped at construction and only moved by the loop, so
    `check_liveness` reports a never-started monitor as stalled after
    `threshold` periods. An entry point that calls neither this nor that has a
    system which stops silently; one that calls this has a running system; one
    that calls both has a running system that says when it stops.
    """
    monitors = list(registry.resolve("monitor:*"))
    threads = []
    for monitor in monitors:
        thread = threading.Thread(
            target=monitor.mainloop, name=f"monitor-{monitor.name}", daemon=True
        )
        thread.start()
        threads.append(thread)
    return RunningMonitors(monitors, threads)


def check_liveness(
    monitors: Any,
    *,
    period: float,
    threshold: int = 3,
    now: Callable[[], float] = time.monotonic,
) -> list[EventRecord]:
    """One `LOOP_STALLED` per monitor whose beat is older than `threshold`
    periods. Pure: takes a clock, returns records, writes nothing.

    **`threshold` consecutive periods, not one** — the `failureThreshold` shape,
    because one slow round is not a death.

    **The checker is a comparison of one float**, called from the main thread,
    which is already sitting there. That is what makes this an answer rather than
    an infinite regress, and it is deliberately the cheapest of the three answers
    the prior art gives: s6 makes its top-level supervisor unable to fail,
    systemd hands the problem to a hardware watchdog, Ray hands it outward to
    KubeRay.

    `time.monotonic`, never wall-clock: a clock adjustment must not read as a
    stalled monitor.
    """
    stale = threshold * period
    at = now()
    out = []
    for monitor in monitors:
        age = at - monitor.last_beat
        if age > stale:
            out.append(
                event(
                    EventKind.LOOP_STALLED,
                    NO_TASK,
                    reported_by=monitor.name,
                    severity=17,
                    attributes={"monitor": monitor.name, "stale_for": age, "threshold": threshold},
                )
            )
    return out


# --------------------------------------------------------------------------- #
# The monitor


@dataclass
class _Watch:
    """What `set_task` accumulates. Insertion-ordered, which is the round-robin
    a global monitor's FIFO already is."""

    ids: list[TaskId] = field(default_factory=list)

    def add(self, task_id: TaskId) -> None:
        if task_id not in self.ids:
            self.ids.append(task_id)

    def __contains__(self, task_id: object) -> bool:
        return task_id in self.ids


class BaseMonitor:
    """A task's event loop, on two channels.

    **Two kinds, one class** (spec §3): the agent-bearing monitor is not a
    different module, it is a different body behind `decide`. **Per-task and
    global are one class as well** — a global monitor is one whose `set_task` has
    been called more than once, and nothing in the loop cares how many ids are in
    the watch set.
    """

    def __init__(
        self,
        name: str,
        registry: Registry,
        *,
        period: float = 1.0,
        sink: UserSink | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self._r = registry
        self._period = period
        self._sink: UserSink = sink if sink is not None else NullUserSink()
        self._clock = clock
        self._watch = _Watch()
        self._planned = PlannedQueue()
        self._buffer = ExceptionBuffer()
        self._current: TaskId | None = None
        self._stopping = False
        self.last_beat = clock()
        self.sweeps = 0

    # ---- collaborators, resolved by name at use time -----------------------
    #
    # Never through a constructor: that is what lets a test swap an
    # implementation after the system is wired, and it is what keeps the import
    # graph acyclic.

    @property
    def _recorder(self) -> Recorder:
        return Recorder(self._r.get("store_mgr"))

    @property
    def _task_mgr(self) -> Any:
        return self._r.get("task_mgr")

    @property
    def _runner(self) -> Any:
        return self._r.get("runner")

    # ---- the interface -----------------------------------------------------

    def set_task(self, task_id: TaskId) -> None:
        """Take this task under watch.

        **The only way a monitor learns what it watches** (criterion 10): it is
        told, and it does not go looking. Nothing else in this class writes the
        watch set — in particular `report` does not, so an event for an unwatched
        task cannot smuggle one in.
        """
        self._watch.add(task_id)

    def watches(self, task_id: TaskId) -> bool:
        return task_id in self._watch

    def report(self, record: EventRecord) -> None:
        """Persist synchronously, then enqueue. Three statements, in that order.

        **The routing is here and nowhere else** — one `if` over one frozenset at
        the single inbound call, which is what makes spec §5's "a reporter never
        picks a queue" structural. A reporter that had to choose the door would
        have to classify the event before reporting it, and classification is the
        thing this module exists to own.

        **A `write` that raises propagates and does not fall through to `add`.**
        A record that is not durable must not become queued work: then the buffer
        holds the only copy of a fact, and rule 3 is a comment rather than an
        invariant.
        """
        self._recorder.write(record)
        if record.kind in PLANNED:
            self._planned.add(record)
        else:
            self._buffer.add(record)

    def mainloop(self) -> None:
        """Drain both queues, planned first. Its own thread, never an agent's.

        **Planned work is taken first and without waiting.** A task waiting to
        advance is a task doing nothing, and a decision can block on the
        scheduler's `RLock` while a transition cannot meaningfully be deferred.
        **It cannot starve the buffer**: a planned advance is a fixed
        non-blocking transition, so the planned queue drains to empty in bounded
        time and the loop reaches `buffer.get` on the next round.
        """
        while True:
            self._beat()  # before either, so a round spent on planned work counts

            record = self._planned.get_nowait()
            if record is not None:
                self._run_guarded(record.task_id, self._advance, record)
                continue

            unit = self._buffer.get(self._period)
            if unit is None:
                if self._stopping:
                    return
                self._sweep()
                continue
            self._run_guarded(unit.task_id, self._handle, unit, release=True)

    def stop(self) -> None:
        """Refuse new reports loudly, drain what is queued, then let the loop
        return. `queue.Queue.shutdown`'s `immediate=True` discards pending items
        silently and is 3.13-only besides; this is the well-behaved variant."""
        self._stopping = True
        self._planned.shutdown()
        self._buffer.shutdown()

    # ---- what a subclass supplies ------------------------------------------

    def decide(self, unit: Unit) -> Decision:
        """What to do about this unit. The one method the two kinds differ in."""
        raise NotImplementedError

    # ---- the loop's internals ----------------------------------------------

    def _run_guarded(
        self,
        task_id: TaskId,
        work: Callable[..., None],
        *args: Any,
        release: bool = False,
    ) -> None:
        """The one place the broad catch and the `_current` bookkeeping live, so
        neither path can forget either.

        **The broad catch is deliberate and is criterion 9.** A handler that
        raises must not take the loop with it — measured: after an unguarded
        handler raises, the thread is dead, further reports are accepted, depth
        grows, and no producer sees an error (`probes-monitor/p4`).
        """
        self._current = task_id
        try:
            work(*args)
        except Exception as exc:
            self._record_handling_failed(task_id, exc)
        finally:
            self._current = None
            if release:
                self._buffer.done(task_id)

    def _record_handling_failed(self, task_id: TaskId, exc: BaseException) -> None:
        try:
            self._recorder.write(
                event(
                    EventKind.HANDLING_FAILED,
                    task_id,
                    reported_by=self.name,
                    exception_type=type(exc).__name__,
                    exception_message=str(exc),
                    exception_stacktrace="".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    ),
                    severity=17,
                )
            )
        except Exception:
            # The recorder itself failed. There is nowhere left to put this, and
            # taking the loop down would turn one lost record into every
            # subsequent task never advancing.
            traceback.print_exc()

    def _beat(self) -> None:
        """Stamp `last_beat`. Once per round, before any work."""
        self.last_beat = self._clock()

    def _sweep(self) -> None:
        """Called once per idle period. **Does nothing, and that is the point.**

        The alpha builds no poller for the never-returning agent: a wedged agent
        is a wedged thread and Python cannot kill a thread, so detection would
        arrive without remedy. The *period* and the *hook* are what make adding
        one later an edit instead of a refactor (spec §4.3).
        """
        self.sweeps += 1

    # ---- the planned channel — no subclass may replace this ----------------

    def _advance(self, record: EventRecord) -> None:
        """One transition, then wake or resume. Program, always.

        Three lines of behaviour and no branch on anything but liveness. There is
        no policy here, no threshold and no model — which is criterion 19 by
        construction rather than by inspecting a prompt: an `AnalysingMonitor`
        inherits this unchanged.
        """
        if record.kind is EventKind.SUBGRAPH_DONE and "from_task" not in record.attributes:
            # It is mine, and it is addressed upward: the is_end subtask's
            # monitor walks the tree. A record that already carries `from_task`
            # has had that hop and names the task to advance.
            self._notify_parent_done(record)
            return

        task = self._task_mgr.get(record.task_id)

        if record.kind is EventKind.PHASE_DONE and record.attributes.get("phase") == LAST_PHASE:
            # **The last phase finishing is a completion, and completion is not
            # an advance.** `PHASE_ORDER` has three entries and therefore two
            # advances, but `agent`'s `_close` reports a third `PHASE_DONE` and
            # then completes the task itself. Without this, `next_phase` refuses
            # — correctly — and `_run_guarded` turns the refusal into a
            # persisted `HANDLING_FAILED` **for every successful task**, in the
            # store this module owns (`demo-2`, on the first task ever to
            # succeed; `probes-monitor/p14`).
            #
            # **The report is not the defect and must not be suppressed**: it
            # carries the final phase's validator evidence, and `report()` makes
            # it durable before the queue sees it. Only the transition has
            # nowhere to go.
            #
            # **Keyed on the record, not on `task.status`.** The status is read
            # after the fact and `on_done` may already have moved it, which is
            # why the old failure alternated between two different refusals.
            # `phase` is `TaskStatus.name`, set before `on_done` runs — and
            # `.name` because `PHASE_ORDER` is name-keyed, as `next_phase`'s own
            # `PHASE_ORDER.index(status.name)` says. `.value` would never match,
            # and the branch would be dead while looking applied.
            #
            # **Scoped to `PHASE_DONE` deliberately.** `_notify_parent_done`
            # rekeys to `SUBGRAPH_DONE` and copies the child's attributes, so a
            # non-leaf re-entry can carry a `phase` belonging to another task.
            # Matching on the kind keeps that record on the advancing path.
            #
            # An absent `phase` falls through and still refuses loudly. That is
            # the old behaviour, kept: a reporter that omits it is not granted a
            # benign default.
            #
            # **And this is where the subgraph's completion is announced —
            # criterion 24's producing half, which had never been built.**
            # `SUBGRAPH_DONE` was declared, consumed at the top of this method,
            # and re-emitted by `_notify_parent_done`; **nothing anywhere
            # created the first one**, so that call was a relay with no source.
            # A non-leaf's `_main` ends its thread at `unfold` with the task in
            # `RUNNING`, and the re-entry is the only thing that moves it — so
            # every non-leaf, the root included, sat in `main: running` for
            # ever and no run of the demo ever terminated cleanly.
            #
            # **The trigger belongs here and nowhere else**: a subgraph is
            # finished exactly when its `is_end` subtask has run out of phases,
            # which is this branch. Reporting it from the runner would make it
            # a second producer of a fact the monitor already holds, and
            # criterion 23 keeps the scheduler out of it — *"it never reads
            # `is_end`, never observes another task's status to decide this
            # one's"*. A monitor reading its **own** task's `is_end` is neither.
            #
            # `_notify_parent_done` returns on a `None` parent, so the root
            # announcing to nobody is already handled there.
            if task.is_end:
                self._notify_parent_done(record)
            return

        self._transition(task.id, "enter_phase", phase=next_phase(task.status))

        # **One verb, because the branch that used to be here was this module
        # reading a neighbour's property and computing with it.** `_advance`
        # read `is_running` for exactly one purpose — choosing between `wake()`
        # and `resume()` — which is `engineer_principle.md` §3's stated symptom,
        # and §4.4's answer is to offer the computation instead of the parts.
        #
        # Design §6.1 argued the opposite, that collapsing them "would hide, at
        # the one place it matters, which of the two shapes a task is". That is
        # **withdrawn**: the branch never revealed the shape. It revealed thread
        # liveness, a *proxy* for leaf-versus-non-leaf — and it is that proxy
        # which was already wrong once, silently, when the absent-attempt branch
        # never fired for a non-leaf.
        #
        # `carry_on` returns what it did. Nothing here reads it: the record for
        # this event is already on disk, written by `report()` before the queue
        # saw it, and this module's store is append-only by design (§3.4), so
        # there is no record left to amend. Kept because the verb being
        # self-describing is what lets a test assert the two shapes without
        # reaching into the runner.
        self._runner.carry_on(task.id)

    def _notify_parent_done(self, record: EventRecord) -> None:
        """One hop of the escalation walk, with a different payload (§7.4).

        **This method is where `SUBGRAPH_DONE` is produced, so grepping the name
        will not find its producers — grep this method instead.** The rekey
        happens below; the callers pass a `PHASE_DONE`, so the constant appears
        nowhere at either call site. That blind spot is not hypothetical: it is
        the shape a producer would have had while `SUBGRAPH_DONE` had none, and
        it would have hidden one. (`task_graph` named it; the conclusion at the
        time was right because the relay genuinely had no source, but the
        instrument could not have shown otherwise.) **An absence needs an
        instrument that could have found a presence**, and for a kind that is
        produced by rekeying, the instrument is the call site of this method.

        The two callers, and they are the whole of criterion 24:

        - `_advance`, on a `SUBGRAPH_DONE` with no `from_task` — the walk
          continuing upward.
        - `_advance`, on the **completion branch** when the task `is_end` — the
          walk starting. This is the producer.

        **The subtask's monitor does not transition the parent.** It reports; the
        parent's monitor decides nothing and advances. That is criterion 24, and
        it is why the planned channel needs no exception to the scope guard.

        **`parent is None` returns rather than escalating.** The root of the task
        tree is the system whole task, and its `is_end` completing means the
        system finished — a completion, not something to surface. The two walks
        share a mechanism and differ at the top; this is where that shows.
        """
        task = self._task_mgr.get(record.task_id)
        if task.parent is None:
            return
        parent = self._task_mgr.get(task.parent)
        monitor_for(parent, self._r).report(rekeyed(record, parent.id, EventKind.SUBGRAPH_DONE))

    # ---- the unplanned channel ---------------------------------------------

    def _handle(self, unit: Unit) -> None:
        self._apply(self.decide(unit), unit)

    def _apply(self, decision: Decision, unit: Unit) -> None:
        """Carry out one decision. **An unrecognised one is an error** — nothing
        here may default to a benign outcome."""
        if isinstance(decision, Push):
            self._push(decision, unit)
        elif isinstance(decision, Escalate):
            self._escalate(unit, decision.why)
        elif isinstance(decision, ReportToUser):
            self._to_user(unit.newest, decision.why)
        elif isinstance(decision, GiveUp):
            self._recorder.write(
                event(
                    EventKind.MONITOR_GAVE_UP,
                    unit.task_id,
                    attempt=unit.newest.attempt,
                    reported_by=self.name,
                    severity=17,
                    attributes={"why": decision.why},
                )
            )
        else:
            raise TypeError(f"{decision!r} is not a Decision")

    def _push(self, decision: Push, unit: Unit) -> None:
        """Recorded **before** the call, so an `instruct` that raises still
        leaves the attempt visible (criterion 9)."""
        self._recorder.write(
            event(
                EventKind.PUSH_ATTEMPTED,
                unit.task_id,
                attempt=unit.newest.attempt,
                reported_by=self.name,
                attributes={"message": decision.message},
            )
        )
        decision.handle.instruct(decision.message)

    def _escalate(self, unit: Unit, why: str) -> None:
        """Up the **task** tree, never the monitor topology (spec §3.1).

        Global monitors are a flat pool; the tree that matters is `task_graph`'s,
        and `Task.parent` is the edge `unfold` sets. So the target is always *the
        monitor of my task's parent*, whichever kind either one is — and it
        always has at least the reporter's scope, because a parent's zone
        contains its children's.

        **The walk needs no visited set.** `unfold` sets `parent` on tasks it has
        just created and therefore cannot close a loop; a second guard for one
        invariant is the two-writers failure.
        """
        newest = unit.newest
        task = self._task_mgr.get(unit.task_id)
        self._recorder.write(
            event(
                EventKind.ESCALATED,
                task.id,
                attempt=newest.attempt,
                reported_by=self.name,
                attributes={"why": why},
            )
        )
        if task.parent is None:
            self._to_user(newest, why)
            return
        parent = self._task_mgr.get(task.parent)
        monitor_for(parent, self._r).report(rekeyed(newest, parent.id))

    def _to_user(self, record: EventRecord, why: str) -> None:
        arrival = event(
            EventKind.ESCALATED,
            record.task_id,
            attempt=record.attempt,
            reported_by=self.name,
            severity=17,
            attributes={"why": why, ESCALATION_TARGET: TARGET_USER},
        )
        self._recorder.write(arrival)
        self._sink.deliver(arrival, why)

    # ---- the scope guard — criterion 8 -------------------------------------

    def _transition(self, task_id: TaskId, verb: str, **kw: Any) -> None:
        """Every monitor action is a transition it **calls**, never a status it
        **assigns**. `verb` names a method on `Task`, every one of which routes
        through the scheduler's single writer under its `RLock`.

        Refusing a task other than the one being handled is **stricter than
        criterion 8 asks** — the criterion says "the task `set_task` gave it",
        and a global monitor was given several. The stricter form is what spec §6
        means by holding one task's scope at a time, and it is also **how rule 5
        spans both queues**: one task is in `_current` at a time regardless of
        which queue its work came from.
        """
        if task_id not in self._watch:
            raise ScopeViolation(
                f"{self.name} was never given {task_id} by set_task; it watches {self._watch.ids}"
            )
        if task_id != self._current:
            raise ScopeViolation(f"{self.name} is handling {self._current}, not {task_id}")
        getattr(self._task_mgr.get(task_id), verb)(**kw)
