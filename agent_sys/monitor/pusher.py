"""The alpha's decision function — design §6.2.

**A status check plus one phrase: *continue, do it until finished*.** That is the
whole reaction, and spec §4.1's definition is what makes it coherent — the agent
returned without delivering, so the corrective action is to tell it to keep
going.

**"Keep going" is expressible on a returned agent. Measured, not assumed.** A
`ResultMessage` ends a *turn*, not the session and not the process; a live probe
pushed a returned agent and got an answer on the same session id, in the same
process, in ~2 s (`scratch/design/findings-monitor-push.md` §1–2).
"""

from __future__ import annotations

from task_graph.ids import TaskId

from .base import BaseMonitor, Decision, Escalate, GiveUp, Push
from .buffer import Unit
from .protocols import AttemptRunner, EventKind, Pushable
from .record import EventRecord, event

__all__ = ["GATE_KINDS", "PusherMonitor"]

#: The three delivery failures of the completeness gate. A budget overrun is a
#: gate failure too and is deliberately not here: it is what *bounds* the loop.
GATE_KINDS = frozenset(
    {
        EventKind.OUTPUT_ABSENT,
        EventKind.OUTPUT_NOT_EXECUTABLE,
        EventKind.SELF_CHECK_UNSET,
    }
)

_TERMINAL = frozenset({EventKind.VALIDATION_FAILED, EventKind.VALIDATION_UNREACHED})


def live_handle(runner: AttemptRunner, task_id: TaskId) -> tuple[Pushable | None, str]:
    """The live agent handle for a task, and **why there is not one** when there
    is not.

    Returns a reason string beside the handle rather than a bare `None`,
    because `None` is **two different real situations** and this module's whole
    thesis is that two situations must not be one observation:

    | | what it means |
    |---|---|
    | no attempt at all | `Runner.stop` already removed it. The agent is gone |
    | an attempt whose `executor` is `None` | it has not reached its main phase — or it is a non-leaf, which never will, because `_main` releases before deploying one |

    **At a gate failure the second is not routine**, and that is why it is worth
    separating. The completeness gate runs at the *end* of the main phase, so an
    executor is normally set when a gate kind is reported; a non-leaf cannot
    report one at all. Flattening the two into "no live agent" would put a
    surprising state and an ordinary one behind one sentence in the record —
    the same shape as the `getattr(runner, "attempt_of", None)` that used to be
    here, one level in.

    That fallback was correct while `agent` was declaration-only and **outlived
    its reason the day `attempt_of` landed**: it made a renamed accessor
    indistinguishable from "no live agent", which is `interfaces.md` §4.11's
    first row — *a check that reports nothing is indistinguishable from a check
    that found nothing.* A missing accessor now raises, and `_run_guarded`
    records it as `HANDLING_FAILED`.
    """
    attempt = runner.attempt_of(task_id)
    if attempt is None:
        return None, "no attempt is live for this task; the agent is gone"
    if attempt.executor is None:
        return None, "the attempt holds no executor: it is not in its main phase"
    if not isinstance(attempt.executor, Pushable):
        # **A `kind: program` body has no level 2 to instruct.** `interfaces.md`
        # §4.4: a program executor implements `Executor` and has nothing to
        # raise from — no `status`, no `instruct`, no `query`. Without this the
        # handle is non-None, `decide` returns `Push`, and `_push` calls
        # `instruct` on something that has none: measured as `PUSH_ATTEMPTED`
        # then `HANDLING_FAILED`, an `AttributeError` where a decision belongs
        # (`scratch/impl-2026-08/monitor/p11_program_body_push.py`).
        #
        # Reachable **today** through `OUTPUT_ABSENT`, with nothing to do with
        # `done_by_self_check`. Found answering `agent`'s question about that
        # field, which is the second time this week a question about a future
        # change surfaced a present defect.
        #
        # This is what `Pushable` being `runtime_checkable` was declared for.
        # Until now the decorator was kept for a check nobody could run.
        return None, "the executor is a program body: there is no agent to instruct"
    return attempt.executor, ""


class PusherMonitor(BaseMonitor):
    """The monitor without an agent: a loop and a fixed reaction (spec §3).

    It overrides `decide` and nothing else. The analysing dispatcher will do the
    same — **`_advance` is inherited unchanged**, so an AI never reaches the
    ordinary path.
    """

    def decide(self, unit: Unit) -> Decision:
        newest = unit.newest
        kind = newest.kind

        if kind in GATE_KINDS:
            return self._decide_gate(newest)

        if kind is EventKind.BUDGET_EXCEEDED:
            # The budget is what bounds the gate loop (spec §4.1.3). Pushing past
            # it would remove the bound, which is the whole reason the exit is a
            # decision rather than a retry count in the runner.
            return Escalate("budget exceeded; pushing past it would remove the loop's bound")

        if kind in _TERMINAL:
            # The task is terminal and no agent is running (`validator` spec
            # §3.4), so none of push / resume / restart applies to the *agent*.
            # The branch is still reported and still recorded: a dead branch
            # nobody is told about is how a graph stops without anyone noticing.
            return Escalate(f"{kind.value}: the task is terminal and there is nothing to push")

        return GiveUp(f"the pusher has no action for {kind.value}")

    def _decide_gate(self, newest: EventRecord) -> Decision:
        if self._pushed_before(newest):
            # **Read back from the record set**, which is the concrete reason
            # spec §8.2 rejected the OpenTelemetry SDK: OTel is emit-only by
            # construction, and this decision has to read what it wrote.
            self._recorder.write(
                event(
                    EventKind.PUSH_INEFFECTIVE,
                    newest.task_id,
                    attempt=newest.attempt,
                    reported_by=self.name,
                    attributes={"then": newest.kind.value},
                )
            )
            return Escalate(
                "a push was already attempted for this attempt and the gate failed again"
            )

        # `self._runner` resolves by name and raises if the root did not
        # register one. A `"runner" in self._r else None` guard here would be
        # §4.11 again: the composition root always supplies it, so its absence
        # is a wiring bug and must not read as "no agent is running".
        handle, why = live_handle(self._runner, newest.task_id)
        if handle is None:
            # The reason travels into the record, because "the agent is gone"
            # and "it never had one" are different facts about a failed gate.
            return Escalate(f"nothing to push: {why}")

        # **Never `restart` first.** Push is ~2 s and lossless; resume is ~5.5 s
        # warm and drops the per-attempt wiring `env_mgr` prepared; restart loses
        # all context plus the zone. An agent that returned nearly-finished work
        # and merely failed to publish is the cheapest possible fault, and
        # restart is the most expensive possible reaction to it.
        return Push(handle)

    def _pushed_before(self, newest: EventRecord) -> bool:
        return any(
            r.kind is EventKind.PUSH_ATTEMPTED
            for r in self._recorder.read(newest.task_id, newest.attempt)
        )
