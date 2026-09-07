"""The completeness gate — `monitor` spec §4.1.0, owned here.

It sits between the main phase and `OUTPUT_VALIDATING`, and it is deliberately
cheap: an admission check, not validation. **It does not ask whether the work is
right, only whether there is something to check.**

Four independent mechanical failures, and the runner **reports** every one of
them and decides nothing (`monitor` design §8). A runner that retried on its own
would be a second failure policy the record cannot see.

Two of the four cannot be fully mechanised against the shipped `HandoffStore`,
and `README.md` reports both rather than this file pretending otherwise:

- **`done_by_self_check` does not exist.** `monitor` spec §4.1.2 says so
  explicitly and §9 carries it as a `handoff` propagation item. The check is
  written and reports only when the field is present and false, so it activates
  the day `handoff` lands it instead of failing every gate today.
- **Executability is not a `HandoffStore` query.** `put` checks the README and
  locality and nothing else (§4.1.1), and the Protocol exposes no mode. The
  check copies the version out **only when a `script` or `command` item is
  declared**, so the common case costs nothing.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from monitor.protocols import Budget, EventKind
from task_graph.ids import HandoffId

__all__ = ["EXECUTABLE_ITEMS", "GateFailure", "run_gate"]

#: The item keys that claim executability. `handoff` spec §5 makes a README
#: required "whenever `script` or `command` is present", which is the same two.
EXECUTABLE_ITEMS: frozenset[str] = frozenset({"script", "command", "entry"})


@dataclass(frozen=True)
class GateFailure:
    """One mechanical failure. **A value, not a log line** — it becomes an
    `EventRecord` and the monitor decides from the record."""

    kind: EventKind
    message: str
    handoff_id: HandoffId | None = None
    #: Why the producer was **never asked**, as against `seal_refused`'s *asked
    #: and refused*. `monitor` criterion 5 turns on that distinction, and one
    #: key cannot carry both: `seal_refused`'s **presence** is what says the
    #: producer was asked at all.
    nothing_to_attempt: str = ""


def run_gate(
    outputs: Sequence[HandoffId],
    usage: Mapping[str, float],
    *,
    store: Any,
    budget: Budget | None,
) -> list[GateFailure]:
    """The four checks, in order, all of them run.

    All four run even after the first fails: they are independent, and reporting
    one absence while a budget was also blown would hide half the situation.
    """
    failures: list[GateFailure] = []
    for hid in outputs:
        failures += _no_store(hid) if store is None else _one_output(hid, store)
    failures += _budget(usage, budget)
    return failures


def _no_store(hid: HandoffId) -> list[GateFailure]:
    """A declared output and no store to have published it into.

    **Two "nothing to say" answers were composing into "nothing was wrong".**
    `_seal_outputs` returned `{}` and this gate returned `[]` when no
    `handoff_store` is registered — each correct about the single fact it
    tests — and `_main`'s `if not failures` then read the pair as a pass. So a
    task that declared outputs and published none was reported **succeeded**,
    with no cause named anywhere. Measured by `monitor`,
    `p15_storeless_outputs.py`.

    **Storeless is a supported mode and stays one.** `bootstrap.py:216` leaves
    the name unregistered rather than rooting a store nobody chose, and
    `task_graph` runs its whole suite that way. The fault is the *conjunction* —
    declares outputs, no store — which is exactly what `_pin_outputs` already
    warns about and what neither of these two guards asked.

    `nothing_to_attempt` rather than `seal_refused`: **the producer was never
    asked.** `monitor`'s criterion 5 needs *refused* and *never attempted* to
    stay apart, and one key cannot carry both without presence losing the
    meaning the distinction depends on.
    """
    return [
        GateFailure(
            kind=EventKind.OUTPUT_ABSENT,
            message=f"declared output {hid} could not be published: no handoff store is configured",
            handoff_id=hid,
            nothing_to_attempt="no handoff store is configured, so nothing was ever asked of the producer",
        )
    ]


def _one_output(hid: HandoffId, store: Any) -> list[GateFailure]:
    if not store.exists(hid):
        return [
            GateFailure(
                kind=EventKind.OUTPUT_ABSENT,
                message=f"declared output {hid} was never delivered",
                handoff_id=hid,
            )
        ]
    versions = store.list_versions(hid)
    if not versions:
        return [
            GateFailure(
                kind=EventKind.OUTPUT_ABSENT,
                message=f"declared output {hid} exists with no version",
                handoff_id=hid,
            )
        ]
    version = versions[-1]
    manifest = store.get_manifest(hid, version)
    return _self_check(hid, manifest) + _executable(hid, version, store)


def _self_check(hid: HandoffId, manifest: Any) -> list[GateFailure]:
    """`done_by_self_check` — a weak check whose description is the mechanism.

    Absent means `handoff` has not landed the field yet, which is not the
    producing agent's fault; only a present-and-false is a failure.

    **Ruled by `main` 2026-08-29, and not yet buildable here — see the blocker
    at the bottom.** Recorded now because the reasoning is the part that gets
    lost: a later reader sees a producer distinction with no branch defending
    it and "simplifies" it away.

    | producer | absent means | gate |
    |---|---|---|
    | agent-bodied | the agent **could have claimed and did not** | fault |
    | `kind: program` | **there was no agent to claim** | not a fault |

    **This is `interfaces.md` §5.13's answer reused, not a new one.** A script
    body has no agent, which is why `Verdict.agent_id` became `AgentId | None` —
    route (a), *"the record says 'no agent' by having no agent"*.
    `done_by_self_check` is **an agent's claim**, so for a program body absent is
    not a missing claim but an **inapplicable question**.

    **It does not violate spec §3.3.1.** The mechanism is identical for both
    executors — same file, same location, same read. What differs is whether an
    agent exists, which is a fact about the producer and not a capability of the
    executor. That is exactly the distinction §5.13 settled.

    **Why the tolerance clause cannot simply be deleted.** Without the producer
    distinction, every `kind: program` output would report `SELF_CHECK_UNSET` on
    every attempt — **a value that arrives so reliably it stops carrying
    information**, which is `interfaces.md` §4.13's family seen from the other
    side, and it would cost the check on the AI path too, where it is the only
    path it was ever worth anything on.

    **`monitor` checked the mechanism this paragraph used to assert, and it is
    not today's monitor.** *"A monitor learns to discount it"* was a prediction;
    `PusherMonitor.decide` is a fixed table — gate kind, pushed-before, live
    handle — with no frequency and no history beyond this attempt, so it cannot
    count a kind and cannot discount one. Two real casualties remain, and they
    are why the concern stands: the analysing dispatcher (`monitor` spec §7.1)
    is the thing that would weigh, and by the time it exists the base rate is
    already in the record it learns from; and **a human reading `read()` is the
    nearer one**, needing no dispatcher at all.

    **`monitor` also ruled out the alternative to the distinction**: for a
    program body the inapplicable case should not become an event, so there is
    nothing for the record to distinguish. Their reason is narrower than the
    base rate and better — §4.1.2 says the field exists to cut round-trips by
    making an agent look once more, a program cannot benefit, and whether it
    wrote its outputs is `OUTPUT_ABSENT`'s question and already answered.

    **One constraint on whoever resolves the blocker**, `monitor`'s preference
    and the shape §5.13 already set: what arrives should say **what the producer
    is**, not whether this check applies. A `self_check_applies` boolean would
    put one consumer's policy in this signature, and the next question about the
    same distinction would need a boolean of its own.

    **The blocker, measured.** This function cannot tell the two producer classes
    apart from anything it is given. `Manifest` is `digest` / `algorithm` /
    `kind` / `producer` / `created_at`, and `producer` is a **`TaskId`, not an
    `AgentId`** (`handoff/protocols.py:126`); `seal` takes `producer: TaskId`
    too, and `run_gate` has no parameter for it. So *"there was no agent"* is not
    computable where the check lives. Reported to `main` and `handoff`; until it
    is resolved the tolerance clause below stays, and it stays **with this note
    on it** rather than silently.
    """
    claimed = getattr(manifest, "done_by_self_check", None)
    if claimed is None or claimed:
        return []
    return [
        GateFailure(
            kind=EventKind.SELF_CHECK_UNSET,
            message=f"the producing agent did not mark {hid} done_by_self_check",
            handoff_id=hid,
        )
    ]


def _executable(hid: HandoffId, version: int, store: Any) -> list[GateFailure]:
    """**`copy_out` is the only way to learn the item keys**, so it always runs.

    This was a cheap pre-check on `manifest.items` until a test drove the real
    `FilesystemStore`: `handoff.protocols.Manifest` is `digest` / `algorithm` /
    `kind` / `producer` / `created_at` and has never had `items`, so the
    `getattr` found nothing, the check returned early and
    `OUTPUT_NOT_EXECUTABLE` was **unreachable in production while its unit tests
    were green**. `Manifest.digest` is a whole-tree digest rather than a
    per-item map and the Protocol exposes no listing call, so there is nothing
    cheaper to ask.
    """
    with tempfile.TemporaryDirectory(prefix="agent-gate-") as tmp:
        # A *child* of the temporary directory: `copy_out` creates its
        # destination and refuses one that exists, which `TemporaryDirectory`
        # has already made. The dead branch had this bug too, unobserved.
        content = store.copy_out(hid, version, Path(tmp) / "out")
        items = getattr(content, "items", None)
        if not isinstance(items, Mapping):
            return []
        return [
            GateFailure(
                kind=EventKind.OUTPUT_NOT_EXECUTABLE,
                message=f"{hid} declares {key!r} executable, and it is not",
                handoff_id=hid,
            )
            for key in sorted(set(items) & EXECUTABLE_ITEMS)
            if not _runnable(items[key])
        ]


def _runnable(item: Any) -> bool:
    path = getattr(item, "path", None)
    return path is not None and os.access(path, os.X_OK)


def _budget(usage: Mapping[str, float], budget: Budget | None) -> list[GateFailure]:
    """`monitor` spec §4.1.3. **This is what bounds the gate's loop** — without
    it an agent that never satisfies the gate cycles forever, and with it the
    cycle has an exit that is the monitor's *decision* rather than a retry count
    buried in the runner."""
    if budget is None:
        return []
    limits = (
        ("tokens", budget.max_tokens),
        ("seconds", budget.max_seconds),
        ("turns", budget.max_turns),
    )
    return [
        GateFailure(
            kind=EventKind.BUDGET_EXCEEDED,
            message=f"{name}: {usage[name]} exceeds the budget of {limit}",
        )
        for name, limit in limits
        if limit is not None and usage.get(name, 0) > limit
    ]
