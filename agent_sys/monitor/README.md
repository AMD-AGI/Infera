# `monitor` — the task's event loop

Everything that happens to a task **and is not the task's own work** arrives
here, through one call: `report`.

| | |
|---|---|
| Specification | [`docs/spec.md`](docs/spec.md) — 26 acceptance criteria |
| Design | [`docs/design.md`](docs/design.md) |
| Seam | [`../docs/interfaces.md`](../docs/interfaces.md) §4.9. Imports `task_graph` and **nothing else of ours** |
| Contract | [`protocols.py`](protocols.py) + `protocols.pyi` |
| Tests | `../tests/monitor/`, plus the seam tests in `../tests/interfaces/` |

## Who uses it

`agent`'s runner reports every phase boundary here — planned and not.
`validator` routes a failed phase here. `cli` starts the monitors and supplies
the user sink. Nothing here calls back into `agent`: the monitor needs to push a
live agent and declares `Pushable` **locally** rather than import it, which is
what keeps the edge one-way.

## Two channels, and one routing rule

| | Handled by | Meaning |
|---|---|---|
| **planned** | code, never a model | an ordinary phase advance. The task is doing what the graph says |
| **unplanned** | a decision | everything else: an exception, a stall, a gate failure, a terminal task |

`PLANNED` is the whole of that rule and is consulted in exactly one place. That
is why **a reporter never has to classify what it is reporting** — it calls
`report` and the routing is not its problem.

The alpha's unplanned handler is a **simple pusher**: a status check plus one
phrase. The *analysing* dispatcher — which picks from a richer action set — is
[`../docs/ROADMAP.md`](../docs/ROADMAP.md) §2, and it is bound to the unplanned
channel alone, so no model is ever on the ordinary path.

## The interface

```python
from monitor import (
    Monitor, BaseMonitor, PusherMonitor,   # the loop
    start_monitors, monitor_for,           # starting them, and finding one
    EventRecord, EventKind, event,         # what is reported
    Recorder, UserSink, NullUserSink,      # where it lands
    Decision, Push, Escalate, GiveUp, ReportToUser,   # what an unplanned outcome becomes
    Budget, check_liveness, install_excepthook,
)
```

**A monitor has no authority over task state.** Every action it takes is a
transition it *calls* on the task; `_move` stays the single writer. That is what
lets the analysing dispatcher arrive later without re-opening the authority rule.

## What is inside

| | |
|---|---|
| `protocols.py` / `.pyi` | the seam: `EventRecord`, `Recorder`, `Monitor`, `UserSink`, `Pushable` |
| `base.py` | the loop itself, the two queues, and the routing |
| `record.py` | the event record, its kinds, and `rekeyed` for a subtask's event reaching its parent |
| `buffer.py` | the exception buffer, and what bounds it |
| `pusher.py` | the alpha's unplanned handler |

Two names appear in both `protocols.py` and an implementation file and mean one
thing each: what this package **re-exports is the implementation**, because a
caller that imports `Monitor` wants the loop, not the declaration.

## Nothing monitors the monitor

Ordinary progress now depends on a monitor being alive, so that is a real
exposure rather than a footnote. `check_liveness` and `install_excepthook` are
the two mechanisms spec §5.4 requires; the excepthook is installed by the
composition root, not on construction, because `threading.excepthook` is
process-global and a library that mutates it would have two monitors fighting
over it.
