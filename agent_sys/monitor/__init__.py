"""monitor — the task's event loop, on two channels.

Everything that happens to a task and is not the task's own work arrives here
through one call, `report`, and lands on one of two queues: **planned** advances,
handled by code and never by a model, and **unplanned** outcomes, which are a
decision. `PLANNED` is the whole of that routing rule, consulted in one place, and
it is why a reporter never has to classify what it is reporting.

Two names appear twice and mean one thing each. `protocols.py` declares
`EventRecord`, `Recorder`, `Monitor` and `UserSink` as the *seam*; `record.py` and
`base.py` hold the implementations that satisfy them. What is re-exported here is
the implementation, because a reporter has to **build** a record rather than only
accept one.

Nothing in this package imports `agent`. See `protocols.Pushable`.
"""

from .base import (
    DEFAULT_MONITOR_NAME,
    ESCALATION_TARGET,
    NO_TASK,
    PUSH_MESSAGE,
    TARGET_USER,
    BaseMonitor,
    Decision,
    Escalate,
    GiveUp,
    NullUserSink,
    Push,
    ReportToUser,
    RunningMonitors,
    check_liveness,
    install_excepthook,
    monitor_for,
    next_phase,
    reached_the_user,
    start_monitors,
)
from .buffer import ExceptionBuffer, PlannedQueue, Unit
from .protocols import (
    PLANNED,
    Budget,
    BufferClosed,
    EventKind,
    Monitor,
    Pushable,
    ScopeViolation,
    UserSink,
)
from .pusher import PusherMonitor
from .record import EventId, EventRecord, Recorder, default_fingerprint, event, rekeyed

__all__ = [
    # the seam — docs/interfaces.md §4.9
    "PLANNED",
    "Budget",
    "BufferClosed",
    "EventKind",
    "EventRecord",
    "Monitor",
    "Pushable",
    "Recorder",
    "ScopeViolation",
    "UserSink",
    # what the composition root builds and calls — docs/interfaces.md §2
    "DEFAULT_MONITOR_NAME",
    "PusherMonitor",
    "check_liveness",
    "install_excepthook",
    "start_monitors",
    # the rest of the module's own surface
    "ESCALATION_TARGET",
    "NO_TASK",
    "PUSH_MESSAGE",
    "TARGET_USER",
    "BaseMonitor",
    "Decision",
    "Escalate",
    "EventId",
    "ExceptionBuffer",
    "GiveUp",
    "NullUserSink",
    "PlannedQueue",
    "Push",
    "RunningMonitors",
    "ReportToUser",
    "Unit",
    "default_fingerprint",
    "event",
    "monitor_for",
    "next_phase",
    "reached_the_user",
    "rekeyed",
]
