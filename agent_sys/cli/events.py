"""The event vocabulary. **This is an interface, not a log format.**

`demo` criterion 14 makes the machine-readable stream sufficient to assert
criteria 2–10 without parsing prose, and something a test asserts over is an
interface whether or not anyone says so. Terraform's answer to owning one is
adopted whole (`materials/08-demo.md` §8): a **closed** enumeration of message
types, a version constant with a comment obliging a bump, and — the part that
matters most — **one stream rendered twice rather than two writers**.

This module imports nothing, from this repository or outside it beyond the
standard library, and that is deliberate. A stream that imported `task_graph`
would tempt someone to put a `Task` in an event instead of the two fields a
renderer needs, and then the JSON view would start rendering a model.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from typing import Any, NamedTuple

__all__ = ["SCHEMA_VERSION", "Event", "EventKind"]

SCHEMA_VERSION = "1.4"
"""The schema of the machine-readable stream.

Criterion 14 makes this an interface: **bump it on any change to `EventKind`,
to `Event`'s fields, or to what `render/machine.py` emits.** Terraform's
`JSON_UI_VERSION` carries the same comment for the same reason, and the comment
is the mechanism — nothing else can notice that a consumer's parser is now
wrong. `tests/cli/test_events.py::test_schema_version_matches_the_emitted_field`
holds the constant and the field together; it cannot hold the constant and the
*meaning* together, which is what this sentence is for.

**1.1** — `HANDOFF_TRANSITION.version` and `VERDICT_RECORDED.version` became
`slot_version` and `store_version`. There are two allocators and nothing
forces them to agree (`docs/interfaces.md` §5.12), and one field named
`version` on both events implied a single number that does not exist. The
bump is the obligation above being honoured on the first change that needed
it, which is the only evidence that the obligation is real.

**1.2** — `EXPECTATION_UNREACHED`. A promised failure that **did not happen**
and one that **never got the chance** are different facts, and the stream had
one kind for both. `main` spotted it in a run where `produce` failed before
`describe` could produce a summary: `UNEXPECTED_SUCCESS` was true about what
was seen and false about what it meant. Three outcomes, two kinds — the same
shape this package reported to `agent` and `validator` as F-D9, in its own
code.

**1.3** — `PERMISSIONS_DISABLED` and `VALIDATION_DROPPED`. Both exist because
**an absence and a decision must not render the same**, which is the one
mistake this stream is built to make impossible.

`PERMISSIONS_DISABLED` is `docs/interfaces.md` §4.17a: with permission
management switched off, `CONFINEMENT_APPLIED` would still be true — the
*machine* has Landlock, the probe finds it — and the run would print
`confined` while running unconfined. **A run whose sandbox was switched off and
a run on a machine with no sandbox are different facts**, and the second
already exits 2 with a refusal (criterion 9). The first must be as visible and
must not be mistaken for the ordinary case, so it is its own kind with its own
shouting label rather than a field on the confinement line.

`VALIDATION_DROPPED` is §4.17: a green run whose green covers fewer checks than
it appears to. The vocabulary already distinguished *observed*, *did not
happen* and *never reached*; it had no way to say **deliberately not run**. A
dropped check that simply vanishes from the output is indistinguishable from
one that was never declared, and the difference is the whole claim the run is
making.

**1.4** — `O11Y_PANEL`. The AgentsView panel's URL, and the notice that its
binary was fetched for the first time, were `log.info` calls. **Nothing in this
repository configures `logging`**, so the root logger sits at `WARNING` with no
handler and both lines were discarded — while the o11y failure paths, being
`log.warning`, reached stderr through `logging.lastResort`. Failures were
visible and successes were not, and the tests did not notice because
`caplog.at_level("INFO")` forces the level from pytest's side. A fact the user
is meant to read belongs in the stream, which is the thing in this package
whose job is being read; `logging` here is for the operator's diary.

`docs/interfaces.md` §5.7: once the whole-system CLI wants the same stream,
two artefacts share this constant with no bump policy. That is open.
"""


class EventKind(str, Enum):
    """Closed, and closed is the point.

    A free-form `logging` record with arbitrary `extra` cannot be asserted over
    without a parser that guesses, which is why `logging` was considered for the
    stream and rejected (`demo` design §13).
    """

    # the run itself
    RUN_START = "run_start"
    RUN_COMPLETE = "run_complete"

    # loading — everything `--dry-run` can reach
    PACKAGE_LOADED = "package_loaded"
    CLOSURE_RESOLVED = "closure_resolved"
    SPEC_REJECTED = "spec_rejected"
    GRAPH_BUILT = "graph_built"

    # the environment
    CONFINEMENT_APPLIED = "confinement_applied"
    PERMISSIONS_DISABLED = "permissions_disabled"
    ZONE_PREPARED = "zone_prepared"
    ACCESS_DENIED = "access_denied"
    O11Y_PANEL = "o11y_panel"

    # what this run did NOT check, and why. Absent is not the same as dropped.
    VALIDATION_DROPPED = "validation_dropped"

    # the task lifecycle
    TASK_DISPATCHED = "task_dispatched"
    PHASE_START = "phase_start"
    PHASE_COMPLETE = "phase_complete"
    VERDICT_RECORDED = "verdict_recorded"
    HANDOFF_TRANSITION = "handoff_transition"
    TASK_COMPLETE = "task_complete"

    # the end
    TASK_FINAL_STATE = "task_final_state"
    EXPECTED_FAILURE = "expected_failure"
    UNEXPECTED_SUCCESS = "unexpected_success"
    EXPECTATION_UNREACHED = "expectation_unreached"


class Event(NamedTuple):
    """One thing that happened, in both renderings at once.

    **`message` and `fields` are produced by the same call**, which is the whole
    of Terraform's S5: a demo whose narration and whose JSON disagree fails at
    the one job it has. There is no path through `Stream.emit` that produces one
    without the other.

    `fields` holds JSON scalars, lists and dicts — **never a `Task`, an
    `Execution` or a `Handoff`**. A renderer that could reach into a model would
    start rendering one, and the JSON view would then carry whatever that model
    happened to hold that week.
    `tests/cli/test_events.py::test_no_event_field_holds_a_model` asserts it.
    """

    kind: EventKind
    message: str
    fields: Mapping[str, Any]
    at: datetime
