"""The narration a person reads. `demo` design §7.3.

Spec §4.2 asks the demo to **teach the taxonomy by using it**, so a verdict line
spells `trustworthiness / strong` rather than abbreviating it, and an expected
failure says `expected` on the line rather than in a footnote.

Every `EventKind` has a rendering here, and
`tests/cli/test_events.py::test_every_event_kind_has_a_human_rendering` is
parametrised over the enum — a new kind that renders as `<Event object>` fails
there rather than in front of a reviewer.
"""

from __future__ import annotations

import sys
from typing import Any, TextIO

from cli.events import Event, EventKind

__all__ = ["HumanRenderer", "line_for"]

#: The left column. Two spaces of indent for anything inside a dispatch, so the
#: three phases of one task read as belonging to it.
_LABEL = {
    EventKind.RUN_START: "run",
    EventKind.RUN_COMPLETE: "done",
    EventKind.PACKAGE_LOADED: "package",
    EventKind.CLOSURE_RESOLVED: "closure",
    EventKind.SPEC_REJECTED: "REJECTED",
    EventKind.GRAPH_BUILT: "graph",
    EventKind.CONFINEMENT_APPLIED: "confined",
    # Shouted, like `REJECTED` and `UNEXPECTED`, and for the same reason: the
    # label is the only part of the line a reviewer is guaranteed to read, and
    # this one says the safety property was switched off for this run.
    EventKind.PERMISSIONS_DISABLED: "NO SANDBOX",
    EventKind.VALIDATION_DROPPED: "DROPPED",
    EventKind.ZONE_PREPARED: "zone",
    EventKind.O11Y_PANEL: "o11y",
    EventKind.ACCESS_DENIED: "  denied",
    EventKind.TASK_DISPATCHED: "dispatch",
    EventKind.PHASE_START: "  phase",
    EventKind.PHASE_COMPLETE: "  phase",
    EventKind.VERDICT_RECORDED: "  verdict",
    EventKind.HANDOFF_TRANSITION: "handoff",
    EventKind.TASK_COMPLETE: "  task",
    EventKind.TASK_FINAL_STATE: "final",
    EventKind.EXPECTED_FAILURE: "expected",
    EventKind.UNEXPECTED_SUCCESS: "UNEXPECTED",
    EventKind.EXPECTATION_UNREACHED: "UNTESTED",
}

_WIDTH = 10


def _taxonomy(fields: Any) -> str:
    """`dimension / strength`, spelled out. **Criterion 10 is this function.**

    Both halves or neither: a verdict line that carried one of them would look
    labelled and be half-labelled, which is worse than an obviously bare line.
    """
    dimension, strength = fields.get("dimension"), fields.get("strength")
    return f"{dimension} / {strength}" if dimension and strength else ""


def line_for(event: Event) -> str:
    """One line, complete on its own.

    `message` is already a whole sentence — `Stream.emit` produces it beside the
    fields, never from them — so this adds the label, the taxonomy where there
    is one, and the `expected` marker, and never reformats the sentence.
    """
    label = _LABEL[event.kind].rjust(_WIDTH)
    parts = [f"{label}  {event.message}"]
    taxonomy = _taxonomy(event.fields)
    if taxonomy:
        parts.append(taxonomy)
    result = event.fields.get("result")
    if result is not None:
        parts.append("PASS" if result else "FAIL")
    mechanism = event.fields.get("mechanism")
    if mechanism:
        parts.append(str(mechanism))
    errno = event.fields.get("errno")
    if errno:
        parts.append(str(errno))
    if event.fields.get("expected"):
        parts.append("expected")
    return "    ".join(parts)


class HumanRenderer:
    """Writes to a stream, one line per event. Defaults to stdout.

    **Not stderr**, and not only stderr: criterion 6's message arrives on stdout
    from the backend (`materials/08-demo.md` §4), and a demo that printed only
    stderr would lose the one message that criterion is about.
    """

    def __init__(self, out: TextIO | None = None) -> None:
        self._out = out if out is not None else sys.stdout

    def on_event(self, event: Event) -> None:
        print(line_for(event), file=self._out, flush=True)
