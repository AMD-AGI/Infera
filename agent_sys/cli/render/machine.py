"""JSON Lines. `demo` design §7.4, and criterion 14's whole surface.

**JSON Lines rather than one document**, and criterion 12 is why: a run that is
interrupted must still have produced valid, complete output for everything that
happened before the interrupt. A single top-level JSON object would be
truncated and unparseable, which would make the resume demonstration
unassertable — the criterion would be true and unprovable.

**Not stdout.** Criterion 6's message arrives on stdout from the backend, and
interleaving a byte stream we do not control with a machine-readable one is how
a parser learns to be lenient. `--json PATH` writes it to a file; with no flag
it is suppressed.
"""

from __future__ import annotations

import json
from typing import Any, TextIO

from cli.events import SCHEMA_VERSION, Event

__all__ = ["JsonLinesRenderer", "as_object"]


def as_object(event: Event) -> dict[str, Any]:
    """One JSON object per event.

    `schema`, `kind`, `at` and `message` are always present; `fields` is
    flattened in beside them rather than nested, because criterion 14's
    assertions read `fields` directly and one level of nesting per read is a
    tax on the thing the format exists for.

    A field colliding with a reserved key would silently win, so it cannot: the
    four are written **after** the flatten and therefore always mean what they
    say.
    """
    out: dict[str, Any] = dict(event.fields)
    out.update(
        {
            "schema": SCHEMA_VERSION,
            "kind": event.kind.value,
            "at": event.at.isoformat().replace("+00:00", "Z"),
            "message": event.message,
        }
    )
    return out


class JsonLinesRenderer:
    """One object per line, flushed per event.

    Flushed rather than buffered, for criterion 12's reason: the interrupt is
    `os._exit`, which runs no `atexit` handler and flushes nothing.
    """

    def __init__(self, out: TextIO) -> None:
        self._out = out

    def on_event(self, event: Event) -> None:
        json.dump(as_object(event), self._out, sort_keys=True, default=str)
        self._out.write("\n")
        self._out.flush()
