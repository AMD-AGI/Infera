"""Two renderings of one stream. `demo` design §7.

Two, not two writers: `Stream.emit` produces the human sentence and the typed
fields in one call, and each renderer reads the same `Event`. A demo whose
narration and whose JSON disagree fails at the one job it has.
"""

from cli.render.human import HumanRenderer, line_for
from cli.render.machine import JsonLinesRenderer, as_object

__all__ = ["HumanRenderer", "JsonLinesRenderer", "as_object", "line_for"]
