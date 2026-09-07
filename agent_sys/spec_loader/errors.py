"""The load-time error vocabulary.

All three classes are declared in `protocols.py`, which is the frozen contract
(`docs/interfaces.md` §8), and are re-exported here rather than redeclared: two
definitions of one exception is exactly the duplication `engineer_principle.md`
§1 forbids, and the seam has two sides.

**`RenderError` was a fourth and is gone with the thing it described.** It meant
*"a jsonnet source failed to evaluate"*, and there is no evaluation: a YAML
syntax error is a fault in a package's document, reported as a `Problem` with a
line and a column like every other one (`yaml_source.read_yaml`). Its one durable
idea survives the deletion — `line` and `column` are `None` when the parser did
not report them, never guessed — and it survives on `Problem`, which is where a
position now belongs.
"""

from __future__ import annotations

from .protocols import SpecInconsistent, SpecInvalid, SpecNotFound

__all__ = [
    "SpecInconsistent",
    "SpecInvalid",
    "SpecNotFound",
]
