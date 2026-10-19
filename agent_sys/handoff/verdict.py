"""`validation.yaml` — the verdict record, beside the artefact and outside the digest.

**This module owns `Verdict`** (declared in `protocols.py`), because the layer
that persists a record is the layer that has to keep it readable.
`validator/history.py` keeps `VerdictRecord` as its own *view* of one; it does
not declare a second shape.

Criterion 7 is a structural fact, not a convenience: the file is a **sibling**
of `content/`, so recording a verdict cannot move the artefact's identity. PyPA
excludes `RECORD` from its own hashes, Debian's `md5sums` breaks the recursion
identically, and Bazel's Action/ActionResult split answers the general question
— *does omitting this let a wrong answer masquerade as a right one?* A verdict
does not, so it is outside.

The file is created **empty at publication** rather than on first verdict,
because absent must be stated, not omitted: an empty `verdicts:` list says
"nothing has checked this yet", a missing file says something is wrong with
this version.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import yaml

from handoff.errors import Malformed
from handoff.protocols import Verdict
from task_graph.ids import AgentId, TaskId

__all__ = ["VERDICT_FILE", "append", "create_empty", "read", "to_row"]

VERDICT_FILE = "validation.yaml"

_EMPTY = "verdicts: []\n"


def to_row(verdict: Verdict) -> dict[str, object]:
    """One verdict as plain YAML-safe data.

    Criterion 8's eight fields: "has this been checked, by what, when, did it
    pass, and in what environment" must be answerable from the handoff alone.

    **`agent_id` is written as YAML `null` when no agent ran** — a script body
    has none. It is the one field that may be absent, and absent is a statement:
    the alternative was the producing agent's id, which says the opposite of the
    truth in the field a reader consults for attribution.

    Not omitted from the mapping when null, because an absent *key* and a null
    *value* are two different records and only one of them is this one. That is
    the same rule `validation.yaml` itself follows — created with an empty
    `verdicts:` list rather than not created.
    """
    return {
        "validator": verdict.validator,
        "result": bool(verdict.result),
        "strength": verdict.strength,
        "dimension": verdict.dimension,
        "task_id": str(verdict.task_id),
        "agent_id": None if verdict.agent_id is None else str(verdict.agent_id),
        "environment": dict(verdict.environment),
        "at": verdict.at.isoformat(),
    }


def _agent_id(value: object) -> AgentId | None:
    """`null` means no agent ran; anything else must parse as an `AgentId`.

    A missing key raises through `_from_row`'s `KeyError`, which is deliberate:
    a row that never mentioned the field is a row written by something that did
    not know the field exists, and that is not the same as one saying `null`.
    """
    return None if value is None else AgentId(str(value))


def _from_row(row: Mapping[str, object], *, origin: Path) -> Verdict:
    try:
        return Verdict(
            validator=str(row["validator"]),
            result=bool(row["result"]),
            strength=str(row["strength"]),
            dimension=str(row["dimension"]),
            task_id=TaskId(str(row["task_id"])),
            agent_id=_agent_id(row["agent_id"]),
            environment=dict(row["environment"] or {}),  # type: ignore[arg-type]
            at=datetime.fromisoformat(str(row["at"])),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Malformed(f"{origin}: unreadable verdict row {row!r}: {exc}") from exc


def create_empty(path: Path) -> None:
    """Write `verdicts: []`. Called once, at publication."""
    Path(path).write_text(_EMPTY, encoding="utf-8")


def read(path: Path) -> list[Verdict]:
    """Every verdict recorded against this version, in the order recorded."""
    path = Path(path)
    if not path.is_file():
        raise Malformed(
            f"{path} is missing. An empty `verdicts:` list says nothing has "
            f"checked this version; a missing file says something is wrong with it"
        )
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = doc.get("verdicts") or []
    if not isinstance(rows, list):
        raise Malformed(f"{path}: `verdicts` must be a list")
    return [_from_row(row, origin=path) for row in rows]


def append(path: Path, verdict: Verdict) -> None:
    """Add one verdict. Read-modify-write, and it does not touch `content/`."""
    rows = [to_row(v) for v in read(path)]
    rows.append(to_row(verdict))
    Path(path).write_text(
        yaml.safe_dump({"verdicts": rows}, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
