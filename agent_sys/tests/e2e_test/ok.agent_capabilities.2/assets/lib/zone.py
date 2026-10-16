#!/usr/bin/env python3
"""The four body-facing files a validation zone carries, and nothing else.

`args.json`, `inputs.json`, `materials.json`, `verdict.json`. A validator body
is started with `cwd` set to a freshly allocated zone holding the first three,
and it owes the fourth.

This package's handoff is `structured_text`, whose payload is one JSON file;
`report` reads it. The store-layout question (`AGENT_SYS_DEMO_STORE`) is kept
out of this module entirely, since neither validator here needs it.
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = [
    "REPORT_ITEM",
    "args",
    "content_of",
    "inputs",
    "materials",
    "readme_of",
    "report",
    "write_verdict",
]

#: The one item an `env_report` handoff carries. `handoff/content.py`'s
#: `structured_text` type requires **one of** `text.json` / `text.yaml` /
#: `text.xml`, and the kind's `items_schema` in `steps/check.yaml` narrows that
#: to this one — argued there.
REPORT_ITEM = "text.json"


def inputs() -> list[str]:
    """The handoff ids this body is validating, as `ScriptBodyRunner` wrote them."""
    return list(json.loads(Path("inputs.json").read_text(encoding="utf-8")))


def args() -> dict:
    """This validator spec's `args` block, or `{}` when it declared none."""
    path = Path("args.json")
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def materials() -> dict[str, Path]:
    """The staged copies of what this phase validates, by handoff id.

    Written unconditionally, so an empty mapping is a record rather than an
    absence.
    """
    path = Path("materials.json")
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return {str(hid): Path(where) for hid, where in loaded.items()}


def content_of(hid: str) -> Path | None:
    """The staged content directory of one handoff, or `None`.

    `None` means the phase staged nothing for this id, and a caller must
    treat that as no content, never as a pass.
    """
    staged = materials().get(hid)
    return staged if staged is not None and staged.is_dir() else None


def readme_of(content: Path) -> str | None:
    """The handoff's own `README.md`, or `None` if it is absent or unreadable."""
    path = content / "README.md"
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def report(content: Path) -> tuple[dict | None, str]:
    """The parsed `items/text.json`, and why not when it is `None`.

    Returns `(payload, reason)`. Absent, unparseable, and not-an-object each
    get a distinct reason.
    """
    path = content / "items" / REPORT_ITEM
    if not path.is_file():
        return None, f"no items/{REPORT_ITEM}"
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        return None, f"items/{REPORT_ITEM}: unreadable: {exc}"
    except json.JSONDecodeError as exc:
        return None, f"items/{REPORT_ITEM}: not valid JSON: {exc}"
    if not isinstance(loaded, dict):
        return None, f"items/{REPORT_ITEM}: top level is {type(loaded).__name__}, needs an object"
    return loaded, ""


def write_verdict(results: dict[str, bool]) -> None:
    """`verdict.json`: one entry per declared handoff. A missing one raises at
    `PhaseRunner`'s seam rather than folding as falsy."""
    Path("verdict.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
