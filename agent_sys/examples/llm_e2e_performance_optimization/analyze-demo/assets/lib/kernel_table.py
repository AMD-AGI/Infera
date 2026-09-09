#!/usr/bin/env python3
"""Read a published `kernel_table`, whichever of its two producers wrote it.

This package has two upstreams for one kind, and they lay the content out
differently because their `content_type`s differ:

| producer | content_type | document | raw CSV |
|---|---|---|---|
| `seed_table` (this package's mock) | `structured_text` | `items/text.json` | `items/gap_analysis/` |
| `profiling-demo`'s `kernel_scan` | `reproducible` | `items/result/text.json` | `items/result/gap_analysis/` |

The difference is not a disagreement anybody chose. `reproducible` requires
`result` and `env` and one of `script`/`command`, which is what makes a real
capture reproducible; `structured_text` permits only `text.json` / `text.yaml` /
`text.xml` / `schema` at the top level, so a handoff that has to carry a
directory of trace-derived evidence cannot use it. The real producer needs the
former and the mock needs nothing.

**So the reader accepts both, and this is the only place that knows there are
two.** `rank` and `check_kernel_table` ask for the document and get the same
records either way. What this deliberately does *not* do is paper over a
different failure: a handoff with neither layout is an error naming both places
it looked, not an empty table.

Imports nothing from `agent_sys`: package data run as a subprocess.
"""

from __future__ import annotations

import json
from pathlib import Path

#: Where a document may sit, in the order tried. `items/text.json` first because
#: it is the mock's and the mock is what a standalone run of this package uses.
DOCUMENT_PATHS = (
    Path("items") / "text.json",
    Path("items") / "result" / "text.json",
)

#: Where the raw CSV may sit. The kind requires it either way: a consumer wanting
#: a column this package did not parse reads it from there rather than re-running
#: the profile.
CSV_DIRS = (
    Path("items") / "gap_analysis",
    Path("items") / "result" / "gap_analysis",
)

#: Fields every record carries, whichever producer wrote it. `csv_io.to_record`
#: and `profiling-demo`'s `kernel_doc.py` both emit exactly these.
REQUIRED_FIELDS = ("name", "calls", "self_us", "avg_us", "pct_total", "input_shapes")


class LayoutError(ValueError):
    """The content is not a `kernel_table` either producer would have written."""


def document_path(content: Path) -> Path:
    """The document inside a published content directory.

    Raises:
        LayoutError: naming every path tried, because "no kernel table here" is
            not actionable and "neither of these two exists" is.
    """
    content = Path(content)
    for candidate in DOCUMENT_PATHS:
        if (content / candidate).is_file():
            return content / candidate
    raise LayoutError(
        "no kernel_table document found. Looked for "
        + " and ".join(str(p) for p in DOCUMENT_PATHS)
        + f" under {content.name}/"
    )


def csv_dir(content: Path) -> Path | None:
    """The directory holding the raw CSV, or None when neither layout has one."""
    content = Path(content)
    for candidate in CSV_DIRS:
        target = content / candidate
        if target.is_dir() and any(target.glob("*.csv")):
            return target
    return None


def read(content: Path) -> dict:
    """The document, checked far enough that a caller can index it.

    Validation here is only what every caller would otherwise repeat: the file
    parses, `kernels` is a list, and the records carry the agreed field names. A
    rename on either producer's side is caught here with both names in hand,
    rather than three tasks later as a column of nulls.
    """
    path = document_path(content)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise LayoutError(f"{path.name} is not valid JSON: {error}") from error
    if not isinstance(document, dict):
        raise LayoutError(f"{path.name} is not a JSON object")

    kernels = document.get("kernels")
    if not isinstance(kernels, list):
        raise LayoutError(f"{path.name} has no 'kernels' array")
    if kernels:
        missing = [field for field in REQUIRED_FIELDS if field not in kernels[0]]
        if missing:
            raise LayoutError(
                f"{path.name} kernel records are missing {missing}. Both producers "
                f"emit {list(REQUIRED_FIELDS)}; a rename upstream stops here"
            )
    return document


def launcher_coverage(document: dict) -> dict:
    """`{available, resolved, rows_with_frames, reason}` for the launcher block.

    Normalised because only the real producer reports on it. A document from the
    mock has no `launchers` summary at all, and reading that as "resolution
    failed" would be wrong -- nobody looked.
    """
    summary = document.get("launchers")
    rows = sum(
        1
        for kernel in document.get("kernels") or []
        if (kernel.get("launcher") or {}).get("source_file")
    )
    if not isinstance(summary, dict):
        return {
            "available": False,
            "resolved": 0,
            "rows_with_frames": rows,
            "reason": "the producer reported no launcher resolution",
        }
    return {
        "available": bool(summary.get("available")),
        "resolved": int(summary.get("resolved") or 0),
        "rows_with_frames": rows,
        "reason": summary.get("reason") or "",
    }
