#!/usr/bin/env python3
"""Read the facts manifest and write a summary with a deliberately ungrounded claim.

The summary says "elapsed_ms: 99999" — a number that cannot appear in the facts
manifest (which contains only file counts, line counts, and 8-char hex SHA-256
prefixes). check_grounded extracts numerals from both sides and finds 99999 in
the summary but not in the facts, so it returns False.

The grounding copy is placed in items/grounding/ so check_grounded can compare
without reaching outside its own staged handoff.
"""

import json
import os
import shutil
import sys
from pathlib import Path


README = """# summary

## Purpose

A text summary of the facts manifest, including a fabricated elapsed time
that cannot be grounded in the facts.

## Schema

`items/text.json` carries the summary text.
`items/grounding/` is a verbatim copy of the facts artefact.
"""


def main() -> int:
    facts_dir = Path(os.environ["AGENT_SYS_INPUT_FACTS"])
    dst = Path(os.environ["AGENT_SYS_OUTPUT_SUMMARY"])

    facts_json = facts_dir / "items" / "text.json"
    data = json.loads(facts_json.read_text(encoding="utf-8"))
    total_files = data["totals"]["files"]
    total_lines = data["totals"]["lines"]

    # The summary text — the last sentence contains the ungrounded number.
    summary_text = (
        f"The package contains {total_files} files with {total_lines} total lines. "
        f"The collection completed in 99999 milliseconds."
    )

    (dst / "items").mkdir(parents=True, exist_ok=True)
    (dst / "items" / "grounding").mkdir(parents=True, exist_ok=True)
    (dst / "README.md").write_text(README, encoding="utf-8")
    (dst / "items" / "text.json").write_text(
        json.dumps({"text": summary_text}, indent=2),
        encoding="utf-8",
    )

    # Copy the entire facts content into grounding/ so check_grounded has both
    # sides in one staged handoff.
    for item in facts_dir.iterdir():
        src = facts_dir / item.name
        dest = dst / "items" / "grounding" / item.name
        if src.is_dir():
            shutil.copytree(src, dest, dirs_exist_ok=True)
        elif src.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

    print(f"transform: summary with ungrounded 99999 -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
