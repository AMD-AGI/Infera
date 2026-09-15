#!/usr/bin/env python3
"""Write valid rows with all required fields (name, value, tag)."""

import json
import os
import sys
from pathlib import Path

README = """# thing_a

## Purpose

Rows produced by emit_a, each carrying name, value, and tag.

## Schema

`items/text.json` has `rows` and `totals`.
"""

ROWS = [
    {"name": "alpha", "value": 10, "tag": "ok"},
    {"name": "beta", "value": 20, "tag": "ok"},
    {"name": "gamma", "value": 30, "tag": "ok"},
]


def main() -> int:
    dst = Path(os.environ["AGENT_SYS_OUTPUT_THING_A"])
    (dst / "items").mkdir(parents=True, exist_ok=True)
    (dst / "README.md").write_text(README, encoding="utf-8")
    (dst / "items" / "text.json").write_text(
        json.dumps({
            "rows": ROWS,
            "totals": {"count": len(ROWS), "value_sum": sum(r["value"] for r in ROWS)},
        }, indent=2),
        encoding="utf-8",
    )
    print(f"emit_a: {len(ROWS)} rows -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
