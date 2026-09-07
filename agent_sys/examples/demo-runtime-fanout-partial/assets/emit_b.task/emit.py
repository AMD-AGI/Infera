#!/usr/bin/env python3
"""Write rows deliberately missing the required `tag` field.

check_thing expects every row to carry [name, value, tag]. This script omits
`tag`, so the validator returns False and thing_b is sealed INVALID.
"""

import json
import os
import sys
from pathlib import Path

README = """# thing_b

## Purpose

Rows produced by emit_b. Deliberately missing the `tag` field so that
check_thing fails validation.

## Schema

`items/text.json` has `rows` and `totals`.
"""

ROWS = [
    {"name": "delta", "value": 40},
    {"name": "epsilon", "value": 50},
    {"name": "zeta", "value": 60},
]


def main() -> int:
    dst = Path(os.environ["AGENT_SYS_OUTPUT_THING_B"])
    (dst / "items").mkdir(parents=True, exist_ok=True)
    (dst / "README.md").write_text(README, encoding="utf-8")
    (dst / "items" / "text.json").write_text(
        json.dumps({
            "rows": ROWS,
            "totals": {"count": len(ROWS), "value_sum": sum(r["value"] for r in ROWS)},
        }, indent=2),
        encoding="utf-8",
    )
    print(f"emit_b: {len(ROWS)} rows (missing tag) -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
