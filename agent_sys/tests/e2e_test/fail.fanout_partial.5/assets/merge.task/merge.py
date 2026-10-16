#!/usr/bin/env python3
"""Merge three producers' outputs. Never runs: thing_b is sealed INVALID."""

import json
import os
import sys
from pathlib import Path


def main() -> int:
    sources = {}
    for kind in ("THING_A", "THING_B", "THING_C"):
        src = Path(os.environ[f"AGENT_SYS_INPUT_{kind}"])
        data = json.loads((src / "items" / "text.json").read_text(encoding="utf-8"))
        sources[kind.lower()] = data["rows"]
    merged = [row for rows in sources.values() for row in rows]
    Path("merged.json").write_text(
        json.dumps({"merged": merged, "count": len(merged)}, indent=2),
        encoding="utf-8",
    )
    print(f"merge: {len(merged)} rows from 3 sources")
    return 0


if __name__ == "__main__":
    sys.exit(main())
