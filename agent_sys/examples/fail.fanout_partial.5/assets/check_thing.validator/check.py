#!/usr/bin/env python3
"""check_thing — completeness, strong.

Every row carries the keys named in args.required_row_keys and the totals
object has a consistent count. One validator shared by three kinds.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import store  # noqa: E402


def check(content: Path, required: list) -> bool:
    document = content / "items" / "text.json"
    if not document.is_file():
        return False
    try:
        data = json.loads(document.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    rows = data.get("rows")
    totals = data.get("totals")
    if not isinstance(rows, list) or not isinstance(totals, dict):
        return False
    if not rows:
        return False
    for row in rows:
        if not isinstance(row, dict) or any(key not in row for key in required):
            return False
    return totals.get("count") == len(rows)


def main() -> int:
    required = list(store.args().get("required_row_keys") or ["name", "value", "tag"])
    results = {}
    for hid in store.inputs():
        content = store.staged_content(hid) or store.content_dir(hid)
        results[hid] = False if content is None else check(content, required)
    store.write_verdict(results)
    print(f"check_thing: {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
