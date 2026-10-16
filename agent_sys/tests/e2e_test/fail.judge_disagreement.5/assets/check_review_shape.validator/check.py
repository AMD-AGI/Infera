#!/usr/bin/env python3
"""check_review_shape — completeness, strong.

Every review row carries the required keys and the verdict is from a closed set.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import store  # noqa: E402


def check(content: Path, required: list, valid_verdicts: list) -> bool:
    document = content / "items" / "text.json"
    if not document.is_file():
        return False
    try:
        data = json.loads(document.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    reviews = data.get("reviews")
    if not isinstance(reviews, list) or not reviews:
        return False
    for row in reviews:
        if not isinstance(row, dict):
            return False
        if any(key not in row for key in required):
            return False
        if valid_verdicts and row.get("verdict") not in valid_verdicts:
            return False
    return True


def main() -> int:
    a = store.args()
    required = list(a.get("required_row_keys") or ["student", "verdict", "comment"])
    valid_verdicts = list(a.get("valid_verdicts") or [])
    results = {}
    for hid in store.inputs():
        content = store.staged_content(hid) or store.content_dir(hid)
        results[hid] = False if content is None else check(content, required, valid_verdicts)
    store.write_verdict(results)
    print(f"check_review_shape: {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
