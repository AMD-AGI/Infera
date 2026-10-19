#!/usr/bin/env python3
"""check_agree — trustworthiness, strong.

The merged review must have an empty disagreed list. A non-empty list means
two independent reviewers reached different conclusions, which is evidence
of inconsistency rather than correctness.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import store  # noqa: E402


def check(content: Path) -> bool:
    document = content / "items" / "text.json"
    if not document.is_file():
        return False
    try:
        data = json.loads(document.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    disagreed = data.get("disagreed")
    if not isinstance(disagreed, list):
        return False
    totals = data.get("totals", {})
    if totals.get("disagreed") != len(disagreed):
        return False
    return len(disagreed) == 0


def main() -> int:
    results = {}
    for hid in store.inputs():
        content = store.staged_content(hid) or store.content_dir(hid)
        results[hid] = False if content is None else check(content)
        if not results[hid]:
            print(f"check_agree: {hid}: disagreements found or no content")
    store.write_verdict(results)
    print(f"check_agree: {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
