#!/usr/bin/env python3
"""`check_facts` — completeness, strong.

Total over the document: every row carries the keys `args.json` names, and both
totals are recomputed rather than trusted. Writes one boolean per handoff id in
`inputs.json`.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import store  # noqa: E402 — the path insert above is what makes it importable


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
    for row in rows:
        if not isinstance(row, dict) or any(key not in row for key in required):
            return False
    return totals.get("files") == len(rows) and totals.get("lines") == sum(
        int(row.get("lines") or 0) for row in rows
    )


def main() -> int:
    required = list(store.args().get("required_row_keys") or ["path", "lines", "sha256_prefix"])
    results = {}
    for hid in store.inputs():
        # `materials.json` first: it is the declared route, the copies are
        # verified on the way out, and a body reading it needs to know neither
        # that a store exists nor where it is. `content_dir` is the fallback for
        # a run with no `env_mgr` wired, where nothing was staged.
        content = store.staged_content(hid) or store.content_dir(hid)
        # A handoff with no published content is not a pass. `dict.get` folding
        # `None` as falsy is what DeepEval's unreached DAG node did, and it
        # reports identically to a real zero.
        results[hid] = False if content is None else check(content, required)
    store.write_verdict(results)
    print(f"check_facts: {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
