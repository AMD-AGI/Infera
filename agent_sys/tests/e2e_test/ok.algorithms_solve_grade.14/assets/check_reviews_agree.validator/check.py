#!/usr/bin/env python3
"""`check_reviews_agree` — trustworthiness, strong.

The reconciled review's `disagreed` list is empty, and its `totals` are what the
two lists actually contain. Total over the document: every count is recomputed
rather than read, and the two lists are checked for the same pair appearing in
both.

**What a pass here means, and does not**: two independent reviewers reached the
same verdict on every pair. That is evidence of *consistency*. It is not
evidence of correctness — see `readme.md` beside this file.

Writes one boolean per handoff id in `inputs.json`.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import store  # noqa: E402 — the path insert above is what makes it importable


def pairs_of(rows: list) -> list[tuple[str, str]]:
    return [
        (str(row.get("student")), str(row.get("problem_id")))
        for row in rows
        if isinstance(row, dict)
    ]


def check(content: Path) -> tuple[bool, str]:
    """`(verdict, why)`. The reason is printed, so a failure names itself."""
    document = content / "items" / "text.json"
    if not document.is_file():
        return False, "items/text.json does not exist"
    try:
        data = json.loads(document.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"items/text.json is not valid JSON: {exc}"
    if not isinstance(data, dict):
        return False, "items/text.json is not an object"

    agreed = data.get("agreed")
    disagreed = data.get("disagreed")
    totals = data.get("totals")
    if not isinstance(agreed, list) or not isinstance(disagreed, list):
        return False, "'agreed' and 'disagreed' must both be lists"
    if not isinstance(totals, dict):
        return False, "'totals' must be an object"

    agreed_pairs = pairs_of(agreed)
    disagreed_pairs = pairs_of(disagreed)
    if len(agreed_pairs) != len(agreed) or len(disagreed_pairs) != len(disagreed):
        return False, "every entry of 'agreed' and 'disagreed' must be an object"

    # Recomputed, never read. A merger that got the arithmetic wrong and a
    # merger that dropped a pair produce the same `totals` if the totals are
    # trusted.
    if totals.get("agreed") != len(agreed):
        return False, f"totals.agreed is {totals.get('agreed')!r}, the list holds {len(agreed)}"
    if totals.get("disagreed") != len(disagreed):
        return (
            False,
            f"totals.disagreed is {totals.get('disagreed')!r}, the list holds {len(disagreed)}",
        )
    if totals.get("pairs") != len(agreed) + len(disagreed):
        return (
            False,
            f"totals.pairs is {totals.get('pairs')!r}, the two lists hold "
            f"{len(agreed) + len(disagreed)}",
        )

    # One pass over both lists together, so *appears in both* and *appears
    # twice in one* are the same test. Counting them separately is how the
    # second one gets forgotten.
    all_pairs = agreed_pairs + disagreed_pairs
    if len(set(all_pairs)) != len(all_pairs):
        repeated = sorted({pair for pair in all_pairs if all_pairs.count(pair) > 1})
        return False, f"pairs {repeated} are listed more than once across the two lists"

    if disagreed:
        listed = ", ".join(f"{s}/{p}" for s, p in disagreed_pairs[:5])
        more = "" if len(disagreed_pairs) <= 5 else f" (and {len(disagreed_pairs) - 5} more)"
        return False, f"the two reviewers disagree on {len(disagreed)} pair(s): {listed}{more}"

    return True, f"{len(agreed)} pairs, all agreed, totals recomputed and correct"


def main() -> int:
    results = {}
    for hid in store.inputs():
        # `materials.json` first: it is the declared route, and a body reading it
        # needs to know neither that a store exists nor where it is.
        content = store.staged_content(hid) or store.content_dir(hid)
        if content is None:
            results[hid], why = False, "no staged content"
        else:
            results[hid], why = check(content)
        print(f"check_reviews_agree: {hid}: {results[hid]} — {why}")
    store.write_verdict(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
