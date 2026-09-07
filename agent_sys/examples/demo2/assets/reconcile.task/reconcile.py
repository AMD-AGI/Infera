#!/usr/bin/env python3
"""What `reconcile` runs: merge two independent reviews into one document.

**This file imports nothing from `agent_sys`.** It is package data, run as a
subprocess by `agent.backends.program.ProgramExecutor`. Everything it needs
arrives as an environment variable:

| | |
|---|---|
| `AGENT_SYS_INPUT_REVIEW_X` | reviewer X's content directory |
| `AGENT_SYS_INPUT_REVIEW_Y` | reviewer Y's content directory |
| `AGENT_SYS_OUTPUT_REVIEW` | where to write `README.md` and `items/` |

It merges and it does not judge — see `readme.md` beside this file for why
those are two jobs. A pair the reviewers answered differently lands in
`disagreed` with both answers intact, and `check_reviews_agree` is what fails
the run over it.
"""

import json
import os
import sys
from pathlib import Path

#: The fields two reviewers must match on. `comment` is deliberately absent:
#: it is prose, and requiring two reviewers to phrase one verdict identically
#: would make agreement a test of language rather than of judgement.
JUDGEMENT_FIELDS = ("implements_claimed_algorithm", "complexity_credible", "verdict")

README = """# review

## Purpose

The two independent reviews of every submitted (student, problem) pair, merged
into one document. Reviewer X and reviewer Y saw the same submissions and could
not see each other's answers; this records where they agree, where they do not,
and how many of each.

**Nothing is resolved here.** A pair the two answered differently is copied into
`disagreed` carrying both answers, because a body that picked a winner would be
manufacturing the very fact `check_reviews_agree` then checks.

## Schema

`items/text.json` is a JSON object:

```json
{{"agreed": [{{"student": "a", "problem_id": "p1",
             "implements_claimed_algorithm": true, "complexity_credible": true,
             "verdict": "accept", "comment_x": "...", "comment_y": "..."}}],
 "disagreed": [{{"student": "b", "problem_id": "p1",
                "x": {{"implements_claimed_algorithm": true,
                      "complexity_credible": true,
                      "verdict": "accept", "comment": "..."}},
                "y": {{...same shape, or null when that reviewer had no row...}}}}],
 "totals": {{"pairs": {pairs}, "agreed": {agreed}, "disagreed": {disagreed}}}}}
```

A pair is `(student, problem_id)`. Two rows agree when
`implements_claimed_algorithm`, `complexity_credible` and `verdict` are all
equal; `comment` is not compared. A pair present in only one review is a
disagreement, with the absent side recorded as `null`.

`totals.pairs` is the size of the union of the two reviews' pairs, so
`agreed + disagreed == pairs` holds by construction and is recomputable from
the two lists.

This run merged **{pairs}** pairs: {agreed} agreed, {disagreed} disagreed.
"""


def _required(name: str) -> str:
    """A named refusal instead of a bare `KeyError`.

    `env_mgr.grants.output_env` exports `AGENT_SYS_OUTPUT_<KIND>` per output
    slot at every dispatch, and `AGENT_SYS_INPUT_<KIND>` per input slot. It
    exports *nothing* for a kind naming two slots, deliberately — so a missing
    variable is a real condition and not something to default away.
    """
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"{name} is not set. A task body reads its inputs from "
            f"AGENT_SYS_INPUT_<KIND> and writes into AGENT_SYS_OUTPUT_<KIND>, "
            f"exported per slot by env_mgr.grants at dispatch."
        )
    return value


def rows_of(content: Path) -> list[dict]:
    """The `reviews` list of one review document.

    A malformed document is a `SystemExit` naming the file rather than a
    traceback: this body is downstream of two model calls, and *which* of the
    two produced unreadable JSON is the first thing anybody debugging will ask.
    """
    document = content / "items" / "text.json"
    if not document.is_file():
        raise SystemExit(f"{document} does not exist; the review was not written")
    try:
        data = json.loads(document.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{document}: not valid JSON: {exc}") from exc
    reviews = data.get("reviews") if isinstance(data, dict) else None
    if not isinstance(reviews, list):
        raise SystemExit(f"{document}: expected an object with a 'reviews' list")
    return [row for row in reviews if isinstance(row, dict)]


def by_pair(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Index rows by `(student, problem_id)`, last write wins.

    A duplicate pair is not repaired here. `check_review_shape` is total over
    each review document and rejects one, and repairing it silently would hide
    a fault from the validator whose whole job is to find it.
    """
    return {(str(row.get("student")), str(row.get("problem_id"))): row for row in rows}


def judgement(row: dict | None) -> dict | None:
    if row is None:
        return None
    out = {field: row.get(field) for field in JUDGEMENT_FIELDS}
    out["comment"] = row.get("comment")
    return out


def merge(x_rows: list[dict], y_rows: list[dict]) -> dict:
    """The whole merge. Sorted output, so two runs are byte-identical."""
    x_by_pair = by_pair(x_rows)
    y_by_pair = by_pair(y_rows)
    agreed: list[dict] = []
    disagreed: list[dict] = []

    for pair in sorted(set(x_by_pair) | set(y_by_pair)):
        student, problem_id = pair
        x_row = x_by_pair.get(pair)
        y_row = y_by_pair.get(pair)
        same = (
            x_row is not None
            and y_row is not None
            and all(x_row.get(f) == y_row.get(f) for f in JUDGEMENT_FIELDS)
        )
        if same:
            assert x_row is not None and y_row is not None  # narrowed by `same`
            entry = {"student": student, "problem_id": problem_id}
            entry.update({field: x_row.get(field) for field in JUDGEMENT_FIELDS})
            entry["comment_x"] = x_row.get("comment")
            entry["comment_y"] = y_row.get("comment")
            agreed.append(entry)
        else:
            disagreed.append(
                {
                    "student": student,
                    "problem_id": problem_id,
                    "x": judgement(x_row),
                    "y": judgement(y_row),
                }
            )

    return {
        "agreed": agreed,
        "disagreed": disagreed,
        "totals": {
            "pairs": len(agreed) + len(disagreed),
            "agreed": len(agreed),
            "disagreed": len(disagreed),
        },
    }


def main() -> int:
    x_content = Path(_required("AGENT_SYS_INPUT_REVIEW_X"))
    y_content = Path(_required("AGENT_SYS_INPUT_REVIEW_Y"))
    dst = Path(_required("AGENT_SYS_OUTPUT_REVIEW"))

    merged = merge(rows_of(x_content), rows_of(y_content))
    totals = merged["totals"]

    (dst / "items").mkdir(parents=True, exist_ok=True)
    (dst / "README.md").write_text(README.format(**totals), encoding="utf-8")
    (dst / "items" / "text.json").write_text(
        json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"reconcile: {totals['pairs']} pairs, {totals['agreed']} agreed, "
        f"{totals['disagreed']} disagreed -> {dst}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
