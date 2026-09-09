#!/usr/bin/env python3
"""Merge two reviews field by field. Exit 0 — the merge is correct.

check_agree then checks whether the disagreed list is empty. A non-empty
list means the two reviewers reached different conclusions on at least one
student, and check_agree returns False.
"""

import json
import os
import sys
from pathlib import Path

README = """# review

## Purpose

The two reviewers' verdicts merged, with every disagreement listed.

## Schema

`items/text.json` has `agreed`, `disagreed`, and `totals`.
"""


def main() -> int:
    rx_dir = Path(os.environ["AGENT_SYS_INPUT_REVIEW_X"])
    ry_dir = Path(os.environ["AGENT_SYS_INPUT_REVIEW_Y"])
    dst = Path(os.environ["AGENT_SYS_OUTPUT_REVIEW"])

    rx = json.loads((rx_dir / "items" / "text.json").read_text(encoding="utf-8"))
    ry = json.loads((ry_dir / "items" / "text.json").read_text(encoding="utf-8"))

    by_student_x = {r["student"]: r for r in rx["reviews"]}
    by_student_y = {r["student"]: r for r in ry["reviews"]}

    agreed = []
    disagreed = []
    for student in sorted(set(by_student_x) | set(by_student_y)):
        vx = by_student_x.get(student, {}).get("verdict")
        vy = by_student_y.get(student, {}).get("verdict")
        if vx == vy:
            agreed.append({
                "student": student,
                "verdict": vx,
                "comment_x": by_student_x.get(student, {}).get("comment", ""),
                "comment_y": by_student_y.get(student, {}).get("comment", ""),
            })
        else:
            disagreed.append({
                "student": student,
                "verdict_x": vx,
                "verdict_y": vy,
                "comment_x": by_student_x.get(student, {}).get("comment", ""),
                "comment_y": by_student_y.get(student, {}).get("comment", ""),
            })

    merged = {
        "agreed": agreed,
        "disagreed": disagreed,
        "totals": {
            "students": len(set(by_student_x) | set(by_student_y)),
            "agreed": len(agreed),
            "disagreed": len(disagreed),
        },
    }

    (dst / "items").mkdir(parents=True, exist_ok=True)
    (dst / "README.md").write_text(README, encoding="utf-8")
    (dst / "items" / "text.json").write_text(
        json.dumps(merged, indent=2),
        encoding="utf-8",
    )
    print(f"reconcile: {len(agreed)} agreed, {len(disagreed)} disagreed -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
