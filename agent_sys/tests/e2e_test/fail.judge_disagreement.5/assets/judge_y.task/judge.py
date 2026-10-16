#!/usr/bin/env python3
"""Reviewer Y: marks student_a as revise, accepts the others.

The disagreement on student_a is what makes check_agree fail.
"""

import json
import os
import sys
from pathlib import Path

README = """# review_y

## Purpose

Reviewer Y's verdict on each student, independently of reviewer X.

## Schema

`items/text.json` has `reviews` (a list of per-student verdicts).
"""

REVIEWS = [
    {"student": "student_a", "verdict": "revise", "comment": "Time complexity is suboptimal."},
    {"student": "student_b", "verdict": "accept", "comment": "Correct solution."},
    {"student": "student_c", "verdict": "accept", "comment": "Well structured."},
]


def main() -> int:
    dst = Path(os.environ["AGENT_SYS_OUTPUT_REVIEW_Y"])
    (dst / "items").mkdir(parents=True, exist_ok=True)
    (dst / "README.md").write_text(README, encoding="utf-8")
    (dst / "items" / "text.json").write_text(
        json.dumps({"reviews": REVIEWS}, indent=2),
        encoding="utf-8",
    )
    print(f"judge_y: {len(REVIEWS)} reviews -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
