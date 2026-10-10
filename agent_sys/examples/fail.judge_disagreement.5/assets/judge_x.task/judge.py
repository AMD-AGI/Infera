#!/usr/bin/env python3
"""Reviewer X: accepts all students."""

import json
import os
import sys
from pathlib import Path

README = """# review_x

## Purpose

Reviewer X's verdict on each student.

## Schema

`items/text.json` has `reviews` (a list of per-student verdicts).
"""

REVIEWS = [
    {"student": "student_a", "verdict": "accept", "comment": "Solution is correct and efficient."},
    {"student": "student_b", "verdict": "accept", "comment": "Clean implementation."},
    {"student": "student_c", "verdict": "accept", "comment": "Good approach."},
]


def main() -> int:
    dst = Path(os.environ["AGENT_SYS_OUTPUT_REVIEW_X"])
    (dst / "items").mkdir(parents=True, exist_ok=True)
    (dst / "README.md").write_text(README, encoding="utf-8")
    (dst / "items" / "text.json").write_text(
        json.dumps({"reviews": REVIEWS}, indent=2),
        encoding="utf-8",
    )
    print(f"judge_x: {len(REVIEWS)} reviews -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
