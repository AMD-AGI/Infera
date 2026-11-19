#!/usr/bin/env python3
"""Compare a PR run with its base run (pytest junit XML): print the counts of both and every test that fails or errors
on the PR but not on the base (new tests included). Exit 1 if there is any such test, or if the PR run is missing or
collected nothing; failures that the base has too are listed as pre-existing and do not fail the step.
  junit_diff.py BASE.xml PR.xml
"""

import sys
import xml.etree.ElementTree as ET


def load(path: str) -> tuple[dict[str, str], dict[str, int]] | None:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return None
    results: dict[str, str] = {}
    for case in root.iter("testcase"):
        name = f"{case.get('classname')}::{case.get('name')}"
        status = "passed"
        for child in case:
            if child.tag in ("failure", "error"):
                status = "failed"
            elif child.tag == "skipped" and status == "passed":
                status = "skipped"
        results[name] = status
    counts = {s: sum(v == s for v in results.values()) for s in ("passed", "failed", "skipped")}
    return results, counts


base = load(sys.argv[1])
pr = load(sys.argv[2])
if pr is None or not pr[0]:
    print(f"PR run {sys.argv[2]} missing or empty")
    sys.exit(1)
# A module that fails to import shows up as one failed case without a class; nothing in it ran.
broken = [n for n, s in pr[0].items() if s == "failed" and n.startswith("::")]
if broken or pr[1]["passed"] == 0:
    print(f"pr:   {pr[1]}")
    for n in broken:
        print(f"COLLECTION ERROR: {n}")
    print("verdict: FAIL (nothing ran)" if not broken else "verdict: FAIL (collection error)")
    sys.exit(1)
print(f"base: {base[1] if base else 'missing'}")
print(f"pr:   {pr[1]}")
base_results = base[0] if base else {}
new = [n for n, s in pr[0].items() if s == "failed" and base_results.get(n) != "failed"]
old = [n for n, s in pr[0].items() if s == "failed" and base_results.get(n) == "failed"]
for n in old:
    print(f"pre-existing failure: {n}")
for n in new:
    print(f"NEW FAILURE: {n}")
print("verdict:", "FAIL" if new else "PASS")
sys.exit(1 if new else 0)
