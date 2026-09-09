#!/usr/bin/env python3
"""`check_worklist_shape` — completeness, strong.

The worklist is a complete, auditable, ordered document. Six rules:

1. Every kernel carries a `bucket` from the known set.
2. Every non-selected kernel carries a non-empty `excluded_reason`, and every
   selected one carries an empty string. The two are complements, so a row that
   is neither selected nor explained is a hole in the audit trail.
3. The selected count is between `min_selected` and the declared `top_n`.
4. The selected rows are ranked 1..N contiguously and ordered by descending
   share of GPU time.
5. Every selected row carries at least one case with a `CASE_ID` selector,
   because a row with no shapes cannot become a workset.
6. The unclassified share is at or below `max_unknown_ratio`.

Rule 6 is the one that ages: a workload the taxonomy has not seen pushes rows
into `unknown`, and past the threshold the rules in
`assets/lib/kernel_taxonomy.yaml` have fallen behind and need a look.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import store  # noqa: E402

BUCKETS = {"collective", "vendor_tuned", "framework_native", "routable", "unknown"}


def check(content: Path, args: dict) -> tuple[bool, str]:
    document = content / "items" / "text.json"
    if not document.is_file():
        return False, "items/text.json is absent"
    try:
        data = json.loads(document.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return False, f"items/text.json is not valid JSON: {error}"

    for name in ("items/schema", "items/worklist.csv"):
        if not (content / name).is_file():
            return False, f"{name} is absent"

    kernels = data.get("kernels")
    if not isinstance(kernels, list) or not kernels:
        return False, "kernels is absent or empty"

    for row in kernels:
        if row.get("bucket") not in BUCKETS:
            return False, f"{row.get('name', '?')[:40]}: bucket {row.get('bucket')!r} is not known"
        selected = bool(row.get("selected"))
        reason = (row.get("excluded_reason") or "").strip()
        if selected and reason:
            return False, f"{row.get('name', '?')[:40]}: selected but carries reason {reason!r}"
        if not selected and not reason:
            return False, f"{row.get('name', '?')[:40]}: excluded with no reason"

    selected = [r for r in kernels if r.get("selected")]
    top_n = int((data.get("thresholds") or {}).get("top_n") or 0)
    minimum = int(args.get("min_selected") or 1)
    if len(selected) < minimum:
        return False, f"{len(selected)} selected, need at least {minimum}"
    if top_n and len(selected) > top_n:
        return False, f"{len(selected)} selected, exceeds declared top_n {top_n}"

    ranks = sorted(int(r.get("rank") or 0) for r in selected)
    if ranks != list(range(1, len(selected) + 1)):
        return False, f"ranks are not 1..{len(selected)}: {ranks}"

    ordered = sorted(selected, key=lambda r: int(r["rank"]))
    percentages = [float(r.get("pct_total") or 0.0) for r in ordered]
    if any(a < b for a, b in zip(percentages, percentages[1:])):
        return False, f"selected rows are not ordered by descending % Total: {percentages}"

    for row in ordered:
        cases = row.get("cases") or []
        if not cases:
            return False, f"{row.get('name', '?')[:40]}: selected but carries no cases"
        for case in cases:
            if "CASE_ID" not in (case.get("selector") or {}):
                return False, f"{row.get('name', '?')[:40]}: a case selector has no CASE_ID"

    unknown = sum(1 for r in kernels if r.get("bucket") == "unknown")
    ratio = unknown / len(kernels)
    ceiling = float(args.get("max_unknown_ratio") or 0.3)
    if ratio > ceiling:
        return False, (
            f"unclassified share {ratio:.3f} exceeds {ceiling}; "
            f"extend assets/lib/kernel_taxonomy.yaml"
        )

    return True, (
        f"{len(kernels)} kernels, {len(selected)} selected, "
        f"unclassified {ratio:.3f}, top rank {percentages[0]:.2f}%"
    )


def main() -> int:
    args = store.args()
    results = {}
    for hid in store.inputs():
        content = store.staged_content(hid) or store.content_dir(hid)
        if content is None:
            results[hid] = False
            print(f"check_worklist_shape: {hid}: no published content")
            continue
        ok, why = check(content, args)
        results[hid] = ok
        print(f"check_worklist_shape: {hid}: {'PASS' if ok else 'FAIL'} — {why}")
    store.write_verdict(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
