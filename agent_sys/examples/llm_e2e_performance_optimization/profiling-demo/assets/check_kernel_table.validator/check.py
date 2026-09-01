#!/usr/bin/env python3
"""`check_kernel_table` — usability, strong.

`usability` and not `completeness`, because a CSV that arrived with every column
and a hundred rows can still be unusable: if the top twenty-five kernels account
for a tenth of the time, there is no operator worth picking and the next stage
would be choosing noise.

`strong` is still honest. Every rule is a count or a sum over the table, and none
of them is approximately right.

Seven rules:

1. Magpie's columns, exactly. A rename upstream stops here rather than at
   whatever parses the CSV next.
2. Enough rows to rank.
3. The shares add up to a whole run — too low and the trace covered a fraction of
   the work, above 100 and something is double-counted across ranks.
4. The head accounts for enough of the run to be worth acting on.
5. `Input Shapes` is populated. That column exists only when the capture asked
   the profiler for `record_shapes`, and without it every roofline downstream is
   impossible — but the CSV looks entirely normal, so nothing else would notice.
6. `text.json` carries **every** row of the CSV, under the field names the next
   stage reads. Rules 1–5 all judge the head; this is the one that checks the
   thing the next stage is actually handed, and the failure it catches is a
   consumer receiving a top-N where it expected a whole table.
7. Enough of the head carries a launcher frame, when the round was asked for a
   stack window. A `launcher` block is what answers "which file do I edit" for a
   symbol that is a compilation artefact, and the way that goes wrong is quietly:
   the resolution reports zero and reads exactly like a profile whose frames were
   genuinely unfindable.
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import store  # noqa: E402 — the path insert above is what makes it importable


def _fail(reasons: list, message: str) -> bool:
    reasons.append(message)
    return False


def check(content: Path, args: dict, reasons: list) -> bool:
    csv_path = content / "items" / "result" / "gap_analysis" / "gap_analysis.csv"
    json_path = content / "items" / "result" / "top_kernels.json"

    if not csv_path.is_file():
        return _fail(reasons, "items/result/gap_analysis/gap_analysis.csv is missing")

    try:
        with csv_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            header = list(reader.fieldnames or [])
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        return _fail(reasons, f"gap_analysis.csv is not readable: {exc}")

    ok = True

    for column in args.get("require_columns") or []:
        if column not in header:
            ok = _fail(reasons, f"gap_analysis.csv has no column {column!r} (header: {header})")
    if any(c not in header for c in ("Name", "Self CUDA total (us)", "% Total", "Input Shapes")):
        # Without these four nothing below can be computed; report and stop
        # rather than emit four more failures that all say the same thing.
        return ok and _fail(reasons, "gap_analysis.csv is missing the columns the remaining rules read")

    min_rows = int(args.get("min_rows", 20))
    if len(rows) < min_rows:
        ok = _fail(reasons, f"gap_analysis.csv has {len(rows)} data row(s), floor is {min_rows}")

    def number(row: dict, key: str) -> float:
        try:
            return float((row.get(key) or "0").strip().replace(",", ""))
        except ValueError:
            return 0.0

    pct_sum = sum(number(r, "% Total") for r in rows)
    low = float(args.get("min_pct_total_sum", 80.0))
    high = float(args.get("max_pct_total_sum", 100.5))
    if not low <= pct_sum <= high:
        ok = _fail(reasons, f"the '% Total' column sums to {pct_sum:.2f}, want {low}..{high}")

    if args.get("require_input_shapes", True):
        with_shapes = sum(1 for r in rows if (r.get("Input Shapes") or "").strip())
        if with_shapes == 0:
            ok = _fail(
                reasons,
                "no row carries Input Shapes — the capture did not ask the profiler "
                "for record_shapes, and no roofline can be built from this table",
            )

    # The head's share, from the summary the producer built. Recomputed from the
    # CSV rather than trusted, for the same reason `check_facts` recomputes its
    # totals: a number a producer wrote about itself is not evidence.
    top_n = 25
    share = 0.0
    if json_path.is_file():
        try:
            summary = json.loads(json_path.read_text(encoding="utf-8"))
            top_n = int(summary.get("top_n") or top_n)
        except (OSError, json.JSONDecodeError) as exc:
            ok = _fail(reasons, f"top_kernels.json is not readable JSON: {exc}")
    else:
        ok = _fail(reasons, "items/result/top_kernels.json is missing")

    times = sorted((number(r, "Self CUDA total (us)") for r in rows), reverse=True)
    total = sum(times)
    if total <= 0:
        ok = _fail(reasons, "the 'Self CUDA total (us)' column sums to zero")
    else:
        share = 100.0 * sum(times[:top_n]) / total
        floor = float(args.get("min_top_n_share_pct", 50.0))
        if share < floor:
            ok = _fail(
                reasons,
                f"the top {top_n} kernels account for {share:.1f}% of CUDA time, "
                f"floor is {floor}% — this table has no head worth acting on",
            )

    ok = _check_document(content, args, rows, reasons) and ok

    reasons.append(f"(note) {len(rows)} kernels, top {top_n} cover {share:.1f}%, shares sum to {pct_sum:.2f}")
    return ok


def _check_document(content: Path, args: dict, rows: list, reasons: list) -> bool:
    """Rules 6 and 7: the document the next stage reads, and its launcher frames."""
    document_path = content / "items" / "result" / "text.json"
    if not document_path.is_file():
        return _fail(
            reasons,
            "items/result/text.json is missing — the CSV is the artefact but this is "
            "what the operator-selection stage reads",
        )
    try:
        document = json.loads(document_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _fail(reasons, f"text.json is not readable JSON: {exc}")

    ok = True
    kernels = document.get("kernels")
    if not isinstance(kernels, list):
        return _fail(reasons, "text.json has no 'kernels' array")

    # Rule 6. Every row, not the head. The next stage buckets every kernel before
    # it sorts, so a truncated table costs it candidates rather than noise.
    if len(kernels) != len(rows):
        ok = _fail(
            reasons,
            f"text.json carries {len(kernels)} kernel(s) against {len(rows)} row(s) in "
            f"the CSV — the next stage classifies every row before it ranks, so a "
            f"partial table silently drops candidates",
        )

    # The consumer's field names, checked here rather than discovered there. A
    # rename on either side turns into a column of nulls three tasks away.
    required = ("name", "calls", "self_us", "avg_us", "pct_total", "input_shapes")
    if kernels:
        missing = [field for field in required if field not in kernels[0]]
        if missing:
            ok = _fail(reasons, f"text.json kernels are missing field(s) {missing}")
    unnamed = sum(1 for kernel in kernels if not (kernel.get("name") or "").strip())
    if unnamed:
        ok = _fail(reasons, f"{unnamed} row(s) in text.json carry no kernel name")

    # Rule 7. Launcher coverage over the head, not over the whole table: a
    # sub-percent kernel may legitimately not appear in a three-second window,
    # and requiring it would fail a correct capture.
    want = int(args.get("min_launchers_in_top_n", 0) or 0)
    summary = document.get("launchers") or {}
    if want > 0:
        if not summary.get("available"):
            return ok and _fail(
                reasons,
                f"no launcher frames were resolved ({summary.get('reason') or 'no reason recorded'}) "
                f"and this round wanted at least {want} in the head. Set "
                f"--var stack_window_s=0 and min_launchers_in_top_n to 0 to say that is intended",
            )
        head = kernels[: int(args.get("launcher_head_n", 25) or 25)]
        with_frames = [k for k in head if (k.get("launcher") or {}).get("source_file")]
        if len(with_frames) < want:
            ok = _fail(
                reasons,
                f"{len(with_frames)} of the top {len(head)} kernel(s) carry a launcher "
                f"frame, floor is {want}. Resolved {summary.get('resolved')} of "
                f"{summary.get('wanted')} overall, {summary.get('unmapped')} outside a "
                f"known container root",
            )
        # A frame with no owner cannot be turned into a repository by the
        # consumer, so it is present without being usable.
        ownerless = [k for k in with_frames if not (k.get("launcher") or {}).get("container_root")]
        if ownerless:
            ok = _fail(
                reasons,
                f"{len(ownerless)} launcher frame(s) name no container_root, so the "
                f"consumer has no repository to resolve them against",
            )
        reasons.append(
            f"(note) launchers: {len(with_frames)}/{len(head)} of the head, "
            f"by owner {summary.get('by_owner')}, by form {summary.get('by_path_form')}"
        )
    return ok


def main() -> int:
    args = store.args()
    results = {}
    for hid in store.inputs():
        content = store.staged_content(hid) or store.content_dir(hid)
        reasons: list = []
        if content is None:
            results[hid] = False
            reasons.append("no published content for this handoff")
        else:
            results[hid] = check(content, args, reasons)
        print(f"check_kernel_table: {hid} {'PASS' if results[hid] else 'FAIL'}")
        for reason in reasons:
            print(f"  - {reason}")
    store.write_verdict(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
