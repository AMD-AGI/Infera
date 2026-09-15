#!/usr/bin/env python3
"""The head of Magpie's gap analysis, as structured data.

The CSV is the artefact; this is the part the next stage reads. Keeping both
means the operator list downstream is built from a parsed number rather than from
a re-implementation of CSV parsing in whatever writes it, and the CSV is still
there for anyone who wants a row this did not keep.

    top_kernels.py <gap_analysis.csv> <top_kernels.json> [n]
"""

import csv
import json
import sys
from pathlib import Path

#: Magpie's columns, exactly. Matched by name and not by position: the reference
#: kit's own invocation passes --no-rank-csv, and a future flag that adds a
#: column would silently shift every positional read.
COLUMNS = ("Name", "Calls", "Self CUDA total (us)", "Avg time (us)", "% Total", "Input Shapes")


def number(raw: str) -> float:
    try:
        return float((raw or "").strip().replace(",", ""))
    except ValueError:
        return 0.0


def main(argv: list[str]) -> int:
    if not 2 <= len(argv) <= 3:
        print("usage: top_kernels.py <gap_analysis.csv> <top_kernels.json> [n]", file=sys.stderr)
        return 2
    src, dst = Path(argv[0]), Path(argv[1])
    top_n = int(argv[2]) if len(argv) == 3 else 25

    with src.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        missing = [c for c in COLUMNS if c not in header]
        if missing:
            print(f"top_kernels: {src.name} is missing columns {missing}", file=sys.stderr)
            print(f"  header was: {header}", file=sys.stderr)
            return 1
        rows = list(reader)

    kernels = [
        {
            "name": (row["Name"] or "").strip(),
            "calls": int(number(row["Calls"])),
            "self_cuda_us": number(row["Self CUDA total (us)"]),
            "avg_us": number(row["Avg time (us)"]),
            "pct_total": number(row["% Total"]),
            # Kept as written. Magpie packs several shape tuples into one cell
            # separated by "; ", and splitting them here would be this file
            # deciding what a shape is on the analyser's behalf.
            "input_shapes": (row["Input Shapes"] or "").strip(),
        }
        for row in rows
        if (row["Name"] or "").strip()
    ]
    # Magpie already sorts by self CUDA time; sorting again makes that
    # independent of it rather than assumed.
    kernels.sort(key=lambda k: k["self_cuda_us"], reverse=True)

    total_us = sum(k["self_cuda_us"] for k in kernels)
    head = kernels[:top_n]
    out = {
        "totals": {
            "kernels": len(kernels),
            "self_cuda_us": round(total_us, 1),
            "pct_total_sum": round(sum(k["pct_total"] for k in kernels), 2),
        },
        "top_n": top_n,
        # How much of the whole the head accounts for. The number that decides
        # whether a top-N list is a summary or a fragment.
        "top_n_share_pct": round(100 * sum(k["self_cuda_us"] for k in head) / total_us, 2)
        if total_us
        else 0.0,
        "kernels": head,
    }
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(
        f"top_kernels: {len(kernels)} kernel(s), top {len(head)} cover "
        f"{out['top_n_share_pct']}% of {out['totals']['self_cuda_us']} us"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
