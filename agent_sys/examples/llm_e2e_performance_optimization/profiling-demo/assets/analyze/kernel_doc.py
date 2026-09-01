#!/usr/bin/env python3
"""The whole gap analysis as one JSON document, with launcher frames merged in.

`top_kernels.py` publishes the head of the ranking, which is what this package's
own validator judges. This publishes **every** row, in the shape the next stage
reads, and it exists because those are two different jobs:

* the head answers "is this table worth acting on", which is a question about
  this round and is `check_kernel_table`'s to answer;
* the whole table is the next stage's input, and it must be whole, because
  `analyze-demo`'s `rank` classifies every row into a bucket before it sorts --
  collectives held 79% of GPU time in the sample profile, so a top-25 hand-off
  would have thrown away most of the routable candidates along with the noise.

**The field names here are the consumer's, not this package's.** They match
`analyze-demo/assets/lib/csv_io.py:to_record` and `seed_table`'s document exactly
-- `self_us` and not `self_cuda_us`, `pct_total`, `input_shapes`, and an optional
`launcher` block per kernel. The consumer's `kernel_table` kind reserves that
block (its DESIGN.md section 4.4) and its `identify` reads it as resolution level
one. Writing the consumer's names means the join between the two packages is a
wiring change rather than a translation layer.

Usage:

    kernel_doc.py --csv <gap_analysis.csv> --out <text.json> \
                  [--launchers <launchers.json>] [--trace-manifest <manifest.json>]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

#: Magpie's base columns, matched by name and never by position. Duplicated from
#: `analyze-demo/assets/lib/csv_io.py:BASE_COLUMNS`, which duplicates them from
#: Magpie; the bound on the duplication is that it is only these six names.
BASE_COLUMNS = [
    "Name",
    "Calls",
    "Self CUDA total (us)",
    "Avg time (us)",
    "% Total",
    "Input Shapes",
]

#: `KernelSourceInfo.csv_headers()`, appended by Magpie's `--find-kernel-sources`.
#: This package does not pass that flag, so the columns are absent and `enriched`
#: is false; the list is here so that a run which does pass it is reported as
#: enriched rather than silently read as a base file.
SOURCE_COLUMNS = [
    "kind",
    "category",
    "source_repo",
    "source_file",
    "upstream_url",
    "test_file",
    "test_cmd",
    "baseline_ref_file",
    "baseline_ref_symbol",
    "baseline_ref_kind",
    "triton_ref_file",
    "triton_ref_symbol",
    "notes",
]


def _number(raw: str, cast):
    text = (raw or "").strip().replace(",", "")
    if not text:
        return None
    try:
        return cast(text)
    except ValueError:
        return None


def read_csv(path: Path) -> tuple[list[dict], list[str], dict]:
    """`(rows, columns, repo_paths)` from a Magpie gap analysis.

    An enriched file is prefixed with `#`-comment rows mapping repo variables to
    paths. Those are host paths and never reach the handoff -- only the variable
    names do -- so they are read here and dropped in `build`.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    repo_paths: dict[str, str] = {}
    first = 0
    for index, line in enumerate(lines):
        if not line.startswith("#"):
            first = index
            break
        body = line.lstrip("#").strip()
        if "=" in body:
            name, _, where = body.partition("=")
            repo_paths[name.strip()] = where.strip()
    else:
        raise SystemExit(f"kernel_doc: {path} is all comments and has no header")

    reader = csv.DictReader(lines[first:])
    columns = list(reader.fieldnames or [])
    missing = [column for column in BASE_COLUMNS if column not in columns]
    if missing:
        raise SystemExit(
            f"kernel_doc: {path} is missing column(s) {missing}\n"
            f"  header was: {columns}"
        )
    return [dict(row) for row in reader], columns, repo_paths


def build(
    csv_path: Path,
    *,
    launchers: dict,
    trace_manifest: dict,
) -> dict:
    rows, columns, repo_paths = read_csv(csv_path)
    payload = csv_path.read_bytes()

    frames = (launchers or {}).get("launchers") or {}
    records = []
    matched = 0
    for row in rows:
        name = (row.get("Name") or "").strip()
        if not name:
            continue
        record = {
            "name": name,
            "calls": _number(row.get("Calls"), int),
            "self_us": _number(row.get("Self CUDA total (us)"), float),
            "avg_us": _number(row.get("Avg time (us)"), float),
            "pct_total": _number(row.get("% Total"), float),
            # Kept exactly as Magpie wrote it. Several shape tuples are packed
            # into one cell separated by "; ", and splitting them here would be
            # this file deciding what a shape is on the consumer's behalf.
            "input_shapes": (row.get("Input Shapes") or "").strip(),
        }
        for column in SOURCE_COLUMNS:
            if column in row and (row.get(column) or "").strip():
                record.setdefault("source_info", {})[column] = (row[column] or "").strip()
        frame = frames.get(name)
        if frame:
            record["launcher"] = frame
            matched += 1
        records.append(record)

    pct_sum = sum(record["pct_total"] or 0.0 for record in records)
    totals = (trace_manifest or {}).get("totals") or {}

    return {
        # Provenance by digest and never by directory: `handoff/locality.py`
        # refuses to seal content naming an absolute host path, so the artefact
        # is identified exactly and the machine is not named at all.
        "source": {
            "filename": csv_path.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
            # Which traces this ranking is of, carried over from the capture's own
            # manifest so the two cannot drift apart unnoticed.
            "trace_files": totals.get("files"),
            "trace_gpu_kernels": totals.get("gpu_kernels"),
        },
        "enriched": all(column in columns for column in SOURCE_COLUMNS),
        "columns": columns,
        # Names only. The paths behind them are this machine's.
        "repo_variables": sorted(repo_paths),
        "totals": {"rows": len(records), "pct_total_sum": round(pct_sum, 3)},
        # How much of the table carries a launcher, so a consumer can tell "the
        # stack window was not taken" from "it was taken and resolved nothing".
        "launchers": {
            "available": bool((launchers or {}).get("available")),
            "reason": (launchers or {}).get("reason") or "",
            "wanted": (launchers or {}).get("wanted") or 0,
            "resolved": (launchers or {}).get("resolved") or 0,
            "matched_rows": matched,
            "by_owner": (launchers or {}).get("by_owner") or {},
            "by_path_form": (launchers or {}).get("by_path_form") or {},
            "unmapped": len((launchers or {}).get("unmapped") or []),
        },
        "kernels": records,
    }


def _load(path: Path | None, label: str) -> dict:
    if path is None:
        return {}
    if not path.is_file():
        print(f"kernel_doc: no {label} at {path.name}; continuing without it", file=sys.stderr)
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        print(f"kernel_doc: {label} is not valid JSON: {error}", file=sys.stderr)
        return {}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--launchers", type=Path, default=None)
    parser.add_argument("--trace-manifest", type=Path, default=None)
    args = parser.parse_args(argv)

    document = build(
        args.csv,
        launchers=_load(args.launchers, "launchers.json"),
        trace_manifest=_load(args.trace_manifest, "trace manifest"),
    )
    args.out.write_text(json.dumps(document, indent=2), encoding="utf-8")

    launcher_summary = document["launchers"]
    print(
        f"kernel_doc: {document['totals']['rows']} kernel(s), "
        f"shares sum to {document['totals']['pct_total_sum']}, "
        f"{'enriched' if document['enriched'] else 'base'} header, "
        f"{launcher_summary['matched_rows']} row(s) carry a launcher -> {args.out.name}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
