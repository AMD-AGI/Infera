#!/usr/bin/env python3
"""Reading Magpie gap_analysis CSV in both of its shapes.

Magpie writes six columns by default. With `--find-kernel-sources` it appends
the thirteen columns of `Magpie/tools/amd_kernel_finder/models.py:KernelSourceInfo`
to the same file, and prefixes the file with `#`-comment rows mapping repo
variables to paths. Both shapes reach this package as the same handoff kind, so
the reader has to accept either and say which it got.

Imports nothing from `agent_sys` and nothing from Magpie: this is package data
run as a subprocess, and the only contract it depends on is the column names.
"""

from __future__ import annotations

import csv
from pathlib import Path

BASE_COLUMNS = [
    "Name",
    "Calls",
    "Self CUDA total (us)",
    "Avg time (us)",
    "% Total",
    "Input Shapes",
]

#: `KernelSourceInfo.csv_headers()`, duplicated. Magpie owns this list; the
#: duplication is bounded to the names and is what lets the reader tell an
#: enriched file from a base one without importing Magpie.
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


class CsvShapeError(ValueError):
    """The file is not a Magpie gap_analysis CSV."""


def read_gap_analysis(path: Path) -> tuple[list[dict], dict]:
    """Return `(rows, meta)`.

    `meta` carries `enriched` (bool), `columns` (the header as read) and
    `repo_paths` (the `# VAR=path` comment rows, empty for a base file).
    """
    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()

    repo_paths: dict[str, str] = {}
    first_data = 0
    for index, line in enumerate(lines):
        if not line.startswith("#"):
            first_data = index
            break
        body = line.lstrip("#").strip()
        if "=" in body:
            name, _, where = body.partition("=")
            repo_paths[name.strip()] = where.strip()
    else:
        raise CsvShapeError(f"{path}: every line is a comment, no header found")

    reader = csv.DictReader(lines[first_data:])
    columns = list(reader.fieldnames or [])
    missing = [c for c in BASE_COLUMNS if c not in columns]
    if missing:
        raise CsvShapeError(
            f"{path}: missing required column(s) {missing}. "
            f"Got {columns}. Expected at least {BASE_COLUMNS}."
        )

    enriched = all(c in columns for c in SOURCE_COLUMNS)
    rows = [dict(r) for r in reader]
    if not rows:
        raise CsvShapeError(f"{path}: header present but no data rows")

    return rows, {
        "enriched": enriched,
        "columns": columns,
        "repo_paths": repo_paths,
        "row_count": len(rows),
    }


def to_record(row: dict) -> dict:
    """One CSV row as the JSON record the rest of the package passes around.

    Numeric columns are parsed here so that a malformed cell fails at read time
    with the row in hand, rather than three tasks later as a TypeError.
    """

    def number(key: str, cast):
        raw = (row.get(key) or "").strip()
        if not raw:
            return None
        try:
            return cast(raw)
        except ValueError:
            return None

    record = {
        "name": (row.get("Name") or "").strip(),
        "calls": number("Calls", int),
        "self_us": number("Self CUDA total (us)", float),
        "avg_us": number("Avg time (us)", float),
        "pct_total": number("% Total", float),
        "input_shapes": (row.get("Input Shapes") or "").strip(),
    }
    for column in SOURCE_COLUMNS:
        if column in row:
            value = (row.get(column) or "").strip()
            if value:
                record.setdefault("source_info", {})[column] = value
    return record


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
