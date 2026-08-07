#!/usr/bin/env python3
"""Validate the patch ↔ upstream status records under deploy/docker/patches/.

Three schemas, three kinds of file:

  _schema/patch.upstream.status.schema.json   -> <patch-prefix>.upstream.status.yaml
  _schema/patch.upstream.index.schema.json    -> deploy/docker/patch.upstream.status.yaml
  _schema/patch.archived.schema.json          -> patches/archived/patch.archived.yaml

Schema validation alone would let the set drift out of agreement with the tree, so
this also cross-checks them against what is actually on disk:

  * every applied patch file has a record, unless the index lists it under not_patches
  * every record points at a patch file that exists, and only speaks for its own directory
  * every record file on disk belongs to a patch — a rename cannot strand one
  * every index entry points at a record that exists, and every record is indexed
  * the index totals match the files found
  * archived entries point at a file that exists, or say `deleted` and name the commit

Dates are written unquoted in the YAML, so PyYAML hands us `datetime.date`; those
are normalised to ISO strings before validation, which means the schema's date
pattern still rejects anything written as a malformed string.

Usage:
    python3 scripts/validate-patch-status.py [--repo-root .] [-v]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("validate-patch-status: PyYAML is required (pip install -e '.[dev]')")

try:
    from jsonschema import Draft202012Validator, FormatChecker
except ImportError:  # pragma: no cover
    sys.exit("validate-patch-status: jsonschema is required (pip install -e '.[dev]')")

PATCH_ROOT = Path("deploy/docker/patches")
SCHEMA_DIR = PATCH_ROOT / "_schema"
INDEX_PATH = Path("deploy/docker/patch.upstream.status.yaml")
ARCHIVED_PATH = PATCH_ROOT / "archived" / "patch.archived.yaml"

RECORD_SUFFIX = ".upstream.status.yaml"
# Extensions that can carry a fix and therefore need a record.
PATCH_EXTS = {".py", ".diff", ".patch", ".sh"}


def normalise(node):
    """Turn YAML's native dates back into ISO strings so the schema can check them."""
    if isinstance(node, dict):
        return {k: normalise(v) for k, v in node.items()}
    if isinstance(node, list):
        return [normalise(v) for v in node]
    if isinstance(node, dt.datetime):
        return node.date().isoformat()
    if isinstance(node, dt.date):
        return node.isoformat()
    return node


class Report:
    def __init__(self, verbose: bool) -> None:
        self.errors: list[str] = []
        self.verbose = verbose

    def fail(self, where: str, message: str) -> None:
        self.errors.append(f"{where}: {message}")

    def ok(self, message: str) -> None:
        if self.verbose:
            print(f"  ok  {message}")


def load_yaml(path: Path, report: Report):
    try:
        return normalise(yaml.safe_load(path.read_text()))
    except yaml.YAMLError as exc:
        report.fail(str(path), f"not valid YAML: {exc}")
        return None


def validate_against(schema_path: Path, doc, where: str, report: Report) -> None:
    schema = json.loads(schema_path.read_text())
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    for err in sorted(validator.iter_errors(doc), key=lambda e: list(e.path)):
        loc = "/".join(str(p) for p in err.path) or "(root)"
        report.fail(where, f"{loc}: {err.message}")


def patch_files(root: Path) -> list[Path]:
    """Every file under patches/ that could carry a fix, excluding schemas and archive."""
    found = []
    for path in sorted((root / PATCH_ROOT).rglob("*")):
        if not path.is_file() or path.suffix not in PATCH_EXTS:
            continue
        rel = path.relative_to(root)
        parts = rel.parts
        if "_schema" in parts or "archived" in parts:
            continue
        found.append(rel)
    return found


def record_for(patch: Path) -> Path:
    """patches/vllm/patch_x.py -> patches/vllm/patch_x.upstream.status.yaml"""
    return patch.with_suffix("").with_name(patch.stem + RECORD_SUFFIX)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".", type=Path)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    root = args.repo_root.resolve()
    report = Report(args.verbose)

    record_schema = root / SCHEMA_DIR / "patch.upstream.status.schema.json"
    index_schema = root / SCHEMA_DIR / "patch.upstream.index.schema.json"
    archived_schema = root / SCHEMA_DIR / "patch.archived.schema.json"
    for schema in (record_schema, index_schema, archived_schema):
        if not schema.is_file():
            report.fail(str(schema.relative_to(root)), "schema missing")
    if report.errors:
        print("\n".join(f"FAIL {e}" for e in report.errors))
        return 1

    # ---- index -------------------------------------------------------------
    index = load_yaml(root / INDEX_PATH, report)
    if index is None:
        print("\n".join(f"FAIL {e}" for e in report.errors))
        return 1
    validate_against(index_schema, index, str(INDEX_PATH), report)

    not_patches = {Path(e["path"]) for e in index.get("not_patches", [])}
    indexed: dict[Path, dict] = {}
    for lib, block in index.get("libraries", {}).items():
        for entry in block.get("patches", []):
            indexed[Path(entry["patch"])] = {"library": lib, **entry}

    for entry in index.get("not_patches", []):
        if not (root / entry["path"]).is_file():
            report.fail(str(INDEX_PATH), f"not_patches lists a missing file: {entry['path']}")

    # ---- per-patch records -------------------------------------------------
    found = patch_files(root)
    records_seen: set[Path] = set()

    # A patch can be several files (a .patch plus the fragment a script appends). The
    # record names them under extra_files, and they must not also demand a record of
    # their own — so collect those before deciding what is missing one.
    owned_extras: set[Path] = set()
    on_disk_records: set[Path] = set()
    for record in sorted((root / PATCH_ROOT).rglob(f"*{RECORD_SUFFIX}")):
        rel_record = record.relative_to(root)
        on_disk_records.add(rel_record)
        doc = load_yaml(record, report) or {}
        for entry in doc.get("patch", {}).get("extra_files", []):
            extra = Path(entry)
            # extra_files is the rest of ONE patch, so a record may only name files
            # beside it. Unbounded, one line in any record excuses a patch anywhere
            # in the tree from having its own — which is the whole gate.
            if extra.parent != rel_record.parent:
                report.fail(
                    str(rel_record),
                    f"patch.extra_files entry {extra} is outside {rel_record.parent}",
                )
                continue
            owned_extras.add(extra)

    for patch in found:
        if patch in not_patches:
            report.ok(f"{patch} (declared not-a-patch)")
            continue
        if patch in owned_extras:
            report.ok(f"{patch} (covered as extra_files)")
            continue
        record = record_for(patch)
        if not (root / record).is_file():
            report.fail(
                str(patch),
                f"no status record — expected {record}, or list it under not_patches in the index",
            )
            continue
        records_seen.add(record)
        doc = load_yaml(root / record, report)
        if doc is None:
            continue
        validate_against(record_schema, doc, str(record), report)

        declared = Path(doc.get("patch", {}).get("path", ""))
        if declared != patch:
            report.fail(str(record), f"patch.path is {declared}, file is {patch}")
        for extra in doc.get("patch", {}).get("extra_files", []):
            if not (root / extra).is_file():
                report.fail(str(record), f"patch.extra_files entry missing: {extra}")
        if patch not in indexed:
            report.fail(str(patch), "has a record but is not listed in the index")
        elif Path(indexed[patch]["record"]) != record:
            report.fail(
                str(INDEX_PATH),
                f"{patch}: record is {indexed[patch]['record']}, expected {record}",
            )
        else:
            report.ok(f"{patch} <-> {record}")

    for patch in indexed:
        if patch not in found:
            report.fail(str(INDEX_PATH), f"indexes a patch that is not on disk: {patch}")

    # The walk above only goes patch -> record, so a record whose patch was renamed or
    # deleted is never loaded and never checked. Close the loop the other way.
    for stale in sorted(on_disk_records - records_seen):
        report.fail(
            str(stale),
            "does not correspond to any patch file — delete it, or restore the patch",
        )

    # ---- archived ----------------------------------------------------------
    archived_declared = Path(index["archived_record"])
    if archived_declared != ARCHIVED_PATH:
        report.fail(str(INDEX_PATH), f"archived_record should be {ARCHIVED_PATH}")
    archived = load_yaml(root / ARCHIVED_PATH, report)
    if archived is not None:
        validate_against(archived_schema, archived, str(ARCHIVED_PATH), report)
        for entry in archived.get("patches", []):
            src = entry.get("source", {})
            current = src.get("current_path")
            where = f"{ARCHIVED_PATH}[{entry.get('name')}]"
            if current == "deleted":
                if not src.get("last_commit_with_file"):
                    report.fail(
                        where,
                        "current_path is 'deleted' but last_commit_with_file is unset, "
                        "so the file cannot be recovered",
                    )
            elif not (root / current).is_file():
                report.fail(where, f"current_path does not exist: {current}")
            for extra in src.get("extra_files", []):
                if not (root / extra).is_file():
                    report.fail(where, f"extra_files entry missing: {extra}")
            if entry.get("retired_reason") == "upstream-fixed-and-in-base" and not entry.get(
                "upstream_fix"
            ):
                report.fail(where, "retired as upstream-fixed but upstream_fix is null")

    # ---- totals ------------------------------------------------------------
    totals = index.get("totals", {})
    active = len([p for p in found if p not in not_patches and p not in owned_extras])
    if totals.get("active_patches") != active:
        report.fail(
            str(INDEX_PATH),
            f"totals.active_patches is {totals.get('active_patches')}, found {active}",
        )
    if totals.get("records") != len(records_seen):
        report.fail(
            str(INDEX_PATH),
            f"totals.records is {totals.get('records')}, found {len(records_seen)}",
        )
    n_archived = len((archived or {}).get("patches", []))
    if totals.get("archived_patches") != n_archived:
        report.fail(
            str(INDEX_PATH),
            f"totals.archived_patches is {totals.get('archived_patches')}, found {n_archived}",
        )

    # ---- result ------------------------------------------------------------
    if report.errors:
        for err in report.errors:
            print(f"FAIL {err}")
        print(f"\n{len(report.errors)} problem(s)")
        return 1
    print(
        f"patch status OK — {len(records_seen)} record(s), "
        f"{len(indexed)} indexed, {n_archived} archived"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
