#!/usr/bin/env python3
"""`check_worklist_shape` — completeness, strong.

The worklist validates against its schema, its buckets are accounted for, and
every exclusion carries a reason.

**The division of labour with the schema is the point of this file.** Most of
what "shape" means is now `assets/schemas/kernel_worklist.schema.json`, which
the producer validated against too, so a required field or a biconditional is
caught on both sides of the seal from one source. What is left here is the part
a JSON Schema cannot express, and it is not a small part:

1. **Arithmetic over the document.** The bucket shares have to add up to a whole
   run, and the bucket counts have to add up to the row count. A schema can
   bound one number; it cannot sum a list.
2. **A document agreeing with itself.** `summary` restates what `kernels`
   contains, and a restatement that has drifted is the signature of a record
   edited after it was produced.
3. **The two exports agreeing with each other.** `items/worklist.csv` and
   `items/text.json` are one ranking in two formats. They are produced from one
   record, so a disagreement means one of them was edited independently — the
   same argument `analyze-demo` made for checking `invocation_spec.json` against
   `forge_task.yaml`.
4. **The schema copy is not a private fork** (CONTRACT.md §3.4). A
   `structured_text` handoff carries its schema in `items/schema`; if that copy
   may drift from `assets/schemas/`, the artefact is self-describing and
   describes something other than what graded it.

What it cannot catch, said plainly: it does not open the profile. A worklist
whose every number is internally consistent and about a different capture passes
here. `source.sha256` is what makes that checkable one stage up, and checking it
needs the profile, which this validator was not handed.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import schema as S  # noqa: E402
import zone  # noqa: E402

#: `%` shares are rounded per row, so a whole run does not sum to exactly 100.
#: The same window `check_kernel_table` uses, and for the same reason: below the
#: floor the ranking saw a fraction of the work, above the ceiling something is
#: being double-counted across ranks.
_PCT_FLOOR = 80.0
_PCT_CEILING = 100.5

#: Columns `items/worklist.csv` must carry for the cross-format check to mean
#: anything. Fewer than this and the CSV is a summary, not an export.
_CSV_COLUMNS = ("rank", "selected", "bucket", "kernel_id", "name", "pct_total", "excluded_reason")


def _check(content: Path, args: dict, problems: list[str]) -> bool:
    document = content / "items" / "text.json"
    if not document.is_file():
        problems.append("items/text.json is absent")
        return False
    try:
        data = json.loads(document.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        problems.append(f"items/text.json is not valid JSON: {error}")
        return False

    name = args.get("schema") or "kernel_worklist"

    # 1. The schema itself. Every problem at once, because one at a time turns a
    #    five-field mistake into five runs.
    try:
        S.validate(name, data)
    except S.SchemaError as error:
        problems.extend(str(error).splitlines()[1:])
        # Keep going: the checks below read fields the schema just rejected, so
        # they would report noise. Return here rather than compound it.
        return False

    # 2. The carried schema is this package's, byte for byte.
    carried = content / "items" / "schema"
    if not carried.is_file():
        problems.append("items/schema is absent; a structured_text handoff carries its own schema (CONTRACT 3.4)")
    else:
        canonical = S.schema_path(name)
        if carried.read_bytes() != canonical.read_bytes():
            problems.append(
                f"items/schema differs from assets/schemas/{name}.schema.json. "
                f"The artefact is then self-describing and describes something other than what graded it"
            )

    kernels = data["kernels"]
    buckets = data["buckets"]

    # 3. Arithmetic. The shares account for a run and the counts account for the
    #    rows.
    share = sum(b["pct_total"] for b in buckets.values())
    if not _PCT_FLOOR <= share <= _PCT_CEILING:
        problems.append(
            f"the buckets account for {share:.2f}% of profiled time, outside "
            f"[{_PCT_FLOOR}, {_PCT_CEILING}]. Below the floor the ranking saw a fraction of "
            f"the work; above the ceiling something is counted twice across ranks"
        )
    counted = sum(b["kernels"] for b in buckets.values())
    if counted != len(kernels):
        problems.append(f"buckets count {counted} kernel(s), the list holds {len(kernels)}")

    per_bucket: dict[str, int] = {}
    for kernel in kernels:
        per_bucket[kernel["bucket"]] = per_bucket.get(kernel["bucket"], 0) + 1
    for bucket, declared in buckets.items():
        actual = per_bucket.get(bucket, 0)
        if declared["kernels"] != actual:
            problems.append(f"buckets.{bucket}.kernels is {declared['kernels']}, the list holds {actual}")
    for bucket in sorted(set(per_bucket) - set(buckets)):
        problems.append(f"bucket {bucket!r} appears in kernels but not in buckets")

    # 4. The document agrees with itself.
    ids = [k["kernel_id"] for k in kernels]
    if len(set(ids)) != len(ids):
        duplicated = sorted({i for i in ids if ids.count(i) > 1})
        problems.append(f"duplicate kernel_id(s): {duplicated}. kernel_id is the join key every later artefact uses")

    selected = [k for k in kernels if k["selected"]]
    if not selected:
        problems.append("no kernel is selected; the next stage has nothing to work on")
    ranks = sorted(k["rank"] for k in selected)
    if ranks != list(range(1, len(selected) + 1)):
        problems.append(f"the selected kernels rank {ranks}, expected 1..{len(selected)} with no gap or repeat")

    summary = data.get("summary") or {}
    for key, actual in (("kernels", len(kernels)), ("selected", len(selected)),
                        ("excluded", len(kernels) - len(selected))):
        if key in summary and summary[key] != actual:
            problems.append(f"summary.{key} is {summary[key]}, the list gives {actual}")

    top_n = (data.get("thresholds") or {}).get("top_n")
    if top_n is not None and len(selected) > top_n:
        problems.append(f"{len(selected)} kernel(s) selected above a top_n of {top_n}")

    # 5. The two exports are one ranking.
    _check_csv(content, kernels, problems)

    return not problems


def _check_csv(content: Path, kernels: list[dict], problems: list[str]) -> None:
    """`items/worklist.csv` against `items/text.json`.

    The CSV is what a person opens and the JSON is what a program reads. They
    are generated from one record; a disagreement can only come from one of them
    having been edited on its own, and the edited one is invariably the CSV.
    """
    path = content / "items" / "worklist.csv"
    if not path.is_file():
        problems.append("items/worklist.csv is absent")
        return
    try:
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    except (csv.Error, UnicodeDecodeError) as error:
        problems.append(f"items/worklist.csv does not parse: {error}")
        return

    missing = [c for c in _CSV_COLUMNS if not rows or c not in rows[0]]
    if missing:
        problems.append(f"items/worklist.csv is missing column(s) {missing}")
        return

    if len(rows) != len(kernels):
        problems.append(f"items/worklist.csv has {len(rows)} row(s), items/text.json has {len(kernels)} kernel(s)")

    by_id = {k["kernel_id"]: k for k in kernels}
    for row in rows:
        kernel = by_id.get(row["kernel_id"])
        if kernel is None:
            problems.append(f"items/worklist.csv row {row['kernel_id']!r} is not in items/text.json")
            continue
        # `str()` on both sides: the CSV writes `True`/`False`, and comparing a
        # string to a bool is how this check silently passes for everything.
        if row["selected"].strip().lower() != str(kernel["selected"]).lower():
            problems.append(
                f"{row['kernel_id']}: worklist.csv says selected={row['selected']!r}, "
                f"text.json says {kernel['selected']!r}"
            )
        if row["bucket"] != kernel["bucket"]:
            problems.append(
                f"{row['kernel_id']}: worklist.csv says bucket={row['bucket']!r}, "
                f"text.json says {kernel['bucket']!r}"
            )


def main() -> int:
    args = zone.args()
    verdicts: dict[str, bool] = {}
    for hid in zone.inputs():
        problems: list[str] = []
        content = zone.content_of(hid)
        if content is None:
            # Staged nothing is *no content*, and it is never a pass.
            problems.append("the phase staged no content for this handoff")
            verdicts[hid] = False
        else:
            verdicts[hid] = _check(content, args, problems)
        for problem in problems:
            print(f"{hid}: {problem}")
    zone.write_verdict(verdicts)
    print(f"check_worklist_shape: {sum(verdicts.values())}/{len(verdicts)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
