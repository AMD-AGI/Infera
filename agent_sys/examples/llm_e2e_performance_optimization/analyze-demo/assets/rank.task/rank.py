#!/usr/bin/env python3
"""What `rank` runs: classify every kernel, then select the routable top-N.

DESIGN.md section 6. Two stages, in this order and not the other way round:

1. **Classify.** Every row lands in exactly one bucket. `collective` and
   `vendor_tuned` and `framework_native` are excluded with a reason;
   `routable` is the candidate pool; `unknown` is excluded but recorded,
   because the unmatched set is where the next taxonomy rule comes from.
2. **Filter and rank.** Within `routable`, drop rows below the percentage and
   call-count floors, drop rows the profiler recorded no shapes for, sort by
   share of GPU time, take the top N.

Classification cannot be skipped. In the sample profile NCCL is 78.98% of GPU
time, so a plain top-N would rank a collective first — and forge-loop's
single-GPU driver contract has no way to measure one.

Every row of the input reaches the output. A reader has to be able to tell a
kernel that was considered and rejected from one the profile never saw.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_PACKAGE = Path(
    os.environ.get("AGENT_SYS_TASK_PACKAGE") or os.environ["AGENT_SYS_DEMO_PACKAGE"]
)
sys.path.insert(0, str(_PACKAGE / "assets" / "lib"))

import csv_io  # noqa: E402
import kernel_table  # noqa: E402
import shapes  # noqa: E402
import store  # noqa: E402
import taxonomy  # noqa: E402

README = """# kernel_worklist

## Purpose

Every profiled kernel, classified into one of five buckets, with the routable
ones ranked by share of GPU time and the top {top_n} marked `selected`.

This is the deliverable `temp/mission.md` section 3 names first: the list of
operators worth optimizing.

Read alongside the numbers below:

| bucket | kernels | % of profiled GPU time | in the candidate pool |
|---|---|---|---|
{bucket_table}

{headline}

## Schema

`items/text.json`:

```json
{{"generated_at": "...",
  "source": {{"rows": 0, "enriched": false}},
  "thresholds": {{"top_n": 5, "min_pct": 0.1, "min_calls": 100, "max_cases": 4}},
  "environment": {{"gpu_target": "gfx950", "gpu_type": "mi355x",
                  "framework": "sglang", "snr_threshold": 30.0,
                  "image": "..."}},
  "buckets": {{"<bucket>": {{"kernels": 0, "pct_total": 0.0}}}},
  "kernels": [
    {{"kernel_id": "k000", "name": "...", "calls": 0, "pct_total": 0.0,
      "bucket": "routable", "routable": true, "selected": true, "rank": 1,
      "category": "moe_gemm", "fellow": "ck-fellow", "language": "ck",
      "precision": "fp4", "dtypes": {{"activation": "fp4"}},
      "cases": [{{"case_id": "case_001", "selector": {{"CASE_ID": "case_001", "M": 288}},
                "shapes": [[288, 6144]], "is_primary": true}}],
      "excluded_reason": ""}}
  ]}}
```

`excluded_reason` is empty exactly when `routable` is true and the row cleared
every threshold. Every other row carries one of:

- `not_routable_by_forge_loop` — a collective; Hyperloom routes these to its own
  collective driver generator instead
- `tuned_by_table_not_source` — Tensile/rocBLAS/rocPRIM assembly, optimized by
  changing a tuning table rather than by editing source
- `shared_aten_kernel` — PyTorch ATen, shared by everything on the machine
- `no_taxonomy_rule_matched` — unclassified; see `assets/lib/kernel_taxonomy.yaml`
- `below_pct_floor` / `below_calls_floor` — routable but too small to be worth a run
- `no_shape_evidence` — the profiler recorded no input shapes, so no correctness
  case can be built from this row

`items/worklist.csv` is the same content flattened for a spreadsheet.
`items/schema` is the JSON Schema of `items/text.json`.

## Watch out

The **selection** is judged by no validator in this package. `check_worklist_shape`
checks that the document is complete, ordered and bounded; whether these are the
right five kernels to optimize is a question about the profile, not about the
document.
"""

CSV_COLUMNS = [
    "rank",
    "selected",
    "bucket",
    "kernel_id",
    "name",
    "calls",
    "self_us",
    "avg_us",
    "pct_total",
    "category",
    "fellow",
    "language",
    "precision",
    "n_cases",
    "excluded_reason",
]

SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "kernel_worklist",
    "type": "object",
    "required": ["generated_at", "thresholds", "environment", "buckets", "kernels"],
    "properties": {
        "generated_at": {"type": "string"},
        "source": {"type": "object"},
        "thresholds": {"type": "object"},
        "environment": {"type": "object"},
        "buckets": {"type": "object"},
        "kernels": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "kernel_id",
                    "name",
                    "bucket",
                    "routable",
                    "selected",
                    "excluded_reason",
                ],
                "properties": {
                    "kernel_id": {"type": "string"},
                    "name": {"type": "string"},
                    "calls": {"type": ["integer", "null"]},
                    "pct_total": {"type": ["number", "null"]},
                    "bucket": {
                        "enum": [
                            "collective",
                            "vendor_tuned",
                            "framework_native",
                            "routable",
                            "unknown",
                        ]
                    },
                    "routable": {"type": "boolean"},
                    "selected": {"type": "boolean"},
                    "rank": {"type": ["integer", "null"]},
                    "excluded_reason": {"type": "string"},
                    "cases": {"type": "array"},
                },
            },
        },
    },
}


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set; this body has nowhere to write.")
    return value


def _precision(dtypes: dict) -> str:
    """One dtype label for the operator, for `forge_task.yaml`'s `dtype` field.

    Preference order is activation, then compute, then weight: forge-loop's
    fellows key their numerical expectations off the activation precision, and
    a bf16 accumulate over fp4 operands is an fp4 kernel to them.
    """
    for role in ("activation", "both", "compute", "weight"):
        if role in dtypes:
            return dtypes[role]
    return ""


def classify_all(records: list[dict]) -> list[dict]:
    rules = taxonomy.load()
    out = []
    for index, record in enumerate(records):
        cls = taxonomy.classify(record["name"], rules)
        row = dict(record)
        row.update(
            {
                "kernel_id": f"k{index:03d}",
                "bucket": cls["bucket"],
                "routable": cls["routable"],
                "category": cls["category"],
                "fellow": cls["fellow"],
                "language": cls["language"],
                "dtypes": cls["dtypes"],
                "precision": _precision(cls["dtypes"]),
                "excluded_reason": cls["excluded_reason"],
                "selected": False,
                "rank": None,
                "cases": [],
            }
        )
        out.append(row)
    return out


def select(rows: list[dict], *, top_n: int, min_pct: float, min_calls: int, max_cases: int) -> None:
    """Mark the selected rows in place, and record why the others were not.

    Order of the filters matters for the reason recorded: a row below the
    percentage floor is reported as such even if it also has no shapes, because
    the percentage is the reason a reader would act on.
    """
    pool = []
    for row in rows:
        if not row["routable"]:
            continue
        if (row["pct_total"] or 0.0) < min_pct:
            row["excluded_reason"] = "below_pct_floor"
            continue
        if (row["calls"] or 0) < min_calls:
            row["excluded_reason"] = "below_calls_floor"
            continue
        cases = shapes.build_cases(row["input_shapes"], row["category"], max_cases)
        if not cases:
            # A workset with no shapes has no correctness case, so this row
            # cannot become one however hot it is. `main_kernel` lands here in
            # the sample profile at 3.29%.
            row["excluded_reason"] = "no_shape_evidence"
            continue
        row["cases"] = cases
        pool.append(row)

    pool.sort(key=lambda r: -(r["pct_total"] or 0.0))
    for position, row in enumerate(pool[:top_n], start=1):
        row["selected"] = True
        row["rank"] = position
        row["excluded_reason"] = ""
    for row in pool[top_n:]:
        row["excluded_reason"] = "below_top_n"


def bucket_summary(rows: list[dict]) -> dict:
    summary: dict = {}
    for row in rows:
        entry = summary.setdefault(row["bucket"], {"kernels": 0, "pct_total": 0.0})
        entry["kernels"] += 1
        entry["pct_total"] += row["pct_total"] or 0.0
    for entry in summary.values():
        entry["pct_total"] = round(entry["pct_total"], 3)
    return summary


def main() -> int:
    staged = store.declared_dir("kernel_table", direction="INPUT")
    if staged is None:
        raise SystemExit(
            "AGENT_SYS_INPUT_KERNEL_TABLE does not name a readable directory. "
            "This task declares `inputs: [kernel_table]`, so env_mgr stages it "
            "and exports the path; its absence means the input was not delivered."
        )
    # Read through `kernel_table` rather than by path: the two producers of this
    # kind lay their content out differently, and that file is the only place
    # which knows there are two of them.
    try:
        document = kernel_table.read(staged)
    except kernel_table.LayoutError as error:
        raise SystemExit(f"the staged kernel_table is not readable: {error}")
    records = document["kernels"]

    top_n = _env_int("AD_TOP_N", 5)
    min_pct = _env_float("AD_MIN_PCT", 0.1)
    min_calls = _env_int("AD_MIN_CALLS", 100)
    max_cases = _env_int("AD_MAX_CASES", 4)

    rows = classify_all(records)
    select(rows, top_n=top_n, min_pct=min_pct, min_calls=min_calls, max_cases=max_cases)
    buckets = bucket_summary(rows)

    environment = {
        "gpu_target": os.environ.get("AD_GPU_TARGET", "gfx950"),
        "gpu_type": os.environ.get("AD_GPU_TYPE", "mi355x"),
        "framework": os.environ.get("AD_FRAMEWORK", "sglang"),
        "snr_threshold": _env_float("AD_SNR_THRESHOLD", 30.0),
        "image": os.environ.get("AD_IMAGE", ""),
    }

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Passed through as the upstream wrote it: filename + sha256 + size, and
        # no directory. `handoff/locality.py` refuses to seal content carrying an
        # absolute host path, so provenance travels as a digest.
        "source": {
            **(document.get("source") or {}),
            "rows": len(records),
            "enriched": document.get("enriched", False),
            # Whether the upstream carried Python call sites, and for how many
            # rows. `identify` reads the per-kernel `launcher` block directly;
            # this is here so a reader of the worklist alone can tell a profile
            # taken with `with_stack` from one taken without, which is the
            # difference between its resolution being evidence and being a grep.
            "launchers": kernel_table.launcher_coverage(document),
        },
        "thresholds": {
            "top_n": top_n,
            "min_pct": min_pct,
            "min_calls": min_calls,
            "max_cases": max_cases,
        },
        "environment": environment,
        "buckets": buckets,
        "kernels": rows,
    }

    dst = Path(_required("AGENT_SYS_OUTPUT_KERNEL_WORKLIST"))
    items = dst / "items"
    items.mkdir(parents=True, exist_ok=True)
    (items / "text.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (items / "schema").write_text(json.dumps(SCHEMA, indent=2), encoding="utf-8")

    csv_io.write_csv(
        items / "worklist.csv",
        [
            {
                **{k: row.get(k) for k in CSV_COLUMNS if k != "n_cases"},
                "n_cases": len(row["cases"]),
            }
            for row in sorted(
                rows,
                key=lambda r: (r["rank"] is None, r["rank"] or 0, -(r["pct_total"] or 0.0)),
            )
        ],
        CSV_COLUMNS,
    )

    selected = [r for r in rows if r["selected"]]
    unknown = buckets.get("unknown", {}).get("kernels", 0)
    bucket_table = "\n".join(
        f"| `{name}` | {entry['kernels']} | {entry['pct_total']:.2f} | "
        f"{'yes' if name == 'routable' else 'no'} |"
        for name, entry in sorted(buckets.items(), key=lambda kv: -kv[1]["pct_total"])
    )
    headline = (
        f"Selected {len(selected)} of {sum(1 for r in rows if r['routable'])} routable "
        f"kernels, out of {len(rows)} profiled. Unclassified share: "
        f"{unknown / len(rows):.3f}."
    )
    if "collective" in buckets and buckets["collective"]["pct_total"] > 50:
        headline += (
            f"\n\n**Collectives are {buckets['collective']['pct_total']:.2f}% of profiled "
            f"GPU time.** That is the largest single item in this profile and it is "
            f"excluded from the candidate pool, because forge-loop's single-GPU driver "
            f"contract cannot measure a collective. It is a statement about the "
            f"parallelism strategy rather than about any one operator, and it belongs "
            f"upstream of this stage."
        )

    (dst / "README.md").write_text(
        README.format(top_n=top_n, bucket_table=bucket_table, headline=headline),
        encoding="utf-8",
    )

    coverage = out["source"]["launchers"]
    print(f"rank: {len(rows)} kernels, buckets {dict((k, v['kernels']) for k, v in buckets.items())}")
    print(
        f"      launcher frames: {coverage['rows_with_frames']} row(s)"
        + (f" — {coverage['reason']}" if coverage["reason"] else "")
    )
    for row in selected:
        print(
            f"  #{row['rank']} {row['pct_total']:.2f}% {row['category']:<10} "
            f"{row['fellow'] or '-':<16} cases={len(row['cases'])} {row['name'][:56]}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
