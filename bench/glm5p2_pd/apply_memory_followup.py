#!/usr/bin/env python3
"""Apply a memory-concurrency action table to an existing projection run.

The command first executes every SUPPLEMENT point into a staging directory.
Only when all supplemental projections succeed does it:

1. remove CUT points from raw/schedule.jsonl and raw/projections.jsonl;
2. add supplemental points to both JSONL files;
3. delete CUT point reports and install supplemental human-readable reports;
4. archive the action tables and remove stale derived analysis directories.

Usage:

    ./apply_memory_followup.py results/projection-1p1d \
      --plan-dir results/projection-1p1d/analysis/memory-plan-v2

    ./apply_memory_followup.py results/projection-1p1d \
      --plan-dir results/projection-1p1d/analysis/memory-plan-v2 \
      --execute
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import projection_sweep as sweep  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return records


def read_actions(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combination_id(record: Mapping[str, Any]) -> str:
    spec = record["spec"]
    return (
        f"g{spec['total_gpus']}_"
        f"p{spec['prefill']['tp']}-{spec['prefill']['mode']}_"
        f"d{spec['decode']['tp']}-{spec['decode']['mode']}"
    )


def spec_from_template(
    record: Mapping[str, Any],
    concurrency: int,
) -> sweep.SweepSpec:
    spec = record["spec"]
    prefill = spec["prefill"]
    decode = spec["decode"]
    workload = spec["workload"]
    return sweep.SweepSpec(
        total_gpus=int(spec["total_gpus"]),
        prefill=sweep.PoolStrategy(
            tp=int(prefill["tp"]),
            ep=int(prefill["ep"]),
            attention_dp=int(prefill["attention_dp"]),
            mode=str(prefill["mode"]),
        ),
        decode=sweep.PoolStrategy(
            tp=int(decode["tp"]),
            ep=int(decode["ep"]),
            attention_dp=int(decode["attention_dp"]),
            mode=str(decode["mode"]),
        ),
        concurrency=concurrency,
        draft_cost_factor=float(spec["draft_cost_factor"]),
        input_len=int(workload["input_len"]),
        output_len=int(workload["output_len"]),
    )


def projection_args(settings: Mapping[str, Any]) -> argparse.Namespace:
    args = sweep.build_argument_parser().parse_args([])
    for key, value in settings.items():
        if not hasattr(args, key):
            continue
        setattr(args, key, Path(value) if key == "config" else value)
    return args


def write_jsonl_atomic(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
    temporary.replace(path)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute projections and commit destructive raw-data changes",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    plan_dir = args.plan_dir.resolve()
    raw_dir = run_dir / "raw"
    schedule_path = raw_dir / "schedule.jsonl"
    projections_path = raw_dir / "projections.jsonl"
    manifest_path = run_dir / "run_config.json"
    actions_path = plan_dir / "memory_point_actions.csv"
    limits_path = plan_dir / "memory_limits_by_combination.csv"
    required = (
        schedule_path,
        projections_path,
        manifest_path,
        actions_path,
        limits_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error(f"missing required file(s): {missing}")

    schedule = read_jsonl(schedule_path)
    projections = read_jsonl(projections_path)
    actions = read_actions(actions_path)
    cut_ids = {
        action["point_id"] for action in actions if action["action"] == "CUT"
    }
    supplement_actions = [
        action for action in actions if action["action"] == "SUPPLEMENT"
    ]
    templates = {combination_id(record): record for record in schedule}
    specs: list[sweep.SweepSpec] = []
    for action in supplement_actions:
        template = templates.get(action["combination_id"])
        if template is None:
            raise SystemExit(
                f"no schedule template for {action['combination_id']}"
            )
        specs.append(spec_from_template(template, int(action["concurrency"])))

    existing_schedule_ids = {record["point_id"] for record in schedule}
    duplicate_supplements = [
        spec.point_id for spec in specs if spec.point_id in existing_schedule_ids
    ]
    if duplicate_supplements:
        raise SystemExit(
            f"supplement points already exist in schedule: {duplicate_supplements}"
        )
    if len(cut_ids) != 96 or len(specs) != 32:
        raise SystemExit(
            f"unexpected action counts: CUT={len(cut_ids)}, "
            f"SUPPLEMENT={len(specs)}; expected 96 and 32"
        )

    final_count = len(schedule) - len(cut_ids) + len(specs)
    print(f"run directory:      {run_dir}")
    print(f"current schedule:   {len(schedule)}")
    print(f"CUT points:         {len(cut_ids)}")
    print(f"SUPPLEMENT points:  {len(specs)}")
    print(f"final schedule:     {final_count}")
    if not args.execute:
        print("dry run only; pass --execute to apply")
        return 0
    if final_count != 384:
        raise SystemExit(f"refusing unexpected final schedule size {final_count}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    project_args = projection_args(manifest["settings"])
    staging = raw_dir / ".memory-followup-staging"
    if staging.exists():
        raise SystemExit(f"staging directory already exists: {staging}")
    staging_reports = staging / "reports"
    staging_reports.mkdir(parents=True)
    staging_records: list[dict[str, Any]] = []
    parser_api, launcher, memory_projector = sweep.load_projection_api()
    try:
        for index, spec in enumerate(specs, 1):
            record, report = sweep.project_one(
                spec,
                project_args,
                attempt=1,
                parser=parser_api,
                launcher=launcher,
                memory_projector=memory_projector,
            )
            staging_records.append(record)
            report_name = f"{spec.point_id}.attempt-1.txt"
            (staging_reports / report_name).write_text(report, encoding="utf-8")
            print(
                f"[{index:>2}/{len(specs)}] {spec.point_id} "
                f"{record['status']}"
            )
        errors = [
            record for record in staging_records if record["status"] == "error"
        ]
        if errors:
            raise SystemExit(
                f"{len(errors)} supplemental projections failed; "
                f"original raw data was not changed; inspect {staging}"
            )

        supplemental_schedule = [
            sweep.schedule_record(spec, project_args) for spec in specs
        ]
        new_schedule = [
            record for record in schedule if record["point_id"] not in cut_ids
        ] + supplemental_schedule
        new_projections = [
            record for record in projections if record["point_id"] not in cut_ids
        ] + staging_records
        if len({record["point_id"] for record in new_schedule}) != 384:
            raise SystemExit("final schedule does not contain 384 unique points")
        if len({record["point_id"] for record in new_projections}) != 384:
            raise SystemExit("final projections do not contain 384 unique points")

        archive_dir = run_dir / "memory-plan-applied"
        archive_dir.mkdir(exist_ok=False)
        shutil.copy2(actions_path, archive_dir / actions_path.name)
        shutil.copy2(limits_path, archive_dir / limits_path.name)

        before = {
            "schedule_sha256": sha256(schedule_path),
            "projections_sha256": sha256(projections_path),
        }
        write_jsonl_atomic(schedule_path, new_schedule)
        write_jsonl_atomic(projections_path, new_projections)

        reports_dir = raw_dir / "reports"
        deleted_reports = 0
        for point_id in cut_ids:
            for path in reports_dir.glob(f"{point_id}.attempt-*"):
                path.unlink()
                deleted_reports += 1
        for path in staging_reports.iterdir():
            path.replace(reports_dir / path.name)

        analysis_dir = run_dir / "analysis"
        if analysis_dir.exists():
            shutil.rmtree(analysis_dir)

        applied = {
            "schema_version": 1,
            "applied_at": datetime.now(timezone.utc).isoformat(),
            "source_plan_dir": str(plan_dir),
            "cut_points": len(cut_ids),
            "supplement_points": len(specs),
            "deleted_report_files": deleted_reports,
            "final_points": 384,
            "before": before,
            "after": {
                "schedule_sha256": sha256(schedule_path),
                "projections_sha256": sha256(projections_path),
            },
        }
        (archive_dir / "applied.json").write_text(
            json.dumps(applied, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    print(f"deleted CUT reports: {deleted_reports}")
    print(f"archived plan:       {archive_dir}")
    print("stale analysis directories removed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
