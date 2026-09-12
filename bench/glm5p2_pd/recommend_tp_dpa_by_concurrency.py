#!/usr/bin/env python3
"""Recommend TP+DPA P/D topologies independently at each concurrency.

The two maximized objectives are interactivity and total tokens/s/GPU.
Every emitted topology is Pareto-nondominated within its concurrency.

Usage:

    ./recommend_tp_dpa_by_concurrency.py \
      results/projection-1p1d/analysis/tp-dpa-all-topologies-v3/\
tp_dpa_all_topologies_points.csv \
      --output-dir results/projection-1p1d/analysis/tp-dpa-recommendations
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


CSV_FIELDS = [
    "concurrency",
    "role",
    "topology",
    "total_gpus",
    "interactivity_tok_s_user",
    "total_throughput_tps_per_gpu",
    "point_id",
]


def read_points(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [
        {
            **row,
            "concurrency": int(row["concurrency"]),
            "total_gpus": int(row["pd_group"].rsplit("|", 1)[1].strip().removesuffix("G")),
            "interactivity_tok_s_user": float(
                row["interactivity_tok_s_user"]
            ),
            "total_throughput_tps_per_gpu": float(
                row["total_throughput_tps_per_gpu"]
            ),
        }
        for row in rows
        if int(row["concurrency"]) >= 8
    ]


def pareto_front(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    front: list[dict[str, Any]] = []
    for candidate in rows:
        candidate_inv = candidate["interactivity_tok_s_user"]
        candidate_tput = candidate["total_throughput_tps_per_gpu"]
        dominated = any(
            other["interactivity_tok_s_user"] >= candidate_inv
            and other["total_throughput_tps_per_gpu"] >= candidate_tput
            and (
                other["interactivity_tok_s_user"] > candidate_inv
                or other["total_throughput_tps_per_gpu"] > candidate_tput
            )
            for other in rows
            if other is not candidate
        )
        if not dominated:
            front.append(candidate)
    return sorted(
        front,
        key=lambda row: (
            -row["total_throughput_tps_per_gpu"],
            -row["interactivity_tok_s_user"],
        ),
    )


def classify_roles(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    max_tput = max(row["total_throughput_tps_per_gpu"] for row in rows)
    max_inv = max(row["interactivity_tok_s_user"] for row in rows)
    output: list[dict[str, Any]] = []
    for row in rows:
        labels: list[str] = []
        if row["total_throughput_tps_per_gpu"] == max_tput:
            labels.append("max_tput_per_gpu")
        if row["interactivity_tok_s_user"] == max_inv:
            labels.append("max_interactivity")
        if not labels:
            labels.append("tradeoff")
        output.append({**row, "role": "+".join(labels)})
    return output


def recommendations(points: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        groups[point["concurrency"]].append(point)
    output: list[dict[str, Any]] = []
    for concurrency in sorted(groups):
        output.extend(classify_roles(pareto_front(groups[concurrency])))
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "concurrency": row["concurrency"],
                    "role": row["role"],
                    "topology": row["pd_group"],
                    "total_gpus": row["total_gpus"],
                    "interactivity_tok_s_user": (
                        f"{float(row['interactivity_tok_s_user']):.2f}"
                    ),
                    "total_throughput_tps_per_gpu": (
                        f"{float(row['total_throughput_tps_per_gpu']):.2f}"
                    ),
                    "point_id": row["point_id"],
                }
            )


def write_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# TP+DPA recommended real-test topologies by concurrency",
        "",
        "Objectives: maximize Interactivity and total tokens/s/GPU.",
        "Only Pareto-nondominated topologies are listed.",
        "",
    ]
    by_concurrency: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_concurrency[int(row["concurrency"])].append(row)
    for concurrency in sorted(by_concurrency):
        lines.extend(
            [
                f"## C{concurrency}",
                "",
                "| Role | Topology | GPUs | Interactivity | Total tok/s/GPU |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in by_concurrency[concurrency]:
            lines.append(
                f"| {row['role']} | {row['pd_group']} | "
                f"{row['total_gpus']} | "
                f"{float(row['interactivity_tok_s_user']):.2f} | "
                f"{float(row['total_throughput_tps_per_gpu']):.2f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("points_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    points_path = args.points_csv.resolve()
    output_dir = args.output_dir.resolve()
    if not points_path.is_file():
        parser.error(f"points CSV does not exist: {points_path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    points = read_points(points_path)
    selected = recommendations(points)
    write_csv(output_dir / "recommended_by_concurrency.csv", selected)
    write_markdown(output_dir / "recommended_by_concurrency.md", selected)
    counts = defaultdict(int)
    for row in selected:
        counts[int(row["concurrency"])] += 1
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "source": str(points_path),
                "candidate_points": len(points),
                "recommended_points": len(selected),
                "recommended_per_concurrency": dict(sorted(counts.items())),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"output directory: {output_dir}")
    print(f"candidate points: {len(points)}")
    print(f"recommended points: {len(selected)}")
    print(f"per concurrency: {dict(sorted(counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
