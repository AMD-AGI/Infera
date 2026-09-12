#!/usr/bin/env python3
"""Compare TP+DPA across 1P1D and multi-replica P/D topologies.

Usage:

    ./plot_tp_dpa_pd_groups.py \
      results/projection-1p1d/analysis/baseline/results.csv \
      --replica-results-csv \
        results/projection-tp8dpa-replicas-24-32/analysis/baseline/results.csv \
      --replica-results-csv \
        results/projection-tp8dpa-replicas-low-conc/analysis/baseline/results.csv \
      --replica-results-csv \
        results/projection-tp8dpa-replicas-2p2d-gap/analysis/baseline/results.csv \
      --output-dir results/projection-1p1d/analysis/tp-dpa-all-topologies-v3
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib  # type: ignore[import-not-found]

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore[import-not-found]  # noqa: E402

from plot_projection_groups import read_results


TARGET_MODE = "TP+DPA"


def topology_shape(row: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
    prefill_replicas = int(row.get("prefill_replicas") or 1)
    decode_replicas = int(row.get("decode_replicas") or 1)
    prefill_worker = int(
        row.get("prefill_gpus_per_replica")
        or int(row["prefill_gpus"]) // prefill_replicas
    )
    decode_worker = int(
        row.get("decode_gpus_per_replica")
        or int(row["decode_gpus"]) // decode_replicas
    )
    return (
        int(row["total_gpus"]),
        prefill_replicas,
        decode_replicas,
        prefill_worker,
        decode_worker,
    )


def group_label(row: Mapping[str, Any]) -> str:
    total, prefill_replicas, decode_replicas, prefill_worker, decode_worker = (
        topology_shape(row)
    )
    return (
        f"{prefill_replicas}P{decode_replicas}D | "
        f"P{prefill_worker}D{decode_worker} | {total}G"
    )


def select_points(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["prefill_mode"] == TARGET_MODE
        and row["decode_mode"] == TARGET_MODE
    ]


def write_tables(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
) -> tuple[Path, Path, Path]:
    concurrencies = sorted({row["concurrency"] for row in rows})
    topology_rows = {group_label(row): row for row in rows}
    groups = sorted(
        topology_rows,
        key=lambda group: topology_shape(topology_rows[group]),
    )
    values = {(group_label(row), row["concurrency"]): row for row in rows}
    headers = ["pd_group", *[f"c={value}" for value in concurrencies]]
    table_rows: list[list[str]] = []
    for group in groups:
        cells = [group]
        for concurrency in concurrencies:
            point = values.get((group, concurrency))
            cells.append(
                "—"
                if point is None
                else (
                    f"{point['interactivity_tok_s_user']:.2f} | "
                    f"{point['total_throughput_tps_per_gpu']:.2f}"
                )
            )
        table_rows.append(cells)

    csv_path = output_dir / "tp_dpa_all_topologies_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(table_rows)

    md_path = output_dir / "tp_dpa_all_topologies_table.md"
    markdown = [
        "# TP+DPA on both pools: topology × concurrency",
        "",
        "Cell format: `interactivity tok/s/user | total tokens/s/GPU`.",
        "A dash means that concurrency is absent after the memory cut.",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *["| " + " | ".join(row) + " |" for row in table_rows],
    ]
    md_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")

    long_path = output_dir / "tp_dpa_all_topologies_points.csv"
    fields = [
        "pd_group",
        "concurrency",
        "interactivity_tok_s_user",
        "total_throughput_tps_per_gpu",
        "point_id",
    ]
    order = {group: index for index, group in enumerate(groups)}
    with long_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda item: (order[group_label(item)], item["concurrency"]),
        ):
            writer.writerow(
                {
                    "pd_group": group_label(row),
                    "concurrency": row["concurrency"],
                    "interactivity_tok_s_user": (
                        f"{row['interactivity_tok_s_user']:.6f}"
                    ),
                    "total_throughput_tps_per_gpu": (
                        f"{row['total_throughput_tps_per_gpu']:.6f}"
                    ),
                    "point_id": row["point_id"],
                }
            )
    return csv_path, md_path, long_path


def render_chart(output_dir: Path, rows: Sequence[dict[str, Any]]) -> Path:
    points_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        points_by_group[group_label(row)].append(row)
    topology_rows = {group_label(row): row for row in rows}
    groups = sorted(
        topology_rows,
        key=lambda group: topology_shape(topology_rows[group]),
    )
    colors = list(plt.get_cmap("tab10").colors[: len(groups)])
    offsets = ((5, 6), (5, -11), (-23, 6), (-23, -11))
    figure, axis = plt.subplots(figsize=(14, 9))
    for group_index, group in enumerate(groups):
        points = sorted(
            points_by_group[group],
            key=lambda row: row["concurrency"],
        )
        color = colors[group_index]
        axis.plot(
            [row["interactivity_tok_s_user"] for row in points],
            [row["total_throughput_tps_per_gpu"] for row in points],
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=5,
            label=group,
        )
        for point_index, point in enumerate(points):
            axis.annotate(
                f"c={point['concurrency']}",
                (
                    point["interactivity_tok_s_user"],
                    point["total_throughput_tps_per_gpu"],
                ),
                xytext=offsets[(group_index + point_index) % len(offsets)],
                textcoords="offset points",
                fontsize=7.5,
                color=color,
            )
    axis.set_title(
        "TP+DPA on Prefill and Decode: 1P1D and Replica Topologies",
        fontsize=15,
        pad=14,
    )
    axis.set_xlabel("Interactivity (tokens/s/user)", fontsize=11)
    axis.set_ylabel("Total throughput (input + output tokens/s/GPU)", fontsize=11)
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    axis.grid(True, alpha=0.25, linewidth=0.7)
    axis.legend(
        title="P/D topology",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8,
        title_fontsize=9,
    )
    figure.text(
        0.01,
        0.01,
        "Source: projection-1p1d final results. Both pools use TP+DPA.",
        fontsize=8,
    )
    figure.subplots_adjust(right=0.76, bottom=0.1)
    path = output_dir / "tp_dpa_all_topologies_interactivity_vs_tput.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("results_csv", type=Path)
    parser.add_argument(
        "--replica-results-csv",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    results_path = args.results_csv.resolve()
    replica_results_paths = [
        path.resolve() for path in args.replica_results_csv
    ]
    output_dir = args.output_dir.resolve()
    if not results_path.is_file():
        parser.error(f"results CSV does not exist: {results_path}")
    missing = [path for path in replica_results_paths if not path.is_file()]
    if missing:
        parser.error(f"replica results CSV does not exist: {missing}")
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        *select_points(read_results(results_path)),
        *[
            row
            for path in replica_results_paths
            for row in select_points(read_results(path))
        ],
    ]
    if len(rows) != 69:
        raise SystemExit(f"expected 69 TP+DPA points, got {len(rows)}")
    if len({(group_label(row), row["concurrency"]) for row in rows}) != len(rows):
        raise SystemExit("duplicate topology/concurrency points")
    table_csv, table_md, long_csv = write_tables(output_dir, rows)
    chart = render_chart(output_dir, rows)
    (output_dir / "REPORT.md").write_text(
        "\n".join(
            [
        "# TP+DPA topology comparison",
                "",
        "Both Prefill and Decode use TP+DPA. Includes 1P1D and replica scaling.",
                "",
                f"![TP+DPA P/D group curves]({chart.name})",
                "",
                f"- Table: [{table_csv.name}]({table_csv.name})",
                f"- Markdown table: [{table_md.name}]({table_md.name})",
                f"- Long-form points: [{long_csv.name}]({long_csv.name})",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"output directory: {output_dir}")
    print(f"selected points: {len(rows)}")
    topology_rows = {group_label(row): row for row in rows}
    for group in sorted(
        topology_rows,
        key=lambda label: topology_shape(topology_rows[label]),
    ):
        concurrencies = sorted(
            row["concurrency"] for row in rows if group_label(row) == group
        )
        print(f"{group}: concurrency={concurrencies}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
