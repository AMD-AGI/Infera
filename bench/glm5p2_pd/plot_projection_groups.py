#!/usr/bin/env python3
"""Render P/D-width grouped projection tables and strategy curves.

For each ``(prefill_gpus, decode_gpus)`` group, this command writes:

* a pivoted CSV/Markdown table whose cells are ``interactivity | tput/GPU``;
* PNG and SVG line charts with one colored line per P/D parallel strategy;
* concurrency labels beside every plotted point.

Usage:

    ./plot_projection_groups.py \
      results/projection-1p1d/analysis/baseline/results.csv \
      --output-dir results/projection-1p1d/analysis/grouped-strategy-plots
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


GROUP_ORDER = ((4, 4), (4, 8), (8, 4), (8, 8))
MODE_ORDER = {
    "TP": 0,
    "TP+EP": 1,
    "TP+EP+DPA": 2,
    "TP+DPA": 3,
}
LONG_FIELDS = [
    "group",
    "parallel_strategy",
    "prefill_mode",
    "decode_mode",
    "concurrency",
    "interactivity_tok_s_user",
    "total_throughput_tps_per_gpu",
    "point_id",
]


def read_results(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    output: list[dict[str, Any]] = []
    for row in rows:
        if row["status"] != "ok":
            continue
        output.append(
            {
                **row,
                "prefill_gpus": int(row["prefill_gpus"]),
                "decode_gpus": int(row["decode_gpus"]),
                "concurrency": int(row["concurrency"]),
                "interactivity_tok_s_user": float(
                    row["interactivity_tok_s_user"]
                ),
                "total_throughput_tps_per_gpu": float(
                    row["total_throughput_tps_per_gpu"]
                ),
            }
        )
    return output


def strategy_label(row: Mapping[str, Any]) -> str:
    return f"P:{row['prefill_mode']} | D:{row['decode_mode']}"


def strategy_sort_key(label: str) -> tuple[int, int]:
    prefill, decode = label.split(" | ")
    return (
        MODE_ORDER[prefill.removeprefix("P:")],
        MODE_ORDER[decode.removeprefix("D:")],
    )


def grouped_rows(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["prefill_gpus"], row["decode_gpus"])].append(row)
    unexpected = sorted(set(groups) - set(GROUP_ORDER))
    if unexpected:
        raise ValueError(f"unexpected P/D GPU groups: {unexpected}")
    missing = [group for group in GROUP_ORDER if group not in groups]
    if missing:
        raise ValueError(f"missing P/D GPU groups: {missing}")
    return groups


def write_table_files(
    output_dir: Path,
    group_name: str,
    rows: Sequence[dict[str, Any]],
) -> tuple[Path, Path]:
    concurrencies = sorted({row["concurrency"] for row in rows})
    strategies = sorted(
        {strategy_label(row) for row in rows},
        key=strategy_sort_key,
    )
    values = {
        (strategy_label(row), row["concurrency"]): row
        for row in rows
    }
    if len(values) != len(rows):
        raise ValueError(f"duplicate strategy/concurrency points in {group_name}")
    headers = ["parallel_strategy", *[f"c={value}" for value in concurrencies]]
    table_rows: list[list[str]] = []
    for strategy in strategies:
        cells = [strategy]
        for concurrency in concurrencies:
            point = values.get((strategy, concurrency))
            cells.append(
                "—"
                if point is None
                else (
                    f"{point['interactivity_tok_s_user']:.2f} | "
                    f"{point['total_throughput_tps_per_gpu']:.2f}"
                )
            )
        table_rows.append(cells)

    csv_path = output_dir / f"{group_name}_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(table_rows)

    markdown_path = output_dir / f"{group_name}_table.md"
    markdown = [
        f"# {group_name.upper()} strategy × concurrency",
        "",
        "Cell format: `interactivity tok/s/user | total tokens/s/GPU`.",
        "A dash means that point is absent after the memory cut.",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    markdown.extend("| " + " | ".join(row) + " |" for row in table_rows)
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return csv_path, markdown_path


def render_chart(
    output_dir: Path,
    group_name: str,
    rows: Sequence[dict[str, Any]],
) -> tuple[Path, Path]:
    by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_strategy[strategy_label(row)].append(row)
    strategies = sorted(by_strategy, key=strategy_sort_key)
    colors = list(plt.get_cmap("tab20").colors[: len(strategies)])
    offsets = ((5, 5), (5, -10), (-22, 5), (-22, -10))

    figure, axis = plt.subplots(figsize=(17, 10))
    for strategy_index, strategy in enumerate(strategies):
        points = sorted(
            by_strategy[strategy],
            key=lambda row: row["concurrency"],
        )
        x_values = [row["interactivity_tok_s_user"] for row in points]
        y_values = [row["total_throughput_tps_per_gpu"] for row in points]
        color = colors[strategy_index]
        axis.plot(
            x_values,
            y_values,
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=4.5,
            label=strategy,
        )
        for point_index, point in enumerate(points):
            axis.annotate(
                f"c={point['concurrency']}",
                (
                    point["interactivity_tok_s_user"],
                    point["total_throughput_tps_per_gpu"],
                ),
                xytext=offsets[(strategy_index + point_index) % len(offsets)],
                textcoords="offset points",
                fontsize=6.5,
                color=color,
            )

    axis.set_title(
        f"{group_name.upper()}: Interactivity vs Total Tokens/s/GPU",
        fontsize=15,
        pad=14,
    )
    axis.set_xlabel("Interactivity (tokens/s/user)", fontsize=11)
    axis.set_ylabel("Total throughput (input + output tokens/s/GPU)", fontsize=11)
    axis.grid(True, alpha=0.25, linewidth=0.7)
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    axis.legend(
        title="P/D parallel strategy",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        fontsize=8,
        title_fontsize=9,
    )
    figure.text(
        0.01,
        0.01,
        "Source: projection-1p1d final 384-point results. "
        "Labels show configured concurrency.",
        fontsize=8,
    )
    figure.subplots_adjust(right=0.73, bottom=0.1)
    png_path = output_dir / f"{group_name}_interactivity_vs_tput.png"
    svg_path = output_dir / f"{group_name}_interactivity_vs_tput.svg"
    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, svg_path


def write_long_csv(output_dir: Path, rows: Sequence[dict[str, Any]]) -> Path:
    path = output_dir / "grouped_points.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LONG_FIELDS)
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda item: (
                item["prefill_gpus"],
                item["decode_gpus"],
                strategy_sort_key(strategy_label(item)),
                item["concurrency"],
            ),
        ):
            writer.writerow(
                {
                    "group": f"p{row['prefill_gpus']}d{row['decode_gpus']}",
                    "parallel_strategy": strategy_label(row),
                    "prefill_mode": row["prefill_mode"],
                    "decode_mode": row["decode_mode"],
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
    return path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    results_path = args.results_csv.resolve()
    output_dir = args.output_dir.resolve()
    if not results_path.is_file():
        parser.error(f"results CSV does not exist: {results_path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_results(results_path)
    if len(rows) != 384:
        raise SystemExit(f"expected 384 feasible final points, got {len(rows)}")
    groups = grouped_rows(rows)
    report = [
        "# GLM-5.2 grouped P/D projection charts",
        "",
        "Source: `results/projection-1p1d/analysis/baseline/results.csv`.",
        "",
        "Table cell format: `interactivity tok/s/user | total tokens/s/GPU`.",
        "",
    ]
    summary: dict[str, Any] = {
        "source": str(results_path),
        "total_points": len(rows),
        "groups": {},
    }
    for prefill_gpus, decode_gpus in GROUP_ORDER:
        group_name = f"p{prefill_gpus}d{decode_gpus}"
        group = groups[(prefill_gpus, decode_gpus)]
        table_csv, table_markdown = write_table_files(
            output_dir,
            group_name,
            group,
        )
        chart_png, chart_svg = render_chart(output_dir, group_name, group)
        report.extend(
            [
                f"## {group_name.upper()}",
                "",
                f"![{group_name.upper()} strategy curves]({chart_png.name})",
                "",
                f"- Table: [{table_csv.name}]({table_csv.name})",
                f"- Markdown table: [{table_markdown.name}]({table_markdown.name})",
                f"- SVG: [{chart_svg.name}]({chart_svg.name})",
                "",
            ]
        )
        summary["groups"][group_name] = {
            "points": len(group),
            "strategies": len({strategy_label(row) for row in group}),
            "concurrencies": sorted({row["concurrency"] for row in group}),
        }

    write_long_csv(output_dir, rows)
    (output_dir / "REPORT.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"output directory: {output_dir}")
    for group_name, details in summary["groups"].items():
        print(
            f"{group_name.upper()}: {details['points']} points, "
            f"{details['strategies']} strategies, "
            f"concurrency={details['concurrencies']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
