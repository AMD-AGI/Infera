#!/usr/bin/env python3
"""Create results.csv and an InferenceX-reference AgentX Pareto plot."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import collect_agentx


X = "P90 Interactivity (tok/s/user)"
Y = "Token Throughput per Chip (tok/s/chip)"
REFERENCE = (
    Path(__file__).resolve().parent
    / "ref"
    / "InferenceX_GLM-5.2_interactivity.csv"
)


def load(path: Path) -> list[tuple[float, float, dict[str, str]]]:
    with path.open(encoding="utf-8") as stream:
        rows = csv.DictReader(line for line in stream if not line.startswith("#"))
        points = []
        for row in rows:
            try:
                points.append((float(row[X]), float(row[Y]), row))
            except (KeyError, TypeError, ValueError):
                continue
    return points


def reference_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["Hardware"],
        row.get("Framework", ""),
        row.get("Precision", ""),
        row.get("Disaggregated", ""),
        row.get("Num Prefill Chips", ""),
        row.get("Num Decode Chips", ""),
        row.get("TP", ""),
        row.get("EP", ""),
        row.get("DP Attention", ""),
    )


def reference_label(key: tuple[str, ...]) -> str:
    hardware, framework, precision, disagg, prefill, decode, tp, ep, dpa = key
    shape = f"{prefill}P-GPU+{decode}D-GPU" if disagg == "true" else f"tp{tp}"
    ep_label = f" ep{ep}" if ep not in ("", "1") else ""
    dpa_label = " dpa" if dpa == "true" else ""
    return f"ref {hardware} {framework} {precision} {shape}{ep_label}{dpa_label}"


def plot(root: Path) -> Path:
    collect_agentx.collect(root)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(10, 6.5), dpi=140)
    reference = load(REFERENCE)
    reference_groups: dict[tuple[str, ...], list[tuple[float, float]]] = {}
    for x, y, row in reference:
        reference_groups.setdefault(reference_key(row), []).append((x, y))
    for key, points in sorted(reference_groups.items()):
        if len(points) < 2:
            continue
        points.sort()
        axes.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            markersize=3,
            linewidth=1,
            alpha=0.45,
            label=reference_label(key),
        )

    ours = load(root / "results.csv")
    own_groups: dict[tuple[str, str, str], list[tuple[float, float, dict[str, str]]]] = {}
    for x, y, row in ours:
        key = (row["Topology"], row["Hardware"], row["Framework"])
        own_groups.setdefault(key, []).append((x, y, row))
    colors = ["#A94D1C", "#1F6F6B", "#6A3D9A", "#8A5A00", "#B2285B"]
    for index, (key, points) in enumerate(sorted(own_groups.items())):
        points.sort(key=lambda point: (point[0], point[1]))
        color = colors[index % len(colors)]
        axes.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="s",
            markersize=6,
            linewidth=2.2,
            color=color,
            label=f"ours {key[1]} {key[2]} {key[0]}",
        )
        for x, y, row in points:
            axes.annotate(
                f"c{row['Concurrency']}",
                (x, y),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=8,
                color=color,
            )

    models = sorted({row["Model"] for _, _, row in ours})
    reference_dates = [
        row["Date"] for _, _, row in reference if row.get("Date")
    ]
    latest_reference = max(reference_dates) if reference_dates else "unknown"
    axes.set_xlabel(X)
    axes.set_ylabel(Y)
    axes.set_title(f"{'/'.join(models)} AgentX · {root.name}")
    axes.grid(alpha=0.25, linewidth=0.6)
    axes.legend(fontsize=7, frameon=False, ncol=2)
    figure.text(
        0.99,
        0.01,
        f"InferenceX official reference through {latest_reference}",
        ha="right",
        va="bottom",
        fontsize=7,
        color="#666666",
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    output = root / "pareto.png"
    figure.savefig(output)
    print(output)
    return output


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: plot_agentx.py <result directory>")
    root = Path(sys.argv[1]).resolve()
    if not root.is_dir():
        raise SystemExit(f"result directory does not exist: {root}")
    if not REFERENCE.is_file():
        raise SystemExit(f"reference data is missing: {REFERENCE}")
    plot(root)


if __name__ == "__main__":
    main()
