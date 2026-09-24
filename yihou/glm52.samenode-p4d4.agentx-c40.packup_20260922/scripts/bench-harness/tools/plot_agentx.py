#!/usr/bin/env python3
"""Purpose: Plot local AgentX groups against the official InferenceX reference.
Usage:
    python3 tools/plot_agentx.py RESULT_DIRECTORY
Artifacts:
    Aggregated CSV and Pareto PNG.
Artifact paths:
    RESULT_DIRECTORY/results.csv and RESULT_DIRECTORY/pareto.png.
"""

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


def load(path: Path):
    with path.open(encoding="utf-8") as stream:
        points = []
        for row in csv.DictReader(line for line in stream if not line.startswith("#")):
            try:
                points.append((float(row[X]), float(row[Y]), row))
            except (KeyError, TypeError, ValueError):
                pass
    return points


def reference_key(row: dict[str, str]) -> tuple[str, ...]:
    fields = (
        "Hardware",
        "Framework",
        "Precision",
        "Disaggregated",
        "Prefill GPUs",
        "Decode GPUs",
        "Prefill TP",
        "Decode TP",
        "Prefill EP",
        "Decode EP",
        "Prefill DPA",
        "Decode DPA",
    )
    return tuple(row.get(field, "") for field in fields)


def reference_label(key: tuple[str, ...]) -> str:
    hardware, framework, precision, disagg, pg, dg, ptp, dtp, pep, dep, pdpa, ddpa = key
    if disagg == "true":
        shape = f"{pg}P-GPU+{dg}D-GPU pTP{ptp}/EP{pep} dTP{dtp}/EP{dep}"
        dpa = " DPA" if pdpa == "true" or ddpa == "true" else ""
    else:
        shape = f"TP{ptp}/EP{pep}"
        dpa = " DPA" if pdpa == "true" else ""
    return f"ref {hardware} {framework} {precision} {shape}{dpa}"


def plot(root: Path) -> Path:
    collect_agentx.collect(root)
    rows = load(root / "results.csv")
    if not rows:
        raise SystemExit("results.csv contains no numeric points")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6), dpi=140)

    if REFERENCE.exists():
        reference_groups = {}
        for x, y, row in load(REFERENCE):
            reference_groups.setdefault(reference_key(row), []).append((x, y))
        for key, points in sorted(reference_groups.items()):
            points.sort()
            ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                ms=3.5,
                lw=1,
                alpha=0.5,
                label=reference_label(key),
            )

    groups = {}
    for x, y, row in rows:
        key = (row["Topology"], row["Hardware"], row["Framework"])
        groups.setdefault(key, []).append((x, y, row))

    for (topology, hardware, framework), points in sorted(groups.items()):
        points.sort(key=lambda item: (item[0], item[1]))
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="s",
            ms=6,
            lw=2,
            label=f"{hardware} {framework} {topology}",
        )
        for x, y, row in points:
            ax.annotate(
                f"c{row['Concurrency']}",
                (x, y),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=8,
            )

    models = sorted({row["Model"] for _, _, row in rows})
    ax.set_xlabel(X)
    ax.set_ylabel(Y)
    ax.set_title(f"{'/'.join(models)} AgentX · {root.name}")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()

    output = root / "pareto.png"
    fig.savefig(output)
    print(output)
    return output


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: plot_agentx.py RESULT_DIRECTORY")
    root = Path(sys.argv[1]).resolve()
    if not root.is_dir():
        raise SystemExit(f"result directory does not exist: {root}")
    plot(root)


if __name__ == "__main__":
    main()
