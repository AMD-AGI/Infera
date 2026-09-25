#!/usr/bin/env python3
"""Compare the 8P4D sweep with the single-node ATOM points (C >= 16) of the InferenceX export.

Usage: scripts/plot_sweep_compare.py [RUN_DIR ...]
The 8P4D runs are resolved as in plot_sweep.py. Writes $RESULTS_DIR/sweep_compare.png and
sweep_compare.csv (default results/). The export has no TTFT p90, prompt-cache or decode
concurrency data, so TTFT and end-to-end latency show p50 and mean, and the last panel is
throughput / GPU against P90 interactivity. Concurrency axes count requests per 4 GPUs, so the
12-GPU 8P4D points sit at a third of their concurrency.
"""
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator, StrMethodFormatter
from matplotlib.transforms import Bbox

from plot_sweep import KIT, MAIN, RESULTS, SECOND, load_sweep, per_point
from plot_ttft import log_ticks

INFERENCEX_CSV = KIT.parent / "refs_performance/20260922/InferenceX_GLM-5.2_interactivity.csv"
# The export has no DCP or MTP columns. Its MI355X ATOM TP4 rows at C >= 16 come from this run of
# InferenceX PR 3359: TP4 + DCP4 with LMCache DRAM offload, forced acceptance 3.33 (MTP K4) up to
# C40 and 2.99 (K3) at C48.
INFERENCEX_RUN = "https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35693365964/attempts/3"
MIN_CONC = 16
# Concurrency axes count requests per this many GPUs, the size of the single-node reference.
PER_GPUS = 4
FIELDS = ("Series", "Concurrency", "ConcurrencyPer4GPUs", "GPUs", "TotalTokensPerSecondPerGPU",
          "OutputTokensPerSecondPerGPU", "TTFTp50Seconds", "TTFTMeanSeconds", "ITLp50Ms", "ITLp90Ms",
          "E2Ep50Seconds", "E2EMeanSeconds", "P90Interactivity", "SimulatedAcceptance", "Source")
ATTRIBUTION = ("InferenceX data: https://github.com/SemiAnalysisAI/InferenceX, "
               "Copyright 2026 SemiAnalysis LLC, Apache License 2.0")
DIRECTIONS = ((7, 0, "left", "center"), (0, 7, "center", "bottom"), (0, -7, "center", "top"),
              (-7, 0, "right", "center"), (5, 5, "left", "bottom"), (-5, 5, "right", "bottom"),
              (5, -5, "left", "top"), (-5, -5, "right", "top"))
LABEL_BOX = {"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.5}


def load_local(args: list[str]) -> list[dict]:
    return [{**row, "Series": f'{row["Topology"]} ATOM + Infera ({row["GPUs"]} GPUs)',
             "ITLp50Ms": row["FullResponseITLp50Ms"], "ITLp90Ms": row["FullResponseITLp90Ms"],
             "Source": row["Run"]} for row in load_sweep(args)]


def load_inferencex(path: Path) -> list[dict]:
    with path.open() as stream:
        table = [r for r in csv.DictReader(line for line in stream if not line.startswith("#"))
                 if r["Hardware Key"] == "mi355x_atom" and r["Disaggregated"] == "false"
                 and int(r["TP"]) == 4 and int(r["Concurrency"]) >= MIN_CONC]
    if not table:
        raise SystemExit(f"{path} has no MI355X ATOM TP4 rows at concurrency >= {MIN_CONC}")
    if {r["Run URL"] for r in table} != {INFERENCEX_RUN}:
        raise SystemExit(f"{path}: MI355X ATOM TP4 rows are not all from {INFERENCEX_RUN}; "
                         "update the run description")
    return sorted(({
        "Series": f'Single-node ATOM TP{r["TP"]} + DCP4 ({r["Physical Chips"]} GPUs, InferenceX)',
        "Concurrency": int(r["Concurrency"]), "GPUs": int(r["Physical Chips"]),
        "TotalTokensPerSecondPerGPU": float(r["Token Throughput per Chip (tok/s/chip)"]),
        "OutputTokensPerSecondPerGPU": float(r["Output Throughput/Chip (tok/s)"]),
        "TTFTp50Seconds": float(r["Median TTFT (s)"]), "TTFTMeanSeconds": float(r["Mean TTFT (s)"]),
        "ITLp50Ms": 1000 * float(r["Median ITL (s)"]),
        "ITLp90Ms": 1000 / float(r["P90 Interactivity (tok/s/user)"]),
        "E2Ep50Seconds": float(r["Median E2E Latency (s)"]),
        "E2EMeanSeconds": float(r["Mean E2E Latency (s)"]),
        "P90Interactivity": float(r["P90 Interactivity (tok/s/user)"]),
        "SimulatedAcceptance": "2.99" if int(r["Concurrency"]) >= 48 else "3.33",
        "Date": r["Date"], "Source": r["Run URL"],
    } for r in table), key=lambda row: row["Concurrency"])


def by_acceptance(rows: list[dict]) -> str:
    """Simulated acceptance with the concurrency range it applies to, e.g. 3.33 for C16-C40."""
    groups = {}
    for row in rows:
        groups.setdefault(row["SimulatedAcceptance"], []).append(row["Concurrency"])
    return ", ".join(f"{accept} for C{concs[0]}" + (f"-C{concs[-1]}" if len(concs) > 1 else "")
                     for accept, concs in groups.items())


def label_points(ax, points: list[tuple]) -> None:
    """Write c<conc> next to each (x, y, conc, color) point at the first position that overlaps no
    marker or earlier label. Call after the layout is fixed."""
    renderer = ax.figure.canvas.get_renderer()
    radius = 5 * ax.figure.dpi / 72
    taken = [Bbox.from_extents(px - radius, py - radius, px + radius, py + radius)
             for px, py in (ax.transData.transform((x, y)) for x, y, _, _ in points)]

    def place(x, y, conc, color, dx, dy, ha, va):
        text = ax.annotate(f"c{conc}", (x, y), xytext=(dx, dy), textcoords="offset points",
                           ha=ha, va=va, fontsize=9, color=color, bbox=LABEL_BOX)
        return text, text.get_window_extent(renderer)

    for point in points:
        for direction in DIRECTIONS:
            text, box = place(*point, *direction)
            if not any(box.overlaps(other) for other in taken):
                break
            text.remove()
        else:
            text, box = place(*point, *DIRECTIONS[0])
        taken.append(box)


def main() -> None:
    local, reference = load_local(sys.argv[1:]), load_inferencex(INFERENCEX_CSV)
    series = ((local, MAIN, "o"), (reference, SECOND, "s"))
    both = local + reference
    for row in both:
        row["ConcurrencyPer4GPUs"] = row["Concurrency"] * PER_GPUS / row["GPUs"]
    scale = local[0]["GPUs"] / PER_GPUS

    def column(rows: list[dict], field: str) -> list:
        return [row[field] for row in rows]

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 14, "axes.labelsize": 11,
                         "legend.fontsize": 10})
    fig, axes = plt.subplots(3, 2, figsize=(12, 12.8), layout="constrained")
    for ax, title, field, fmt in (
        (axes[0, 0], "Total throughput / GPU", "TotalTokensPerSecondPerGPU", "{:,.0f}"),
        (axes[0, 1], "Output throughput / GPU", "OutputTokensPerSecondPerGPU", "{:,.1f}"),
    ):
        for rows, color, marker in series:
            ax.plot(column(rows, "ConcurrencyPer4GPUs"), column(rows, field), color=color,
                    marker=marker, linewidth=2)
            peak = max(rows, key=lambda row: row[field])
            ax.annotate(fmt.format(peak[field]), (peak["ConcurrencyPer4GPUs"], peak[field]),
                        xytext=(0, 9),
                        textcoords="offset points", ha="center", va="bottom", fontsize=9,
                        color=color, bbox=LABEL_BOX)
        ax.set(title=title, ylabel="tokens/s/GPU", ylim=(0, max(column(both, field)) * 1.21))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    for ax, title, low, high, high_label, unit, log in (
        (axes[1, 0], "Time to first token", "TTFTp50Seconds", "TTFTMeanSeconds", "mean",
         "seconds (log scale)", True),
        (axes[1, 1], "Inter-token latency (ITL)", "ITLp50Ms", "ITLp90Ms", "p90", "milliseconds",
         False),
        (axes[2, 0], "End-to-end latency", "E2Ep50Seconds", "E2EMeanSeconds", "mean", "seconds",
         False),
    ):
        for rows, color, marker in series:
            ax.plot(column(rows, "ConcurrencyPer4GPUs"), column(rows, low), color=color,
                    marker=marker, linewidth=2)
            ax.plot(column(rows, "ConcurrencyPer4GPUs"), column(rows, high), color=color,
                    marker=marker, linewidth=2, linestyle="--", markerfacecolor="white")
        values = column(both, low) + column(both, high)
        if log:
            ticks = log_ticks(min(values) / 1.25, max(values) * 1.25)
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.set_ylim(ticks[0], ticks[-1])
        else:
            ax.set_ylim(0, max(values) * 1.15)
        ax.set(title=title, ylabel=unit)
        ax.legend(handles=[Line2D([], [], color="0.4", linewidth=2, label="p50"),
                           Line2D([], [], color="0.4", linewidth=2, linestyle="--", marker="o",
                                  markerfacecolor="white", label=high_label)],
                  frameon=False, loc="upper left")
    trade = axes[2, 1]
    for rows, color, marker in series:
        trade.plot(column(rows, "P90Interactivity"), column(rows, "TotalTokensPerSecondPerGPU"),
                   color=color, marker=marker, linewidth=2)
    interactivity = column(both, "P90Interactivity")
    trade.set(title="Throughput / GPU vs P90 interactivity",
              xlabel="P90 interactivity (tok/s/user)", ylabel="tokens/s/GPU",
              xlim=(5 * math.floor((min(interactivity) - 5) / 5),
                    5 * math.ceil((max(interactivity) + 7) / 5)),
              ylim=(0, max(column(both, "TotalTokensPerSecondPerGPU")) * 1.21))
    trade.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    trade.grid(axis="x", alpha=0.2)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        if ax is trade:
            continue
        ax.set(xlabel=f"Concurrency per {PER_GPUS} GPUs",
               xticks=sorted({row["ConcurrencyPer4GPUs"] for row in reference}))
        ax.margins(x=0.04)
        top = ax.secondary_xaxis("top", functions=(lambda x: x * scale, lambda x: x / scale))
        top.set_ticks(column(local, "Concurrency"),
                      labels=[f'C{row["Concurrency"]}' for row in local])
        top.tick_params(colors=MAIN, labelsize=9)
        top.spines[:].set_visible(False)

    fig.suptitle(f'GLM-5.2 MXFP4 | {local[0]["Topology"]} ATOM + Infera vs single-node ATOM '
                 f'(C ≥ {MIN_CONC}) | MI355X', fontsize=17, fontweight="bold")
    axes[0, 0].legend(handles=[Line2D([], [], color=color, marker=marker, linewidth=2,
                                      label=rows[0]["Series"]) for rows, color, marker in series],
                      frameon=False, loc="lower right")
    fig.supxlabel("\n".join((
        f'{local[0]["Series"]}: {local[0]["Deployment"]}; '
        f'{per_point(local, "SendingWindowS", "{:,} s")} sending window per point; '
        f'simulated acceptance {by_acceptance(local)}.',
        f'{reference[0]["Series"]}: InferenceX PR 3359 run of {reference[0]["Date"]}, '
        f'LMCache DRAM offload; simulated acceptance {by_acceptance(reference)}.',
        f'Concurrency axes count requests per {PER_GPUS} GPUs: {local[0]["Topology"]} sits at '
        f'concurrency × {PER_GPUS} / {local[0]["GPUs"]}, its actual concurrency is on the blue '
        "top axis and in the last panel.",
        "Throughput / GPU divides by all GPUs of each deployment and includes cached input tokens. "
        "P90 interactivity = 1 / P90 ITL.",
        "The InferenceX export has no TTFT p90, prompt-cache or decode-concurrency data, "
        "so TTFT and end-to-end latency show p50 and mean.",
        ATTRIBUTION + ".",
    )), fontsize=9.5)
    fig.canvas.draw()
    fig.set_layout_engine("none")
    label_points(trade, [(row["P90Interactivity"], row["TotalTokensPerSecondPerGPU"],
                          row["Concurrency"], color) for rows, color, _ in series for row in rows])

    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "sweep_compare.png", dpi=180)
    plt.close(fig)
    with (RESULTS / "sweep_compare.csv").open("w", newline="") as stream:
        stream.write(f"# {ATTRIBUTION}; rows whose Source is an InferenceX run URL are derived from it.\n")
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()}
                         for row in both)
    print(RESULTS / "sweep_compare.png")
    print(RESULTS / "sweep_compare.csv")


if __name__ == "__main__":
    main()
