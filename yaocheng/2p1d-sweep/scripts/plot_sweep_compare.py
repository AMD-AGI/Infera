#!/usr/bin/env python3
"""Render the six-panel sweep comparison from archived JSON and AIPerf CSV.

Topology labels and GPU divisors come from each run's measured metadata.
Both decode curves use client Effective Decode Concurrency because the
reference package does not retain per-point server running gauges.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
REFERENCE = KIT.parents[1] / "yihou/glm52.p8d8.agentx-sweep.packup_20260920/results"
os.environ.setdefault("MPLCONFIGDIR", str(KIT / ".tmp/cache/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter


def read_profile(path):
    """Handle the statistics and scalar sections of AIPerf's CSV export."""
    metrics = {}
    header = None
    with path.open() as stream:
        for values in csv.reader(stream):
            if not values:
                continue
            if values[0] == "Metric":
                header = values
                continue
            if header is None or len(values) != len(header):
                raise ValueError(f"Unexpected CSV row in {path}")
            row = dict(zip(header, values))
            value = row.get("avg", row.get("Value"))
            if value:
                metrics[row["Metric"]] = float(value)
    return metrics


def load_series(directory):
    rows = []
    for point in sorted(directory.glob("c[0-9]*")):
        files = list(point.glob("agentx_conc*.json"))
        if len(files) != 1:
            raise ValueError(f"Expected one aggregate JSON in {point}")
        result = json.loads(files[0].read_text())
        profile = read_profile(point / "profile_export_aiperf.csv")
        p, d = int(result["prefill_num_workers"]), int(result["decode_num_workers"])
        gpus = int(result["num_prefill_gpu"]) + int(result["num_decode_gpu"])
        metrics = result["request_metrics"]
        latency, throughput = metrics["latency"], metrics["throughput"]
        hit = profile["Overall Usage Prompt Cache Read % (%)"]
        counted_hit = (100 * profile["Total Usage Prompt Cache Read Tokens (tokens)"]
                       / profile["Total Usage Prompt Tokens (tokens)"])
        if abs(hit - counted_hit) > 0.006:
            raise ValueError(f"Cache accounting differs in {point}")
        row = {
            "Shape": f"{p}P{d}D", "GPUs": gpus,
            "Concurrency": int(result["conc"]),
            "TotalTokensPerSecondPerGPU": throughput["total"]["tokens_per_second"] / gpus,
            "OutputTokensPerSecondPerGPU": throughput["output"]["tokens_per_second"] / gpus,
            "TTFTp50Seconds": latency["ttft"]["p50"],
            "TTFTp90Seconds": latency["ttft"]["p90"],
            "FullResponseITLp50Ms": latency["full_response_itl"]["p50"] * 1000,
            "FullResponseITLp90Ms": latency["full_response_itl"]["p90"] * 1000,
            "UsageCacheHitPct": hit,
            "TheoreticalCacheHitPct": metrics["cache"]["theoretical_cache_hit_rate"] * 100,
            "ClientEffectiveDecodeMean": profile["Effective Decode Concurrency"],
            "DecodeMetric": "AIPerf Effective Decode Concurrency (client mean)",
            "AggregateSource": str(files[0].resolve()),
        }
        if not all(math.isfinite(v) and v >= 0 for v in row.values() if isinstance(v, (int, float))):
            raise ValueError(f"Invalid metric in {point}")
        rows.append(row)
    if not rows or len({(r["Shape"], r["GPUs"]) for r in rows}) != 1:
        raise ValueError(f"Missing or inconsistent topology in {directory}")
    return sorted(rows, key=lambda row: row["Concurrency"])


def annotate(ax, x, values, color, fmt, offset):
    for conc, value in zip(x, values):
        ax.annotate(fmt.format(value), (conc, value), xytext=offset,
                    textcoords="offset points", ha="center", fontsize=9,
                    color=color, annotation_clip=False,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.5})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-results", type=Path, default=KIT / "results")
    parser.add_argument("--reference-results", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path, default=KIT / "results/sweep_compare.png")
    args = parser.parse_args()
    current, reference = load_series(args.current_results), load_series(args.reference_results)
    x = [r["Concurrency"] for r in current]
    if x != [r["Concurrency"] for r in reference]:
        raise ValueError("The two sweeps must contain the same concurrency points")
    styles = ((current, "#2563a8"), (reference, "#c66020"))
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 14,
                         "axes.labelsize": 11, "legend.fontsize": 10})
    fig, axes = plt.subplots(3, 2, figsize=(14.5, 13.8), layout="constrained")
    panels = (
        (axes[0, 0], "Total throughput / GPU", "TotalTokensPerSecondPerGPU", "tokens/s/GPU"),
        (axes[0, 1], "Output throughput / GPU", "OutputTokensPerSecondPerGPU", "tokens/s/GPU"),
    )
    for ax, title, field, unit in panels:
        for rows, color in styles:
            values = [r[field] for r in rows]
            label = f"{rows[0]['Shape']} ({rows[0]['GPUs']} GPUs)"
            ax.plot(x, values, color=color, marker="o", linewidth=2, label=label)
            best = max(range(len(values)), key=values.__getitem__)
            annotate(ax, [x[best]], [values[best]], color, "{:,.0f}", (0, 11))
        ax.set(title=title, ylabel=unit)
        ax.set_ylim(0, max(r[field] for rows, _ in styles for r in rows) * 1.21)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.legend(frameon=False, loc="lower left")

    latency_panels = (
        (axes[1, 0], "Time to first token", "TTFTp50Seconds", "TTFTp90Seconds", "seconds"),
        (axes[1, 1], "Full-response inter-token latency (ITL)",
         "FullResponseITLp50Ms", "FullResponseITLp90Ms", "milliseconds"),
    )
    for ax, title, p50, p90, unit in latency_panels:
        for index, (rows, color) in enumerate(styles):
            for field, percentile, linestyle, marker in (
                (p50, "p50", "-", "o"), (p90, "p90", "--", "s"),
            ):
                values = [r[field] for r in rows]
                ax.plot(x, values, color=color, linestyle=linestyle, marker=marker,
                        linewidth=2, label=f"{rows[0]['Shape']} {percentile}")
                if unit == "milliseconds" and percentile == "p50":
                    annotate(ax, x, values, color, "{:.2f}", (0, 9 if index == 0 else -17))
                elif percentile == "p90":
                    annotate(ax, [x[-1]], [values[-1]], color, "{:.2f}", (0, 10))
        maximum = max(r[p90] for rows, _ in styles for r in rows)
        ax.set(title=title, ylabel=unit, ylim=(0, maximum * 1.22))
        ax.legend(frameon=False, loc="upper left", ncol=2, columnspacing=1.2)

    cache_ax, decode_ax = axes[2]
    for index, (rows, color) in enumerate(styles):
        for field, label, linestyle, marker in (
            ("UsageCacheHitPct", "API usage", "-", "o"),
            ("TheoreticalCacheHitPct", "theoretical", "--", "s"),
        ):
            values = [r[field] for r in rows]
            cache_ax.plot(x, values, color=color, linestyle=linestyle, marker=marker,
                          linewidth=2, label=f"{rows[0]['Shape']} {label}")
            if field == "UsageCacheHitPct":
                for conc, value in zip(x, values):
                    offset = 22 if index == 0 and conc <= 144 else (10 if index == 0 else -17)
                    annotate(cache_ax, [conc], [value], color, "{:.2f}%", (0, offset))
        values = [r["ClientEffectiveDecodeMean"] for r in rows]
        decode_ax.plot(x, values, color=color, marker="o", linewidth=2,
                       label=f"{rows[0]['Shape']} client mean")
        annotate(decode_ax, x, values, color, "{:.2f}", (0, 10 if index == 0 else -17))
    cache_ax.set(title="Prompt cache hit rate", ylabel="cached input tokens / input tokens", ylim=(69, 101))
    cache_ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    cache_ax.legend(frameon=False, loc="lower left", ncol=2, columnspacing=1.2)
    decode_ax.set(title="Effective decode concurrency", ylabel="concurrent requests",
                  ylim=(0, max(r["ClientEffectiveDecodeMean"] for rows, _ in styles for r in rows) * 1.23))
    decode_ax.legend(frameon=False, loc="lower left")
    for ax in axes.flat:
        ax.set(xlabel="Concurrency (AgentX session trees)", xticks=x)
        ax.margins(x=0.06)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    current_label = f"{current[0]['Shape']} ({current[0]['GPUs']} GPUs)"
    reference_label = f"{reference[0]['Shape']} ({reference[0]['GPUs']} GPUs)"
    fig.suptitle(f"GLM-5.2 MXFP4 | {current_label} vs {reference_label} | MI355X",
                 fontsize=18, fontweight="bold")
    fig.supxlabel(
        "3,600 s sending window per point | Simulated acceptance 3.61 | Total includes cached input tokens\n"
        "Throughput / GPU divides by all GPUs in each deployment. Decode: client effective mean for both runs.\n"
        "2P1D C256: manual warmup recovery; 210 requests cancelled after the sending window.", fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        path = args.output.with_suffix("." + extension)
        fig.savefig(path, dpi=180)
        print(path)
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(current[0]))
        writer.writeheader()
        writer.writerows(current + reference)
    plt.close(fig)


if __name__ == "__main__":
    main()
