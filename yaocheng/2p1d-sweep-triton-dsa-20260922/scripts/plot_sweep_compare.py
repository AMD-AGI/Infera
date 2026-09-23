#!/usr/bin/env python3
"""Compare the Triton DSA sweep with the archived TileLang 2P1D baseline.

Read throughput/latency from aggregate JSON and cache/decode metrics from
AIPerf CSV, matching the original six-panel sweep comparison. Export PNG,
SVG, source metrics CSV, and a CSV of Triton changes relative to TileLang.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
REFERENCE = KIT.parent / "2p1d-sweep/results"
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


def load_series(directory, label):
    run = json.loads((directory / "run.json").read_text())
    if run["status"] != "complete":
        raise ValueError(f"Sweep is not complete: {directory}")
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
            "Series": label, "RunID": run["run_id"],
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
            "ProfileSource": str((point / "profile_export_aiperf.csv").resolve()),
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


def annotate_pair(ax, x, styles, field, fmt, above=10, below=-18):
    """Place close values on opposite sides so both runs remain readable."""
    for i, conc in enumerate(x):
        values = [rows[i][field] for rows, _ in styles]
        higher = max(range(2), key=values.__getitem__)
        for j, (_, color) in enumerate(styles):
            annotate(ax, [conc], [values[j]], color, fmt,
                     (0, above if j == higher else below))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(path)


def comparison_deltas(current, reference):
    fields = (
        "TotalTokensPerSecondPerGPU", "OutputTokensPerSecondPerGPU",
        "TTFTp50Seconds", "TTFTp90Seconds", "FullResponseITLp50Ms",
        "FullResponseITLp90Ms", "ClientEffectiveDecodeMean",
    )
    rows = []
    for new, old in zip(current, reference):
        row = {"Concurrency": new["Concurrency"]}
        for field in fields:
            row[field + "ChangePct"] = 100 * (new[field] / old[field] - 1)
        for field in ("UsageCacheHitPct", "TheoreticalCacheHitPct"):
            row[field + "ChangePercentagePoints"] = new[field] - old[field]
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-results", type=Path, default=KIT / "results")
    parser.add_argument("--reference-results", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path, default=KIT / "results/sweep_compare.png")
    args = parser.parse_args()
    current = load_series(args.current_results, "Triton DSA")
    reference = load_series(args.reference_results, "TileLang")
    current_run = json.loads((args.current_results / "run.json").read_text())
    reference_run = json.loads((args.reference_results / "run.json").read_text())
    for field in ("topology", "image_id", "requested_duration_s",
                  "simulation_accept_length", "warmup_requests_per_lane"):
        if current_run[field] != reference_run[field]:
            raise ValueError(f"Comparison conditions differ: {field}")
    if (current[0]["Shape"], current[0]["GPUs"]) != (reference[0]["Shape"], reference[0]["GPUs"]):
        raise ValueError("The DSA comparison requires matching topologies")
    x = [r["Concurrency"] for r in current]
    if x != [r["Concurrency"] for r in reference]:
        raise ValueError("The two sweeps must contain the same concurrency points")
    styles = ((reference, "#2563a8"), (current, "#c66020"))
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 14,
                         "axes.labelsize": 11, "legend.fontsize": 10})
    fig, axes = plt.subplots(3, 2, figsize=(16, 14.4), layout="constrained")
    panels = (
        (axes[0, 0], "Total throughput / GPU", "TotalTokensPerSecondPerGPU", "tokens/s/GPU"),
        (axes[0, 1], "Output throughput / GPU", "OutputTokensPerSecondPerGPU", "tokens/s/GPU"),
    )
    for ax, title, field, unit in panels:
        for rows, color in styles:
            values = [r[field] for r in rows]
            label = rows[0]["Series"]
            ax.plot(x, values, color=color, marker="o", linewidth=2, label=label)
        annotate_pair(ax, x, styles, field,
                      "{:,.0f}" if field.startswith("Total") else "{:.2f}")
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
        for rows, color in styles:
            for field, percentile, linestyle, marker in (
                (p50, "p50", "-", "o"), (p90, "p90", "--", "s"),
            ):
                values = [r[field] for r in rows]
                ax.plot(x, values, color=color, linestyle=linestyle, marker=marker,
                        linewidth=2, label=f"{rows[0]['Series']} {percentile}")
                if percentile == "p90":
                    annotate(ax, [x[-1]], [values[-1]], color, "{:.2f}", (0, 10))
        if unit == "milliseconds":
            annotate_pair(ax, x, styles, p50, "{:.2f}")
        else:
            annotate_pair(ax, [x[-1]], tuple(([rows[-1]], color) for rows, color in styles),
                          p50, "{:.2f}")
        maximum = max(r[p90] for rows, _ in styles for r in rows)
        ax.set(title=title, ylabel=unit, ylim=(0, maximum * 1.22))
        ax.legend(frameon=False, loc="upper left", ncol=2, columnspacing=1.2)

    cache_ax, decode_ax = axes[2]
    for rows, color in styles:
        for field, label, linestyle, marker in (
            ("UsageCacheHitPct", "API usage", "-", "o"),
            ("TheoreticalCacheHitPct", "theoretical", "--", "s"),
        ):
            values = [r[field] for r in rows]
            cache_ax.plot(x, values, color=color, linestyle=linestyle, marker=marker,
                          linewidth=2, label=f"{rows[0]['Series']} {label}")
        values = [r["ClientEffectiveDecodeMean"] for r in rows]
        decode_ax.plot(x, values, color=color, marker="o", linewidth=2,
                       label=f"{rows[0]['Series']} client mean")
    annotate_pair(cache_ax, x, styles, "UsageCacheHitPct", "{:.2f}%", above=24)
    annotate_pair(decode_ax, x, styles, "ClientEffectiveDecodeMean", "{:.2f}")
    cache_min = min(r["UsageCacheHitPct"] for rows, _ in styles for r in rows)
    cache_ax.set(title="Prompt cache hit rate", ylabel="cached input tokens / input tokens",
                 ylim=(max(0, math.floor(cache_min) - 5), 101))
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

    topology = f"{current[0]['Shape']} ({current[0]['GPUs']} GPUs each)"
    fig.suptitle(f"GLM-5.2 MXFP4 | TileLang vs Triton DSA | {topology} | MI355X",
                 fontsize=17, fontweight="bold")
    duration = float(current_run["requested_duration_s"])
    acceptance = current_run["simulation_accept_length"]
    fig.supxlabel(
        f"{duration:,.0f} s sending window per point | Simulated acceptance {acceptance} | Total includes cached input tokens\n"
        "Both runs: throughput divided by 24 GPUs; decode concurrency is the client effective mean.\n"
        "TileLang C256: manual warmup recovery. C256 end-of-window cancellations: TileLang 210 / Triton 160.\n"
        "Latency percentiles cover completed requests. Single sweep per backend; differences are observed values.",
        fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        path = args.output.with_suffix("." + extension)
        fig.savefig(path, dpi=180)
        print(path)
    write_csv(args.output.with_suffix(".csv"), reference + current)
    write_csv(args.output.with_name(args.output.stem + "_delta.csv"),
              comparison_deltas(current, reference))
    plt.close(fig)


if __name__ == "__main__":
    main()
