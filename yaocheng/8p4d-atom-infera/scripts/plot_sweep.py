#!/usr/bin/env python3
"""Draw the six-panel AgentX sweep figure as results/sweep.png and its values as results/sweep.csv.

Usage: scripts/plot_sweep.py [RUN_DIR ...]
Without arguments, uses .tmp/runs/<run> for every row of results/summary.csv. Each RUN_DIR needs
workers.json and agentx/{agentx_conc<N>.json, runtime.env, aiperf_artifacts/}. The output
directory is $RESULTS_DIR (default results/).

Decode running concurrency is the mean of the atom:requests_running samples that AIPerf keeps
for the decode endpoint in the profiling phase (warmup excluded), from server_metrics_export.csv.
"""
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter

from plot_ttft import deployment

KIT = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("RESULTS_DIR", KIT / "results"))
MAIN, SECOND = "#2563a8", "#c66020"
LABEL_HEIGHT_PT = 12
PROFILING_END = re.compile(r"Phase profiling \(profiling\) complete \| completed=[\d,]+, cancelled=(\d+)")


def read_profile(path: Path) -> dict:
    """Return the avg column of the statistics section and the Value column of the scalar section."""
    metrics, header = {}, None
    with path.open() as stream:
        for values in csv.reader(stream):
            if not values:
                continue
            if values[0] == "Metric":
                header = values
                continue
            if header is None or len(values) != len(header):
                raise ValueError(f"unexpected CSV row in {path}")
            row = dict(zip(header, values))
            value = row.get("avg", row.get("Value"))
            if value:
                metrics[row["Metric"]] = float(value)
    return metrics


def decode_running(run: Path) -> float:
    workers = json.loads((run / "workers.json").read_text())["workers"]
    decode = {w["worker_id"] for w in workers if w["disagg_mode"] == "decode"}
    path = run / "agentx/aiperf_artifacts/server_metrics_export.csv"
    with path.open() as stream:
        rows = [r for r in csv.DictReader(line for line in stream if not line.startswith("#"))
                if r["Metric"] == "atom:requests_running" and r["Endpoint"] in decode]
    if len(rows) != len(decode):
        raise ValueError(f"{path}: expected one atom:requests_running row per decode endpoint "
                         f"{sorted(decode)}")
    return sum(float(r["avg"]) for r in rows)


def load(run: Path) -> dict:
    agentx = run / "agentx"
    artifacts = agentx / "aiperf_artifacts"
    paths = list(agentx.glob("agentx_conc*.json"))
    if len(paths) != 1:
        raise ValueError(f"expected one agentx_conc*.json in {agentx}")
    r = json.loads(paths[0].read_text())
    env = dict(line.split("=", 1) for line in (agentx / "runtime.env").read_text().splitlines()
               if "=" in line)
    profile = read_profile(artifacts / "profile_export_aiperf.csv")
    hit = profile["Overall Usage Prompt Cache Read % (%)"]
    counted = (100 * profile["Total Usage Prompt Cache Read Tokens (tokens)"]
               / profile["Total Usage Prompt Tokens (tokens)"])
    if abs(hit - counted) > 0.006:
        raise ValueError(f"cache accounting differs in {artifacts}")
    end = PROFILING_END.search((artifacts / "logs/aiperf.log").read_text())
    gpus = r["num_prefill_gpu"] + r["num_decode_gpu"]
    metrics = r["request_metrics"]
    latency, throughput = metrics["latency"], metrics["throughput"]
    return {
        "Concurrency": r["conc"], "GPUs": gpus,
        "TotalTokensPerSecondPerGPU": throughput["total"]["tokens_per_second"] / gpus,
        "OutputTokensPerSecondPerGPU": throughput["output"]["tokens_per_second"] / gpus,
        "TTFTp50Seconds": latency["ttft"]["p50"], "TTFTp90Seconds": latency["ttft"]["p90"],
        "TTFTMeanSeconds": latency["ttft"]["mean"],
        "FullResponseITLp50Ms": latency["full_response_itl"]["p50"] * 1000,
        "FullResponseITLp90Ms": latency["full_response_itl"]["p90"] * 1000,
        "P90Interactivity": 1 / latency["full_response_itl"]["p90"],
        "E2Ep50Seconds": latency["e2el"]["p50"], "E2EMeanSeconds": latency["e2el"]["mean"],
        "UsageCacheHitPct": hit,
        "TheoreticalCacheHitPct": metrics["cache"]["theoretical_cache_hit_rate"] * 100,
        "DecodeRunningMean": decode_running(run),
        "ClientEffectiveDecodeMean": profile["Effective Decode Concurrency"],
        "ErrorRecords": r["request_accounting"]["records_error_dropped"],
        "CancelledAfterWindow": int(end.group(1)) if end else None,
        "SendingWindowS": int(env["DURATION"]),
        "SimulatedAcceptance": env.get("SIMULATE_ACC_LEN") or "off",
        "Topology": f'{r["num_prefill_gpu"]}P{r["num_decode_gpu"]}D',
        "Hardware": r["hw"].upper(), "Deployment": deployment(r), "Run": run.name,
    }


def load_sweep(args: list[str]) -> list[dict]:
    """Rows for the RUN_DIR arguments, or for .tmp/runs/<run> of every row of results/summary.csv."""
    if not args:
        with (RESULTS / "summary.csv").open() as stream:
            args = [KIT / ".tmp/runs" / row["run"] for row in csv.DictReader(stream)]
    rows = sorted((load(Path(run)) for run in args), key=lambda row: row["Concurrency"])
    if len({(row["Deployment"], row["GPUs"]) for row in rows}) != 1:
        raise SystemExit("the runs mix deployments")
    return rows


def per_point(rows: list[dict], field: str, fmt: str = "{}") -> str:
    """One value when all points share it, otherwise C<conc> <value> for each point."""
    if len({row[field] for row in rows}) == 1:
        return fmt.format(rows[0][field])
    return ", ".join(f'C{row["Concurrency"]} {fmt.format(row[field])}' for row in rows)


def annotate(ax, x: list, values: list, color: str, fmt: str, dy: float) -> None:
    """Label points above (dy > 0) or below them; a label below that would cross the x axis goes
    right of its point. Call after the layout is fixed."""
    room = (abs(dy) + LABEL_HEIGHT_PT) * ax.figure.dpi / 72
    for conc, value in zip(x, values):
        if dy < 0 and ax.transData.transform((conc, value))[1] - ax.bbox.y0 < room:
            place = {"xytext": (7, 0), "ha": "left", "va": "center"}
        else:
            place = {"xytext": (0, dy), "ha": "center", "va": "bottom" if dy > 0 else "top"}
        ax.annotate(fmt.format(value), (conc, value), textcoords="offset points", fontsize=9,
                    color=color, annotation_clip=False,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.5},
                    **place)


def main() -> None:
    rows = load_sweep(sys.argv[1:])
    x = [row["Concurrency"] for row in rows]

    def column(field: str) -> list:
        return [row[field] for row in rows]

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 14, "axes.labelsize": 11,
                         "legend.fontsize": 10})
    fig, axes = plt.subplots(3, 2, figsize=(12, 11.8), layout="constrained")
    labels = []
    for ax, title, field, fmt in (
        (axes[0, 0], "Total throughput / GPU", "TotalTokensPerSecondPerGPU", "{:,.0f}"),
        (axes[0, 1], "Output throughput / GPU", "OutputTokensPerSecondPerGPU", "{:,.1f}"),
    ):
        ax.plot(x, column(field), color=MAIN, marker="o", linewidth=2)
        labels.append((ax, x, column(field), MAIN, fmt, 9))
        ax.set(title=title, ylabel="tokens/s/GPU", ylim=(0, max(column(field)) * 1.21))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    for ax, title, p50, p90, unit in (
        (axes[1, 0], "Time to first token", "TTFTp50Seconds", "TTFTp90Seconds", "seconds"),
        (axes[1, 1], "Full-response inter-token latency (ITL)", "FullResponseITLp50Ms",
         "FullResponseITLp90Ms", "milliseconds"),
    ):
        ax.plot(x, column(p50), color=MAIN, marker="o", linewidth=2, label="p50")
        ax.plot(x, column(p90), color=SECOND, marker="s", linewidth=2, label="p90")
        labels += [(ax, x, column(p50), MAIN, "{:.2f}", -9), (ax, x, column(p90), SECOND, "{:.2f}", 9)]
        ax.set(title=title, ylabel=unit, ylim=(0, max(column(p90)) * 1.22))
        ax.legend(frameon=False, loc="upper left")
    cache_ax, decode_ax = axes[2]
    usage, theory = column("UsageCacheHitPct"), column("TheoreticalCacheHitPct")
    cache_ax.plot(x, usage, color=MAIN, marker="o", linewidth=2, label="API usage (measured)")
    cache_ax.plot(x, theory, color=SECOND, marker="s", linestyle="--", linewidth=2,
                  label="Theoretical prefix")
    labels += [(cache_ax, x, usage, MAIN, "{:.2f}%", -9), (cache_ax, x, theory, SECOND, "{:.2f}%", 9)]
    cache_ax.set(title="Prompt cache hit rate", ylabel="cached input tokens / input tokens",
                 ylim=(math.floor(min(usage + theory)) - 3, 100))
    cache_ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    cache_ax.legend(frameon=False, loc="lower left")
    running = column("DecodeRunningMean")
    decode_ax.plot(x, running, color=MAIN, marker="o", linewidth=2,
                   label="Server mean, atom:requests_running")
    labels.append((decode_ax, x, running, MAIN, "{:.2f}", 9))
    decode_ax.set(title="Decode running concurrency", ylabel="running requests",
                  ylim=(0, max(running) * 1.23))
    decode_ax.legend(frameon=False, loc="lower left")
    for ax in axes.flat:
        ax.set(xlabel="Concurrency", xticks=x)
        ax.margins(x=0.06)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    first = rows[0]
    fig.suptitle(f'GLM-5.2 MXFP4 | {first["Topology"]} (ATOM + Infera) | '
                 f'{first["GPUs"]} {first["Hardware"]} GPUs', fontsize=18, fontweight="bold")
    accounting = f'Error records excluded: {per_point(rows, "ErrorRecords")}'
    if all(row["CancelledAfterWindow"] is not None for row in rows):
        accounting = (f'In-flight requests cancelled after the sending window: '
                      f'{per_point(rows, "CancelledAfterWindow")} | {accounting}')
    fig.supxlabel("\n".join((
        f'{per_point(rows, "SendingWindowS", "{:,} s")} sending window per point | Simulated acceptance '
        f'{per_point(rows, "SimulatedAcceptance")} | Total includes cached input tokens',
        f'Deployment: {first["Deployment"]}; throughput / GPU divides by all {first["GPUs"]} GPUs',
        "Decode: server-sampled mean of atom:requests_running on the decode endpoint over the "
        "profiling phase (warmup excluded)",
        accounting,
    )), fontsize=10)
    fig.canvas.draw()
    fig.set_layout_engine("none")
    for label in labels:
        annotate(*label)

    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "sweep.png", dpi=180)
    plt.close(fig)
    with (RESULTS / "sweep.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(first))
        writer.writeheader()
        writer.writerows({k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()}
                         for row in rows)
    print(RESULTS / "sweep.png")
    print(RESULTS / "sweep.csv")


if __name__ == "__main__":
    main()
