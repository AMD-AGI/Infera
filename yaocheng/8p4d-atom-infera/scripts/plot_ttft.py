#!/usr/bin/env python3
"""Plot the AgentX TTFT statistics of results/c*/agentx_conc*.json against concurrency.

Usage: scripts/plot_ttft.py
Reads the aggregates under $RESULTS_DIR (default results/) and writes ttft_vs_conc.png there.
The statistics are request_metrics.latency.ttft, which InferenceX computes from the
per-request time to first token of the measured requests (warmup and failed requests excluded).
"""
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, StrMethodFormatter

RESULTS = Path(os.environ.get("RESULTS_DIR", Path(__file__).resolve().parent.parent / "results"))
# (statistic in request_metrics.latency.ttft, color, linestyle, marker)
STATS = (("p50", "tab:blue", "-", "o"), ("p75", "tab:green", "-", "o"),
         ("p90", "tab:orange", "-", "o"), ("p95", "tab:red", "-", "o"),
         ("mean", "black", "--", "D"))
LABEL_GAP_PT = 11


def deployment(r: dict) -> str:
    prefill = f'TP{r["prefill_tp"]}' + (" + DPA" if r["prefill_dp_attention"] == "true" else "")
    decode = f'TP{r["decode_tp"]}' + (f' + DCP{r["decode_dcp_size"]}' if r["decode_dcp_size"] > 1 else "")
    return (f'prefill {r["prefill_num_workers"]}× {prefill} ({r["num_prefill_gpu"]} GPUs), '
            f'decode {r["decode_num_workers"]}× {decode} ({r["num_decode_gpu"]} GPUs)')


def load(results: Path) -> list[dict]:
    rows = []
    for path in results.glob("c*/agentx_conc*.json"):
        r = json.loads(path.read_text())
        rows.append({"conc": r["conc"], "ttft": r["request_metrics"]["latency"]["ttft"],
                     "requests": r["request_accounting"]["records_profiled"],
                     "duration": r["request_metrics"]["throughput"]["duration_seconds"],
                     "deployment": deployment(r)})
    if not rows:
        raise SystemExit(f"no c*/agentx_conc*.json under {results}")
    if len({row["deployment"] for row in rows}) > 1:
        raise SystemExit(f"{results} mixes deployments")
    return sorted(rows, key=lambda row: row["conc"])


def log_ticks(lo: float, hi: float) -> list[float]:
    """1-2-5 ticks from the last one at or below lo to the first one at or above hi."""
    ticks = [m * 10 ** e for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1)
             for m in (1, 2, 5)]
    start = max(i for i, t in enumerate(ticks) if t <= lo)
    stop = min(i for i, t in enumerate(ticks) if t >= hi)
    return ticks[start:stop + 1]


def label_values(ax, rows: list[dict]) -> None:
    """Write each value right of its marker, lifting labels of one concurrency apart."""
    dpi = ax.figure.dpi
    gap = LABEL_GAP_PT * dpi / 72
    for row in rows:
        top = -math.inf
        for value, color in sorted((row["ttft"][key], color) for key, color, _, _ in STATS):
            y = ax.transData.transform((row["conc"], value))[1]
            lift = max(0.0, top + gap - y)
            top = y + lift
            ax.annotate(f"{value:.2f}" if value < 10 else f"{value:.1f}", (row["conc"], value),
                        xytext=(7, lift * 72 / dpi), textcoords="offset points", ha="left",
                        va="center", fontsize=9, color=color, zorder=5,
                        bbox={"boxstyle": "square,pad=0.1", "facecolor": "white",
                              "edgecolor": "none", "alpha": 0.8})


def main() -> None:
    rows = load(RESULTS)
    concs = [row["conc"] for row in rows]
    values = [row["ttft"][key] for row in rows for key, *_ in STATS]
    ticks = log_ticks(min(values) / 1.25, max(values) * 1.25)
    span = max(concs[-1] - concs[0], 1)

    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    for key, color, style, marker in STATS:
        ax.plot(concs, [row["ttft"][key] for row in rows], color=color, linestyle=style,
                marker=marker, linewidth=2, markersize=6, label=key)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set(xlim=(concs[0] - 0.1 * span, concs[-1] + 0.18 * span), ylim=(ticks[0], ticks[-1]),
           xticks=concs, xlabel="AgentX concurrency", ylabel="TTFT (s, log scale)",
           title=rows[0]["deployment"])
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="TTFT statistic", loc="upper left")
    fig.suptitle("GLM-5.2 MXFP4 8P4D (ATOM + Infera) AgentX: TTFT vs concurrency",
                 fontsize=13, fontweight="bold")
    fig.supxlabel(
        "TTFT statistics of the measured requests (warmup and failed requests excluded), "
        "from request_metrics.latency.ttft.\n"
        + "; ".join(f'C{row["conc"]}: {row["requests"]:,} requests in {row["duration"]:,.0f} s'
                    for row in rows) + ".", fontsize=8.5)

    fig.canvas.draw()
    fig.set_layout_engine("none")
    label_values(ax, rows)
    out = RESULTS / "ttft_vs_conc.png"
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
