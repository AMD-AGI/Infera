#!/usr/bin/env python3
"""Plot the completed no-profile sweep; matplotlib is available in the image."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--note", default="", help="Optional observation printed below the figure")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = json.loads((args.run_dir / "analysis/summary.json").read_text())
    groups = defaultdict(list)
    for row in data["points"]:
        config = (row.get("server_max_running", 64), row.get("mem_fraction_static", .85))
        groups[(config, row["concurrency"])].append(row)
    cs = sorted({c for _, c in groups})
    configs = sorted({config for config, _ in groups})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    def series(ax, key, label, color):
        for index, config in enumerate(configs):
            xs = [c for c in cs if (config, c) in groups]
            values = [[r[key] for r in groups[(config, c)]] for c in xs]
            mid = [statistics.median(v) for v in values]
            errors = [[m - min(v) for m, v in zip(mid, values)], [max(v) - m for m, v in zip(mid, values)]]
            suffix = f" (server cap {config[0]})" if len(configs) > 1 else ""
            ax.errorbar(xs, mid, yerr=errors, marker="o" if index == 0 else "s", color=color,
                        linestyle="-" if index == 0 else "--", capsize=3, linewidth=1.5, label=label + suffix)

    ax = axes[0, 0]
    series(ax, "decode_tokens_s_p50", "P50 request speed", "tab:blue")
    series(ax, "decode_tokens_s_p10", "P10 request speed", "tab:orange")
    ax.axhspan(70, 80, alpha=.18, color="green", label="70–80 tokens/s")
    ax.set_title("Per-request decode speed")
    ax.set_ylabel("Output tokens/s per request")
    ax = axes[0, 1]
    series(ax, "output_tokens_s", "4-GPU output throughput", "tab:blue")
    ax.set_title("Instance output throughput")
    ax.set_ylabel("Output tokens/s")
    ax = axes[1, 0]
    series(ax, "tpot_ms_p50", "P50 TPOT", "tab:blue")
    series(ax, "tpot_ms_p90", "P90 TPOT", "tab:orange")
    ax.axhline(12.5, color="green", linestyle="--", linewidth=1, label="80 tok/s: 12.5 ms")
    ax.axhline(1000 / 70, color="brown", linestyle=":", linewidth=1, label="70 tok/s: 14.29 ms")
    ax.set_title("Request-average token latency")
    ax.set_ylabel("TPOT (ms)")
    ax = axes[1, 1]
    for q, color in zip(("p50", "p90", "p99"), ("tab:blue", "tab:orange", "tab:green")):
        series(ax, "itl_ms_" + q, q.upper() + " ITL", color)
    ax.set_title("Native SGLang ITL (chunk-gap / token count)")
    ax.set_ylabel("Adjusted ITL (ms)")
    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.set_xticks(cs, [str(c) for c in cs], rotation=35)
        ax.set_xlabel("Client concurrency")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("GLM-5.2 fake decode · TP4/EP4 · 10000/500 · no profiling", fontsize=15)
    fig.text(.01, .014, "Points: median across runs with the same server capacity; error bars: min/max. CUDA Graph and packet capture enabled.", fontsize=9)
    if args.note:
        fig.text(.01, .001, args.note, fontsize=9)
    fig.tight_layout(rect=(0, .025, 1, .965))
    for ext in ("png", "svg"):
        fig.savefig(args.run_dir / f"analysis/goodput_sweep.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
