###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Plot KV-cliff sweep CSVs (one or more arms) onto a single chart
that mirrors LMCache's June 1 slide format.

Inputs: one or more CSVs produced by ``run_cliff.py`` (each CSV has
its own ``arm`` column — `vram_only`, `kvd_v2`, etc.). Outputs a
PNG / SVG showing throughput vs concurrency per arm with the cliff
inflection visible.

Usage:
  python -u -m bench.kvcache.kv_cache_cliff.plot_cliff \\
      results/cliff-vram-only.csv \\
      results/cliff-kvd-v2.csv \\
      --out results/cliff.png \\
      --title "KV Cache Cliff — MI355X TP=1 — gpt-oss-120b — ISL=20K OSL=1"
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median


def _arm_label(arm: str) -> str:
    return {
        "vram_only": "vLLM VRAM-only (baseline)",
        "vram_dram": "vLLM VRAM+DRAM (CPU offload)",
        "kvd_v2": "vLLM VRAM+kvd-v2 (NVMe)",
        "kvd_v2_posix": "vLLM VRAM+kvd-v2 async POSIX",
        "kvd_v2_hipfile": "vLLM VRAM+kvd-v2 async hipFile",
        "kvd_v2_nfs": "vLLM VRAM+kvd-v2 (Vast NFS)",
    }.get(arm, arm)


def _arm_color(arm: str) -> str:
    return {
        "vram_only": "tab:orange",
        "vram_dram": "tab:red",
        "kvd_v2": "tab:blue",
        "kvd_v2_posix": "tab:blue",
        "kvd_v2_hipfile": "tab:green",
        "kvd_v2_nfs": "tab:purple",
    }.get(arm, "tab:gray")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csvs", nargs="+", help="one or more cliff CSV outputs")
    parser.add_argument("--out", required=True, help="output image path (.png or .svg)")
    parser.add_argument("--title", default="KV Cache Cliff")
    parser.add_argument(
        "--xlim", type=int, default=None, help="cap x-axis at this concurrency (default: auto)"
    )
    parser.add_argument(
        "--ylim", type=int, default=None, help="cap y-axis at this throughput (default: auto)"
    )
    args = parser.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    # arm → concurrency → list of throughput samples (across iters)
    by_arm: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for csv_path in args.csvs:
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                arm = row["arm"]
                c = int(row["concurrency"])
                tput = float(row["throughput_tok_s_total"])
                by_arm[arm][c].append(tput)

    if not by_arm:
        print("no rows found", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(11, 6))
    for arm in sorted(by_arm):
        xs = sorted(by_arm[arm])
        ys = [median(by_arm[arm][c]) for c in xs]
        ax.plot(
            xs,
            ys,
            marker="s",
            linewidth=2,
            markersize=6,
            color=_arm_color(arm),
            label=_arm_label(arm),
        )
        # Annotate each data point with its throughput value (LMCache
        # slide style).
        for x, y in zip(xs, ys):
            ax.annotate(
                f"{int(y)}",
                xy=(x, y),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=_arm_color(arm),
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.7),
            )

    ax.set_xlabel("Number of clients (concurrency)")
    ax.set_ylabel("Total Token Throughput (tok/s)")
    ax.set_title(args.title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.9)
    if args.xlim:
        ax.set_xlim(0, args.xlim)
    if args.ylim:
        ax.set_ylim(0, args.ylim)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
