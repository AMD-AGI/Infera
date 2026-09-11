#!/usr/bin/env python3
"""Render standalone PNG/SVG summaries (requires matplotlib, present in image)."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Output path without suffix")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = json.loads(args.summary.read_text())
    if any(t["graph_launches_without_kernels"] for t in data["traces"]):
        raise RuntimeError("Refusing to plot incomplete graph traces as a full distribution")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.8, 1]})
    for ax, key, title in zip(axes, ("categories", "stages"),
                              ("GPU kernel categories", "MTP stage attribution")):
        rows = data[key]
        names = [r["name"] for r in rows]
        values = [r["percent_kernel_time"] for r in rows]
        bars = ax.barh(names, values, color="#4472a8")
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.22)
        ax.set_xlabel("Share of summed GPU kernel duration (%)")
        ax.set_title(title, loc="left", fontsize=13)
        ax.bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=4, fontsize=9)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle("GLM-5.2 fake decode · TP4/EP4 · concurrency 32 · 10000/500 tokens", fontsize=15, y=1.01)
    fig.text(.01, -.025,
             "CUDA Graph enabled; ROCm packet capture disabled for complete kernel tracing. "
             "Percentages are not wall-clock latency shares.", fontsize=9)
    fig.tight_layout(w_pad=3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(args.output.with_suffix("." + suffix), bbox_inches="tight", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
