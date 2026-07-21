#!/usr/bin/env python3
"""Regenerate the two result figures in images/ from the benchmark numbers.

Data are 3-run averages from the disaggregation-vs-single-node sweep (Kimi-K2.6-MXFP4
on MI355X); edit the arrays below to plot your own run_sweep.sh results. Usage:

    pip install matplotlib
    python plot_results.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT, exist_ok=True)

GRAY, TEAL, BLUE, RED = "#808080", "#1b9e77", "#3b7dd8", "#d1603d"
conc = [4, 8, 16, 32, 64, 96]
x = np.arange(len(conc))

# --- Figure 1: relative goodput at the ~30 tok/s per-user SLA ---
# TP8 meets the SLA only through 16 sessions; beyond that its throughput is not goodput.
tp8_goodput = [0.35, 0.62, 1.00, 1.46, 1.58, 1.57]
pd_goodput = [0.37, 0.67, 1.22, 1.92, 2.51, 2.51]
tp8_meets_sla = [True, True, True, False, False, False]

fig, ax = plt.subplots(figsize=(10, 4.3))
w = 0.38
for i, (v, meets) in enumerate(zip(tp8_goodput, tp8_meets_sla)):
    if meets:
        ax.bar(x[i] - w / 2, v, w, color=GRAY)
    else:
        ax.bar(x[i] - w / 2, v, w, facecolor="none", edgecolor=GRAY, linestyle="--", linewidth=1.4)
ax.bar(x + w / 2, pd_goodput, w, color=TEAL)
for i, v in enumerate(tp8_goodput):
    ax.text(x[i] - w / 2, v + 0.03, f"{v:.2f}x", ha="center", va="bottom", fontsize=11, color=GRAY)
for i, v in enumerate(pd_goodput):
    ax.text(
        x[i] + w / 2, v + 0.03, f"{v:.2f}x", ha="center", va="bottom", fontsize=11, color="#111"
    )
ax.legend(
    handles=[
        Patch(color=GRAY, label="Single-node TP8 (meets SLA)"),
        Patch(
            facecolor="none",
            edgecolor=GRAY,
            linestyle="--",
            label="TP8 below SLA - not counted as goodput",
        ),
        Patch(color=TEAL, label="Infera PD 12-GPU (meets SLA)"),
    ],
    loc="upper left",
    frameon=False,
    fontsize=11,
)
ax.set_xticks(x)
ax.set_xticklabels(conc)
ax.set_ylim(0, 2.8)
ax.set_xlabel("Concurrent agent sessions", fontsize=12)
ax.set_ylabel("Relative goodput (TP8 @ 16 = 1.0x)", fontsize=12)
ax.set_title("Relative goodput by concurrency (~30 tok/s per-user SLA)", fontsize=14)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_relative_goodput.png"), dpi=150)
plt.close(fig)

# --- Figure 2: per-user generation speed vs concurrency ---
tp8_speed = [48.3, 43.0, 34.5, 25.2, 14.3, 11.9]
pd_speed = [53.6, 51.1, 45.3, 37.0, 32.2, 30.6]

fig, ax = plt.subplots(figsize=(10, 4.3))
ax.axhline(50, color=BLUE, linestyle="--", linewidth=1.3)
ax.text(len(conc) - 1, 51.0, "Interactive SLA (~50 tok/s)", color=BLUE, fontsize=11, ha="right")
ax.axhline(30, color=RED, linestyle="--", linewidth=1.3)
ax.text(0, 30.8, "Relaxed SLA (~30 tok/s)", color=RED, fontsize=11, ha="left")
ax.plot(x, tp8_speed, "-o", color=GRAY, linewidth=2.2, markersize=7, label="Single-node TP8")
ax.plot(x, pd_speed, "-o", color=TEAL, linewidth=2.2, markersize=7, label="Infera PD 12-GPU")
for i, v in enumerate(tp8_speed):
    ax.text(x[i], v - 2.4, f"{v:.1f}", ha="center", color=GRAY, fontsize=11)
for i, v in enumerate(pd_speed):
    ax.text(x[i], v + 1.3, f"{v:.1f}", ha="center", color="#111", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(conc)
ax.set_ylim(0, 60)
ax.set_xlabel("Concurrent agent sessions", fontsize=12)
ax.set_ylabel("Per-user generation speed (tokens/s)", fontsize=12)
ax.set_title("Per-user generation speed vs concurrency", fontsize=14)
ax.legend(loc="lower left", frameon=False, fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_per_user_speed.png"), dpi=150)
plt.close(fig)

print(f"wrote fig1_relative_goodput.png and fig2_per_user_speed.png to {OUT}")
