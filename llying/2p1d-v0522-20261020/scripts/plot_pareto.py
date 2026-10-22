"""Tokens/s/GPU against interactivity p90, InferenceX style: our 2P1D / 1P1D sweep (and the mixed
TP4 C48 point) next to InferenceX's GLM-5.2 agentic lines for B300 and GB300 Dynamo + SGLang and
MI355X SGLang and ATOM, plus ATOM's MI355X PD points from InferenceX PR #4045 (not yet on the site).
Each line is the Pareto frontier of its points; dominated points are faint.
  python3 plot_pareto.py REPORT_DIR INFERENCEX_API_JSON OUT_PREFIX"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

report, api, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
ours = [  # (results dir, label, label offset in points)
    ("results/c176", "2P(TP4/DPA)+1D(TP8/DPA) C176, 16 GPU", (-150, -16)),
    ("results/c128", "2P1D C128, 16 GPU", (8, -2)),
    ("results/c96", "2P1D C96, 16 GPU", (-95, -12)),
    ("results/c64", "2P1D C64, 16 GPU", (6, -13)),
    ("results/pp4-ck8k-2p-c176", "2P(PP4)+1D(TP8/DPA) C176, prefill MTP, 16 GPU", (8, 8)),
    ("results/pp4-ck8k-2p-c192", "2P(PP4)+1D C192", (-80, 4)),
    ("results/pp4-ck8k-d4-c80", "1P(PP4)+1D(TP4/DPA) C80, prefill MTP, 8 GPU", (10, -6)),
    ("results/pp4-ck8k-d4-c64", "1P(PP4)+1D(TP4/DPA) C64, prefill MTP, 8 GPU", (8, 4)),
    ("results/pp4-ck8k-c112", "1P(PP4)+1D C112, prefill MTP, 12 GPU", (8, 4)),
    ("results/pp4-ck8k-c88", "1P(PP4)+1D(TP8/DPA) C88, prefill MTP, 12 GPU", (8, -12)),
    ("results/pp4-ck8k-c64", "1P(PP4)+1D C64, prefill MTP, 12 GPU", (8, 4)),
    ("results/pp4-ck8k-c32", "1P(PP4)+1D C32, prefill MTP, 12 GPU", (8, 4)),
    ("results/1p1d-c88", "1P(TP4/DPA)+1D(TP8/DPA) C88, 12 GPU", (8, -12)),
    ("results/1p1d-c64", "1P1D C64, 12 GPU", (8, 4)),
    ("results/1p1d-c32", "1P(TP4/DPA)+1D(TP8/DPA) C32, 12 GPU", (6, -13)),
]
ZOOM_X, ZOOM_Y = 130, 46000
mixed = report.parent / "mix-v0522-correctness-20261019/results/example-c48-dcar-topkv2-140"
atom_pd = sorted((report.parent / "gb300-atom-pd-analysis-20261021/evidence/atom-pd-pr4045/run39028510364").glob("*.json"))


def our_point(d: Path):
    j = json.loads(next(d.glob("agentx_conc*.json")).read_text())
    return j["request_metrics"]["latency"]["intvty"]["p90"], j["request_metrics"]["throughput"]["per_gpu"]["total_tput_tps"]


def shape(r):
    def leg(tp, ep, dpa):
        return f"TP{tp}" + (f"/EP{ep}" if ep > 1 else "") + ("/DPA" if dpa else "")

    if r["disagg"]:
        return (f"{r['prefill_num_workers']}P({leg(r['prefill_tp'], r['prefill_ep'], r['prefill_dp_attention'])})"
                f"+{r['decode_num_workers']}D({leg(r['decode_tp'], r['decode_ep'], r['decode_dp_attention'])})")
    return leg(r["decode_tp"], r["decode_ep"], r["decode_dp_attention"])


def frontier(pts):
    """Points not dominated in (x up, y up), sorted by x."""
    keep = [p for p in pts if not any(q[0] >= p[0] and q[1] >= p[1] and q != p for q in pts)]
    return sorted(keep)


rows = [r for r in json.loads(api.read_text()) if r["benchmark_type"] == "agentic_traces"]
lines = [
    ("b300", "dynamo-sglang", "B300 Dynamo+SGLang", "tab:green"),
    ("gb300", "dynamo-sglang", "GB300 Dynamo+SGLang", "tab:blue"),
    ("mi355x", "sglang", "MI355X SGLang (InferenceX)", "tab:orange"),
    ("mi355x", "atom", "MI355X ATOM (InferenceX)", "tab:purple"),
]

fig, axes = plt.subplots(1, 2, figsize=(22, 9))
for ax, zoom in zip(axes, (False, True)):
    for hw, fw, name, color in lines:
        pts = [(r["metrics"]["p90_intvty"], r["metrics"]["tput_per_gpu"], f"{shape(r)} c{r['conc']}")
               for r in rows if r["hardware"] == hw and r["framework"] == fw]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], color=color, alpha=0.25, s=18)
        f = frontier(pts)
        ax.plot([p[0] for p in f], [p[1] for p in f], "-o", color=color, ms=5, lw=1.6, label=name)
        for x, y, lab in f:
            if zoom == (x <= ZOOM_X and y <= ZOOM_Y):
                ax.annotate(lab, (x, y), textcoords="offset points", xytext=(4, 3), fontsize=6.5, color=color)
    pts = []
    for p in atom_pd:
        j = json.loads(p.read_text())
        dcp = f"+DCP{j['decode_dcp_size']}" if j["decode_dcp_size"] > 1 else ""
        pts.append((j["request_metrics"]["latency"]["intvty"]["p90"], j["request_metrics"]["throughput"]["per_gpu"]["total_tput_tps"],
                    f"1P(PP4)+{j['decode_num_workers']}D(TP{j['decode_tp']}{dcp}) c{j['conc']}, {j['num_gpus']} GPU"))
    if pts:
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], color="tab:purple", alpha=0.25, s=18, marker="D")
        f = frontier(pts)
        ax.plot([p[0] for p in f], [p[1] for p in f], "--D", color="tab:purple", ms=5, lw=1.6,
                label="MI355X ATOM PD (InferenceX PR #4045, unofficial run)")
        for x, y, lab in f:
            if zoom == (x <= ZOOM_X and y <= ZOOM_Y):
                ax.annotate(lab, (x, y), textcoords="offset points", xytext=(4, 3), fontsize=6.5, color="tab:purple")
    pts = [(*our_point(report / d), lab, off) for d, lab, off in ours]
    f = frontier([p[:3] for p in pts])
    ax.plot([p[0] for p in f], [p[1] for p in f], "-", color="tab:red", lw=2.4,
            label="Infera + SGLang v0.5.22, MI355X PD (this sweep)")
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], color="tab:red", s=55, zorder=5, edgecolor="black")
    mx, my = our_point(mixed)
    ax.scatter([mx], [my], marker="s", color="tab:red", facecolor="none", s=60, zorder=5)
    if zoom:
        for x, y, lab, off in pts:
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=off, fontsize=8.5, color="tab:red",
                        fontweight="bold")
        ax.annotate("mixed TP4/DPA C48, 4 GPU (10-20)", (mx, my), textcoords="offset points", xytext=(-30, 12),
                    fontsize=7.5, color="tab:red")
    ax.set_xlabel("Interactivity p90 (tok/s/user)")
    ax.set_ylabel("Throughput (tokens/s/GPU)")
    ax.grid(alpha=0.3)
    if zoom:
        ax.set_xlim(0, ZOOM_X)
        ax.set_ylim(0, ZOOM_Y)
        ax.set_title("Zoom: MI355X region")
    else:
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 68000)
        ax.set_title("GLM-5.2 agentic (AgentX), tokens/s/GPU vs interactivity p90")
        ax.legend(loc="upper right", fontsize=9)
fig.text(0.01, 0.005, f"InferenceX: {api.name} (api/v1/benchmarks?model=GLM-5.2, agentic_traces); "
         "ours: results/*/agentx_conc*.json, all with --disable-custom-all-reduce, forced MTP acceptance 3.61",
         fontsize=8, color="gray")
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(f"{out}.png", dpi=130)
fig.savefig(f"{out}.svg")
print(f"wrote {out}.png / .svg")
