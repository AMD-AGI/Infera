"""InferenceX-style Pareto plot of the PD example's first baseline (tokens/s/GPU against interactivity p90):
our frontier points labelled with their parallelism and concurrency, against the frontier lines of InferenceX's
GLM-5.2 agentic results for B300 / GB300 Dynamo + SGLang and MI355X SGLang, and ATOM's MI355X PD run from
InferenceX PR #4045 (not on the site). One panel; reference lines carry no labels.
  python3 plot_pareto_baseline.py REPORT_DIR INFERENCEX_API_JSON OUT_PREFIX"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

report, api, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
ours = [  # (results dir, label, label offset in points, horizontal alignment)
    ("pp4-ck8k-2p-c176", "2P+1D(TP8/DPA) C176", (-10, 8), "right"),
    ("pp4-ck8k-d4-c80", "1P+1D(TP4/DPA) C80", (12, 2), "left"),
    ("pp4-ck8k-d4-c64", "1P+1D(TP4/DPA) C64", (-10, -12), "right"),
    ("pp4-ck8k-rb-c88", "1P+1D(TP8/DPA) C88", (6, 10), "left"),
    ("ex-rb-c64", "1P+1D(TP8/DPA) C64", (10, 6), "left"),
    ("ex-rb-c48", "1P+1D(TP8/DPA) C48", (10, 4), "left"),
    ("ex-rb-c32", "1P+1D(TP8/DPA) C32", (10, -4), "left"),
]
refs = [  # (hardware, framework, legend, color)
    ("gb300", "dynamo-sglang", "GB300 Dynamo+SGLang (InferenceX)", "tab:blue"),
    ("b300", "dynamo-sglang", "B300 Dynamo+SGLang (InferenceX)", "tab:green"),
    ("mi355x", "sglang", "MI355X SGLang (InferenceX)", "tab:orange"),
]
atom_pd_dir = report.parent / "gb300-atom-pd-analysis-20261021/evidence/atom-pd-pr4045/run39028510364"


def frontier(pts):
    """Points not dominated in (x up, y up), sorted by x."""
    return sorted(p for p in pts if not any(q[0] >= p[0] and q[1] >= p[1] and q != p for q in pts))


def xy(j):
    return j["request_metrics"]["latency"]["intvty"]["p90"], j["request_metrics"]["throughput"]["per_gpu"]["total_tput_tps"]


rows = [r for r in json.loads(api.read_text()) if r["benchmark_type"] == "agentic_traces"]
fig, ax = plt.subplots(figsize=(12, 7.5))
for hw, fw, name, color in refs:
    f = frontier([(r["metrics"]["p90_intvty"], r["metrics"]["tput_per_gpu"])
                  for r in rows if r["hardware"] == hw and r["framework"] == fw])
    ax.plot(*zip(*f), "-o", color=color, ms=4, lw=1.4, alpha=0.8, label=name)
f = frontier([xy(json.loads(p.read_text())) for p in atom_pd_dir.glob("*.json")])
ax.plot(*zip(*f), "--D", color="tab:purple", ms=4, lw=1.4, alpha=0.8,
        label="MI355X ATOM PD (InferenceX PR #4045, unofficial run)")

pts = [(*xy(json.loads(next((report / "results" / d).glob("agentx_conc*.json")).read_text())), lab, off, ha)
       for d, lab, off, ha in ours]
ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color="tab:red", lw=2.6, ms=7, mec="black", zorder=5,
        label="Infera + SGLang v0.5.22 PD, MI355X (PR #210)")
for x, y, lab, off, ha in pts:
    ax.annotate(lab, (x, y), textcoords="offset points", xytext=off, ha=ha, va="center", fontsize=10,
                color="tab:red", fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

ax.set_xlim(0, 200)
ax.set_ylim(0, 66000)
ax.set_xlabel("Interactivity p90 (tok/s/user)")
ax.set_ylabel("Throughput (tokens/s/GPU)")
ax.set_title("GLM-5.2 agentic (InferenceX AgentX): tokens/s/GPU vs interactivity p90")
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=9)
fig.text(0.01, 0.005, "Ours: every P is a TP1×PP4 prefill (4 GPUs) with MTP; D is the decode; the 1P+1D(TP8/DPA) points use "
         f"decode request balance.\n3600 s per point, forced MTP acceptance 3.61. InferenceX: {api.name}. "
         "Each line is its Pareto frontier.", fontsize=8, color="gray")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(f"{out}.png", dpi=150)
fig.savefig(f"{out}.svg")
print(f"wrote {out}.png / .svg")
