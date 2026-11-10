"""InferenceX-style Pareto plot (tokens/s/GPU against interactivity p90) of the second-round results on AUS
MI355X next to the first baseline (PR #210, 10-23) and the reference lines of that figure
(2p1d-v0522-20261020/scripts/plot_pareto_baseline.py): InferenceX's GLM-5.2 agentic frontiers for
GB300 / B300 Dynamo + SGLang and MI355X SGLang, and ATOM's MI355X PD run from InferenceX PR #4045.
The second round is drawn twice: the frontier over all its points (labelled), and the one-node 4+4
DP-balance series on its own.
  python3 plot_pareto_v2.py LLYING_DIR OUT_PREFIX"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

llying, out = Path(sys.argv[1]), sys.argv[2]
v1_dir = llying / "2p1d-v0522-20261020"
api = v1_dir / "evidence/inferencex-api-glm52-20261021.json"
atom_pd_dir = llying / "gb300-atom-pd-analysis-20261021/evidence/atom-pd-pr4045/run39028510364"
v2_dir = llying / "dp-graph-variant-fix-aus-20261103/results/v2/pareto"

v1 = [  # results dir under 2p1d-v0522-20261020/results (as in plot_pareto_baseline.py)
    "pp4-ck8k-2p-c176", "pp4-ck8k-d4-c80", "pp4-ck8k-d4-c64", "pp4-ck8k-rb-c88", "ex-rb-c64", "ex-rb-c48", "ex-rb-c32",
]
# Second round: (source, label). A source is an AgentX result JSON in results/v2/pareto, or (interactivity p90,
# tokens/s/GPU) taken from a score file when the run directory stayed on the node's /scratch.
v2 = [
    ((70.9, 56593), "1P(PP8)+1D(TP8/DPA) K3 C208"),  # vfix-logs/opt/score-pp8.txt (11-06)
    ((65.7, 54094), "2P+1D(TP8/DPA) K3 C208"),  # vfix-logs/opt/score-2p1d.txt (11-04)
    ("dpbal-k3-c104", "1P+1D(TP4/DPA) K3 C104"),
    ("dpbal-k3-c88", "1P+1D(TP4/DPA) K3 C88"),
    ("dpbal-k5-c64", "1P+1D(TP4/DPA) K5 C64"),
    ("1p2d-c128", "1P+2D(TP4/DPA) K3 C128"),
    ("1p2d-c96", "1P+2D(TP4/DPA) K3 C96"),
    ("dpbal-k5-c48", "1P+1D(TP4/DPA) K5 C48"),
    ("dpbal-k5-c32", "1P+1D(TP4/DPA) K5 C32"),
    ("4p4-2node-c104", "1P+1D(TP4/DPA) K3 C104, two nodes"),
]
one_node = ["dpbal-k5-c32", "dpbal-k5-c48", "dpbal-k5-c64", "dpbal-k3-c88", "dpbal-k3-c104"]
label_offset = {  # frontier labels: (offset in points, horizontal alignment)
    "1P(PP8)+1D(TP8/DPA) K3 C208": ((10, 8), "left"),
    "1P+1D(TP4/DPA) K5 C64": ((12, 4), "left"),
    "1P+2D(TP4/DPA) K3 C96": ((12, -2), "left"),
    "1P+1D(TP4/DPA) K5 C48": ((12, 4), "left"),
    "1P+1D(TP4/DPA) K5 C32": ((12, -4), "left"),
}
refs = [  # (hardware, framework, legend, color)
    ("gb300", "dynamo-sglang", "GB300 Dynamo+SGLang (InferenceX)", "tab:blue"),
    ("b300", "dynamo-sglang", "B300 Dynamo+SGLang (InferenceX)", "tab:green"),
    ("mi355x", "sglang", "MI355X SGLang (InferenceX)", "tab:orange"),
]


def frontier(pts):
    """Points not dominated in (x up, y up), sorted by x."""
    return sorted(p for p in pts if not any(q[0] >= p[0] and q[1] >= p[1] and q[:2] != p[:2] for q in pts))


def xy(j):
    return j["request_metrics"]["latency"]["intvty"]["p90"], j["request_metrics"]["throughput"]["per_gpu"]["total_tput_tps"]


def point(src):
    return src if isinstance(src, tuple) else xy(json.loads((v2_dir / f"{src}.json").read_text()))


rows = [r for r in json.loads(api.read_text()) if r["benchmark_type"] == "agentic_traces"]
fig, ax = plt.subplots(figsize=(12, 7.5))
for hw, fw, name, color in refs:
    f = frontier([(r["metrics"]["p90_intvty"], r["metrics"]["tput_per_gpu"])
                  for r in rows if r["hardware"] == hw and r["framework"] == fw])
    ax.plot(*zip(*f), "-o", color=color, ms=4, lw=1.4, alpha=0.8, label=name)
f = frontier([xy(json.loads(p.read_text())) for p in atom_pd_dir.glob("*.json")])
ax.plot(*zip(*f), "--D", color="tab:purple", ms=4, lw=1.4, alpha=0.8,
        label="MI355X ATOM PD (InferenceX PR #4045, unofficial run)")

p1 = [xy(json.loads(next((v1_dir / "results" / d).glob("agentx_conc*.json")).read_text())) for d in v1]
ax.plot(*zip(*sorted(p1)), "--o", color="tab:red", lw=1.8, ms=5, alpha=0.75, mec="black", zorder=4,
        label="Round 1: Infera + SGLang v0.5.22 PD, MI355X (PR #210, 10-23)")

p2 = [(*point(src), lab) for src, lab in v2]
f2 = frontier(p2)
ax.plot([p[0] for p in f2], [p[1] for p in f2], "-o", color="darkred", lw=2.8, ms=8, mec="black", zorder=6,
        label="Round 2: frontier over all shapes, MI355X AUS (11-09)")
p4 = sorted(point(s) for s in one_node)
ax.plot(*zip(*p4), ":s", color="darkred", lw=1.6, ms=5, alpha=0.8, zorder=5,
        label="Round 2: one-node 1P+1D(TP4/DPA) with DP balance, C32-C104")
for x, y, lab in f2:
    off, ha = label_offset.get(lab, ((10, 6), "left"))
    ax.annotate(lab, (x, y), textcoords="offset points", xytext=off, ha=ha, va="center", fontsize=9.5,
                color="darkred", fontweight="bold", zorder=7,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

ax.set_xlim(0, 200)
ax.set_ylim(0, 66000)
ax.set_xlabel("Interactivity p90 (tok/s/user)")
ax.set_ylabel("Throughput (tokens/s/GPU)")
ax.set_title("GLM-5.2 agentic (InferenceX AgentX): tokens/s/GPU vs interactivity p90, round 1 vs round 2")
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=8.5)
fig.text(0.01, 0.005,
         "Round 2: every P is a TP1×PP4 prefill (4 GPUs) except PP8 (8 GPUs); D is the decode; one-node points run P and D on one "
         "node (MC_DISABLE_HIP=1); DP balance 400k on the 1P+1D(TP4) points;\nK = MTP draft steps, forced acceptance 2.99 (K3) / "
         "3.61 (K5); draft MoE in FP8 (not InferenceX-submittable); 3600 s per point. Round 1 and references as in "
         "pareto-p90-baseline-v4-20261023. Each line is a Pareto frontier.",
         fontsize=7.5, color="gray")
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(f"{out}.png", dpi=150)
fig.savefig(f"{out}.svg")
print(f"wrote {out}.png / .svg")
for x, y, lab in f2:
    print(f"round 2 frontier: {lab:40s} {y:8.0f} @ {x:6.1f}")
