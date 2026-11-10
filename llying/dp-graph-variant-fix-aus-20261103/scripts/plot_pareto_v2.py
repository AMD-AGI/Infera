"""InferenceX-style Pareto plot (tokens/s/GPU against interactivity p90) of the second round, the kit v2
points of Infera PR #235 (examples/sglang_pd_glm5.2_agentx: one-node 4+4, DP balance), next to the first
baseline (PR #210, 10-23), the reference lines of that figure (2p1d-v0522-20261020/scripts/
plot_pareto_baseline.py: InferenceX's GLM-5.2 agentic frontiers for GB300 / B300 Dynamo + SGLang and
MI355X SGLang, ATOM's MI355X PD run from InferenceX PR #4045), and ATOM's 4P4D kit of Infera PR #209.
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
# Second round: the kit v2 points of PR #235's table (AgentX result JSONs in results/v2/pareto), with their labels
# and label offsets (points, horizontal alignment).
v2 = [
    ("dpbal-k3-c104", "1P+1D(TP4/DPA) K3 C104", (-10, 8), "right"),
    ("dpbal-k3-c88", "1P+1D(TP4/DPA) K3 C88", (10, 2), "left"),
    ("dpbal-k5-c64", "1P+1D(TP4/DPA) K5 C64", (10, 4), "left"),
    ("dpbal-k5-c48", "1P+1D(TP4/DPA) K5 C48", (10, 4), "left"),
    ("dpbal-k5-c32", "1P+1D(TP4/DPA) K5 C32", (10, -4), "left"),
]
# Infera PR #209 (ATOM 4P4D kit, new defaults), its Performance table: (CONC, input tok/s/GPU, output tok/s/GPU,
# P90 interactivity). The JSONs are not in the repo; total = input + output, as total_tput_tps in our JSONs.
pr209 = [(56, 32925, 246.2, 64.0), (72, 40075, 321.9, 52.1), (96, 46234, 387.3, 45.6), (112, 48154, 406.0, 42.1)]
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

p9 = sorted((x, i + o) for _, i, o, x in pr209)
ax.plot(*zip(*p9), "-^", color="tab:brown", lw=2.0, ms=7, mec="black", zorder=5,
        label="MI355X ATOM 4P4D (Infera PR #209, one node)")
for c, i, o, x in pr209:
    ax.annotate(f"C{c}", (x, i + o), textcoords="offset points", xytext=(-8, -2), ha="right", va="center",
                fontsize=8.5, color="tab:brown", zorder=6)

p2 = [(*xy(json.loads((v2_dir / f"{src}.json").read_text())), lab, off, ha) for src, lab, off, ha in v2]
ax.plot(*zip(*sorted(p[:2] for p in p2)), "-o", color="darkred", lw=2.8, ms=8, mec="black", zorder=6,
        label="Round 2: Infera + SGLang v0.5.22 PD, kit v2, MI355X one node (PR #235, 11-09)")
for x, y, lab, off, ha in p2:
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
         "Round 2: P is a TP1×PP4 prefill (GPUs 0-3) and D a TP4/DP4-attention decode (GPUs 4-7) on one node (MC_DISABLE_HIP=1), "
         "DP balance 400k; K = MTP draft steps, forced acceptance 2.99 (K3) / 3.61 (K5);\ndraft MoE in FP8 (not InferenceX-"
         "submittable); 3600 s per point. PR #209: forced acceptance 2.99, total = input + output tok/s/GPU of its table. "
         "Round 1 and references as in pareto-p90-baseline-v4-20261023.",
         fontsize=7.5, color="gray")
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(f"{out}.png", dpi=150)
fig.savefig(f"{out}.svg")
print(f"wrote {out}.png / .svg")
for x, y, lab, _, _ in sorted(p2):
    print(f"round 2: {lab:28s} {y:8.0f} @ {x:6.1f}")
for x, y in p9:
    print(f"PR #209:                              {y:8.0f} @ {x:6.1f}")
