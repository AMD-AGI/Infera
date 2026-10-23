"""Pareto frontier of our points (interactivity p90 vs tok/s/GPU), each point's lane cycle split
(TTFT / decode / client gap, Little's-law decode in-flight), and our frontier against InferenceX
references at fixed p90 (linear interpolation between frontier points).
  python3 frontier_gap.py REPORT_DIR INFERENCEX_API_JSON"""

import json
import sys
from pathlib import Path

report, api = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
ours = [  # (label, results dir, GPUs)
    ("4+8 C32", "pp4-ck8k-c32", 12),
    ("4+8 C64", "pp4-ck8k-c64", 12),
    ("4+8 C88", "pp4-ck8k-c88", 12),
    ("4+8 C112", "pp4-ck8k-c112", 12),
    ("4+4 C64", "pp4-ck8k-d4-c64", 8),
    ("4+4 C80", "pp4-ck8k-d4-c80", 8),
    ("2P C176", "pp4-ck8k-2p-c176", 16),
    ("2P C192", "pp4-ck8k-2p-c192", 16),
]
# Infera PR #209 has no raw JSON; points from its README "Reference numbers" (input tok/s/GPU).
atom209 = [(42.1, 48154), (45.6, 46234), (52.1, 40075), (64.0, 32925)]
P90 = [43.2, 45, 48.6, 50.7, 55, 60, 66, 75, 85, 95, 106.7, 120, 140]


def frontier(pts):
    return sorted(p for p in pts if not any(q[0] >= p[0] and q[1] >= p[1] and q != p for q in pts))


def interp(f, x):
    for (x0, y0), (x1, y1) in zip(f, f[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return None


print("| 点 | intvty p90 | tok/s/GPU | C/qps 周期 s | TTFT 均值 | decode 段 s | 客户端间隔 s | ITL 均值 ms | decode 在跑（Little） |")
print("|---|---|---|---|---|---|---|---|---|")
pts = []
for lab, d, gpus in ours:
    j = json.loads(next((report / "results" / d).glob("agentx_conc*.json")).read_text())
    r = j["request_metrics"]
    lat, qps = r["latency"], r["qps"]["mean"]
    x, y = lat["intvty"]["p90"], r["throughput"]["per_gpu"]["total_tput_tps"]
    pts.append((x, y))
    cyc, ttft, e2e = j["conc"] / qps, lat["ttft"]["mean"], lat["e2el"]["mean"]
    print(f"| {lab} | {x:.1f} | {y:,.0f} | {cyc:.1f} | {ttft:.2f} | {e2e - ttft:.1f} | {cyc - e2e:.1f} "
          f"| {lat['itl']['mean'] * 1e3:.1f} | {qps * (e2e - ttft):.0f} |")
ours_f = frontier(pts)
print("\nfrontier:", ", ".join(f"{y / 1000:.1f}k @ {x:.1f}" for x, y in ours_f))

rows = [r for r in json.loads(api.read_text()) if r["benchmark_type"] == "agentic_traces"]


def ix(hw, fw):
    return frontier([(r["metrics"]["p90_intvty"], r["metrics"]["tput_per_gpu"])
                     for r in rows if r["hardware"] == hw and r["framework"] == fw])


atom_pd = frontier([
    (j["request_metrics"]["latency"]["intvty"]["p90"], j["request_metrics"]["throughput"]["per_gpu"]["total_tput_tps"])
    for j in (json.loads(p.read_text()) for p in
              (report.parent / "gb300-atom-pd-analysis-20261021/evidence/atom-pd-pr4045/run39028510364").glob("*.json"))
])
refs = [("ATOM PD #4045", atom_pd), ("ATOM #209", atom209), ("B300 单机", ix("b300", "dynamo-sglang")),
        ("MI355X SGLang 单机", ix("mi355x", "sglang")), ("MI355X ATOM 单机", ix("mi355x", "atom"))]
print("\n| p90 | 我们 | " + " | ".join(n for n, _ in refs) + " |")
print("|---" * (len(refs) + 2) + "|")
for x in P90:
    o = interp(ours_f, x)
    cells = []
    for _, f in refs:
        v = interp(f, x)
        cells.append("—" if v is None else f"{v / 1000:.1f}k" + (f"（{(o / v - 1) * 100:+.0f}%）" if o else ""))
    print(f"| {x} | {'—' if o is None else f'{o / 1000:.1f}k'} | " + " | ".join(cells) + " |")
