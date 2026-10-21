"""Per-point comparison of GB300 Dynamo+SGLang PD (InferenceX API rows), ATOM MI355X PD (InferenceX PR #4045
run 39028510364) and our 2P1D / 1P1D sweep, as two Markdown tables:
  1. throughput, interactivity, ITL, TTFT, requests/s per prefill GPU and per decode GPU;
  2. decode load by Little's law: running = qps x mean output tokens x mean ITL, per decode GPU, and the
     step time ITL x acceptance length (golden AL: K=3 2.99, K=4 3.33, K=5 3.61).
  python3 compare_pd.py"""

import glob
import json
from pathlib import Path

here = Path(__file__).resolve().parents[1]
sweep = here.parent / "2p1d-v0522-20261020"
api = sweep / "evidence/inferencex-api-glm52-20261021.json"
atom = here / "evidence/atom-pd-pr4045/run39028510364"
ours = [("c176", 8), ("c128", 8), ("c96", 8), ("c64", 8), ("1p1d-c88", 4), ("1p1d-c64", 4), ("1p1d-c32", 4)]

pts = []  # name, prefill GPUs, decode GPUs, conc, AL, request metrics in AgentX aggregate form


def agentx(name, prefill, decode, al, j):
    rm = j["request_metrics"]
    pts.append(dict(name=name, P=prefill, D=decode, conc=j["conc"], al=al, tpg=rm["throughput"]["per_gpu"]["total_tput_tps"],
                    ip90=rm["latency"]["intvty"]["p90"], itl50=rm["latency"]["itl"]["p50"], itl=rm["latency"]["itl"]["mean"],
                    tt50=rm["latency"]["ttft"]["p50"], tt90=rm["latency"]["ttft"]["p90"], qps=rm["qps"]["mean"],
                    out=rm["tokens"]["output_actual"]["mean"], inp=rm["tokens"]["input"]["mean"]))


for r in json.loads(api.read_text()):
    if r["benchmark_type"] != "agentic_traces" or r["hardware"] != "gb300" or r["framework"] != "dynamo-sglang" or not r["disagg"]:
        continue
    m = r["metrics"]
    dec = f"TP{r['decode_tp']}" + (f"/EP{r['decode_ep']}/DPA" if r["decode_dp_attention"] else "")
    pts.append(dict(name=f"GB300 {r['prefill_num_workers']}P(TP{r['prefill_tp']}xPP{m['prefill_pp']})+{r['decode_num_workers']}D({dec})",
                    P=r["num_prefill_gpu"], D=r["num_decode_gpu"], conc=r["conc"], al=3.61, tpg=m["tput_per_gpu"], ip90=m["p90_intvty"],
                    itl50=m["median_itl"], itl=m["mean_itl"], tt50=m["median_ttft"], tt90=m["p90_ttft"], qps=m["mean_qps"],
                    out=m["mean_output_tokens_actual"], inp=m["mean_input_tokens"]))
for f in sorted(atom.glob("*.json")):
    j = json.loads(f.read_text())
    nd = j["decode_num_workers"]
    dcp = "+DCP4" if j["decode_dcp_size"] > 1 else ""
    agentx(f"ATOM 1P(PP4)+{nd}D(TP4{dcp})", j["num_prefill_gpu"], j["num_decode_gpu"], {1: 2.99, 2: 3.33, 3: 3.61}[nd], j)
for d, p in ours:
    j = json.loads(Path(glob.glob(str(sweep / "results" / d / "agentx_conc*.json"))[0]).read_text())
    agentx(f"ours {p // 4}P(TP4/DPA)+1D(TP8/DPA)", p, 8, 3.61, j)

print("| config | P+D GPUs | C | tok/s/GPU | intvty p90 | ITL p50 (ms) | TTFT p50 / p90 (s) | qps | qps / prefill GPU | qps / decode GPU "
      "| output tok/s / decode GPU | mean input |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for x in sorted(pts, key=lambda x: (x["name"][:4], -x["tpg"])):
    print(f"| {x['name']} | {x['P']}+{x['D']} | {x['conc']} | {x['tpg']:,.0f} | {x['ip90']:.1f} | {x['itl50'] * 1e3:.1f} "
          f"| {x['tt50']:.2f} / {x['tt90']:.2f} | {x['qps']:.2f} | {x['qps'] / x['P']:.2f} | {x['qps'] / x['D']:.2f} "
          f"| {x['qps'] * x['out'] / x['D']:.0f} | {x['inp'] / 1e3:.0f}k |")
print()
print("| config | C | decode GPUs | running = qps x output x ITL | running / decode GPU | step = ITL x AL (ms) |")
print("|---|---|---|---|---|---|")
for x in sorted(pts, key=lambda x: (x["name"][:4], -x["tpg"])):
    run = x["qps"] * x["out"] * x["itl"]
    print(f"| {x['name']} | {x['conc']} | {x['D']} | {run:.0f} | {run / x['D']:.1f} | {x['itl'] * 1e3 * x['al']:.0f} |")
