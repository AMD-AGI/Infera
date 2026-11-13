"""Measured vs simulated dLLM-over-AR speedup (Dream-7B vs Qwen2.5-7B, same backbone)."""

import json
import math
import os

R = os.path.expanduser("~/dllm/results/")
ar = {r["batch"]: r for r in json.load(open(R + "qwen_ar.json"))["rows"]}
out = []
for name in ("llada", "dream"):
    rows = json.load(open(R + f"{name}_replay.json"))["rows"]
    by = {}
    for r in rows:
        by.setdefault((r["cache"], r["batch"]), []).append(r["ratio"])
    print(f"== {name} replay: sim/meas")
    for (c, b), v in by.items():
        g = math.exp(sum(map(math.log, v)) / len(v))
        print(f"  {c:6s} b{b}: geomean {g:.2f}  range {min(v):.2f}-{max(v):.2f}")
print("== Dream over Qwen AR (P=256, G=256), speedup = AR latency / dLLM latency")
print(f"  {'mode':6s} {'tok/step':>8s} {'batch':>5s} {'measured':>9s} {'simulated':>9s}")
for r in json.load(open(R + "dream_replay.json"))["rows"]:
    a = ar[r["batch"]]
    meas = a["meas_ms"] / r["meas_ms"]
    sim = a["sim_ms"] / r["sim_ms"]
    tps = 256 / r["steps"]
    out.append({**r, "tok_per_step": tps, "speedup_meas": meas, "speedup_sim": sim})
    if r["block"] == 32:
        print(f"  {r['cache']:6s} {tps:8.0f} {r['batch']:5d} {meas:9.2f} {sim:9.2f}")
right = sum((o["speedup_meas"] > 1) == (o["speedup_sim"] > 1) for o in out)
print(f"  winner (dLLM vs AR) called right in {right}/{len(out)} configurations")
json.dump(out, open(R + "crossover_measured.json", "w"), indent=1)
