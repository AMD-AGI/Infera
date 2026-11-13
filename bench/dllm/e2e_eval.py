"""Whole generations: measured vs simulate mode vs anchored mode.

usage: python e2e_eval.py <steps_dir> <replay_dir> <anchor_store> <out.json>

Writes each model's four-point anchor (the sweep's 4pt set) into the store,
then prices every replayed generation twice with the same static-batch
simulator: on the step model alone, and on the anchored step model.
"""

import json
import math
import os
import sys

from infera.projection.core.projection.inference_projection.dllm import (
    DLLMHardware,
    DLLMStepModel,
    Parallel,
    load_model,
)
from infera.projection.core.projection.inference_projection.dllm.anchor import (
    AnchoredStepModel,
    AnchorPoint,
    DLLMAnchor,
    find_anchor,
    save_anchor,
    target_regime,
)
from infera.projection.core.projection.inference_projection.dllm.runtime import get_runtime
from infera.projection.core.projection.inference_projection.dllm.simulator import simulate_static
from infera.projection.core.projection.inference_projection.dllm.workload import DLLMWorkload

steps_dir, replay_dir, store, out = sys.argv[1:5]
hw = DLLMHardware.from_arch("mi325x")
FOUR = [(8, 1024, 1024), (8, 32, 2048), (1, 512, 512), (1, 32, 1024)]
# The reference code's MoE dispatch, per model (model code, not a fit).
DISPATCH = {"llada2_mini": "loop_touched"}
os.makedirs(store, exist_ok=True)


def runtime_for(preset):
    return get_runtime({"preset": "hf_eager", "moe_dispatch": DISPATCH.get(preset, "loop_all")})


for f in sorted(os.listdir(steps_dir)):
    if not f.endswith(".json") or f.startswith("anchor_eval"):
        continue
    d = json.load(open(os.path.join(steps_dir, f)))
    spec = load_model(d["preset"])
    pts = {(r["batch"], r["q"], r["kv"]): r for r in d["rows"]}
    chosen = [pts[k] for k in FOUR if k in pts]
    anchor = DLLMAnchor(
        regime=target_regime(
            spec, runtime="hf_eager", attention_backend="sdpa", graph_mode="eager"
        ),
        points=[
            AnchorPoint(r["batch"], r["q"], r["kv"], r["logits"], r["step_ms"]) for r in chosen
        ],
        gpu_arch="mi325x",
        backend="hf",
    )
    save_anchor(anchor, os.path.join(store, f"{d['preset']}.json"))
    print(f"anchor {d['preset']}: {[(p.batch, p.q, p.kv) for p in anchor.points]}")

summary = {}
for f in sorted(os.listdir(replay_dir)):
    if not f.endswith(".json"):
        continue
    d = json.load(open(os.path.join(replay_dir, f)))
    spec = load_model(d["preset"])
    rt = runtime_for(d["preset"])
    sim = DLLMStepModel(spec, Parallel(), hw, runtime=rt)
    anchor = find_anchor(store, target_regime(spec, runtime=rt.name), gpu_arch="mi325x")
    anc = AnchoredStepModel(anchor, spec, Parallel(), hw, runtime=rt)
    rows = []
    for r in d["rows"]:
        block = 0 if r["cache"] == "none" else r["block"]
        wl = DLLMWorkload(
            prompt_len=d["P"], gen_len=d["G"], block_len=block, steps=r["steps"], cache=r["cache"]
        )
        r["sim_ms"] = simulate_static(sim, wl, r["batch"]).latency_ms
        r["anchored_ms"] = simulate_static(anc, wl, r["batch"]).latency_ms
        rows.append(r)
        print(
            f"{d['preset']:10} {r['cache']:6} blk{r['block']:>4} steps{r['steps']:>4} b{r['batch']:>3}: "
            f"meas {r['meas_ms']:9.1f}  sim {r['sim_ms']:9.1f} ({r['sim_ms'] / r['meas_ms']:.2f}x)  "
            f"anchored {r['anchored_ms']:9.1f} ({r['anchored_ms'] / r['meas_ms']:.2f}x)"
        )

    def score(key, rows=rows):
        e = [abs(r[key] / r["meas_ms"] - 1) for r in rows]
        g = math.exp(sum(math.log(r[key] / r["meas_ms"]) for r in rows) / len(rows))
        return {"mape": sum(e) / len(e), "worst": max(e), "geomean": g, "n": len(e)}

    summary[d["preset"]] = {
        "simulate": score("sim_ms"),
        "anchored": score("anchored_ms"),
        "rows": rows,
    }
    s, a = summary[d["preset"]]["simulate"], summary[d["preset"]]["anchored"]
    print(
        f"== {d['preset']}: simulate MAPE {s['mape']:.1%} (worst {s['worst']:.1%}, geo {s['geomean']:.2f}) | "
        f"anchored MAPE {a['mape']:.1%} (worst {a['worst']:.1%}, geo {a['geomean']:.2f})"
    )
json.dump(summary, open(out, "w"), indent=1)
print("wrote", out)
