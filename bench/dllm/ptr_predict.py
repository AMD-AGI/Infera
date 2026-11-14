"""Predict, then run: score every candidate before any is measured.

usage: python ptr_predict.py <anchor_store> <out.json>

Study A, one GPU: per model, the cache mode and batch with the most tokens/s
under a per-request latency target. Study B, one node: Dream's tensor-parallel
degree, replica count and batch with the most tokens/s per node under a
target. Every candidate is priced in simulate and in anchored mode and the
memory model drops those that do not fit; the picks are written with a
timestamp so the runs cannot change them.
"""

import datetime
import json
import sys

from infera.projection.core.projection.inference_projection.dllm import (
    DLLMHardware,
    DLLMStepModel,
    Parallel,
    load_model,
)
from infera.projection.core.projection.inference_projection.dllm.anchor import (
    AnchoredStepModel,
    find_anchor,
    target_regime,
)
from infera.projection.core.projection.inference_projection.dllm.memory import memory_report
from infera.projection.core.projection.inference_projection.dllm.runtime import get_runtime
from infera.projection.core.projection.inference_projection.dllm.simulator import simulate_static
from infera.projection.core.projection.inference_projection.dllm.workload import DLLMWorkload

store, out = sys.argv[1:3]
hw = DLLMHardware.from_arch("mi325x")
P = G = 256
STEPS = 64  # four tokens a step: the schedule is the user's, not the tuner's
FULL_BATCHES = (1, 2, 4, 8, 16, 32, 64)
# Dual-cache and block steps recompute a few positions; their batches go further.
SMALL_BATCHES = FULL_BATCHES + (128, 256)
SLOS_S = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, None)
DISPATCH = {"llada2_mini": "loop_touched"}
MODES = {
    "dream_7B": [("none", 0), ("prefix", 32), ("dual", 32)],
    "llada_8B": [("none", 0)],
    "llada_moe_7B_A1B": [("none", 0)],
    "llada2_mini": [("none", 0)],
    "sdar_1.7B": [("block", 4)],
    "sdar_8B": [("block", 4)],
    "sdar_30B_A3B": [("block", 4)],
}


def runtime_for(preset):
    return get_runtime({"preset": "hf_eager", "moe_dispatch": DISPATCH.get(preset, "loop_all")})


def pick(cands, key, slo):
    ok = [c for c in cands if c["fits"] and (slo is None or c[key] / 1e3 <= slo)]
    if not ok:
        return None
    best = max(ok, key=lambda c: c["batch"] * G / c[key])
    return {k: best[k] for k in ("cache", "block", "batch")}


res = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "P": P, "G": G, "steps": STEPS}

# Study A
study_a = {}
for preset, modes in MODES.items():
    spec = load_model(preset)
    rt = runtime_for(preset)
    sim = DLLMStepModel(spec, Parallel(), hw, runtime=rt)
    anc = AnchoredStepModel(
        find_anchor(store, target_regime(spec, runtime=rt.name), gpu_arch="mi325x"),
        spec,
        Parallel(),
        hw,
        runtime=rt,
    )
    cands = []
    for cache, block in modes:
        wl = DLLMWorkload(prompt_len=P, gen_len=G, block_len=block, steps=STEPS, cache=cache)
        for b in SMALL_BATCHES if cache in ("dual", "block") else FULL_BATCHES:
            mem = memory_report(spec, Parallel(), hw, wl, b)
            cands.append(
                dict(
                    cache=cache,
                    block=block,
                    batch=b,
                    sim_ms=simulate_static(sim, wl, b).latency_ms,
                    anc_ms=simulate_static(anc, wl, b).latency_ms,
                    mem_gb=mem.weights_gb + mem.kv_cache_gb + mem.activations_gb,
                    fits=mem.fits,
                )
            )
    picks = {
        str(slo): {"simulate": pick(cands, "sim_ms", slo), "anchored": pick(cands, "anc_ms", slo)}
        for slo in SLOS_S
    }
    study_a[preset] = {"candidates": cands, "picks": picks}
    print(preset, json.dumps(picks))
res["study_a"] = study_a

# Study B: Dream, no cache, one node of 8 GPUs
spec = load_model("dream_7B")
rt = runtime_for("dream_7B")
anchor = find_anchor(store, target_regime(spec, runtime=rt.name), gpu_arch="mi325x")
wl = DLLMWorkload(prompt_len=P, gen_len=G, block_len=0, steps=STEPS, cache="none")
cands = []
for tp in (1, 2, 4, 8):
    par = Parallel(tp=tp)
    sim = DLLMStepModel(spec, par, hw, runtime=rt)
    anc = AnchoredStepModel(anchor, spec, par, hw, runtime=rt)
    for b in (1, 2, 4, 8, 16, 32):
        mem = memory_report(spec, par, hw, wl, b)
        cands.append(
            dict(
                cache="none",
                block=0,
                tp=tp,
                replicas=8 // tp,
                batch=b,
                sim_ms=simulate_static(sim, wl, b).latency_ms,
                anc_ms=simulate_static(anc, wl, b).latency_ms,
                fits=mem.fits,
            )
        )


def pick_node(key, slo):
    ok = [c for c in cands if c["fits"] and (slo is None or c[key] / 1e3 <= slo)]
    if not ok:
        return None
    best = max(ok, key=lambda c: c["replicas"] * c["batch"] * G / c[key])
    return {k: best[k] for k in ("tp", "replicas", "batch")}


NODE_SLOS = (0.5, 1.0, 2.0, 5.0, None)
res["study_b"] = {
    "model": "dream_7B",
    "candidates": cands,
    "picks": {
        str(s): {"simulate": pick_node("sim_ms", s), "anchored": pick_node("anc_ms", s)}
        for s in NODE_SLOS
    },
}
print("study_b", json.dumps(res["study_b"]["picks"]))
json.dump(res, open(out, "w"), indent=1)
print("wrote", out, res["created_utc"])
