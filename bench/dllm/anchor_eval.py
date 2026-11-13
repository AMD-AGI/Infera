"""Simulate vs anchored mode on measured refinement steps.

usage: python anchor_eval.py <out.json> <wsteps.json> [more ...]

Simulate mode prices every step from the model alone. Anchored mode transports
the nearest of a few measured steps by the simulator's ratio. Both are scored
on the same held-out steps: every measured step except the four-point
anchor set's.
"""

import json
import math
import sys

from infera.projection.core.projection.inference_projection.dllm import (
    DLLMHardware,
    DLLMStepModel,
    Parallel,
    SeqShape,
    load_model,
)
from infera.projection.core.projection.inference_projection.dllm.anchor import (
    AnchoredStepModel,
    AnchorPoint,
    DLLMAnchor,
    target_regime,
)
from infera.projection.core.projection.inference_projection.dllm.runtime import get_runtime

hw = DLLMHardware.from_arch("mi325x")
print(f"mi325x priced at {hw.gpu_clock_mhz} MHz, runtime hf_eager")
# The reference code's MoE dispatch, per model (model code, not a fit).
DISPATCH = {"llada2_mini": "loop_touched"}


def runtime_for(preset):
    return get_runtime({"preset": "hf_eager", "moe_dispatch": DISPATCH.get(preset, "loop_all")})


# Nested anchor sets: a mid full step; + a mid cached step; + the small corners.
SETS = {
    "1pt": [(8, 1024, 1024)],
    "2pt": [(8, 1024, 1024), (8, 32, 2048)],
    "4pt": [(8, 1024, 1024), (8, 32, 2048), (1, 512, 512), (1, 32, 1024)],
}


def stats(ratios):
    if not ratios:
        return {}
    ape = [abs(r - 1) for r in ratios]
    return {
        "n": len(ratios),
        "geomean": math.exp(sum(math.log(r) for r in ratios) / len(ratios)),
        "min": min(ratios),
        "max": max(ratios),
        "mape": sum(ape) / len(ape),
        "worst": max(ape),
        "within10": sum(a <= 0.10 for a in ape) / len(ape),
    }


def fmt(s):
    if not s:
        return "-"
    return (
        f"MAPE {s['mape']:6.1%}  worst {s['worst']:6.1%}  geo {s['geomean']:.2f} "
        f"[{s['min']:.2f}-{s['max']:.2f}]  <=10%: {s['within10']:.0%}  (n={s['n']})"
    )


summary = {}
for path in sys.argv[2:]:
    d = json.load(open(path))
    spec = load_model(d["preset"])
    rt = runtime_for(d["preset"])
    sim = DLLMStepModel(spec, Parallel(), hw, runtime=rt)
    regime = target_regime(spec)
    rows = d["rows"]
    pts = [AnchorPoint(r["batch"], r["q"], r["kv"], r["logits"], r["step_ms"]) for r in rows]
    for r in rows:
        st = sim.step([SeqShape(r["q"], r["kv"], r["logits"])] * r["batch"])
        r["sim_ms"] = st.total_ms
        r["sim_by_kind"] = st.as_dict()["by_kind_ms"]
    key = {(p.batch, p.q, p.kv): p for p in pts}
    sim_ms = {(r["batch"], r["q"], r["kv"]): r["sim_ms"] for r in rows}
    full = [p for p in pts if p.q == p.kv]
    has_cache = any(p.q < p.kv for p in pts)
    sets = {}
    for name, keys in SETS.items():
        chosen = [key[k] for k in keys if k in key]
        if not has_cache:
            chosen = [p for p in chosen if p.q == p.kv] or [full[len(full) // 2]]
        sets[name] = chosen
    anchors4 = {(p.batch, p.q, p.kv) for p in sets["4pt"]}
    held = [p for p in pts if (p.batch, p.q, p.kv) not in anchors4]

    def anchored(points, target, spec=spec, regime=regime, rt=rt):
        a = DLLMAnchor(regime=regime, points=points, gpu_arch="mi325x")
        m = AnchoredStepModel(a, spec, Parallel(), hw, runtime=rt)
        return m.step(target.seqs()).total_ms

    res = {"simulate": {}, "anchored": {}}
    for subset, pick in (
        ("all", lambda p: True),
        ("full", lambda p: p.q == p.kv),
        ("cached", lambda p: p.q < p.kv),
    ):
        hs = [p for p in held if pick(p)]
        if not hs:
            continue
        res["simulate"][subset] = stats([sim_ms[(p.batch, p.q, p.kv)] / p.step_ms for p in hs])
        for name, chosen in sets.items():
            res["anchored"].setdefault(name, {})[subset] = stats(
                [anchored(chosen, p) / p.step_ms for p in hs]
            )
    loo = []
    for p in pts:
        rest = [x for x in pts if x is not p]
        loo.append(anchored(rest, p) / p.step_ms)
    res["anchored"]["leave_one_out"] = {"all": stats(loo)}

    print(f"\n== {d['preset']} ({d['repo']}), {len(pts)} steps, {len(held)} held out")
    for subset in ("all", "full", "cached"):
        if subset in res["simulate"]:
            print(f"  simulate        {subset:6}: {fmt(res['simulate'][subset])}")
    for name in SETS:
        for subset in ("all", "full", "cached"):
            s = res["anchored"][name].get(subset)
            if s:
                print(f"  anchored {name:5}  {subset:6}: {fmt(s)}")
    print(f"  leave-one-out   all   : {fmt(res['anchored']['leave_one_out']['all'])}")
    worst = sorted(rows, key=lambda r: abs(math.log(r["sim_ms"] / r["step_ms"])), reverse=True)[:5]
    for r in worst:
        print(
            f"    worst sim: b{r['batch']} q{r['q']} kv{r['kv']}: meas {r['step_ms']:.2f} sim {r['sim_ms']:.2f} "
            f"({r['sim_ms'] / r['step_ms']:.2f}x) {r['sim_by_kind']}"
        )
    summary[d["preset"]] = {
        "repo": d["repo"],
        "anchor_sets": {k: [(p.batch, p.q, p.kv) for p in v] for k, v in sets.items()},
        "result": res,
        "rows": rows,
    }
json.dump(summary, open(sys.argv[1], "w"), indent=1)
print("\nwrote", sys.argv[1])
