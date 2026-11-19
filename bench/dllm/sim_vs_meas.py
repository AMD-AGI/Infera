# ruff: noqa: B023, F821 - each closure is timed in the iteration that builds it
"""Simulated vs measured refinement steps, and anchor transport error.

usage: python sim_vs_meas.py results/llada_steps.json [more.json ...]
"""

import json
import math
import os
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

ARCH = os.getenv("ARCH", "mi325x")
RUNTIME = os.getenv("RUNTIME", "engine")
CLOCK = int(os.getenv("CLOCK", "0")) or None
summary = {}
for path in sys.argv[1:]:
    d = json.load(open(path))
    spec = load_model(d["preset"])
    rows = d["rows"]
    res = {}
    for backend in ("origami", "roofline"):
        hw = DLLMHardware.from_arch(ARCH, backend=backend, gpu_clock_mhz=CLOCK)
        m = DLLMStepModel(spec, Parallel(), hw, runtime=RUNTIME)
        errs = []
        for r in rows:
            st = m.step([SeqShape(r["seq"], r["seq"], r["seq"])] * r["batch"])
            r[f"sim_{backend}_ms"] = st.total_ms
            if backend == "origami":
                r["sim_by_kind"] = st.as_dict()["by_kind_ms"]
            errs.append(st.total_ms / r["step_ms"])
        res[backend] = {
            "ratio_min": min(errs),
            "ratio_max": max(errs),
            "ratio_geomean": math.exp(sum(math.log(e) for e in errs) / len(errs)),
        }
    hw = DLLMHardware.from_arch(ARCH, gpu_clock_mhz=CLOCK)
    regime = target_regime(spec)
    pts = [AnchorPoint(r["batch"], r["seq"], r["seq"], r["seq"], r["step_ms"]) for r in rows]

    def transport(anchor_pts, name):
        a = DLLMAnchor(regime=regime, points=anchor_pts, gpu_arch=ARCH)
        am = AnchoredStepModel(a, spec, Parallel(), hw, runtime=RUNTIME)
        e = []
        held = [p for p in pts if p not in anchor_pts]
        for p in held:
            e.append(abs(am.step(p.seqs()).total_ms / p.step_ms - 1))
        out = {
            "anchors": [(p.batch, p.q) for p in anchor_pts],
            "n": len(held),
            "mape": sum(e) / len(e) if e else 0.0,
            "max": max(e) if e else 0.0,
        }
        print(
            f"  transport [{name}] from {out['anchors']}: {out['n']} held-out, MAPE {out['mape']:.1%}, max {out['max']:.1%}"
        )
        return out

    print(f"== {d['preset']} on {d['device']} ({ARCH} model)")
    for r in rows:
        print(
            f"  b{r['batch']:>3} s{r['seq']:>5}: meas {r['step_ms']:8.2f}  origami {r['sim_origami_ms']:8.2f} "
            f"({r['sim_origami_ms'] / r['step_ms']:.2f}x)  roofline {r['sim_roofline_ms']:8.2f} ({r['sim_roofline_ms'] / r['step_ms']:.2f}x)"
        )
    for k, v in res.items():
        print(
            f"  {k}: sim/meas geomean {v['ratio_geomean']:.2f}, range {v['ratio_min']:.2f}-{v['ratio_max']:.2f}"
        )
    byp = {(p.batch, p.q): p for p in pts}
    tr = {
        "one_point_b1_s1024": transport([byp[(1, 1024)]], "1 point"),
        "one_point_b8_s1024": transport([byp[(8, 1024)]], "1 point"),
        "two_points": transport([byp[(1, 512)], byp[(8, 1024)]], "2 points"),
        "corners": transport([byp[(1, 256)], byp[(16, 2048)]], "corners"),
    }
    # leave-one-out with all other points as the anchor
    e = []
    for p in pts:
        a = DLLMAnchor(regime=regime, points=[q for q in pts if q is not p], gpu_arch=ARCH)
        e.append(
            abs(
                AnchoredStepModel(a, spec, Parallel(), hw, runtime=RUNTIME).step(p.seqs()).total_ms
                / p.step_ms
                - 1
            )
        )
    tr["leave_one_out"] = {"mape": sum(e) / len(e), "max": max(e)}
    print(
        f"  leave-one-out: MAPE {tr['leave_one_out']['mape']:.1%}, max {tr['leave_one_out']['max']:.1%}"
    )
    prof = d["profile"]
    pr = next(r for r in rows if r["batch"] == prof["batch"] and r["seq"] == prof["seq"])
    print(
        f"  profile b{prof['batch']} s{prof['seq']}: measured {prof['by_cat_ms']}  sim {pr['sim_by_kind']}"
    )
    summary[d["preset"]] = {"sim": res, "transport": tr, "rows": rows, "profile": prof}

out = os.getenv("OUT", os.path.expanduser("~/dllm/results/sim_vs_meas.json"))
json.dump(summary, open(out, "w"), indent=1)
print("wrote", out)
