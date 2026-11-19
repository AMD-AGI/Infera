"""TP steps (tp_step.py) re-priced with the switched default vs the mesh profile."""

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

rows = json.load(open(sys.argv[1]))
spec = load_model("dream_7B")
switched = {"node_bw": 1024.0, "node_topology": "switch"}
res = {"mesh": [], "switch": []}
for r in rows:
    for name, over in (("mesh", {}), ("switch", switched)):
        hw = DLLMHardware.from_arch("mi325x", gpu_clock_mhz=1510, collective_overrides=over)
        m = DLLMStepModel(spec, Parallel(tp=r["tp"]), hw, runtime="hf_eager")
        res[name].append(
            m.step([SeqShape(r["seq"], r["seq"], r["seq"])] * r["batch"]).total_ms / r["meas_ms"]
        )
for name, v in res.items():
    tp = [x for x, r in zip(v, rows) if r["tp"] > 1]
    g = math.exp(sum(map(math.log, tp)) / len(tp))
    print(f"{name:6s}: TP>1 sim/meas geomean {g:.2f}, range {min(tp):.2f}-{max(tp):.2f}")
for r, a, b in zip(rows, res["mesh"], res["switch"]):
    if r["tp"] > 1:
        print(f"  TP{r['tp']} b{r['batch']} s{r['seq']}: mesh {a:.2f}  switch {b:.2f}")
json.dump(
    {"rows": rows, "mesh": res["mesh"], "switch": res["switch"]},
    open(os.path.expanduser("~/dllm/results/tp_compare.json"), "w"),
    indent=1,
)
