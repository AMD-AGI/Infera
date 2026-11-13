"""Measured RCCL (comm_bench.py) vs InferaSim's collective model, as DLLMCommModel calls it."""

import json
import math
import sys

from infera.projection.core.projection.inference_projection.dllm import (
    DLLMHardware,
    Parallel,
    load_model,
)
from infera.projection.core.projection.inference_projection.dllm.comm import (
    CommConfig,
    DLLMCommModel,
)

m = json.load(open(sys.argv[1]))
hw = DLLMHardware.from_arch("mi325x", backend="roofline")
dense = load_model("llada_8B")
assert dense.hidden_size == 4096
moe = load_model("llada_moe_7B") if False else None
res = {"allreduce": [], "quant_allreduce": [], "sendrecv": [], "alltoall": []}


def line(kind, key, meas, sim):
    r = sim / meas
    res[kind].append({**key, "meas_ms": meas, "sim_ms": sim, "ratio": r})
    print(f"  {kind:16s} {key}: meas {meas:8.3f}  sim {sim:8.3f}  ({r:.2f}x)")


for row in m["allreduce"]:
    cm = DLLMCommModel(dense, Parallel(tp=row["tp"]), hw)
    sim = cm._icm.tp_allreduce_ms(1, row["tokens"], 2.0) / 2
    line("allreduce", {"tp": row["tp"], "tokens": row["tokens"]}, row["ms"], sim)
for row in m["quant_allreduce"]:
    cm = DLLMCommModel(
        dense, Parallel(tp=8), hw, CommConfig(payload_dtype="fp8", quant_scope="all")
    )
    c = cm.layer("dense", row["tokens"], "fp8")
    wire, pack = c.wire_ms / 2, c.quant_ms / 2
    line("quant_allreduce", {"part": "wire", "tokens": row["tokens"]}, row["wire_ms"], wire)
    line("quant_allreduce", {"part": "kernels", "tokens": row["tokens"]}, row["kernel_ms"], pack)
for row in m["sendrecv"]:
    cm = DLLMCommModel(dense, Parallel(pp=2), hw)
    # pp_p2p_ms = one boundary + the sampled-token return; the return is tiny.
    from infera.projection.core.projection.module_profilers import collective_model as cmod

    sim = cmod.sendrecv(cm._icm._args, row["bytes"]) / 1e3
    line(
        "sendrecv",
        {"tokens": row["tokens"], "engine_ms": cm._icm.pp_p2p_ms(1, row["tokens"], 2.0)},
        row["ms"],
        sim,
    )
for row in m["alltoall"]:
    icm = DLLMCommModel(dense, Parallel(tp=8, ep=8), hw)._icm
    icm.topk = 8
    sim = icm.ep_a2a_ms(1, row["tokens"], 2.0) / 2
    line("alltoall", {"ep": 8, "tokens": row["tokens"]}, row["ms"], sim)
for k, rows in res.items():
    if rows:
        g = math.exp(sum(math.log(r["ratio"]) for r in rows) / len(rows))
        print(
            f"{k}: geomean sim/meas {g:.2f}, range {min(r['ratio'] for r in rows):.2f}-{max(r['ratio'] for r in rows):.2f}"
        )
json.dump(res, open(sys.argv[2], "w"), indent=1)
