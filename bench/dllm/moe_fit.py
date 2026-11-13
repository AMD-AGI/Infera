"""Fit expert_iter_us: what a reference MoE loop costs per expert visit beyond
what the simulator already charges for the layer.

usage: python moe_fit.py moe_loop.json
"""

import json
import statistics
import sys

from infera.projection.core.projection.inference_projection.dllm import (
    DLLMHardware,
    DLLMModelSpec,
    Parallel,
    SeqShape,
)
from infera.projection.core.projection.inference_projection.dllm.model import build_stage_graph
from infera.projection.core.projection.inference_projection.dllm.ops import MOE_MLP, group_ms
from infera.projection.core.projection.inference_projection.dllm.runtime import get_runtime

hw = DLLMHardware.from_arch("mi325x")
rows = json.load(open(sys.argv[1]))
fit = {}
for r in rows:
    spec = DLLMModelSpec(
        name="probe",
        num_layers=1,
        hidden_size=2048,
        num_attention_heads=16,
        num_experts=r["experts"],
        moe_router_topk=8,
        moe_ffn_hidden_size=r["ffn"],
    )
    rt = get_runtime({"preset": "hf_eager", "moe_dispatch": r["style"], "expert_iter_us": 0.0})
    g = build_stage_graph(
        spec,
        [SeqShape(r["tokens"], r["tokens"], 0)],
        Parallel(),
        first=False,
        last=False,
        runtime=rt,
    )
    sim = group_ms(g.ops, hw)[MOE_MLP]
    visits = r["experts"] if r["style"] == "loop_all" else r["touched"]
    r["sim_moe_ms"] = sim
    r["fit_iter_us"] = (r["block_ms"] - sim) * 1e3 / visits
    fit.setdefault(r["style"], []).append(r)
    print(
        f"{r['style']:12} E{r['experts']:>3} ffn{r['ffn']:>5} T{r['tokens']:>6} touched {r['touched']:>3}: "
        f"block {r['block_ms']:8.2f} ms  sim(no loop) {sim:7.2f}  -> {r['fit_iter_us']:6.1f} us/visit"
    )
for style, rs in fit.items():
    # Steps a dLLM takes reach most experts; fit where the loop, not the math, dominates.
    small = [r["fit_iter_us"] for r in rs if r["tokens"] <= 1024]
    print(
        f"{style}: median {statistics.median(small):.1f} us/visit over T<=1024 (n={len(small)}), "
        f"all {statistics.median(r['fit_iter_us'] for r in rs):.1f}"
    )
