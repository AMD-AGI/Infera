#!/usr/bin/env python3
"""2x2 differential for the forge-vs-independent speedup gap: {workset indices, random indices} x
{CUDA-graph timing as in driver.py, eager CUDA-event timing as in independent_check.registry.py}.
Same KV pools (driver's make_kv, 8 layers) and q for all four cells; pristine vs candidate per cell.
Usage (in the container, from this dir): python3 diff_check.registry.py <candidate_kernel.py>
"""
import importlib.util
import os
import statistics
import sys

import torch

import driver as D
from graph_harness import cuda_graph_bench
_spec = importlib.util.spec_from_file_location("ic", os.path.join(D.HERE, "independent_check.registry.py"))
_ic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ic)
load, time_ms = _ic.load, _ic.time_ms


def graph_ms(fn, q_nope, q_rope, pools, idx):
    cap = {}

    def step():
        for kv in pools:
            cap["o"] = fn(q_nope, q_rope, kv, idx, D.SM_SCALE, d_v=D.D_V, topk_length=None, max_topk_length=D.TOPK)

    res = cuda_graph_bench(step, warmup=20, iters=100)
    return statistics.median(t / len(pools) for t in res["times_ms"]), res["mode"]


def main():
    cand = load(sys.argv[1], "cand")
    prist = load(os.path.join(D.HERE, "sparse_mla_kernel.pristine.py"), "prist")
    pools = D.make_kv(D.N_LAYERS, "cuda")
    g = torch.Generator(device="cuda").manual_seed(777)
    for bs in (8, 10, 12, 14):
        t = 6 * bs
        q_nope, q_rope, ws_idx = D.make_case(bs, 0, "cuda")
        rows = []
        for _ in range(t):
            start = int(torch.randint(0, D.NUM_TOKENS_POOL - 600_000, (1,), generator=g, device="cuda").item())
            length = int(torch.randint(4096, 600_000, (1,), generator=g, device="cuda").item())
            rows.append(start + torch.randperm(length, generator=g, device="cuda")[:D.TOPK])
        rnd_idx = torch.stack(rows).to(torch.int32).unsqueeze(1).contiguous()
        for name, idx in (("workset", ws_idx), ("random", rnd_idx)):
            cells = []
            for f in (prist, cand):
                call = lambda qn, qr, kv, ix, f=f: f(qn, qr, kv, ix, D.SM_SCALE, d_v=D.D_V, topk_length=None,
                                                    max_topk_length=D.TOPK)
                gm, mode = graph_ms(f, q_nope, q_rope, pools, idx)
                em = time_ms(call, [(q_nope, q_rope, kv, idx) for kv in pools])
                cells.append((gm, em))
            (pg, pe), (cg, ce) = cells
            print(f"T{t}_bs{bs} {name:8s} graph: prist {pg:.4f} cand {cg:.4f} x{pg / cg:.3f} ({mode}) | "
                  f"eager: prist {pe:.4f} cand {ce:.4f} x{pe / ce:.3f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
