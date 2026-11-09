#!/usr/bin/env python3
"""AKLR's committed GLM-5.2 bf16 GEMM kernels (source embedded in their schema JSON, loaded without the aklr package)
against the production path (aiter tgemm.mm with the production tuned CSVs mounted, as sglang's unquantized linear
on ROCm calls it), at our per-rank decode rows. CUDA-graph replay over NSETS weight sets.

Usage (production image): aklr_gemm_bench.py --json <schema.json> --n 6144 --k 16384 --ms 64,72,88,96,112,128
"""
import argparse
import importlib.util
import json
import os
import statistics as st
import sys
import tempfile

import torch


def load_kernel(path):
    d = json.load(open(path))
    tmp = tempfile.mkdtemp(prefix="aklr_kernel_")
    for s in d["kernel"]["sources"]:
        with open(os.path.join(tmp, s["path"]), "w") as f:
            f.write(s["content"])
    sys.path.insert(0, tmp)
    spec = importlib.util.spec_from_file_location("aklr_kernel", os.path.join(tmp, "kernel.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, [w["axes"]["m"] for w in d["workloads"] if w.get("speedup")]


def graph_us(fn, n_calls, reps=30):
    fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    t = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        t.append(a.elapsed_time(b) * 1e3 / n_calls)
    return st.median(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--ms", default="64,72,80,88,96,104,112,128")
    ap.add_argument("--nsets", type=int, default=4)
    a = ap.parse_args()
    from aiter.tuned_gemm import tgemm

    kern, served = load_kernel(a.json)
    print(f"AKLR kernel {os.path.basename(a.json)}: CI-served m = {served}", flush=True)
    torch.manual_seed(0)
    ws = [torch.randn(a.n, a.k, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(a.nsets)]
    gb = a.n * a.k * 2 / 1e9
    for M in [int(x) for x in a.ms.split(",")]:
        x = torch.randn(M, a.k, device="cuda", dtype=torch.bfloat16)
        outs = [torch.empty(M, a.n, device="cuda", dtype=torch.bfloat16) for _ in ws]
        ref = (x.float() @ ws[0].float().t())

        def prod():
            for w in ws:
                tgemm.mm(x, w, None, torch.bfloat16)

        def aklr():
            for w, o in zip(ws, outs):
                kern.run(a=x, b=w, trans_b=True, bias=None, out_dtype=torch.bfloat16, out=o)

        def err(o):
            return ((o.float() - ref).abs().max() / ref.abs().max()).item()

        tp = graph_us(prod, len(ws))
        ep = err(tgemm.mm(x, ws[0], None, torch.bfloat16))
        try:
            ta = graph_us(aklr, len(ws))
            kern.run(a=x, b=ws[0], trans_b=True, bias=None, out_dtype=torch.bfloat16, out=outs[0])
            ea = err(outs[0])
            res = f"AKLR {ta:6.1f} us ({gb / ta * 1e3:4.2f} TB/s, rel err {ea:.1e})  ratio prod/AKLR {tp / ta:5.3f}"
        except Exception as e:  # noqa: BLE001
            res = f"AKLR failed: {type(e).__name__}: {str(e)[:120]}"
        print(f"M {M:4d}: production {tp:6.1f} us ({gb / tp * 1e3:4.2f} TB/s, rel err {ep:.1e})  {res}", flush=True)


if __name__ == "__main__":
    main()
