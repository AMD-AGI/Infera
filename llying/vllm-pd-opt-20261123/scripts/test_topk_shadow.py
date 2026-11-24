#!/usr/bin/env python3
"""GPU check (1 GPU, in the vLLM image) of the shadow comparison in vpd_triton_topk_decode.py (VPD_TOPK_SHADOW=1):
  PYTHONPATH=<dir with vpd_triton_topk_decode.py> VPD_TOPK_SHADOW_QUIET=2 VPD_TOPK_SHADOW_EVERY=1 test_topk_shadow.py
- Triton output vs vLLM's in-tree top-k (reference) on random rows: no value mismatch;
- one injected wrong index: one val_bad row, recorded in the sample ring with the right row / length;
- rows of ties (mostly 0): tie_rows counted, still no value mismatch;
- the comparison captured in a CUDA graph and replayed twice adds two calls' counts each time;
- the reader thread prints the counters.
"""
import time

import torch
import vllm._custom_ops  # noqa: F401  (registers torch.ops._C.top_k_per_row_decode; vllm._C alone does not on ROCm)

import vpd_triton_topk_decode as tk


def native(logits, lens, k):
    def f(out):
        torch.ops._C.top_k_per_row_decode(logits, 1, lens, out, logits.shape[0], logits.stride(0), logits.stride(1), k)
    return f


def stats(sc=0):
    S = next(iter(tk._SH.values()))
    torch.cuda.synchronize()
    return dict(zip(tk._SH_NAMES, S.st[sc].tolist())), S.ring[sc].tolist()


def main():
    k, W = 2048, 1 << 20
    g = torch.Generator(device="cuda").manual_seed(0)
    lens = torch.tensor([1000, 5000, 150_000, 150_001, 700_000, 300_000, 40_000, 2048], dtype=torch.int32, device="cuda")
    rows = lens.numel()
    logits = torch.full((rows, W), float("-inf"), device="cuda")
    for r in range(rows):
        logits[r, : int(lens[r])] = torch.randn(int(lens[r]), device="cuda", generator=g)
    out = torch.full((rows, k), -1, dtype=torch.int32, device="cuda")

    tk.triton_top_k_per_row_decode(logits, lens, out, k)
    tk.shadow_compare(logits, lens, out, k, False, native(logits, lens, k), None)
    v, _ = stats()
    assert v["calls"] == 1 and v["rows"] == rows and v["t_val_bad"] == 0 and v["t_cnt_bad"] == 0 and v["r_cnt_bad"] == 0, v
    assert v["t_dup_oor"] == 0 and v["nan_rows"] == 0, v

    # inject a wrong index into row 4 (the 700K row): replace one selected position by one below the k-th value
    bad = out.clone()
    sel = set(bad[4].tolist())
    low = int(torch.argmin(logits[4, :700_000]))
    assert low not in sel
    bad[4, 7] = low
    tk.shadow_compare(logits, lens, bad, k, False, native(logits, lens, k), None)
    v, ring = stats()
    assert v["calls"] == 2 and v["t_val_bad"] == 1 and v["t_set_diff"] == 1 and v["t_bad_calls"] == 1, v
    m = dict(zip(tk._SH_META, ring[0]))
    assert m["row"] == 4 and m["L"] == 700_000 and m["n_t_below"] == 1, m

    # tie rows: mostly exact zeros (the a11 shape), draft scope
    tl = logits.clone()
    for r in (2, 4, 5):
        n = int(lens[r])
        x = torch.where(torch.rand(n, device="cuda", generator=g) < 0.0005, tl[r, :n].abs() + 0.1, torch.zeros(n, device="cuda"))
        tl[r, :n] = x
    out2 = torch.full_like(out, -1)
    tk.triton_top_k_per_row_decode(tl, lens, out2, k)
    tk.shadow_compare(tl, lens, out2, k, True, native(tl, lens, k), None)
    v1, _ = stats(1)
    assert v1["calls"] == 1 and v1["tie_rows"] == 3 and v1["kth_zero"] == 3 and v1["t_val_bad"] == 0, v1

    # aiter's decode top-k as the second comparison (the production shadow call passes it when aiter is the choice)
    from vllm._aiter_ops import rocm_aiter_ops

    def aiter(out_):
        rocm_aiter_ops.indexer_top_k_decode(logits, 1, lens, out_, k)

    v0, _ = stats()
    tk.shadow_compare(logits, lens, out, k, False, native(logits, lens, k), aiter)
    v, _ = stats()
    assert v["a_calls"] == v0["a_calls"] + 1 and v["a_val_bad"] == v0["a_val_bad"] and v["a_cnt_bad"] == v0["a_cnt_bad"], (v0, v)

    # CUDA graph: capture (after the eager calls above set the state up), replay twice
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        tk.triton_top_k_per_row_decode(logits, lens, out, k)  # warm-up on the side stream
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=s):
            tk.triton_top_k_per_row_decode(logits, lens, out, k)
            tk.shadow_compare(logits, lens, out, k, False, native(logits, lens, k), None)
    torch.cuda.synchronize()
    v0, _ = stats()
    graph.replay()
    graph.replay()
    v, _ = stats()
    assert v["calls"] == v0["calls"] + 2 and v["rows"] == v0["rows"] + 2 * rows and v["t_val_bad"] == v0["t_val_bad"], (v0, v)

    time.sleep(tk._SH_QUIET + 2 * tk._SH_EVERY + 2)
    print(f"shadow counters target {dict((n, int(x)) for n, x in v.items() if x)}")
    print(f"shadow counters draft {dict((n, int(x)) for n, x in v1.items() if x)}")
    print("topk shadow test passed")


if __name__ == "__main__":
    main()
