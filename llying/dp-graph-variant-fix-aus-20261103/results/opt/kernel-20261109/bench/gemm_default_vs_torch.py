import sys, torch
sys.path.insert(0, "/perf_apps/liyingli/bench_agentx/kopt-20261109/bench")
from aklr_gemm_bench import graph_us
from aiter.tuned_gemm import tgemm
for N, K, ms, name in ((2624, 6144, (64, 72, 88, 96, 112, 128), "fused_qkv_a"), (16384, 2048, (64, 72, 88, 96, 112, 128), "q_b_proj"),
                       (4096, 2048, (72, 96), "idx_wq_b"), (256, 6144, (288, 416, 640, 704, 768), "router")):
    ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(8)]
    for M in ms:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        ta = graph_us(lambda: [tgemm.mm(x, w, None, torch.bfloat16) for w in ws], len(ws))
        tt = graph_us(lambda: [torch.mm(x, w.t()) for w in ws], len(ws))
        print(f"{name:11s} M {M:4d}: aiter default {ta:6.1f} us ({N*K*2/ta/1e3:4.2f} TB/s)  torch.mm {tt:6.1f} us  ratio {ta/tt:5.2f}", flush=True)
