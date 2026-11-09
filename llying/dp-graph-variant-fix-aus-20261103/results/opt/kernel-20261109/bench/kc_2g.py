import sys, statistics as st, torch
from aiter.ops.triton.fusions.fused_bmm_rope_kv_cache import fused_fp8_bmm_rope_cat_and_cache_mla as bmm_kc
sys.path.insert(0, "/perf_apps/liyingli/bench_agentx/kopt-20261109/bench")
import absorb_bench as ab
FP8 = torch.float8_e4m3fn; QH, P, R, DPE = 64, 192, 512, 64
lim = (2**31) // 576; B = 88
w = [(torch.randn(QH, R, P, device="cuda") * 0.05).to(FP8) for _ in range(48)]
ws = torch.tensor([1.0], device="cuda"); ks = torch.tensor(1.0, device="cuda")
q = torch.randn(B, QH, P, device="cuda", dtype=torch.bfloat16); qpe = torch.randn(B, QH, DPE, device="cuda", dtype=torch.bfloat16)
kn = torch.randn(B, 1, R, device="cuda", dtype=torch.bfloat16); kp = torch.randn(B, 1, DPE, device="cuda", dtype=torch.bfloat16)
pos = torch.randint(0, 1 << 18, (B,), device="cuda"); fr = torch.randn(1 << 18, 32, device="cuda"); cos, sin = fr.cos(), fr.sin()
cfg = {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}
for slots in (lim - 64, lim + 64, 4094720):
    kv = torch.empty(slots, 1, R + DPE, device="cuda", dtype=FP8)
    sl = torch.randperm(1 << 20, device="cuda")[:B].to(torch.int64)
    for name, c in (("default", None), ("tuned", cfg)):
        f = lambda: [bmm_kc(q.transpose(0, 1), x, ws, qpe, kn, kp, kv, sl, pos, cos, sin, group_size=128, k_scale=ks, is_neox=False, q_out_dtype=FP8, config=None if c is None else dict(c)) for x in w]
        print(f"pool {slots:8d} ({'<' if slots*576 < 2**31 else '>='} 2^31) {name:7s}: {ab.graph_time(f, len(w)):5.2f} us", flush=True)
    del kv; torch.cuda.empty_cache()
