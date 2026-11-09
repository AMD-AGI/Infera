"""EP8 MoE (33 x i2048, rank-0 mask) at one M: us/call and the gemm1 / gemm2 kernel times (Kineto, graph replay).
The kernels come from the fmoe CSV rows in AITER_CONFIG_FMOE.  Usage: ep_sweep_one.py [M]"""
import collections, os, sys, torch
sys.path.insert(0, "/perf_apps/liyingli/bench_agentx/kopt-20261109/bench")
import ep_moe_bench as eb
from torch.profiler import ProfilerActivity, profile
M = int(sys.argv[1]) if len(sys.argv) > 1 else 704
g = torch.Generator(device="cuda").manual_seed(M)
x = torch.randn(M, 6144, dtype=torch.bfloat16, device="cuda", generator=g)
ids = torch.cat([torch.rand(M, 256, device="cuda", generator=g).topk(8, dim=1).indices.to(torch.int32),
                 torch.full((M, 1), 256, dtype=torch.int32, device="cuda")], 1).contiguous()
tw = torch.rand(M, 9, device="cuda", generator=g).float().contiguous()
mask = torch.zeros(257, dtype=torch.int32, device="cuda"); mask[:32] = 1; mask[256] = 1
ws = [eb.make_local_weights(33, 2048, 70 + i) for i in range(3)]
fn = lambda: [eb.call(x, w, ids, tw, mask) for w in ws]
t = eb.timed(fn, len(ws))
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
gr = torch.cuda.CUDAGraph()
with torch.cuda.graph(gr, stream=s): fn()
gr.replay(); torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for _ in range(4): gr.replay()
    torch.cuda.synchronize()
per = collections.Counter()
for e in p.events():
    if e.device_type.name == "CUDA": per[e.name.split("(")[0][:90]] += e.device_time_total
g1 = sum(v for k, v in per.items() if "moe1" in k or "mxmoe_g1" in k) / 12
g2 = sum(v for k, v in per.items() if "gemm2" in k or "moe2" in k or "reduce" in k.lower()) / 12
print(f"RESULT M {M} total {t:.1f} gemm1 {g1:.1f} gemm2 {g2:.1f} | " + " ; ".join(f"{v / 12:.1f} {k[:60]}" for k, v in per.most_common(3)))
if int(os.environ.get("TOPN", "0")):
    for k, v in per.most_common(int(os.environ["TOPN"])): print(f"   {v / 12:7.1f} us  {k}")
