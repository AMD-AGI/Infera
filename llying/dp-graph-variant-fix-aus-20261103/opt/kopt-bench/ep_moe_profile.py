"""Per-kernel breakdown (Kineto, graph replay) of the EP8 (33 x i2048, rank-0 mask) and TP8 (257 x i256) MoE at one M."""
import collections, sys, torch
sys.path.insert(0, "/perf_apps/liyingli/bench_agentx/kopt-20261109/bench")
import ep_moe_bench as eb, moe_bench as mb
from torch.profiler import ProfilerActivity, profile
M = int(sys.argv[1]) if len(sys.argv) > 1 else 704
g = torch.Generator(device="cuda").manual_seed(M)
x = torch.randn(M, 6144, dtype=torch.bfloat16, device="cuda", generator=g)
ids = torch.cat([torch.rand(M, 256, device="cuda", generator=g).topk(8, dim=1).indices.to(torch.int32),
                 torch.full((M, 1), 256, dtype=torch.int32, device="cuda")], 1).contiguous()
tw = torch.rand(M, 9, device="cuda", generator=g).float().contiguous()
mask = torch.zeros(257, dtype=torch.int32, device="cuda"); mask[:32] = 1; mask[256] = 1
for name, ws, m, key in (("ep", [eb.make_local_weights(33, 2048, 70 + i) for i in range(3)], mask, None),
                         ("tp-k512", [mb.make_weights(256, 7 + i) for i in range(3)], None, "k512")):
    if key: mb.set_key(key)
    fn = lambda: [eb.call(x, w, ids, tw, m) for w in ws]
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
        if e.device_type.name == "CUDA": per[e.name.split("(")[0][:95]] += e.device_time_total
    print(f"== {name} M {M}: {t:.1f} us/call")
    for k, v in per.most_common(9): print(f"   {v / 12:7.1f} us  {k}")
    mb.set_key("auto")
