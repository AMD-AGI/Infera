import sys, torch
sys.path.insert(0, "/perf_apps/liyingli/bench_agentx/kopt-20261109/bench")
import sparse_bench as sb, sparse_probe as sp
kf = sb.load_kf(); g = torch.Generator(device="cuda").manual_seed(0)
lim = (2**31) // 576
rows = 88
qn = (torch.randn(rows, sb.H, sb.DV, device="cuda", generator=g) * 0.5).to(torch.float8_e4m3fn)
qr = (torch.randn(rows, sb.H, sb.DR, device="cuda", generator=g) * 0.5).to(torch.float8_e4m3fn)
idx = [sb.make_indices(rows, 1 << 20, "request", g) for _ in range(8)]
for slots in (lim - 1024, lim - 1, lim, lim + 1, lim + 1024, 4094720):
    kv = torch.empty(slots, 1, 576, device="cuda", dtype=torch.float8_e4m3fn); kv[: 1 << 20].copy_((torch.randn(1 << 20, 1, 576, device="cuda") * 0.5).to(torch.float8_e4m3fn))
    print(f"pool {slots:8d} slots = {slots*576:11d} B ({'<' if slots*576 < 2**31 else '>='} 2^31): {sp.timed(kf, qn, qr, kv, idx):5.1f} us", flush=True)
    del kv; torch.cuda.empty_cache()
