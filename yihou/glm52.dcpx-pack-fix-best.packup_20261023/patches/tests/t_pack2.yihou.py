# t_pack2 (task dcpx10 2026-10-22): compare stock copy_mla_rows_into_pack with the cached-metadata kernel of
# dcp_pack.yihou-nowait2.py on random data (one GPU). Run inside the v0522 image with the overlay on PYTHONPATH.
import importlib.util, os, sys, time, torch, numpy as np
os.environ["YIHOU_PACK_NOWAIT"] = "2"
spec = importlib.util.spec_from_file_location("yhpack", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from sglang.kernels.ops.kvcache.pd_dcp_gather import copy_mla_rows_into_pack
dev = torch.device("cuda:0"); torch.manual_seed(0)
layers, rows_total = 79, 20000
widths = [656] * 78 + [576]; strides = widths
bufs = [torch.randint(0, 255, (rows_total * w,), dtype=torch.uint8, device=dev) for w in widths]
ptrs = [b.data_ptr() for b in bufs]
for n in (1, 777, 8192):
    idx = torch.from_numpy(np.random.choice(rows_total, n, replace=False).astype(np.int64)).to(dev)
    a = torch.zeros(n * sum(widths), dtype=torch.uint8, device=dev); b = torch.zeros_like(a)
    copy_mla_rows_into_pack(ptrs, idx, a, widths, strides)
    m._yh_copy_rows_cached(ptrs, idx, b, widths, strides)
    torch.cuda.synchronize()
    print("n", n, "equal", bool(torch.equal(a, b)))
    t = time.perf_counter()
    for _ in range(20): m._yh_copy_rows_cached(ptrs, idx, b, widths, strides)
    torch.cuda.synchronize(); print("  cached kernel ms", round((time.perf_counter() - t) / 20 * 1e3, 3))
    t = time.perf_counter()
    for _ in range(20): copy_mla_rows_into_pack(ptrs, idx, a, widths, strides)
    torch.cuda.synchronize(); print("  stock kernel ms", round((time.perf_counter() - t) / 20 * 1e3, 3))
