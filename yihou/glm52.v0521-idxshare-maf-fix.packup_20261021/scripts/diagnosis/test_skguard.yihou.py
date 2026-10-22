# Task idxshare debug 2026-10-20: unit test of YIHOU-SKGUARD in sgl_kernel/top_k.py (mounted copy).
# Checks: eager counting, CUDA-graph replay counting, mask clamping. No out-of-buffer reads: every
# over-long row stays inside the [B, W] score buffer (row 0 overruns into row 1 only).
import os
import torch

os.environ.setdefault("YIHOU_TOPK_GUARD", "count")
import sgl_kernel.top_k as tk  # noqa: E402

dev = torch.device("cuda", 0)
torch.cuda.set_device(dev)
B, W = 3, 4096
score = torch.randn(B, W, device=dev)
lengths = torch.tensor([W + 500, 100, 3000], dtype=torch.int32, device=dev)  # row 0 bad
pt = torch.arange(B * W, dtype=torch.int32, device=dev).view(B, W)
cu = torch.arange(B + 1, dtype=torch.int32, device=dev)


def call():
    return tk.fast_topk_transform_fused(score, lengths, pt, cu, 2048)


call()  # eager: allocates counters
torch.cuda.synchronize()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    call()
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out = call()
for _ in range(5):
    g.replay()
torch.cuda.synchronize()
ctr = tk._YG_CTR[0].cpu()
for k, i in sorted(tk._YG_SITES.items(), key=lambda kv: kv[1]):
    print(i, k.split("|")[0:2], ctr[i].tolist())
print("RESULT eager calls (2 expected incl. side-stream warmup):", sum(ctr[i][0].item() for k, i in tk._YG_SITES.items() if "|eager|" in k))
print("RESULT graph replays counted (5 expected):", sum(ctr[i][0].item() for k, i in tk._YG_SITES.items() if "|graph|" in k))
print("RESULT bad_rows per call 1 expected:", [ctr[i][2].item() / max(ctr[i][0].item(), 1) for k, i in tk._YG_SITES.items()])
tk._YG_MODE = "mask"
clamped = tk._yg_guard("t", score, lengths, None)
print("RESULT mask:", clamped.tolist(), "expect", [W, 100, 3000])
