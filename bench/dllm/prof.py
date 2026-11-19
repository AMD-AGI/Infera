"""Kernel census of one eager HF refinement step: counts, GPU busy vs wall, categories."""

import json
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModel

repo, out = sys.argv[1], sys.argv[2]
shapes = [(1, 256), (1, 1024), (8, 256), (8, 1024), (4, 2048)]
dev = torch.device("cuda")
model = (
    AutoModel.from_pretrained(repo, trust_remote_code=True, torch_dtype=torch.bfloat16)
    .to(dev)
    .eval()
)
vocab = model.config.vocab_size


def step(ids):
    o = model(ids)
    lg = o.logits if hasattr(o, "logits") else o[0]
    conf, pred = torch.softmax(lg.float(), dim=-1).max(dim=-1)
    return conf.topk(max(1, conf.shape[-1] // 8), dim=-1).indices


def cat(name):
    n = name.lower()
    if any(s in n for s in ("cijk", "gemm", "matmul", "gemv", "wvsplit")):
        return "gemm"
    if any(s in n for s in ("fmha", "attn", "attention", "flash")):
        return "attention"
    if any(
        s in n
        for s in (
            "softmax",
            "topk",
            "sort",
            "radix",
            "reduce_kernel<512, 1, at::native::reduceop<float, at::native::argmaxops",
            "maxops",
        )
    ):
        return "sampler"
    return "elementwise"


res = []
with torch.inference_mode():
    for b, s in shapes:
        ids = torch.randint(0, vocab - 1000, (b, s), device=dev)
        for _ in range(3):
            step(ids)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            step(ids)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / 5 * 1e3
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            step(ids)
            torch.cuda.synchronize()
        n_kern, busy, cats, counts = 0, 0.0, {}, {}
        for ev in prof.events():
            if ev.device_type == torch.autograd.DeviceType.CUDA:
                dt = ev.device_time / 1e3 if hasattr(ev, "device_time") else ev.cuda_time / 1e3
                n_kern += 1
                busy += dt
                c = cat(ev.name)
                cats[c] = cats.get(c, 0.0) + dt
                counts[c] = counts.get(c, 0) + 1
        r = dict(
            batch=b,
            seq=s,
            wall_ms=wall,
            gpu_busy_ms=busy,
            kernels=n_kern,
            by_cat_ms=cats,
            by_cat_n=counts,
        )
        res.append(r)
        print(
            f"b{b} s{s}: wall {wall:7.2f} ms  gpu busy {busy:7.2f} ms ({busy / wall:.0%})  kernels {n_kern}  "
            + "  ".join(f"{k} {v:.2f}ms/{counts[k]}" for k, v in sorted(cats.items())),
            flush=True,
        )
json.dump(res, open(out, "w"), indent=1)
