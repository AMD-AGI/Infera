# ruff: noqa: B023 - each closure is timed in the iteration that builds it
"""What the HF reference MoE loops cost beyond their expert GEMMs, on one GPU.

usage: python moe_loop.py <out.json>

loop_all: the Mixtral/OLMoE/Qwen3-MoE block -- every expert visited, each with
a torch.where (a device-to-host sync), a gather, gate/up/down, the activation,
two multiplies and an index_add_. loop_touched: DeepSeek's moe_infer as
LLaDA2.0 ships it -- tokens sorted by expert, one .cpu() of the counts, then
only the experts that got tokens. Both against the same GEMMs issued alone.
"""

import json
import sys

import torch
import torch.nn.functional as F

dev = torch.device("cuda")
H = 2048
dt = torch.bfloat16


def timed(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


class Experts:
    def __init__(self, n, ffn):
        self.w = [
            tuple(
                torch.randn(*shape, dtype=dt, device=dev) * 0.02
                for shape in ((ffn, H), (ffn, H), (H, ffn))
            )
            for _ in range(n)
        ]
        self.gate = torch.randn(n, H, dtype=dt, device=dev) * 0.02

    def mlp(self, i, x):
        g, u, d = self.w[i]
        return F.linear(F.silu(F.linear(x, g)) * F.linear(x, u), d)


def loop_all(ex, x, topk):
    logits = F.linear(x, ex.gate)
    w = F.softmax(logits, dim=1, dtype=torch.float)
    w, sel = torch.topk(w, topk, dim=-1)
    w = w.to(dt)
    out = torch.zeros_like(x)
    mask = F.one_hot(sel, num_classes=len(ex.w)).permute(2, 1, 0)
    for i in range(len(ex.w)):
        idx, top_x = torch.where(mask[i])
        cur = x[None, top_x].reshape(-1, H)
        out.index_add_(0, top_x, ex.mlp(i, cur) * w[top_x, idx, None])
    return out


def loop_touched(ex, x, topk):
    logits = F.linear(x, ex.gate)
    w = F.softmax(logits, dim=1, dtype=torch.float)
    w, sel = torch.topk(w, topk, dim=-1)
    cnts = sel.new_zeros((sel.shape[0], len(ex.w)))
    cnts.scatter_(1, sel, 1)
    per = cnts.sum(dim=0).cpu().numpy()
    idxs = sel.view(-1).argsort()
    tokens = x[idxs // topk]
    outs, start = [], 0
    for i, n in enumerate(per):
        if n == 0:
            continue
        outs.append(ex.mlp(i, tokens[start : start + n]))
        start += n
    cat = torch.cat(outs, dim=0)
    new = torch.empty_like(cat)
    new[idxs] = cat
    return new.view(*sel.shape, -1).type(w.dtype).mul_(w.unsqueeze(-1)).sum(dim=1).type(x.dtype)


rows = []
with torch.inference_mode():
    for n_exp, ffn in ((64, 1024), (128, 768), (256, 512)):
        ex = Experts(n_exp, ffn)
        for T in (4, 32, 256, 1024, 4096, 16384):
            x = torch.randn(T, H, dtype=dt, device=dev)
            sel = torch.topk(F.linear(x, ex.gate).float(), 8, dim=-1).indices
            per = torch.bincount(sel.view(-1), minlength=n_exp).tolist()
            touched = sum(c > 0 for c in per)
            slices = [x[:c] for c in per]
            gemms = timed(
                lambda ex=ex, slices=slices: [
                    ex.mlp(i, s) for i, s in enumerate(slices) if s.shape[0]
                ]
            )
            for style, fn in (("loop_all", loop_all), ("loop_touched", loop_touched)):
                ms = timed(lambda fn=fn, ex=ex, x=x: fn(ex, x, 8))
                iters = n_exp if style == "loop_all" else touched
                r = dict(
                    style=style,
                    experts=n_exp,
                    ffn=ffn,
                    tokens=T,
                    touched=touched,
                    block_ms=ms,
                    gemms_ms=gemms,
                    iter_us=(ms - gemms) * 1e3 / iters,
                )
                rows.append(r)
                print(json.dumps(r), flush=True)
        del ex
        torch.cuda.empty_cache()
json.dump(rows, open(sys.argv[1], "w"), indent=1)
