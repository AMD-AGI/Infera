#!/usr/bin/env python3
"""Microbenchmark of SGLang's Triton _sparse_mla_fwd_split_dim_kernel at the 4+4 C80 decode shape.

Per DP rank on the decode worker: 12 requests x 6 verify tokens = 72 query rows, 64 heads,
top-2048 indices into an fp8 KV pool of [slots, 1, 576]. Times every config the autotuner
may pick (FP8 dot prunes it to BLOCK_N 32/64 x num_stages 1/2), in two layouts:
  local : each request's indices fall in a 16k-slot window (KV working set fits in cache)
  spread: each request's indices fall in its own 180k-slot window of a 2.2M-slot pool
KV pools rotate across LAYERS buffers so each launch starts with a cold cache, like the
layer-by-layer decode. Also reports which config the autotuner picks for each layout and
for an all-invalid (-1) index tensor.
"""
import argparse
import json
import math

import torch

from sglang.kernels.ops.attention.dsa import triton_sparse_mla as tsm

SEQ, REQS, H, D_V, D_TAIL, TOPK = 72, 12, 64, 512, 64, 2048
DIM = D_V + D_TAIL


def make_indices(pool, window, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    rows = []
    per_req = SEQ // REQS
    for r in range(REQS):
        base = (r * window) % max(pool - window, 1)
        for _ in range(per_req):
            rows.append(base + torch.randperm(window, generator=g)[:TOPK])
    return torch.stack(rows).to(torch.int32).cuda()


def launch(fn, args, kw, cfg=None):
    if cfg is None:
        fn[(SEQ,)](*args, **kw)
    else:
        bn, w, ns = cfg
        fn[(SEQ,)](*args, **kw, BLOCK_N=bn, num_warps=w, num_stages=ns)


def time_it(run, iters):
    for _ in range(3):
        run(0)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for i in range(iters):
        run(i)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=2_200_000)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--out", default=None)
    ap.add_argument("--async-copy", choices=("module", "off", "on"), default="module",
                    help="module: what SGLang's _no_async_copy() does; off/on: force the Triton knob")
    a = ap.parse_args()
    import contextlib

    import triton

    knobs = getattr(triton, "knobs", None)
    amd = getattr(knobs, "amd", None)
    has_knob = hasattr(amd, "use_async_copy")
    print(f"triton.knobs.amd.use_async_copy present={has_knob} value={getattr(amd, 'use_async_copy', None)!r}; "
          f"sglang _async_copy_is_harmful()={tsm._async_copy_is_harmful()}")
    if a.async_copy == "module":
        scope = tsm._no_async_copy()
    elif not has_knob:
        raise SystemExit("this Triton has no knobs.amd.use_async_copy")
    else:
        @contextlib.contextmanager
        def scope_fn():
            with amd.scope():
                amd.use_async_copy = a.async_copy == "on"
                yield
        scope = scope_fn()
    with scope:
        print(f"inside scope: use_async_copy={getattr(amd, 'use_async_copy', None)!r} (mode {a.async_copy})")
        run(a)


def run(a):
    torch.manual_seed(0)
    dev = torch.device("cuda")
    kvs = [
        (torch.randn(a.pool, 1, DIM, device=dev) * 0.5).to(torch.float8_e4m3fn)
        for _ in range(a.layers)
    ]
    q_nope = torch.randn(SEQ, H, D_V, device=dev, dtype=torch.bfloat16)
    q_rope = torch.randn(SEQ, H, D_TAIL, device=dev, dtype=torch.bfloat16)
    out = torch.empty(SEQ, H, D_V, device=dev, dtype=torch.bfloat16)
    qk_scale = (1.0 / math.sqrt(DIM)) * 1.4426950408889634
    layouts = {
        "local": make_indices(a.pool, 16_384, 1),
        "spread": make_indices(a.pool, 180_000, 2),
        "invalid": torch.full((SEQ, TOPK), -1, dtype=torch.int32, device=dev),
    }

    auto = tsm._sparse_mla_fwd_split_dim_kernel
    fn = auto.fn
    has_scaled = "KV_SCALED" in fn.arg_names
    kw = dict(H=H, DIM=DIM, D_V=D_V, D_TAIL=D_TAIL, NUM_GROUPS=D_V // 128,
              STRIDE_QN_T=q_nope.stride(0), STRIDE_QN_H=q_nope.stride(1),
              STRIDE_QR_T=q_rope.stride(0), STRIDE_QR_H=q_rope.stride(1),
              USE_FP8_DOT=True, SEQ_BUCKET=0)
    if has_scaled:
        kw["KV_SCALED"] = False

    def args_for(kv, idx):
        return (q_nope, q_rope, kv, idx, out, qk_scale, 448.0, TOPK)

    res = {"torch": torch.__version__, "triton": __import__("triton").__version__,
           "async_copy": a.async_copy, "configs": {}, "autotune": {}}
    configs = [(bn, w, ns) for bn in (32, 64) for w in (1, 2, 4) for ns in (1, 2)]
    print(f"torch {res['torch']} triton {res['triton']} pool {a.pool} layers {a.layers}")
    print(f"{'config':>18s} " + " ".join(f"{k:>10s}" for k in ("local", "spread")))
    for cfg in configs:
        row = {}
        for name in ("local", "spread"):
            idx = layouts[name]
            try:
                us = time_it(lambda i: launch(fn, args_for(kvs[i % a.layers], idx), kw, cfg), a.iters)
            except Exception as e:
                us = float("nan")
                print(f"  {cfg} {name} failed: {type(e).__name__}: {str(e)[:120]}")
            row[name] = us
        res["configs"][str(cfg)] = row
        print(f"{str(cfg):>18s} " + " ".join(f"{row[k]:10.1f}" for k in ("local", "spread")))

    for name in ("invalid", "local", "spread"):
        auto.cache.clear()
        launch(auto, args_for(kvs[0], layouts[name]), kw)
        torch.cuda.synchronize()
        best = auto.best_config
        pick = (best.kwargs.get("BLOCK_N"), best.num_warps, best.num_stages)
        res["autotune"][name] = str(pick)
        sp = res["configs"].get(str(pick), {}).get("spread")
        print(f"autotune on {name:7s} indices picks {pick}  -> spread-layout time {sp:.1f} us" if sp else
              f"autotune on {name:7s} indices picks {pick}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
