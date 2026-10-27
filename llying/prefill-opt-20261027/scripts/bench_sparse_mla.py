"""Microbenchmark of the DSA sparse-MLA prefill forward at GLM-5.2 TP1 prefill shapes (H = 64, d_v = 512, rope 64,
FP8 KV of 576 B per token, top-k 2048), run inside the serving image on one GPU.

GLM-5.2 at TP1 has 64 heads per rank, so `triton_sparse_mla_fwd` skips the gfx950 FP8 tuning (H in {8, 16} only)
and takes the single-pass split-dim kernel, whose autotune space was set for other shapes. Compared here:
  prod            triton_sparse_mla_fwd as the server calls it (autotuned single pass)
  single/BN,W,S   the single-pass kernel at an explicit BLOCK_N / num_warps / num_stages (also outside the autotune space)
  splitk/ks       the split-K kernel the TP4 / TP8 shapes use (BLOCK_H = 16, so each KV row is read once per 16 heads)
Each variant's output is compared with prod (max |diff|, and the share of elements off by more than 2 % of max |prod|).

Index modes: "uniform" draws each token's 2048 positions at random from its request's context; "overlap" gives the
tokens of one request a shared hot set (a fraction OVERLAP of each row) and fills the rest at random, which is
closer to real DSA top-k where neighbouring query tokens select mostly the same keys.
  python3 bench_sparse_mla.py [--seq 8192] [--reqs 4] [--ctx 131072] [--mode overlap] [--overlap 0.7] [--iters 10]"""

import argparse
import itertools
import time

import torch

from sglang.kernels.ops.attention.dsa import triton_sparse_mla as tsm

ap = argparse.ArgumentParser()
ap.add_argument("--seq", type=int, default=8192)
ap.add_argument("--reqs", type=int, default=4)
ap.add_argument("--ctx", type=int, default=131072)
ap.add_argument("--heads", type=int, default=64)
ap.add_argument("--topk", type=int, default=2048)
ap.add_argument("--mode", default="overlap", choices=["uniform", "overlap"])
ap.add_argument("--overlap", type=float, default=0.7)
ap.add_argument("--iters", type=int, default=10)
ap.add_argument("--quick", action="store_true", help="only prod, the best-known singles and split-K")
ap.add_argument("--prod-only", action="store_true")
ap.add_argument("--pool-gb", type=float, default=0, help="KV pool size; 0 = just the requests' tokens")
ap.add_argument("--scatter-run", type=int, default=0,
                help="place each request's tokens in the pool in random runs of this many tokens (1 = fully random); 0 = contiguous")
a = ap.parse_args()

torch.manual_seed(0)
dev = "cuda"
fp8 = torch.float8_e4m3fnuz if tsm._IS_FNUZ else torch.float8_e4m3fn
H, DV, DT, DIM, TOPK = a.heads, 512, 64, 576, a.topk
n_rows = max(a.reqs * a.ctx, int(a.pool_gb * 1e9 / DIM))
kv = torch.empty(n_rows, 1, DIM, device=dev, dtype=fp8)
kv.view(torch.uint8).random_(0, 0x40)  # finite positive e4m3 values below 2, no temporaries for large pools
q = (torch.randn(a.seq, H, DV + DT, device=dev) * 0.5).to(fp8)
q_nope, q_rope = q[..., :DV], q[..., DV:]

per_req = a.seq // a.reqs
idx = torch.empty(a.seq, TOPK, dtype=torch.int32, device=dev)
for r in range(a.reqs):
    base = r * a.ctx
    rows = slice(r * per_req, (r + 1) * per_req)
    rnd = torch.randint(0, a.ctx, (per_req, TOPK), device=dev, dtype=torch.int32)
    if a.mode == "overlap":
        hot = torch.randperm(a.ctx, device=dev)[:TOPK].to(torch.int32)
        n_hot = int(a.overlap * TOPK)
        rnd[:, :n_hot] = hot[:n_hot]
    idx[rows] = base + rnd
if a.scatter_run > 0:
    # Logical token t of the requests -> a physical pool row; runs of scatter_run consecutive tokens stay together.
    run = a.scatter_run
    n_runs = (a.reqs * a.ctx + run - 1) // run
    slots = torch.randperm(n_rows // run, device=dev)[:n_runs]
    t = torch.arange(a.reqs * a.ctx, device=dev)
    phys = (slots[t // run] * run + t % run).to(torch.int32)
    idx = phys[idx.long()]
indices = idx.unsqueeze(1)
scale = 1.0 / 16.0


def bench(fn):
    for _ in range(2):
        out = fn()
    torch.cuda.synchronize()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    ev[0].record()
    for _ in range(a.iters):
        out = fn()
    ev[1].record()
    torch.cuda.synchronize()
    return ev[0].elapsed_time(ev[1]) / a.iters, out


def prod():
    return tsm.triton_sparse_mla_fwd(q_nope, q_rope, kv, indices, scale, DV, topk_length=None, max_topk_length=TOPK)


def single(bn, w, s):
    qn, sqn_t, sqn_h = tsm._row_strides(q_nope)
    qr, sqr_t, sqr_h = tsm._row_strides(q_rope)
    out = torch.empty(a.seq, H, DV, device=dev, dtype=torch.bfloat16)

    def run():
        with tsm._no_async_copy():
            tsm._sparse_mla_fwd_split_dim_kernel.fn[(a.seq,)](
                qn, qr, kv, idx, out, scale * tsm._LOG2E, tsm._FP8_MAX, TOPK,
                H=H, DIM=DIM, D_V=DV, D_TAIL=DT, NUM_GROUPS=DV // 128,
                STRIDE_QN_T=sqn_t, STRIDE_QN_H=sqn_h, STRIDE_QR_T=sqr_t, STRIDE_QR_H=sqr_h,
                USE_FP8_DOT=True, SEQ_BUCKET=0, BLOCK_N=bn, num_warps=w, num_stages=s)
        return out.unsqueeze(0)
    return run


def fused(bh, bk, w, ps, i64=True):
    """The head-blocked single-pass kernel (what split-K uses at kv_splits = 1), at an explicit configuration."""
    qn, sqn_t, sqn_h = tsm._row_strides(q_nope)
    qr, sqr_t, sqr_h = tsm._row_strides(q_rope)
    out = torch.empty(a.seq, H, DV, device=dev, dtype=torch.bfloat16)

    def run():
        with tsm._no_async_copy():
            tsm._sparse_mla_fused_kernel[(a.seq, (H + bh - 1) // bh)](
                qn, qr, kv, idx, idx, out, scale * tsm._LOG2E, tsm._FP8_MAX,
                topk=TOPK, H=H, KV_DIM=DIM, D_V=DV, D_TAIL=DT, NUM_GROUPS=DV // 128,
                STRIDE_QN_T=sqn_t, STRIDE_QN_H=sqn_h, STRIDE_QR_T=sqr_t, STRIDE_QR_H=sqr_h,
                USE_FP8_DOT=True, USE_TOPK_LENGTH=False, BLOCK_H=bh, BLOCK_K=bk,
                USE_I64_PAGE=i64, PIPE_STAGES=ps, num_warps=w, num_stages=2)
        return out.unsqueeze(0)
    return run


def splitk(ks):
    def run():
        with tsm._no_async_copy():
            return tsm._triton_sparse_mla_fwd_splitk(q_nope, q_rope, kv, indices, scale, DV, kv_splits=ks)
    return run


t_prod, ref = bench(prod)
ref = ref.float()
ref_max = ref.abs().max().item()
gbytes = a.seq * TOPK * DIM / 1e9
print(f"seq {a.seq} H {H} topk {TOPK} reqs {a.reqs} ctx {a.ctx} mode {a.mode} overlap {a.overlap} "
      f"pool {n_rows * DIM / 1e9:.1f} GB scatter-run {a.scatter_run}; KV bytes gathered per call {gbytes:.1f} GB")
cfg = tsm._sparse_mla_fwd_split_dim_kernel.best_config if hasattr(tsm._sparse_mla_fwd_split_dim_kernel, "best_config") else None
print(f"{'prod':24s} {t_prod:8.3f} ms  {gbytes / t_prod:6.2f} TB/s  autotune pick: {cfg}")

variants = []
if a.prod_only:
    raise SystemExit
if a.quick:
    variants += [(f"single/{bn},{w},{s}", single(bn, w, s)) for bn, w, s in [(32, 4, 2), (64, 4, 2), (64, 8, 2), (32, 8, 2), (128, 8, 2)]]
else:
    variants += [(f"single/{bn},{w},{s}", single(bn, w, s))
                 for bn, w, s in itertools.product((16, 32, 64, 128), (2, 4, 8), (1, 2))]
variants += [(f"splitk/{ks}", splitk(ks)) for ks in (1, 2)]
if a.quick:
    variants += [(f"fused/{bh},{bk},{w},{ps}", fused(bh, bk, w, ps)) for bh, bk, w, ps in [(16, 64, 2, 3), (32, 64, 4, 3), (64, 64, 8, 3)]]
else:
    variants += [(f"fused/{bh},{bk},{w},{ps}", fused(bh, bk, w, ps))
                 for bh, bk, w, ps in itertools.product((16, 32, 64), (32, 64, 128), (2, 4, 8), (1, 2, 3))]
    variants += [(f"fused-i32/{bh},64,{w},3", fused(bh, 64, w, 3, i64=False)) for bh, w in [(16, 2), (16, 4), (32, 4)]]
rows = []
for name, fn in variants:
    try:
        t, out = bench(fn)
    except Exception as e:  # compile failure (registers / LDS) for some configs
        print(f"{name:24s} failed: {type(e).__name__}: {str(e).splitlines()[0][:120]}")
        continue
    d = (out.float() - ref).abs()
    bad = (d > 0.02 * ref_max).float().mean().item()
    rows.append((t, name))
    print(f"{name:24s} {t:8.3f} ms  {gbytes / t:6.2f} TB/s  x{t_prod / t:5.2f}  max|diff| {d.max().item():.4f}  >2%: {bad:.2e}")
rows.sort()
print("best:", ", ".join(f"{n} {t:.3f} ms (x{t_prod / t:.2f})" for t, n in rows[:5]))
