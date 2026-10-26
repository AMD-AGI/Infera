#!/usr/bin/env python3
"""MLA absorbed-BMM config microbenchmark: our image's aiter configs vs ATOM PR #209's per-M table.

Kernels timed (GLM-5.2 decode on one DP-attention rank: 64 heads, kv_lora 512, nope 192, v 256):

  qup_fused  aiter fused_fp8_bmm_rope_cat_and_cache_mla, N=512 K=192.
             This is what our decode / target-verify / draft forwards run for the q up-projection
             (sglang forward_mla_rocm.py:678-682 + 721-724 + 839-847, gated by
             _can_fuse_bmm_rope_cat_and_cache at :444-454). Config family in the image:
             FUSED-FP8-BMM-ROPE-CAT-AND-CACHE-MLA (aiter _triton_kernels/fusions/fused_bmm_rope_kv_cache.py:1041-1053).
  qup_bmm    aiter batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant, N=512 K=192
             (rocm_absorb_q_bmm, forward_mla_rocm.py:229-243; in our image only taken when the fused path is
             not, e.g. extend). This is the kernel ATOM's table was swept for.
  vup_bmm    same batched kernel, N=256 K=512, transpose_bm=True into a (M, H, N) buffer
             (rocm_absorb_v_bmm, forward_mla_rocm.py:284-303).

Configs compared per M:
  image  config=None -> aiter picks from the image's JSON (exactly what sglang does: _absorb_bmm_config
         returns None, forward_mla_rocm.py:156-160).
  pr209  the TUNED table of Infera commit adbb9dc5
         deploy/docker/patches/atom_nightly/patch_aiter_mla_bmm_configs.py, looked up the way aiter's
         get_gemm_config does with that patch's M_BOUNDS (first bound >= M). For qup_fused the PR table
         (swept for the standalone kernel) is applied to the fused kernel's BMM half -- an untested transfer.
  --sweep additionally times a 72-candidate grid per M and reports the fastest output-identical one.

Timing: CUDA-graph replay of `--iters` back-to-back calls rotating over `--nbuf` distinct weight/input
sets (so weights are not all L2-resident), median over `--reps` replays, reported in us per call.
Correctness: each non-image config's outputs are compared with the image config's outputs on identical
inputs (bitwise flag + max abs diff), and the BMM outputs also against an fp32 torch reference.

Read-only with respect to the image: nothing in aiter is patched; configs are passed via config=.
"""

import argparse
import json
import math
import os
import statistics
import sys

import torch

try:  # sglang's import path (forward_mla_rocm.py:131-133)
    from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant as aiter_bmm,
    )
except ImportError:
    from aiter.ops.triton.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant as aiter_bmm,
    )
from aiter.ops.triton._triton_kernels.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
    _get_config as bmm_get_config,
)
from aiter.ops.triton.fusions.fused_bmm_rope_kv_cache import (
    fused_fp8_bmm_rope_cat_and_cache_mla as aiter_fused,
)
from aiter.ops.triton._triton_kernels.fusions.fused_bmm_rope_kv_cache import (
    _get_fp8_config as fused_get_config,
)

H, KV_LORA, NOPE, ROPE, VDIM = 64, 512, 192, 64, 256
FP8 = torch.float8_e4m3fn
DEV = "cuda"

# ---- ATOM PR #209 table, copied verbatim from adbb9dc5:deploy/docker/patches/atom_nightly/patch_aiter_mla_bmm_configs.py
# (N, K): [(M, BLOCK_SIZE_M, BLOCK_SIZE_N, num_warps, waves_per_eu, cache_modifier)]
# fmt: off
PR_TUNED = {
    (512, 192): [
        (16, 16, 128, 4, 2, None), (20, 16, 128, 4, 2, None), (24, 16, 128, 4, 2, None),
        (28, 16, 128, 4, 2, ".cg"), (32, 16, 128, 4, 2, ".cg"),
        (36, 32, 128, 8, 1, ".cg"), (40, 32, 128, 8, 1, None),
        (44, 32, 128, 8, 1, ".cg"), (48, 16, 256, 4, 2, None),
        (52, 32, 128, 8, 1, None), (56, 32, 128, 8, 1, None),
        (60, 32, 128, 8, 1, ".cg"), (64, 32, 128, 4, 2, None),
        (80, 16, 128, 8, 1, None), (96, 32, 256, 8, 1, ".cg"),
        (112, 16, 128, 8, 1, None), (128, 32, 256, 8, 1, None),
        (144, 16, 128, 8, 1, None), (160, 32, 128, 8, 1, ".cg"),
        (176, 16, 128, 8, 1, None), (192, 32, 128, 8, 1, ".cg"),
        (208, 16, 128, 8, 1, None), (224, 32, 128, 8, 1, ".cg"),
        (240, 16, 128, 8, 1, None), (256, 64, 256, 4, 2, None),
        (272, 16, 128, 8, 1, None), (288, 32, 128, 8, 1, None),
        (304, 16, 128, 8, 1, None), (320, 32, 128, 8, 1, ".cg"),
        (336, 16, 128, 8, 1, ".cg"), (352, 32, 128, 8, 1, ".cg"),
        (368, 16, 128, 8, 1, ".cg"), (384, 64, 256, 8, 1, ".cg"),
        (400, 16, 128, 8, 1, None), (416, 32, 128, 8, 1, ".cg"),
        (432, 16, 128, 8, 1, ".cg"), (448, 32, 128, 8, 1, None),
        (464, 16, 128, 8, 1, ".cg"), (480, 32, 128, 8, 1, ".cg"),
        (496, 16, 128, 8, 1, None), (512, 64, 256, 4, 2, None),
    ],
    (256, 512): [
        (16, 16, 64, 4, 2, None), (20, 16, 64, 4, 2, None), (24, 16, 64, 4, 2, None),
        (28, 16, 64, 4, 2, None), (32, 16, 128, 4, 1, None), (36, 16, 128, 4, 2, None),
        (40, 16, 128, 4, 2, None), (44, 16, 128, 4, 2, None), (48, 16, 128, 4, 2, None),
        (52, 16, 128, 4, 2, None), (56, 16, 128, 4, 2, None), (60, 16, 128, 4, 2, None),
        (64, 16, 128, 4, 2, None), (80, 16, 256, 4, 2, None), (96, 32, 256, 8, 1, None),
        (112, 16, 256, 4, 2, None), (128, 32, 256, 8, 2, None),
        (144, 16, 256, 4, 2, None), (160, 32, 128, 8, 1, None),
        (176, 16, 256, 4, 2, None), (192, 32, 128, 8, 1, None),
        (208, 16, 256, 4, 2, None), (224, 32, 256, 4, 2, None),
        (240, 16, 256, 4, 2, None), (256, 32, 256, 4, 2, None),
        (272, 16, 256, 4, 2, None), (288, 32, 256, 8, 1, None),
        (304, 16, 256, 4, 2, None), (320, 32, 256, 8, 1, None),
        (336, 16, 256, 4, 2, None), (352, 32, 256, 8, 1, None),
        (368, 16, 256, 4, 2, None), (384, 32, 256, 8, 1, None),
        (400, 16, 256, 4, 2, None), (416, 32, 256, 8, 1, None),
        (432, 16, 256, 4, 2, None), (448, 64, 256, 8, 1, None),
        (464, 16, 256, 4, 2, None), (480, 32, 256, 8, 1, None),
        (496, 16, 256, 4, 2, None), (512, 64, 256, 8, 1, None),
    ],
}
# fmt: on


def mk_cfg(bm, bn, warps, waves, cache, stages=2, nonk=16):
    return {
        "BLOCK_SIZE_M": bm,
        "BLOCK_SIZE_N": bn,
        "GROUP_SIZE_M": 1,
        "num_warps": warps,
        "num_stages": stages,
        "waves_per_eu": waves,
        "matrix_instr_nonkdim": nonk,
        "cache_modifier": cache,
    }


def pr_cfg(n, k, m):
    """PR table lookup: M_BOUNDS = table Ms (+1024..8192 which keep stock); first bound >= M."""
    for row in PR_TUNED[(n, k)]:
        if m <= row[0]:
            return mk_cfg(*row[1:])
    return None  # M > 512: stock entries keep serving under the PR


def sweep_cfgs():
    out = []
    for bm in (16, 32, 64):
        for bn in (64, 128, 256):
            for warps, waves in ((4, 1), (4, 2), (8, 1), (8, 2)):
                for cache in (None, ".cg"):
                    out.append(mk_cfg(bm, bn, warps, waves, cache))
    return out  # 72


def quant_fp8_per_tensor(w):
    amax = w.abs().amax().float().clamp(min=1e-6)
    scale = amax / torch.finfo(FP8).max
    return (w.float() / scale).to(FP8), scale.reshape(1).to(torch.float32)


# ---------------------------------------------------------------- inputs (layouts as in sglang)
def make_weights(nbuf, gen):
    """Per layer: w_kc as WQ (H, 512, 192) and w_vc as WQ (H, 256, 512), contiguous (forward_mla_rocm.py:232, 293)."""
    ws = []
    for _ in range(nbuf):
        # kv_b_proj rows per head: [nope(192) | v(256)] x kv_lora(512) -> deepseek_weight_loader.py:724-757
        w = torch.randn(H * (NOPE + VDIM), KV_LORA, generator=gen, device=DEV, dtype=torch.float32) * 0.02
        wq, ws_ = quant_fp8_per_tensor(w)
        w_kc, w_vc = wq.unflatten(0, (-1, NOPE + VDIM)).split([NOPE, VDIM], dim=1)  # (H,192,512),(H,256,512)
        wq_k = w_kc.transpose(1, 2).contiguous()  # (H, 512, 192) == attn.w_kc.transpose(-1,-2)
        wq_v = w_vc.contiguous()  # (H, 256, 512) == attn.w_vc.transpose(-1,-2)
        ws.append((wq_k, wq_v, ws_))
    return ws


def make_acts(m, nbuf, gen, num_slots):
    acts = []
    for _ in range(nbuf):
        q = torch.randn(m, H, NOPE + ROPE, generator=gen, device=DEV, dtype=torch.float32).to(torch.bfloat16)
        latent = (torch.randn(m, KV_LORA + ROPE, generator=gen, device=DEV, dtype=torch.float32) / 20).to(torch.bfloat16)
        attn_out = torch.randn(m, H, KV_LORA, generator=gen, device=DEV, dtype=torch.float32).to(torch.bfloat16)
        acts.append(
            dict(
                q_nope=q[..., :NOPE],  # strided view, as _split_q_nope_pe gives
                q_pe=q[..., NOPE:],
                k_nope=latent[..., :KV_LORA].unsqueeze(1),
                k_pe=latent[..., KV_LORA:].unsqueeze(1),
                attn_out=attn_out,
                slot=torch.randperm(num_slots, generator=gen, device=DEV)[:m].contiguous(),
                pos=torch.randint(0, 16384, (m,), generator=gen, device=DEV),
            )
        )
    return acts


class Rope:
    def __init__(self, gen):
        inv = 1.0 / (10000 ** (torch.arange(0, ROPE, 2, device=DEV, dtype=torch.float32) / ROPE))
        t = torch.arange(32768, device=DEV, dtype=torch.float32)
        f = torch.outer(t, inv)  # (max_pos, ROPE/2) -> reuse_freqs_front_part
        self.cos = f.cos().to(torch.bfloat16)
        self.sin = f.sin().to(torch.bfloat16)
        self.k_scale = torch.ones([1], dtype=torch.float32, device=DEV)[0]


# ---------------------------------------------------------------- kernel calls
def call_qup_bmm(w, a, cfg):
    wq_k, _, s = w
    return aiter_bmm(
        X=a["q_nope"], WQ=wq_k, w_scale=s, group_size=128, YQ=None,
        transpose_bm=False, transpose_bm_in=True, dtype=torch.bfloat16,
        config=None if cfg is None else dict(cfg),
    )


def call_vup_bmm(w, a, cfg, out):
    _, wq_v, s = w
    aiter_bmm(
        X=a["attn_out"], WQ=wq_v, w_scale=s, group_size=128, YQ=out,
        transpose_bm=True, transpose_bm_in=True, dtype=torch.bfloat16,
        config=None if cfg is None else dict(cfg),
    )
    return out


def call_qup_fused(w, a, cfg, kv_cache, rope):
    wq_k, _, s = w
    # forward_mla_rocm.py:476-492 (q_out_dtype = fp8 because kv_cache_dtype fp8_e4m3)
    return aiter_fused(
        a["q_nope"].transpose(0, 1), wq_k, s, a["q_pe"], a["k_nope"], a["k_pe"], kv_cache,
        a["slot"], a["pos"], rope.cos, rope.sin, group_size=128,
        config=None if cfg is None else dict(cfg), k_scale=rope.k_scale, is_neox=False, q_out_dtype=FP8,
    )


# ---------------------------------------------------------------- timing / checks
def time_graph(fn_list, iters, reps):
    """fn_list: callables (one per buffer set); captured round-robin `iters` times."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for fn in fn_list:  # JIT + autotune-free warmup outside capture
            fn()
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(iters):
            fn_list[i % len(fn_list)]()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(reps):
        st.record()
        g.replay()
        en.record()
        en.synchronize()
        ts.append(st.elapsed_time(en) * 1e3 / iters)
    del g
    return statistics.median(ts)


def diff(a, b):
    if a.dtype == FP8:
        bit = torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    else:
        bit = torch.equal(a, b)
    return bit, (a.float() - b.float()).abs().max().item()


def ref_rel_err(y, x, wq, s):
    """fp32 reference x @ (wq*s)^T per head; y laid out (H, M, N)."""
    xr = x.float().transpose(0, 1)  # (H, M, K)
    r = torch.bmm(xr, (wq.float() * s.float()).transpose(1, 2))
    return ((y.float() - r).abs().max() / r.abs().max().clamp(min=1e-6)).item()


def cfg_str(c):
    if c is None:
        return "-"
    return "BM{BLOCK_SIZE_M}/BN{BLOCK_SIZE_N}/w{num_warps}/wpe{waves_per_eu}/st{num_stages}/nk{matrix_instr_nonkdim}/{cache_modifier}".format(**c)


def run_case(kind, m, ws, args, gen, rope, sweep):
    nbuf = args.nbuf
    acts = make_acts(m, nbuf, gen, args.num_slots)
    if kind == "qup_fused":
        n, k = 512, 192
        image_cfg, _ = fused_get_config(m, n, k)
    elif kind == "qup_bmm":
        n, k = 512, 192
        image_cfg, _ = bmm_get_config(m, n, k)
    else:
        n, k = 256, 512
        image_cfg, _ = bmm_get_config(m, n, k)
    cands = [("image", None), ("pr209", pr_cfg(n, k, m))]
    if sweep:
        cands += [("sweep%02d" % i, c) for i, c in enumerate(sweep_cfgs())]

    caches = [torch.zeros(args.num_slots, 1, KV_LORA + ROPE, dtype=FP8, device=DEV) for _ in range(nbuf)]
    vbufs = [torch.empty(m, H, VDIM, dtype=torch.bfloat16, device=DEV) for _ in range(nbuf)]

    def runner(cfg, i):
        w, a = ws[i], acts[i]
        if kind == "qup_fused":
            return lambda: call_qup_fused(w, a, cfg, caches[i], rope)
        if kind == "qup_bmm":
            return lambda: call_qup_bmm(w, a, cfg)
        return lambda: call_vup_bmm(w, a, cfg, vbufs[i])

    def outputs(cfg):
        # buffer set 0, fresh cache, single call
        if kind == "qup_fused":
            kc = torch.zeros_like(caches[0])
            q_out, _, k_pe_out, _ = call_qup_fused(ws[0], acts[0], cfg, kc, rope)
            return [q_out.clone(), k_pe_out.clone(), kc[acts[0]["slot"]].clone()]
        if kind == "qup_bmm":
            return [call_qup_bmm(ws[0], acts[0], cfg).clone()]
        vb = torch.empty(m, H, VDIM, dtype=torch.bfloat16, device=DEV)
        return [call_vup_bmm(ws[0], acts[0], cfg, vb).clone()]

    base = outputs(None)
    torch.cuda.synchronize()
    res = {"kind": kind, "M": m, "image_cfg": image_cfg, "rows": []}
    if kind == "qup_bmm":
        res["image_ref_rel_err"] = ref_rel_err(base[0], acts[0]["q_nope"], ws[0][0], ws[0][2])
    elif kind == "vup_bmm":
        res["image_ref_rel_err"] = ref_rel_err(base[0].transpose(0, 1), acts[0]["attn_out"], ws[0][1], ws[0][2])
    for name, cfg in cands:
        if name != "image" and cfg is None:
            continue
        try:
            outs = outputs(cfg)
            torch.cuda.synchronize()
            checks = [diff(o, b) for o, b in zip(outs, base)]
            bit = all(c[0] for c in checks)
            maxd = max(c[1] for c in checks)
            us = time_graph([runner(cfg, i) for i in range(nbuf)], args.iters, args.reps)
            res["rows"].append({"name": name, "cfg": cfg if cfg is not None else image_cfg,
                                "us": us, "bitwise_eq_image": bit, "max_abs_diff_vs_image": maxd})
        except Exception as e:  # a bad sweep candidate must not kill the run
            res["rows"].append({"name": name, "cfg": cfg, "error": repr(e)[:300]})
        torch.cuda.synchronize()
    return res


def parse_list(s):
    out = []
    for part in s.split(","):
        if ":" in part:
            a, b, *st = part.split(":")
            out += list(range(int(a), int(b) + 1, int(st[0]) if st else 1))
        elif part:
            out.append(int(part))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--draft-bs", default="8:24", help="draft M = bs (DECODE draft forwards)")
    p.add_argument("--verify-bs", default="8:24", help="verify M = bs * --draft-tokens")
    p.add_argument("--draft-tokens", type=int, default=6)
    p.add_argument("--kinds", default="qup_fused,qup_bmm,vup_bmm")
    p.add_argument("--nbuf", type=int, default=8, help="distinct weight/input sets rotated in the graph")
    p.add_argument("--iters", type=int, default=64, help="calls per captured graph")
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--num-slots", type=int, default=65536)
    p.add_argument("--sweep", action="store_true", help="also time 72 candidate configs per M")
    p.add_argument("--out", required=True, help="JSON result path")
    p.add_argument("--max-used-gib", type=float, default=2.0, help="abort if the GPU already has more VRAM in use")
    args = p.parse_args()

    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 2**30
    print(f"device {torch.cuda.get_device_name()} HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')} "
          f"used {used:.1f} GiB of {total / 2**30:.0f}")
    if used > args.max_used_gib:
        print(f"ABORT: GPU not idle ({used:.1f} GiB in use > {args.max_used_gib})")
        return 2

    gen = torch.Generator(device=DEV)
    gen.manual_seed(1234)
    ws = make_weights(args.nbuf, gen)
    rope = Rope(gen)
    draft_ms = parse_list(args.draft_bs)
    verify_ms = [b * args.draft_tokens for b in parse_list(args.verify_bs)]
    ms = sorted(set(draft_ms + verify_ms))
    results = []
    for kind in args.kinds.split(","):
        for m in ms:
            r = run_case(kind, m, ws, args, gen, rope, args.sweep)
            r["phase"] = ",".join(x for x, ok in (("draft", m in draft_ms), ("verify", m in verify_ms)) if ok)
            results.append(r)
            rows = {x["name"]: x for x in r["rows"]}
            im, pr = rows.get("image"), rows.get("pr209")
            ok_rows = [x for x in r["rows"] if "us" in x and x["bitwise_eq_image"]]
            best = min(ok_rows, key=lambda x: x["us"]) if ok_rows else None
            line = f"{kind:9s} M={m:4d} [{r['phase']:12s}] image {im['us']:7.2f}us {cfg_str(r['image_cfg'])}"
            if pr and "us" in pr:
                line += (f" | pr209 {pr['us']:7.2f}us x{im['us'] / pr['us']:.2f} {cfg_str(pr['cfg'])}"
                         f" bit_eq={pr['bitwise_eq_image']} maxdiff={pr['max_abs_diff_vs_image']:.3g}")
            elif pr:
                line += f" | pr209 ERROR {pr['error']}"
            if args.sweep and best:
                line += f" | best {best['name']} {best['us']:7.2f}us {cfg_str(best['cfg'])}"
            if "image_ref_rel_err" in r:
                line += f" | ref_rel_err {r['image_ref_rel_err']:.3g}"
            print(line, flush=True)
    # per-step estimate: one q-up + one v-up per layer
    summary = {}
    for kind in args.kinds.split(","):
        for name in ("image", "pr209"):
            tot = {}
            for r in results:
                if r["kind"] != kind:
                    continue
                row = next((x for x in r["rows"] if x["name"] == name and "us" in x), None)
                if row:
                    tot[r["M"]] = row["us"]
            summary[f"{kind}/{name}"] = tot
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "results": results, "summary": summary}, f, indent=1, default=str)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
