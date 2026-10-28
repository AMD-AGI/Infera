"""Measurement driver: GLM-5.2-MXFP4 decode-leg routed MoE, a4w4 (TARGET_VERIFY shape).

forge-loop contract (stdout only):
  * correctness  ``python driver.py``                -> case SNRs + ``SNR: <min> dB``
  * benchmark    ``python driver.py --bench-mode``   -> ``wall_ms`` samples, one ``case_ms`` per case
                                                        (bs8/bs14 ``unscored``), ``median_ms`` aggregate
  * profiling    ``python driver.py --profile-run``  -> only the target op, bs12 case

Production call (sglang v0.5.21 Quark MXFP4 MoE -> aiter runner, see program.md):
  aiter.fused_moe(hidden_states, w13, w2, topk_weights, topk_ids, quant_type=per_1x32, activation=Silu,
                  gate_mode="separated", moe_sorting_dispatch_policy=2, w1_scale=..., w2_scale=...)
Shapes per rank (TP4 / attention-DP4 / EP1): M = 4 ranks x 6 verify tokens x bs_graph = 24*bs rows,
hidden 6144, 257 experts (256 routed + fused shared id 256), top-9, per-rank intermediate 512.
Weights: random bf16 quantized with aiter's MXFP4 quantizer (== Quark 'even' scale + RNE, the checkpoint
format), then pre-shuffled once by the kernel file's prepare_weights() (weight-load time, not timed).
Routing: topk_ids / topk_weights from the GLM workset blobs (data/). Activations: random bf16.

Timing: one graph replay runs the op over N_SETS distinct weight sets (1.29 GB each, all 257 experts touched
per call), so no call can find its weights in the 256 MB MALL -- as in the model, where 75 MoE layers and
everything between them stream through. case_ms = replay time / N_SETS.
Oracle: fp32 MoE on the dequantized weights with MXFP4 quant-dequant of both GEMM inputs, using aiter's
measured rounding rules (program.md "Oracle adaptations").

This file is the measurement oracle: forge must not edit it.
"""
from __future__ import annotations

import os

# ---- deployment environment (must precede any aiter import) ---------------------------------------------
os.environ.setdefault("SGLANG_USE_AITER", "1")
os.environ.setdefault("AITER_USE_FLYDSL_MOE_SORTING", "1")          # set in the GLM-5.2 deployment
# Non-root container: aiter would re-merge tuned_fmoe.csv into root-owned REDACTED_LOCAL_PATH and fail.
# The image's pre-merged copy is row-for-row identical to a fresh merge (program.md), so point at it.
os.environ.setdefault("AITER_CONFIG_FMOE", "REDACTED_LOCAL_PATH")
# The image's prebuilt FlyDSL cache is root-only (mode 600); compile into a writable node-local cache.
os.environ.setdefault("FLYDSL_RUNTIME_CACHE_DIR", "REDACTED_LOCAL_PATH")

import argparse  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from aiter.ops.triton.quant import dynamic_mxfp4_quant  # noqa: E402

from graph_harness import cuda_graph_bench  # noqa: E402
from moe_mxfp4_kernel import glm52_moe_mxfp4, prepare_weights  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
E, H, I, TOPK = 257, 6144, 512, 9
N_SETS = 4
W13_STD, W2_STD = 1.0 / math.sqrt(H), 1.0 / math.sqrt(I)
# bs_graph -> P(bs_graph) (GLM workset batch model, renormalized over the exported points).
CASES = {8: 0.019908, 10: 0.423407, 12: 0.487302, 14: 0.069383}
SCORED = {10, 12}            # 91 % of the probability mass; 8 and 14 are reported unscored
PROFILE_BS = 12
SEED = 20261020
FP4_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

torch.backends.cuda.matmul.allow_tf32 = False


def case_id(bs: int) -> str:
    return f"T{6 * bs}_bs{bs}"


# ---- weights ---------------------------------------------------------------------------------------------
def make_weight_set(k: int, device: str):
    """Random bf16 weights -> MXFP4 (packed e2m1 + e8m0) with aiter's quantizer. Returns logical layout."""
    g = torch.Generator(device=device).manual_seed(SEED + 1000 * k)
    out = []
    for rows, cols, std in ((E * 2 * I, H, W13_STD), (E * H, I, W2_STD)):
        w = (torch.randn(rows, cols, device=device, generator=g) * std).to(torch.bfloat16)
        q, s = dynamic_mxfp4_quant(w)
        del w
        out.append(q.contiguous().view(E, rows // E, cols // 2))
        out.append(s.contiguous().view(E, rows // E, cols // 32))
    return tuple(out)          # w13 [E,2I,H/2], w13_s [E,2I,H/32], w2 [E,H,I/2], w2_s [E,H,I/32]


def make_case(bs: int, variant: int, device: str):
    m = 24 * bs
    d = load_file(os.path.join(DATA, f"moe_mxfp4_routed_topk9_e257_h6144_i512_bs{bs}_verify.safetensors"))
    ids = d["topk_ids"].to(device=device, dtype=torch.int32).contiguous()
    wts = d["topk_weights"].to(device=device, dtype=torch.float32).contiguous()
    assert ids.shape == (m, TOPK) and wts.shape == (m, TOPK), (ids.shape, wts.shape)
    g = torch.Generator(device=device).manual_seed(SEED + bs * 10 + variant)
    x = torch.randn(m, H, device=device, generator=g).to(torch.bfloat16)
    return x, ids, wts


def run_kernel(x, ids, wts, prepped):
    w13, w2, w13_s, w2_s = prepped
    return glm52_moe_mxfp4(x, w13, w2, w13_s, w2_s, wts, ids)


# ---- oracle ----------------------------------------------------------------------------------------------
def dequant_mxfp4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """packed uint8 [..., K/2] (low nibble = even element, bit 3 = sign); scales uint8 UE8M0 [..., K/32]."""
    pos = torch.tensor(FP4_GRID, dtype=torch.float32, device=packed.device)
    lut = torch.cat([pos, -pos])
    b = packed.view(torch.uint8).to(torch.int64)
    vals = torch.stack([lut[b & 0xF], lut[b >> 4]], dim=-1).flatten(-2)
    s = torch.pow(2.0, scales.view(torch.uint8).to(torch.float32) - 127.0)
    return vals * s.repeat_interleave(32, dim=-1)


def quant_dequant_mxfp4(x: torch.Tensor) -> torch.Tensor:
    """aiter's dynamic activation MXFP4 quant (measured bit-exact against the HIP kernel, program.md):
    scale = 2**ceil(log2(amax/6)) (smallest power of two with amax/scale <= 6, never clips),
    elements = round-to-nearest-even onto the E2M1 grid."""
    shape = x.shape
    g = x.float().reshape(-1, 32)
    amax = g.abs().amax(dim=-1, keepdim=True)
    e = torch.ceil(torch.log2(amax / 6.0)).clamp(-127.0, 127.0)
    e = torch.where(amax > 0, e, torch.full_like(e, -127.0))
    scale = torch.pow(2.0, e)
    q = (g / scale).clamp(-6.0, 6.0)
    a = q.abs()
    grid = torch.tensor(FP4_GRID, dtype=torch.float32, device=x.device)
    d = (a.unsqueeze(-1) - grid).abs()
    cand = d == d.min(dim=-1, keepdim=True).values
    even = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.bool, device=x.device)   # 0, 1, 2, 4
    pick = torch.where((cand.sum(-1) > 1).unsqueeze(-1), cand & even, cand)
    r = grid[pick.to(torch.float32).argmax(dim=-1)]
    return (r * q.sign() * scale).reshape(shape)


@torch.no_grad()
def reference(x, ids, wts, w13, w13_s, w2, w2_s) -> torch.Tensor:
    """fp32 routed MoE, 9 = 8 routed + fused shared (id 256); w13 rows [0,I) gate, [I,2I) up;
    out = sum_k w_k * W2_e(q(silu(gate) * up)), q = MXFP4 quant-dequant; x is MXFP4-quantized too."""
    xq = quant_dequant_mxfp4(x)
    out = torch.zeros(x.shape[0], H, dtype=torch.float32, device=x.device)
    ids64 = ids.to(torch.int64)
    for e in torch.unique(ids64).tolist():
        rows, slots = torch.nonzero(ids64 == e, as_tuple=True)
        W13 = dequant_mxfp4(w13[e], w13_s[e])          # [2I, H]
        W2 = dequant_mxfp4(w2[e], w2_s[e])             # [H, I]
        g1 = xq[rows] @ W13.t()
        act = torch.nn.functional.silu(g1[:, :I]) * g1[:, I:]
        o = quant_dequant_mxfp4(act) @ W2.t()
        out.index_add_(0, rows, o * wts[rows, slots].unsqueeze(1))
    return out


def snr_db(ref: torch.Tensor, test: torch.Tensor) -> float:
    ref, test = ref.float(), test.float()
    noise = torch.mean((test - ref) ** 2).item()
    signal = torch.mean(ref ** 2).item()
    if not math.isfinite(noise):
        return -100.0
    if noise <= 0.0:
        return 100.0
    return 10.0 * math.log10(signal / noise) if signal > 0 else 0.0


# ---- modes -----------------------------------------------------------------------------------------------
def correctness(device: str) -> int:
    raw = make_weight_set(0, device)
    prepped = prepare_weights(*raw)
    snrs = []
    for bs in CASES:
        for variant in (0, 1):
            x, ids, wts = make_case(bs, variant, device)
            out = run_kernel(x, ids, wts, prepped)
            assert out.shape == (24 * bs, H) and out.dtype == torch.bfloat16, (out.shape, out.dtype)
            s = snr_db(reference(x, ids, wts, *raw), out)
            snrs.append(s)
            print(f"case_snr: {case_id(bs)}_x{variant} {s:.2f}")
    print(f"SNR: {min(snrs):.2f} dB")
    print(f"allclose: {min(snrs) >= 30.0}")
    return 0


def bench(warmup: int, iters: int, device: str) -> int:
    sets = [prepare_weights(*make_weight_set(k, device)) for k in range(N_SETS)]
    torch.cuda.synchronize()
    weighted = 0.0
    for bs, p in CASES.items():
        x, ids, wts = make_case(bs, 0, device)
        ref_out = run_kernel(x, ids, wts, sets[-1]).clone()
        captured: dict = {}

        def step() -> None:
            for prepped in sets:                        # one call per weight set ("layer")
                captured["out"] = run_kernel(x, ids, wts, prepped)

        # stage2 accumulates with bf16 atomics (order is not deterministic): verify by SNR, not equality
        res = cuda_graph_bench(step, warmup=warmup, iters=iters,
                               dirty=lambda: captured["out"].zero_(),
                               verify=lambda: snr_db(ref_out, captured["out"]) > 30.0)
        times = sorted(t / N_SETS for t in res["times_ms"])
        print(f"# {case_id(bs)} mode: {res['mode']}")
        for t in times:
            print(f"wall_ms: {t:.6f}")
        med = times[len(times) // 2]
        weighted += p * med
        tag = "" if bs in SCORED else " unscored"
        print(f"case_ms: {case_id(bs)} {med:.6f}{tag}")
    print(f"median_ms: {weighted:.6f}")
    return 0


def profile(device: str) -> int:
    prepped = prepare_weights(*make_weight_set(0, device))
    x, ids, wts = make_case(PROFILE_BS, 0, device)
    for _ in range(6):
        run_kernel(x, ids, wts, prepped)
    torch.cuda.synchronize()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="GLM-5.2 MXFP4 MoE forge driver")
    ap.add_argument("--bench-mode", action="store_true")
    ap.add_argument("--profile-run", action="store_true")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    args, _ = ap.parse_known_args()
    if not torch.cuda.is_available():
        print("error: no GPU available")
        return 1
    device = "cuda"
    if args.profile_run:
        return profile(device)
    if args.bench_mode:
        return bench(args.warmup, args.iters, device)
    return correctness(device)


if __name__ == "__main__":
    sys.exit(main())
