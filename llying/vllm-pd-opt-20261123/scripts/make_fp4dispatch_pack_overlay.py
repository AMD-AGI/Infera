#!/usr/bin/env python3
"""Build the FP4-dispatch overlay (same two files as make_fp4dispatch_overlay.py) with one more switch, item 8:
VPD_FP4_DISPATCH_PACK=1 = ONE all-gather per MoE layer instead of four (FP4 hidden, topk_weights, topk_ids, scales).
  make_fp4dispatch_pack_overlay.py NAIVE_ORIG NAIVE_OUT AITER_ORIG AITER_OUT
The base tables (make_fp4dispatch_overlay.NAIVE / AITER) are applied first, unchanged, then PACK on top of the
patched naive_dp_ep.py; rocm_aiter_moe.py is exactly the base overlay's. VPD_FP4_DISPATCH_PACK unset = the base
overlay's behaviour (the added module code only runs under the switch; no Triton import otherwise).

Only inside the base FP4 branch (VPD_FP4_DISPATCH=1, MXFP4 W4A4, deferred quant, no LoRA, 128 < gathered <= 2048),
with bf16 hidden, int32 topk_ids, float32 topk_weights and an all-gather/reduce-scatter all2all manager; otherwise
the base FP4 path runs (reason logged once, "[vpd-fp4pack]").
Send buffer: uint8 [M, ROW] per rank, one row per token (K = hidden, T = topk; K=6144, T=8 -> ROW 3328):
    [0, K/2)            MXFP4 e2m1 pairs        (as aiter per_1x32 quant, shuffle=False)
    [K/2, K/2+K/32)     e8m0 scales, unshuffled
    [IOFF, IOFF+4T)     topk_ids  int32         IOFF = K/2+K/32 rounded up to 4
    [WOFF, WOFF+4T)     topk_weights float32    WOFF = IOFF+4T
    pad to a multiple of 16 B (aiter custom AG needs bytes % 16 == 0 for any M)
  written by ONE Triton kernel (_vpd_fp4_pack_kernel) that quantizes the rows with aiter's own Triton MXFP4 op
  (_mxfp4_quant_op, "even" scale) and stores ids / weights next to them; it replaces the aiter HIP quant launch, so
  the send side has the same kernel count as today. VPD_FP4_DISPATCH_PACK_QUANT=hip keeps aiter's HIP quant and
  the same kernel then only copies q / scales / ids / weights into the row (one extra kernel; fallback if the
  Triton quant is not bit-identical on GPU).
Gather: the buffer viewed as bf16 [M, ROW/2] through the all2all manager's own comm group / sizes / all_gatherv
  (aiter custom AG under FULL graphs, else pynccl), one call instead of the four of AgRsAll2AllManager.dispatch.
Receive (VPD_FP4_DISPATCH_PACK_RECV): view (default) = column-slice views of the gathered [N, ROW] uint8 (row
  stride ROW, no copy, no kernel); split = one Triton kernel copying the four fields into contiguous tensors, for
  the case where an aiter consumer ignores the row stride (scripts/test_fp4dispatch_pack.py moe checks that).
Graph-safe: static shapes per M, no host sync, buffers from the caching allocator as today.
"""
import importlib.util
import os
import sys

sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location(
    "make_fp4dispatch_overlay", os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_fp4dispatch_overlay.py"))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

PACK_ENV = "_VPD_FP4_DISPATCH_PACK = _vpd_os.environ.get(\"VPD_FP4_DISPATCH_PACK\") == \"1\"\n"

PACK_CODE = r'''

_VPD_PACK_QUANT = _vpd_os.environ.get("VPD_FP4_DISPATCH_PACK_QUANT", "triton")
_VPD_PACK_RECV = _vpd_os.environ.get("VPD_FP4_DISPATCH_PACK_RECV", "view")
_VPD_PACK_BLOCK_M = 4
_VPD_PACK_BLOCK_N = 256
_vpd_pack_logged: set = set()


def _vpd_pack_log(key: str, msg: str) -> None:
    _vpd_pack_logged.add(key)
    from vllm.logger import init_logger

    init_logger(__name__).info("[vpd-fp4pack] %s", msg)


if _VPD_FP4_DISPATCH_PACK:
    import triton
    import triton.language as tl

    try:
        from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op
    except ImportError:
        _VPD_PACK_QUANT = "hip"

        @triton.jit
        def _mxfp4_quant_op(x, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP4_QUANT_BLOCK_SIZE, SCALING_MODE: tl.constexpr = 0):
            tl.static_assert(False, "aiter _mxfp4_quant_op not importable")
            return x, x

    @triton.jit
    def _vpd_fp4_pack_kernel(
        x_ptr, q_ptr, s_ptr, ids_ptr, w_ptr, out_ptr, out_i32_ptr, out_f32_ptr,
        M, stride_x, stride_q, stride_s, stride_ids, stride_w,
        K: tl.constexpr, ROW: tl.constexpr, SOFF: tl.constexpr, IOFF: tl.constexpr, WOFF: tl.constexpr,
        TOPK: tl.constexpr, TOPK_P2: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        PREQ: tl.constexpr, SCALING_MODE: tl.constexpr,
    ):
        """One launch per MoE layer: MXFP4 quant of BLOCK_N columns of BLOCK_M rows into the packed rows (or, with
        PREQ, a copy of aiter's HIP-quantized q / scales); the pid_n == 0 programs also store the rows' topk ids
        and weights at their row offsets."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rows = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        rmask = rows < M
        qc = pid_n * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
        sc = pid_n * (BLOCK_N // 32) + tl.arange(0, BLOCK_N // 32)
        if PREQ:
            q = tl.load(q_ptr + rows[:, None] * stride_q + qc[None, :], mask=rmask[:, None])
            s = tl.load(s_ptr + rows[:, None] * stride_s + sc[None, :], mask=rmask[:, None])
        else:
            xc = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            x = tl.load(x_ptr + rows[:, None] * stride_x + xc[None, :], mask=rmask[:, None], other=0.0)
            q, s = _mxfp4_quant_op(x.to(tl.float32), BLOCK_N, BLOCK_M, 32, SCALING_MODE)
        tl.store(out_ptr + rows[:, None] * ROW + qc[None, :], q, mask=rmask[:, None])
        tl.store(out_ptr + rows[:, None] * ROW + SOFF + sc[None, :], s, mask=rmask[:, None])
        if pid_n == 0:
            k = tl.arange(0, TOPK_P2)
            m2 = rmask[:, None] & (k < TOPK)[None, :]
            ids = tl.load(ids_ptr + rows[:, None] * stride_ids + k[None, :], mask=m2)
            w = tl.load(w_ptr + rows[:, None] * stride_w + k[None, :], mask=m2)
            tl.store(out_i32_ptr + rows[:, None] * (ROW // 4) + (IOFF // 4) + k[None, :], ids, mask=m2)
            tl.store(out_f32_ptr + rows[:, None] * (ROW // 4) + (WOFF // 4) + k[None, :], w, mask=m2)

    @triton.jit
    def _vpd_fp4_unpack_kernel(
        in_ptr, hid_ptr, sc_ptr, ids_ptr, w_ptr,
        ROW: tl.constexpr, QB: tl.constexpr, SB: tl.constexpr, IOFF: tl.constexpr, WOFF: tl.constexpr,
        TB: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_E: tl.constexpr,
    ):
        """VPD_FP4_DISPATCH_PACK_RECV=split: one gathered row -> contiguous hidden / scales / ids / weights bytes."""
        row = tl.program_id(0).to(tl.int64)
        pid_n = tl.program_id(1)
        src = in_ptr + row * ROW
        c = pid_n * BLOCK_Q + tl.arange(0, BLOCK_Q)
        tl.store(hid_ptr + row * QB + c, tl.load(src + c, mask=c < QB), mask=c < QB)
        if pid_n == 0:
            e = tl.arange(0, BLOCK_E)
            tl.store(sc_ptr + row * SB + e, tl.load(src + QB + e, mask=e < SB), mask=e < SB)
            tl.store(ids_ptr + row * TB + e, tl.load(src + IOFF + e, mask=e < TB), mask=e < TB)
            tl.store(w_ptr + row * TB + e, tl.load(src + WOFF + e, mask=e < TB), mask=e < TB)


def _vpd_pack_layout(a1: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor):
    """(K, T, K/2, K/32, IOFF, WOFF, ROW) of the packed row, or None when the inputs do not fit it."""
    if (
        a1.dim() != 2
        or a1.dtype != torch.bfloat16
        or a1.stride(-1) != 1
        or a1.shape[1] % _VPD_PACK_BLOCK_N != 0
        or topk_ids.dtype != torch.int32
        or topk_weights.dtype != torch.float32
        or topk_ids.dim() != 2
        or topk_ids.shape != topk_weights.shape
        or topk_ids.shape[0] != a1.shape[0]
        or topk_ids.stride(-1) != 1
        or topk_weights.stride(-1) != 1
    ):
        return None
    k, t = a1.shape[1], topk_ids.shape[1]
    qb, sb = k // 2, k // 32
    ioff = (qb + sb + 3) // 4 * 4
    woff = ioff + 4 * t
    return (k, t, qb, sb, ioff, woff, (woff + 4 * t + 15) // 16 * 16)


def _vpd_pack(a1, topk_ids, topk_weights, lay, quant=None, scaling=0):
    """uint8 [M, ROW]: each rank's rows quantized and packed with their topk ids / weights (one kernel)."""
    k, t, qb, sb, ioff, woff, row = lay
    m = a1.shape[0]
    buf = torch.empty((m, row), dtype=torch.uint8, device=a1.device)
    if m == 0:
        return buf
    if (quant or _VPD_PACK_QUANT) == "hip":
        from aiter import QuantType, get_hip_quant

        q, s = get_hip_quant(QuantType.per_1x32)(a1, shuffle=False)
        q, s, preq = q.view(torch.uint8), s.view(torch.uint8), True
    else:
        q, s, preq = buf, buf, False
    _vpd_fp4_pack_kernel[(triton.cdiv(m, _VPD_PACK_BLOCK_M), k // _VPD_PACK_BLOCK_N)](
        a1, q, s, topk_ids, topk_weights, buf, buf.view(torch.int32), buf.view(torch.float32),
        m, a1.stride(0), q.stride(0), s.stride(0), topk_ids.stride(0), topk_weights.stride(0),
        K=k, ROW=row, SOFF=qb, IOFF=ioff, WOFF=woff, TOPK=t, TOPK_P2=triton.next_power_of_2(t),
        BLOCK_M=_VPD_PACK_BLOCK_M, BLOCK_N=_VPD_PACK_BLOCK_N, PREQ=preq, SCALING_MODE=scaling,
    )
    return buf


def _vpd_unpack(gathered, lay, recv=None):
    """(hidden fp4x2 [N, K/2], scales e8m0 [N, K/32], topk_ids int32 [N, T], topk_weights fp32 [N, T])."""
    k, t, qb, sb, ioff, woff, row = lay
    u8 = gathered.view(torch.uint8)
    assert u8.dim() == 2 and u8.shape[1] == row, (tuple(gathered.shape), row)
    if (recv or _VPD_PACK_RECV) == "split":
        n, dev = u8.shape[0], u8.device
        hid = torch.empty((n, qb), dtype=torch.uint8, device=dev)
        sc = torch.empty((n, sb), dtype=torch.uint8, device=dev)
        ids = torch.empty((n, t), dtype=torch.int32, device=dev)
        w = torch.empty((n, t), dtype=torch.float32, device=dev)
        if n:
            _vpd_fp4_unpack_kernel[(n, triton.cdiv(qb, 1024))](
                u8, hid, sc, ids.view(torch.uint8), w.view(torch.uint8),
                ROW=row, QB=qb, SB=sb, IOFF=ioff, WOFF=woff, TB=4 * t,
                BLOCK_Q=1024, BLOCK_E=triton.next_power_of_2(max(sb, 4 * t)),
            )
    else:
        hid, sc = u8[:, :qb], u8[:, qb : qb + sb]
        ids = u8[:, ioff : ioff + 4 * t].view(torch.int32)
        w = u8[:, woff : woff + 4 * t].view(torch.float32)
    return hid.view(torch.float4_e2m1fn_x2), sc.view(torch.float8_e8m0fnu), ids, w


def _vpd_pack_dispatch(pf, a1, topk_ids, topk_weights):
    """Packed FP4 dispatch: one all-gather; returns prepare()'s result, or None for the unpacked FP4 path."""
    lay = _vpd_pack_layout(a1, topk_ids, topk_weights)
    if lay is None:
        if "layout" not in _vpd_pack_logged:
            _vpd_pack_log("layout", f"not packing: a1 {a1.dtype} {tuple(a1.shape)} stride {a1.stride()}, "
                          f"topk_ids {topk_ids.dtype} {tuple(topk_ids.shape)}, topk_weights {topk_weights.dtype}")
        return None
    mgr = getattr(get_ep_group().device_communicator, "all2all_manager", None)
    if not (hasattr(mgr, "_get_comm_group") and hasattr(mgr, "_get_sizes")):
        if "mgr" not in _vpd_pack_logged:
            _vpd_pack_log("mgr", f"not packing: all2all manager {type(mgr).__name__}")
        return None
    if "on" not in _vpd_pack_logged:
        _vpd_pack_log("on", f"one all-gather per MoE layer: row {lay[6]} B = FP4 {lay[2]} + scales {lay[3]} + "
                      f"ids@{lay[4]} / weights@{lay[5]} ({lay[1]} x 4 B each), quant {_VPD_PACK_QUANT}, "
                      f"recv {_VPD_PACK_RECV}, M {a1.shape[0]}")
    buf = _vpd_pack(a1, topk_ids, topk_weights, lay)
    with pf._comm_region():
        group = mgr._get_comm_group(pf.is_sequence_parallel)
        sizes = mgr._get_sizes(a1.shape[0], group)
        assert sizes[group.rank_in_group] == a1.shape[0]
        gathered = group.all_gatherv(buf.view(torch.bfloat16), dim=0, sizes=sizes)
    a1q, a1q_scale, topk_ids, topk_weights = _vpd_unpack(gathered, lay)
    return a1q, a1q_scale, None, topk_ids, topk_weights
'''

PACK = [
    (base.ENV, base.ENV + PACK_ENV),
    ("    return 128 < total <= 2048\n", "    return 128 < total <= 2048\n" + PACK_CODE),
    ("            q, s = get_hip_quant(QuantType.per_1x32)(a1, shuffle=False)\n",
     "            if _VPD_FP4_DISPATCH_PACK:\n"
     "                packed = _vpd_pack_dispatch(self, a1, topk_ids, topk_weights)\n"
     "                if packed is not None:\n"
     "                    return packed\n"
     "            q, s = get_hip_quant(QuantType.per_1x32)(a1, shuffle=False)\n"),
]


def apply(s, reps, name):
    for old, new in reps:
        n = s.count(old)
        if n != 1:
            sys.exit(f"expected exactly one {old.strip()[:60]!r} in {name}, found {n}")
        s = s.replace(old, new)
    return s


def main():
    naive_src, naive_out, aiter_src, aiter_out = sys.argv[1:5]
    naive = apply(apply(open(naive_src).read(), base.NAIVE, naive_src), PACK, naive_src + " (FP4-patched)")
    open(naive_out, "w").write(naive)
    print(f"wrote {naive_out}")
    open(aiter_out, "w").write(apply(open(aiter_src).read(), base.AITER, aiter_src))
    print(f"wrote {aiter_out}")


if __name__ == "__main__":
    main()
