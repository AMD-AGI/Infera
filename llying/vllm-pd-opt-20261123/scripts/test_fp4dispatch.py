#!/usr/bin/env python3
"""GPU check (1 GPU, in the vLLM image) of the pieces the FP4-dispatch overlay relies on:
  test_fp4dispatch.py PATCHED_NAIVE_DP_EP_PY
- the patched module imports and sees VPD_FP4_DISPATCH;
- aiter's per-1x32 quant (shuffle=False) gives packed FP4 [M, K/2] of dtype torch.float4_e2m1fn_x2 (what
  rocm_aiter_fused_experts recognises as pre-quantized) and e8m0 scales with >= M rows of K/32;
- the BF16 views used for the all-gather round-trip bit-exactly (also after a torch.cat over 4 "ranks", which is what
  the gather does), and are what the aiter custom all-gather accepts (BF16, contiguous, even widths).
"""
import importlib.util
import os
import sys

import torch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    os.environ["VPD_FP4_DISPATCH"] = "1"
    m = load("vpd_naive_dp_ep", sys.argv[1])
    assert m._VPD_FP4_DISPATCH is True
    assert m._vpd_fp4_tokens_ok(64) is False  # no forward context -> keep BF16 dispatch
    ram = load("vpd_rocm_aiter_moe", sys.argv[2])
    from types import SimpleNamespace

    stub = SimpleNamespace(moe_config=SimpleNamespace(use_mori_epv1_kernels=False, hidden_dim=6144),
                           quant_config=SimpleNamespace(use_mxfp4_w4a4=True))
    ws = lambda K: ram.AiterExperts.workspace_shapes(stub, 256, 1024, K, 9, 257, 257, None, None)[2]
    assert ws(3072) == (256, 6144) and ws(6144) == (256, 6144), (ws(3072), ws(6144))
    stub.quant_config.use_mxfp4_w4a4 = False
    assert ws(6144) == (256, 6144)
    print("workspace output width: packed FP4 K=3072 -> 6144, BF16 K=6144 -> 6144, non-MXFP4 unchanged")
    from aiter import QuantType, get_hip_quant

    quant = get_hip_quant(QuantType.per_1x32)
    for M in (1, 6, 64, 256):
        x = torch.randn(M, 6144, dtype=torch.bfloat16, device="cuda") * 3
        q, s = quant(x, shuffle=False)
        assert q.dtype == torch.float4_e2m1fn_x2 and tuple(q.shape) == (M, 3072), (q.dtype, q.shape)
        assert s.shape[0] >= M and s.shape[1] == 6144 // 32, s.shape
        s = s[:M].contiguous()
        parts_q = [q.view(torch.bfloat16)] * 4
        parts_s = [s.view(torch.bfloat16)] * 4
        gq, gs = torch.cat(parts_q), torch.cat(parts_s)
        assert gq.is_contiguous() and gq.shape == (4 * M, 1536) and gs.shape == (4 * M, 96)
        bq, bs = gq.view(q.dtype), gs.view(s.dtype)
        assert torch.equal(bq[:M].view(torch.uint8), q.view(torch.uint8))
        assert torch.equal(bs[3 * M:].view(torch.uint8), s.view(torch.uint8))
        print(f"M={M}: fp4 {tuple(q.shape)} {q.dtype}, scales {tuple(s.shape)} {s.dtype}; bf16 views round-trip")
    print("fp4 dispatch test passed")


if __name__ == "__main__":
    main()
