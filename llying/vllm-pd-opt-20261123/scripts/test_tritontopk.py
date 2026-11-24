#!/usr/bin/env python3
"""GPU check (1 GPU, in the vLLM image) for the Triton decode top-k overlay:
  PYTHONPATH=<dir with vpd_triton_topk_decode.py> VPD_TRITON_TOPK=1 test_tritontopk.py PATCHED_OPS_FILE
- the patched vllm/v1/attention/ops/rocm_aiter_mla_sparse.py imports and binds the Triton module;
- on rows shaped like the decode call (fp32 logits, max_model_len wide, int32 lengths, int32 out with the topk buffer's
  row stride) the result is an exact top-2048 per row, -1 padded, for short rows (< k), long rows and a 700K row.
"""
import importlib.util
import sys

import torch


def main():
    spec = importlib.util.spec_from_file_location("vpd_ops_rocm_aiter_mla_sparse", sys.argv[1])
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m._VPD_TRITON_TOPK is True and hasattr(m, "_vpd_tk")
    tk, k, W = m._vpd_tk, 2048, 1048576
    lens = torch.tensor([1000, 1001, 1002, 1003, 150_000, 150_001, 150_002, 150_003, 700_000, 700_001, 700_002, 700_003,
                         700_000, 700_001, 5_000, 300_000], dtype=torch.int32, device="cuda")
    rows = lens.numel()
    logits = torch.full((rows, W), float("-inf"), device="cuda")
    g = torch.Generator(device="cuda").manual_seed(0)
    for r in range(rows):
        n = int(lens[r])
        x = torch.randn(n, device="cuda", generator=g)
        if r >= 12:  # tie rows: all 0 except a few positives (fewer than k on the long ones), the a11 failure shape
            frac = (0.0015, 0.0015, 0.0, 0.02)[r - 12]
            x = torch.where(torch.rand(n, device="cuda", generator=g) < frac, x.abs() + 0.1, torch.zeros_like(x))
        logits[r, :n] = x
    buf = torch.full((rows, 2 * k), -7, dtype=torch.int32, device="cuda")
    out = buf[:, :k]
    assert tk.supported(logits, k)
    tk.triton_top_k_per_row_decode(logits, lens, out, k)
    torch.cuda.synchronize()
    for r in range(rows):
        n = int(lens[r]); kk = min(n, k)
        sel, pad = out[r, :kk], out[r, kk:]
        assert (pad == -1).all() and (sel >= 0).all() and (sel < n).all() and sel.unique().numel() == kk, r
        kth = torch.topk(logits[r, :n], kk).values[-1]
        assert (logits[r, sel.long()] >= kth).all(), r
    print(f"triton top-k exact on {rows} rows (lengths 1000 .. 700003), out row stride {out.stride(0)}")
    print("triton topk overlay test passed")


if __name__ == "__main__":
    main()
