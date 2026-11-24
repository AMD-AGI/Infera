#!/usr/bin/env python3
"""Which tied elements does each decode top-k pick? (1 GPU, in the vLLM image)
  topk_tie_positions.py VPD_DIR
Rows of length L that are 0 except a few positives (fewer than k = 2048), so k - (#positive) zeros must be chosen
among ~L tied zeros. Prints, per candidate, where the chosen zeros sit in the row (share in the first / last 10%,
median position / L). a14's real acceptance was ~1% below the runs without the Triton top-k; if aiter prefers recent
(late) positions, the tie-guarded Triton select should take ties from the end of the row too.
"""
import sys

import torch

sys.path.insert(0, sys.argv[1])
K, W = 2048, 1048576


def main():
    from aiter.ops.topk import top_k_per_row_decode as at
    import vpd_triton_topk_decode as tv
    import vllm._custom_ops  # noqa: F401

    cands = {
        "aiter": lambda lg, ln, o: at(lg, 1, ln, o, lg.shape[0], lg.stride(0), lg.stride(1), k=K),
        "vllm_c": lambda lg, ln, o: torch.ops._C.top_k_per_row_decode(lg, 1, ln, o, lg.shape[0], lg.stride(0), lg.stride(1), K),
        "triton_v": lambda lg, ln, o: tv.triton_top_k_per_row_decode(lg, ln, o, K),
    }
    g = torch.Generator(device="cuda").manual_seed(0)
    for L in (20_000, 150_000, 700_000):
        lens = torch.full((8,), L, dtype=torch.int32, device="cuda")
        logits = torch.full((8, W), float("-inf"), device="cuda")
        for r in range(8):
            x = torch.zeros(L, device="cuda")
            pos = torch.randperm(L, device="cuda", generator=g)[:300]
            x[pos] = torch.rand(300, device="cuda", generator=g) + 0.1
            logits[r, :L] = x
        for name, fn in cands.items():
            out = torch.full((8, K), -7, dtype=torch.int32, device="cuda")
            fn(logits, lens, out)
            torch.cuda.synchronize()
            z = []
            for r in range(8):
                sel = out[r][out[r] >= 0].long()
                zsel = sel[logits[r, sel] == 0].float() / L
                z.append(zsel)
            zz = torch.cat(z)
            print(f"L {L:>7} {name:>8}: zeros chosen {zz.numel() // 8}/row; first 10% {float((zz < 0.1).float().mean()):.2f}, "
                  f"last 10% {float((zz >= 0.9).float().mean()):.2f}, median pos/L {float(zz.median()):.3f}", flush=True)


if __name__ == "__main__":
    main()
