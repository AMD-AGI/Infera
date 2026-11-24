#!/usr/bin/env python3
"""In what order do the decode top-k kernels write their indices? (1 GPU, in the vLLM image)
  topk_output_order.py VPD_DIR
The Triton top-k writes in candidate-arrival order; aiter / vLLM's in-tree kernel may write ascending positions or
descending scores. With the Triton top-k the gates' real acceptance was ~1.2% lower (3.41 / 3.427 / 3.416 vs
3.455-3.477) although the selected sets are exact; if the others are ordered, the order is the suspect.
"""
import sys

import torch

sys.path.insert(0, sys.argv[1])
K, W = 2048, 1048576


def describe(name, out, logits, lens):
    asc = desc_score = 0
    rows = out.shape[0]
    for r in range(rows):
        n = min(int(lens[r]), K)
        idx = out[r, :n].long()
        asc += int(bool((idx[1:] > idx[:-1]).all()))
        sc = logits[r, idx]
        desc_score += int(bool((sc[1:] <= sc[:-1]).all()))
    print(f"{name:>8}: rows with ascending positions {asc}/{rows}, descending scores {desc_score}/{rows}; "
          f"row 0 head {out[0, :8].tolist()}", flush=True)


def main():
    from aiter.ops.topk import top_k_per_row_decode as at
    import vllm._custom_ops  # noqa: F401
    import vpd_triton_topk_decode as tv

    g = torch.Generator(device="cuda").manual_seed(0)
    lens = torch.tensor([14_000, 14_001, 14_002, 14_003, 150_000, 150_001, 150_002, 150_003], dtype=torch.int32, device="cuda")
    logits = torch.full((8, W), float("-inf"), device="cuda")
    for r in range(8):
        logits[r, : int(lens[r])] = torch.randn(int(lens[r]), device="cuda", generator=g)
    for name, fn in (("aiter", lambda o: at(logits, 1, lens, o, 8, logits.stride(0), logits.stride(1), k=K)),
                     ("vllm_c", lambda o: torch.ops._C.top_k_per_row_decode(logits, 1, lens, o, 8, logits.stride(0), logits.stride(1), K)),
                     ("triton_v", lambda o: tv.triton_top_k_per_row_decode(logits, lens, o, K))):
        out = torch.full((8, K), -7, dtype=torch.int32, device="cuda")
        fn(out)
        torch.cuda.synchronize()
        describe(name, out, logits, lens)


if __name__ == "__main__":
    main()
