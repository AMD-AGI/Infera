#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Per-M Triton configs for the MLA absorbed batched GEMMs of GLM-5.2 decode.

ATOM's MLA decode runs aiter's batched_gemm_a8w8_a_per_token_group_prequant_w_
per_batched_tensor_quant twice per layer over all 64 heads of the DCP group:
the q up-projection (N=512, K=192) and the v up-projection (N=256, K=512), with
M = query rows (4 x graph size in the MTP verify, the graph size in a draft).
Its gfx950 configs cover only the standard M buckets, and from M_LEQ_64 up they
use BLOCK_SIZE_M 64: any M that is not a multiple of 64 -- graph 36, 40, 44
(M 144, 160, 176), every draft from 36 up -- takes the masked path at 2-2.5x
the time of a neighbouring multiple of 64 (q-up 19.1 us at M 160 vs 10.6 at
192). That is most of the irregular step-time staircase between graph sizes.

This adds an M_BOUNDS list with one bucket per decode M (multiples of 4 up to
64, of 16 up to 512) and the fastest config for each from a sweep on MI355X
(72 candidates, CUDA-graph timing, output checked against the stock config):
at graph 40 q-up 19.1 -> 8.4 us and v-up 15.2 -> 8.1 us, ~1.4 ms per decode
step over 78 layers. The stock M_GEQ_*/any entries still serve M > 512.

Written against aiter 9205e1059 in rocm/atom-dev:nightly_202610131505. Strict:
a file that no longer holds the stock buckets fails the build.
"""

import importlib.util
import json
import os
import sys

TAG = "aiter-patch mla-bmm-configs"
CFG_DIR = (
    "ops/triton/configs/gfx950/triton/gemm/"
    "batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant"
)
NAME = "BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT"
# The stock M_LEQ_256 entry of both files; checked before anything is written.
STOCK_LEQ_256 = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 256,
    "GROUP_SIZE_M": 1,
    "num_warps": 8,
    "num_stages": 2,
    "waves_per_eu": 1,
    "matrix_instr_nonkdim": 16,
    "cache_modifier": ".cg",
}
# (N, K): [(M, BLOCK_SIZE_M, BLOCK_SIZE_N, num_warps, waves_per_eu, cache_modifier)]
TUNED = {
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
# Above 512 the stock M_GEQ_* / any entries keep serving (prefill sizes).
UPPER_BOUNDS = [1024, 2048, 4096, 8192]


def main() -> int:
    spec = importlib.util.find_spec("aiter")
    if spec is None or not spec.submodule_search_locations:
        print(f"[{TAG}] FAILED: package 'aiter' not found")
        return 1
    root = os.path.join(list(spec.submodule_search_locations)[0], CFG_DIR)
    pending = {}
    for (n, k), rows in TUNED.items():
        path = os.path.join(root, f"{NAME}-N={n}-K={k}.json")
        if not os.path.isfile(path):
            print(f"[{TAG}] FAILED: {path} not found")
            return 1
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        bounds = [m for m, *_ in rows] + UPPER_BOUNDS
        if cfg.get("M_BOUNDS") == bounds:
            continue  # already applied
        if "M_BOUNDS" in cfg or cfg.get("M_LEQ_256") != STOCK_LEQ_256:
            print(f"[{TAG}] FAILED: {path} is not the stock config")
            return 1
        for key in [key for key in cfg if key.startswith("M_LEQ_")]:
            del cfg[key]
        cfg["M_BOUNDS"] = bounds
        for m, bm, bn, warps, waves, cache in rows:
            cfg[f"M_LEQ_{m}"] = {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "GROUP_SIZE_M": 1,
                "num_warps": warps,
                "num_stages": 2,
                "waves_per_eu": waves,
                "matrix_instr_nonkdim": 16,
                "cache_modifier": cache,
            }
        pending[path] = cfg
    for path, cfg in pending.items():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4)
            f.write("\n")
    print(f"[{TAG}] {'patched ' + ', '.join(pending) if pending else 'already applied'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
