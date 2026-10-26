#!/usr/bin/env python3
"""gen_mlacfg (task llrepro 2026-10-26): write PR #209's per-M MLA bmm configs (TUNED table, M_BOUNDS) into COPIES of
the aiter gfx950 config JSONs: the two batched files (as the PR does) and the fused bmm+rope+cache N512K192 file (our
decode q-up path). Run inside the image (CPU only); reads the image files, writes only to --out. Same bucket logic as
the PR's main(); refuses non-stock inputs."""
import importlib.util, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pr209_patch as P
out = sys.argv[1]
root = os.path.join(list(importlib.util.find_spec("aiter").submodule_search_locations)[0], "ops/triton/configs/gfx950/triton/gemm")
files = [(os.path.join(root, P.CFG_DIR.split("gemm/")[1], f"{P.NAME}-N={n}-K={k}.json"), (n, k)) for (n, k) in P.TUNED]
files.append((os.path.join(root, "fused_fp8_bmm_rope_cat_and_cache_mla", "FUSED-FP8-BMM-ROPE-CAT-AND-CACHE-MLA-N=512-K=192.json"), (512, 192)))
for path, nk in files:
    cfg = json.load(open(path))
    if "M_BOUNDS" in cfg or cfg.get("M_LEQ_256") != P.STOCK_LEQ_256:
        print("NOT STOCK", path, cfg.get("M_LEQ_256")); sys.exit(1)
    rows = P.TUNED[nk]
    for key in [k for k in cfg if k.startswith("M_LEQ_")]:
        del cfg[key]
    cfg["M_BOUNDS"] = [m for m, *_ in rows] + P.UPPER_BOUNDS
    for m, bm, bn, w, wv, c in rows:
        cfg[f"M_LEQ_{m}"] = {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "GROUP_SIZE_M": 1, "num_warps": w, "num_stages": 2,
                             "waves_per_eu": wv, "matrix_instr_nonkdim": 16, "cache_modifier": c}
    dst = os.path.join(out, os.path.basename(path))
    json.dump(cfg, open(dst, "w"), indent=4); open(dst, "a").write("\n")
    print("wrote", dst, "<-", path)
