#!/usr/bin/env python3
"""One line per (capture, captured graph batch) from ana2 JSONs: grouped verify-graph time (ms/step, mean over ranks)."""
import collections, glob, json, os, statistics as st, sys
G = [("moe", ["moe_router", "moe_prep", "moe_gemm1", "moe_gemm2", "moe_red", "mega_moe"]),
     ("comm", ["comm_ag", "comm_rs"]),
     ("dense", ["qkv_a", "q_b", "o_proj"]),
     ("absorb", ["absorb_kc", "absorb_vc"]),
     ("sparse", ["sparse_mla"]),
     ("idx_lg", ["idx_logits"]),
     ("idx_tk", ["idx_topk"]),
     ("idx_ot", ["idx_gemm", "idx_prep"]),
     ("norm", ["norm_in", "norm_qk", "norm_post", "norm_final"]),
     ("mlp", ["mlp_gemm", "mlp_act"]),
     ("tail", ["lm_head", "comm_tail", "tail_misc", "embed_misc", "other", "norm_unexpected", "gemm_unexpected"])]
pat = sys.argv[1] if len(sys.argv) > 1 else "*"
print(f"{'capture':46s} {'gbs':>3s} {'n':>3s} {'bs̄':>5s} {'wall':>6s} " + " ".join(f"{g:>6s}" for g, _ in G) + f" {'wait':>5s} {'nk':>5s} {'drft':>5s} {'dext':>5s} {'eagr':>5s} {'period':>6s}")
for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "../out/ana2", pat + ".json"))):
    r = json.load(open(f)); R = r["ranks"]
    by = collections.defaultdict(list)
    for s in r["steps"]:
        by[s["gbs"]].append(s)
    ex = r["extra"]
    for g in sorted(by):
        S = by[g]
        if len(S) < 3:
            continue
        wall = st.mean(st.mean(s["wall"]) for s in S) / 1e3
        row = []
        for _, cs in G:
            row.append(st.mean(st.mean(sum(s["cls_t"][k].get(c, 0.0) for c in cs) for k in range(R)) for s in S) / 1e3)
        w = 0.0
        for s in S:
            sy = s["sync"]
            kind = "mega" if "mega_min" in sy else "ag"
            w += (sy.get(kind + "_mean", 0) - sy.get(kind + "_min", 0) + sy.get("rs_mean", 0) - sy.get("rs_min", 0)) / len(S) / 1e3
        nk = st.mean(st.mean(s["nk"]) for s in S)
        bsm = st.mean(st.mean(s["bs"]) for s in S)
        print(f"{os.path.basename(f)[:-5][:46]:46s} {g:3d} {len(S):3d} {bsm:5.1f} {wall:6.2f} " + " ".join(f"{x:6.2f}" for x in row)
              + f" {w:5.2f} {nk:5.0f} {ex.get('draft', {}).get('ms', 0):5.2f} {ex.get('draft_extend', {}).get('ms', 0):5.2f} {ex['eager_ms_per_step']:5.2f} {ex['period_ms']['median']:6.2f}")
