#!/usr/bin/env python3
"""Structural breakdown of decode verify graphs from compact traces (extract.py output).

Differences from the earlier kdeep.py / kcat.py / step_cats.py:
  * kernels are classified by their POSITION in the layer (state machine over the op order of a GlmMoeDsa decoder
    layer), not only by name, so tuned-GEMM kernel names that change with the batch cannot land in the wrong class;
  * results are keyed by the captured graph batch of the step (busiest rank, rounded up to the capture list), never
    pooled over batches;
  * time spent inside collective kernels (all-gather, reduce-scatter, MegaMoeTP) is split into the part every rank
    needs (minimum over ranks at that layer and step) and the part that is waiting for slower ranks (own - minimum);
  * the local (per-rank) part of each layer is reported for the mean rank and for the slowest rank, since the step
    advances at the pace of the slowest rank between two collectives;
  * steps are aligned across ranks by matching the sequence of verify-graph durations (profiles start within a step
    of each other), and the alignment is printed.

Usage: ana2.py CAPTURE_DIR [--json OUT] [--quiet]
"""
import argparse, bisect, collections, json, os, re, statistics as st, sys
sys.path.insert(0, os.path.dirname(__file__))
import kl

CAPTURE = list(range(1, 9)) + list(range(10, 33, 2)) + list(range(40, 65, 8)) + [96, 128, 160, 192, 224, 256]
N_LAYERS = 78


def graph_bs(raw):
    return CAPTURE[bisect.bisect_left(CAPTURE, raw)]


RX = [(re.compile(p), c) for p, c in [
    (r"mega_moe_tp", "mega_moe"),
    (r"allgather", "comm_ag"),
    (r"reduce_scatter", "comm_rs"),
    (r"cross_device_reduce|ncclDevKernel", "comm_tail"),
    (r"add_rmsnorm_quant", "NORM"),
    (r"fused_qk_rmsnorm", "norm_qk"),
    (r"_fused_fp8_bmm_rope_cat_and_cache_mla|_fused_qk_rope_cat_and_cache_mla", "absorb_kc"),
    (r"_sparse_mla_", "sparse_mla"),
    (r"_batched_gemm_a8w8", "absorb_vc"),
    (r"mqa_logits|logits_hist", "idx_logits"),
    (r"_topk_est|_topk_collect|_topk_select|k_scatter|topk_transform", "idx_topk"),
    (r"Layernorm2dFwd|kn_entry_2c|fast_hadamard|act_quant_kernel|indexer_k_quant|triton_poi_fused_mul|qk_rope_hadamard_quant", "idx_prep"),
    (r"dual_gemv", "idx_gemm"),
    (r"topk_reg|append_shared|opus_moe_sorting|dynamic_per_group_scaled_quant|mxfp4_moe_sort|moe_sort_fill", "moe_prep"),
    (r"gemm1_a4w4|mfma_moe1", "moe_gemm1"),
    (r"gemm2_a4w4", "moe_gemm2"),
    (r"moe_reduction", "moe_red"),
    (r"act_and_mul", "mlp_act"),
    (r"^Cijk_SB|PostGSU", "POSTGSU"),
    (r"^hgemm_|^Cijk_|_gemm_a16_w16|bf16gemm|gemm_a16w16", "GEMM"),
]]
_cache = {}


def base_class(name):
    c = _cache.get(name)
    if c is None:
        c = "OTHER"
        for rx, cls in RX:
            if rx.search(name):
                c = cls
                break
        _cache[name] = c
    return c


def parse_verify(names, kernels):
    """-> (classes per kernel, layers) where layers = list of dicts with local/sync split and flags."""
    out = []
    layers = []
    state = "A0"
    nl = 0
    cur = {"local": 0.0, "sync1": None, "mid": 0.0, "sync2": None, "idx": False, "dense": False, "ffn_i": []}
    last_gemm_cls = None
    for i, (nid, ts, dur, _s) in enumerate(kernels):
        b = base_class(names[nid])
        cls = b
        if state == "T":
            if b == "GEMM":
                cls = "lm_head"; last_gemm_cls = cls
            elif b == "POSTGSU":
                cls = last_gemm_cls or "tail_misc"
            elif b == "NORM":
                cls = "norm_final"
            elif b == "OTHER":
                cls = "tail_misc"
            out.append(cls)
            continue
        if b == "NORM":
            if state == "A0":
                cls = "norm_in"; state = "A1"
            elif state == "O":
                cls = "norm_post"; state = "F"
            else:
                cls = "norm_unexpected"
        elif b == "norm_qk":
            state = "B"
        elif b == "GEMM":
            if state == "A1":
                cls = "qkv_a"
            elif state == "B":
                cls = "q_b"; state = "I"
            elif state == "I":
                cls = "idx_gemm"
            elif state == "O":
                cls = "o_proj"
            elif state == "F":
                cls = "ffn_gemm"; cur["ffn_i"].append(i)
            else:
                cls = "gemm_unexpected"
            last_gemm_cls = cls
        elif b == "POSTGSU":
            cls = last_gemm_cls or "gemm_unexpected"
            if cls == "ffn_gemm":
                cur["ffn_i"].append(i)
        elif b == "absorb_kc":
            state = "S"
        elif b == "absorb_vc":
            state = "O"
        elif b in ("idx_logits",):
            cur["idx"] = True
        elif b == "mlp_act":
            cur["dense"] = True
        elif b == "OTHER":
            cls = "embed_misc" if nl == 0 and state == "A0" else "other"
        out.append(cls)
        # local / collective bookkeeping
        if b in ("comm_ag", "mega_moe"):
            cur["sync1"] = dur
        elif b == "comm_rs":
            cur["sync2"] = dur
        elif cur["sync1"] is None:
            cur["local"] += dur
        else:
            cur["mid"] += dur
        if b in ("comm_rs", "mega_moe"):
            for j in cur["ffn_i"]:
                out[j] = "mlp_gemm" if cur["dense"] else "moe_router"
            cur["mega"] = b == "mega_moe"
            layers.append(cur)
            nl += 1
            cur = {"local": 0.0, "sync1": None, "mid": 0.0, "sync2": None, "idx": False, "dense": False, "ffn_i": []}
            state = "A0" if nl < N_LAYERS else "T"
    return out, layers


def align(ranks):
    """Shift each rank's verify list so that verify durations line up with rank 0. Returns (shifts, n common)."""
    def durs(r):
        return [l["k"][-1][1] + l["k"][-1][2] - l["k"][0][1] for l in r]
    d0 = durs(ranks[0])
    shifts = [0]
    for r in ranks[1:]:
        d = durs(r)
        best = None
        for s in range(-3, 4):
            pairs = [(d0[i], d[i + s]) for i in range(len(d0)) if 0 <= i + s < len(d)]
            if len(pairs) < 10:
                continue
            err = st.mean(abs(a - b) for a, b in pairs)
            if best is None or err < best[0]:
                best = (err, s)
        shifts.append(best[1])
    lo = max(0, max(-s for s in shifts))
    hi = min(len(r) - s for r, s in zip(ranks, shifts))
    hi = min(hi, len(ranks[0]))
    return shifts, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--json")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    files = kl.rank_files(a.capture)
    data = [kl.load(f) for f in files]
    R = len(data)
    V = [[l for l in d["launches"] if l["kind"] == "verify" and l["k"]] for d in data]
    shifts, lo, hi = align(V)
    steps = []  # per aligned step
    unknown = collections.Counter()
    for i in range(lo, hi):
        ls = [V[r][i + shifts[r]] for r in range(R)]
        bs = [l["bs"] for l in ls]
        parsed = [parse_verify(data[r]["names"], ls[r]["k"]) for r in range(R)]
        nl = [len(p[1]) for p in parsed]
        cls_t = [collections.Counter() for _ in range(R)]
        cls_n = [collections.Counter() for _ in range(R)]
        for r in range(R):
            for (nid, ts, dur, _s), c in zip(ls[r]["k"], parsed[r][0]):
                cls_t[r][c] += dur
                cls_n[r][c] += 1
                if c in ("other", "norm_unexpected", "gemm_unexpected", "tail_misc", "embed_misc"):
                    unknown[(c, kl.short(data[r]["names"][nid], 80))] += dur
        sync = collections.Counter()
        if len(set(nl)) == 1:
            for li in range(nl[0]):
                L = [parsed[r][1][li] for r in range(R)]
                loc = [x["local"] for x in L]
                s1 = [x["sync1"] or 0.0 for x in L]
                mid = [x["mid"] for x in L]
                s2 = [x["sync2"] or 0.0 for x in L]
                kind = "mega" if L[0].get("mega") else "ag"
                sync["local_mean"] += st.mean(loc); sync["local_max"] += max(loc)
                sync[f"{kind}_min"] += min(s1); sync[f"{kind}_mean"] += st.mean(s1)
                sync["mid_mean"] += st.mean(mid); sync["mid_max"] += max(mid)
                sync["rs_min"] += min(s2); sync["rs_mean"] += st.mean(s2)
                if L[0]["idx"]:
                    sync["idxlayer_local_mean"] += st.mean(loc); sync["idxlayer_local_max"] += max(loc)
                    sync["idxlayer_n"] += 1
        wall = [l["k"][-1][1] + l["k"][-1][2] - l["k"][0][1] for l in ls]
        steps.append({"bs": bs, "gbs": graph_bs(max(bs)), "wall": wall, "nk": [len(l["k"]) for l in ls],
                      "cls_t": [dict(c) for c in cls_t], "cls_n": [dict(c) for c in cls_n], "sync": dict(sync),
                      "layers_ok": len(set(nl)) == 1 and nl[0] == N_LAYERS, "nl": nl})
    # other graphs + step period on rank 0
    extra = {}
    for kind in ("draft", "draft_extend"):
        t = [l["k"][-1][1] + l["k"][-1][2] - l["k"][0][1] for d in data for l in d["launches"] if l["kind"] == kind and l["k"]]
        n = [len(l["k"]) for d in data for l in d["launches"] if l["kind"] == kind and l["k"]]
        if t:
            extra[kind] = {"ms": st.mean(t) / 1e3, "nk": st.mean(n), "per_step": len(t) / R / max(1, len(V[0]))}
    starts = [l["k"][0][1] for l in V[0]]
    per = [(b - a_) / 1e3 for a_, b in zip(starts, starts[1:])]
    extra["period_ms"] = {"median": st.median(per), "p10": sorted(per)[len(per) // 10], "p90": sorted(per)[len(per) * 9 // 10]}
    # eager kernel time per step, rank 0
    e0 = [k for k in data[0]["eager"] if starts[0] <= k[1] <= starts[-1]]
    extra["eager_ms_per_step"] = sum(k[2] for k in e0) / 1e3 / max(1, len(starts) - 1)
    extra["eager_n_per_step"] = len(e0) / max(1, len(starts) - 1)
    res = {"capture": a.capture, "ranks": R, "shifts": shifts, "n_steps": len(steps), "steps": steps, "extra": extra,
           "unknown": sorted(((k[0], k[1], v) for k, v in unknown.items()), key=lambda x: -x[2])[:40]}
    if a.json:
        os.makedirs(os.path.dirname(a.json), exist_ok=True)
        json.dump(res, open(a.json, "w"))
    if a.quiet:
        return
    report(res)


ORDER = ["norm_in", "qkv_a", "norm_qk", "q_b", "idx_gemm", "idx_prep", "idx_logits", "idx_topk", "absorb_kc", "sparse_mla",
         "absorb_vc", "o_proj", "norm_post", "comm_ag", "moe_router", "moe_prep", "moe_gemm1", "moe_gemm2", "moe_red",
         "mega_moe", "mlp_gemm", "mlp_act", "comm_rs", "norm_final", "lm_head", "comm_tail", "tail_misc", "embed_misc",
         "other", "norm_unexpected", "gemm_unexpected"]


def report(res):
    R = res["ranks"]
    print(f"# {res['capture']}\nranks {R}  aligned steps {res['n_steps']}  shifts {res['shifts']}  "
          f"layers parsed ok in {sum(s['layers_ok'] for s in res['steps'])} steps")
    ex = res["extra"]
    print(f"step period (rank0 verify start->next): median {ex['period_ms']['median']:.2f} ms (p10 {ex['period_ms']['p10']:.2f}, p90 {ex['period_ms']['p90']:.2f}); "
          + "; ".join(f"{k} {v['ms']:.2f} ms x{v['per_step']:.2f}/step ({v['nk']:.0f} kernels)" for k, v in ex.items() if k in ("draft", "draft_extend"))
          + f"; eager {ex['eager_ms_per_step']:.2f} ms/step ({ex['eager_n_per_step']:.0f} kernels)")
    by = collections.defaultdict(list)
    for s in res["steps"]:
        by[s["gbs"]].append(s)
    print("per-step bs across ranks (first 4):", [s["bs"] for s in res["steps"][:4]])
    for g in sorted(by):
        S = by[g]
        n = len(S)
        tot = st.mean(st.mean(s["wall"]) for s in S) / 1e3
        bsm = st.mean(st.mean(s["bs"]) for s in S)
        bmx = st.mean(max(s["bs"]) for s in S)
        nk = st.mean(st.mean(s["nk"]) for s in S)
        print(f"\n## graph bs {g}: {n} steps, mean rank bs {bsm:.1f}, busiest {bmx:.1f} (real rows {100 * bsm / g:.0f}% of padded), verify wall {tot:.2f} ms, {nk:.0f} kernels")
        print(f"   {'class':16s} {'ms/step':>8s} {'%':>6s} {'calls':>7s} {'us/call':>8s} {'max-rank ms':>11s}")
        acc = 0.0
        for c in ORDER:
            t = st.mean(st.mean(s["cls_t"][r].get(c, 0.0) for r in range(R)) for s in S) / 1e3
            if t == 0:
                continue
            k = st.mean(st.mean(s["cls_n"][r].get(c, 0) for r in range(R)) for s in S)
            mx = st.mean(max(s["cls_t"][r].get(c, 0.0) for r in range(R)) for s in S) / 1e3
            acc += t
            print(f"   {c:16s} {t:8.3f} {100 * t / tot:6.1f} {k:7.1f} {1e3 * t / k:8.2f} {mx:11.3f}")
        print(f"   {'(sum)':16s} {acc:8.3f}")
        sy = collections.Counter()
        for s in S:
            for k, v in s["sync"].items():
                sy[k] += v / n
        if sy:
            kind = "mega" if "mega_min" in sy else "ag"
            print(f"   collectives (ms/step): local part mean-rank {sy['local_mean'] / 1e3:.2f} / slowest-rank {sy['local_max'] / 1e3:.2f}; "
                  f"{kind} needed {sy[kind + '_min'] / 1e3:.2f} + waiting {(sy[kind + '_mean'] - sy[kind + '_min']) / 1e3:.2f}; "
                  f"between {sy['mid_mean'] / 1e3:.2f} (slowest {sy['mid_max'] / 1e3:.2f}); rs needed {sy['rs_min'] / 1e3:.2f} + waiting {(sy['rs_mean'] - sy['rs_min']) / 1e3:.2f}")
            if sy.get("idxlayer_n"):
                print(f"   indexer layers ({sy['idxlayer_n']:.0f}/step): local part mean-rank {sy['idxlayer_local_mean'] / 1e3:.2f} ms, slowest-rank {sy['idxlayer_local_max'] / 1e3:.2f} ms")
    if res["unknown"]:
        print("\nunclassified / unexpected (total us over all steps and ranks):")
        for c, nme, v in res["unknown"][:12]:
            print(f"   {v:10.0f}  {c:16s} {nme}")


if __name__ == "__main__":
    main()
