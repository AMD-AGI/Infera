#!/usr/bin/env python3
"""KF sparse MLA decode with a > 2 GiB KV pool, fixes against the production copy: "split" reads the pool as two
32-bit-addressable halves (kf_sparse_mla_split2g.py, make_kf_sparse_split2g.py); every further argument V loads
kf_sparse_mla_V.py (e.g. u32off from make_kf_sparse_u32off.py, nm_u32_and from make_kf_sparse_nomask.py).
  correctness: outputs bit-compared with indices over the whole pool, packed around the split row, and including the
               first / last slot and -1 (masked) entries;
  time:        CUDA-graph us per call at the TP4 (3.09M, < 2 GiB, unchanged path) and TP8 (4.09M) pool sizes;
  ISA:         buffer_load / global_load (scalar base vs 64-bit per-lane address) counts and registers of every compiled
               split-K kernel variant.

Usage (production image, overlay at /yihou_overlay): python3 sparse_split2g.py [V ...]
"""
import importlib.util
import sys

import torch

sys.path.insert(0, "/perf_apps/liyingli/bench_agentx/kopt-20261109/bench")
import sparse_bench as sb  # noqa: E402
import sparse_probe as sp  # noqa: E402

FP8 = torch.float8_e4m3fn
POOLS = (3093376, 4094720)
ROWS = (60, 72, 88, 96)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def isa(mod):
    out = set()
    for caches in mod._sparse_mla_split_k_kernel.device_caches.values():
        for ck in caches[0].values():
            lines = [ln for ln in ck.asm.get("amdgcn", "").splitlines() if "global_load" in ln]
            out.add((ck.asm.get("amdgcn", "").count("buffer_load"), sum(", s[" in ln for ln in lines),
                     sum(ln.rstrip().endswith("off") or ", off" in ln for ln in lines), getattr(ck, "n_regs", "?")))
    return sorted(out)


def boundary_indices(rows, pool, split, g):
    """Every row: half its slots within +-4096 of the split row, the rest over the pool, plus 0, pool - 1 and -1."""
    near = torch.randint(max(split - 4096, 0), min(split + 4096, pool), (rows, 1, sb.TOPK), device="cuda",
                         dtype=torch.int32, generator=g)
    far = torch.randint(0, pool, (rows, 1, sb.TOPK), device="cuda", dtype=torch.int32, generator=g)
    idx = torch.where(torch.rand(rows, 1, sb.TOPK, device="cuda", generator=g) < 0.5, near, far)
    idx[:, :, 0] = 0
    idx[:, :, 1] = pool - 1
    idx[:, :, 2] = split - 1
    idx[:, :, 3] = split
    idx[:, :, -7:] = -1
    return idx


def main():
    orig = load("/yihou_overlay/yihou_kf/kf_sparse_mla.py", "kf_orig")
    bench = "/perf_apps/liyingli/bench_agentx/kopt-20261109/bench/"
    new = load(bench + "kf_sparse_mla_split2g.py", "kf_split2g")
    cands = {"split": new}
    for v in sys.argv[1:]:
        cands[v] = load(bench + f"kf_sparse_mla_{v}.py", f"kf_{v}")
    g = torch.Generator(device="cuda").manual_seed(0)
    ok = True
    for pool in POOLS:
        kv = (torch.randn(pool, 1, sb.DV + sb.DR, device="cuda", dtype=torch.bfloat16, generator=g) * 0.5).to(FP8)
        split = new._kv_split_rows(kv)
        print(f"pool {pool} ({pool * 576 / 2**30:.3f} GiB): split row {split}", flush=True)
        for rows in ROWS:
            qn = (torch.randn(rows, sb.H, sb.DV, device="cuda", generator=g) * 0.5).to(FP8)
            qr = (torch.randn(rows, sb.H, sb.DR, device="cuda", generator=g) * 0.5).to(FP8)
            sets = {"uniform": sb.make_indices(rows, pool, "uniform", g),
                    "request": sb.make_indices(rows, pool, "request", g)}
            if split:
                sets["boundary"] = boundary_indices(rows, pool, split, g)
            checks = []
            for name, idx in sets.items():
                ro = orig.triton_sparse_mla_fwd(qn, qr, kv, idx, 0.0625, sb.DV)
                for cname, cm in cands.items():
                    rn = cm.triton_sparse_mla_fwd(qn, qr, kv, idx, 0.0625, sb.DV)
                    same = torch.equal(ro, rn)
                    ok &= same
                    checks.append(f"{cname}/{name} "
                                  + ("ok" if same else "DIFF %.3g" % (ro.float() - rn.float()).abs().max().item()))
            timing = [sb.make_indices(rows, pool, "request", g) for _ in range(8)]
            t_o = sp.timed(orig, qn, qr, kv, timing)
            ts = {cname: sp.timed(cm, qn, qr, kv, timing) for cname, cm in cands.items()}
            print(f"  rows {rows:3d}: production {t_o:5.1f} us, "
                  + ", ".join(f"{c} {t:5.1f} us ({100 * (t / t_o - 1):+5.1f}%)" for c, t in ts.items())
                  + "; bit-exact: " + " ".join(checks), flush=True)
        print("  split-K variants so far (buffer_load, global_load scalar-base, global_load 64-bit, regs): "
              f"production {isa(orig)}; " + "; ".join(f"{c} {isa(cm)}" for c, cm in cands.items()), flush=True)
        del kv
        torch.cuda.empty_cache()
    for cname, cm in [("production", orig)] + list(cands.items()):
        for caches in cm._sparse_mla_split_k_kernel.device_caches.values():
            for i, ck in enumerate(caches[0].values()):
                with open(f"{bench}out/sparse_asm_{cname}_{i}.s", "w") as f:
                    f.write(ck.asm.get("amdgcn", ""))
    print("ALL IDENTICAL" if ok else "MISMATCH", flush=True)


if __name__ == "__main__":
    main()
