#!/usr/bin/env python3
"""GPU test (1 GPU, gfx950, in the vLLM image) of the Triton PREFILL top-k (vpd_triton_topk_prefill.py) and its
overlay hooks (make_tritontopk_prefill_overlay.py).
  PYTHONPATH=<dir with vpd_triton_topk_{decode,prefill}.py> VPD_TOPK_PREFILL=1 \
      test_tritontopk_prefill.py MODE [OPS [MXFP4_OPS]]   (hooks / off: the patched modules, as vllm module names when
      mounted over the image's files, or as .py paths)
MODE
  hooks  the patched files import with the switch on and bind the kernel module; each hook gives an exact top-2048 on
         a prefill-shaped call (MXFP4: lengths only; FP8: [ks, ke) bounds) and declines what it does not cover
         (compressed / candidate layers, int64 bounds, wrong out width) -- run with VPD_TOPK_PREFILL=1;
  off    the same files with VPD_TOPK_PREFILL unset: they import, the kernel module is not imported, every hook
         call returns False (behaviour unchanged);
  exact  exactness against torch.topk on prefill shapes: rows 1 .. 4096, row lengths 2K .. 1M (causal-cached,
         fresh-prompt, two requests, FP8-style [ks, ke) bounds, lengths around k), distributions randn / AR(0.995) /
         segment-ReLU / sparse with exact-0 ties / quantized (few distinct values) / all-equal / hot tail / shared
         rows / mostly -inf; garbage (+1e30) outside every row's bounds; prints the per-path row counts (the
         whole-row fallback must stay rare);
  quick  a subset of exact (~1 min), for the job scripts before serving;
  time   time against aiter's one-block kernels on full-budget prefill shapes (rows x width = 128M fp32, the
         VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=512 chunk): top_k_per_row_decode (what the MXFP4 path runs) and
         top_k_per_row_prefill (the FP8 path), median of N calls with a 1 GiB write between calls (the logits do
         not stay in the 256 MB MALL); also a small collect BLOCK / num_warps sweep.
Exit status 1 on any failure; the last line says "... passed" otherwise.
"""
import importlib.util
import os
import sys

import torch

K = 2048
DEV = "cuda"
GARBAGE = 1e30


def tkp():
    import vpd_triton_topk_prefill as m

    return m


# ---------------------------------------------------------------------------------------------------------------- data
def gen(dist: str, rows: int, W: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn((rows, W), device=DEV, generator=g)
    if dist == "randn":
        return x
    if dist == "ar":  # AR(1), rho 0.995, in 1024-element blocks with a random level each (vectorised)
        Wp = -(-W // 1024) * 1024
        e = torch.randn((rows, Wp // 1024, 1024), device=DEV, generator=g) * (1 - 0.995**2) ** 0.5
        w = 0.995 ** torch.arange(1024, device=DEV, dtype=torch.float32)
        c0 = torch.randn((rows, Wp // 1024, 1), device=DEV, generator=g)
        y = w * (torch.cumsum(e / w, -1) + c0)
        return y.reshape(rows, Wp)[:, :W].contiguous()
    if dist == "segrelu":  # 2K-token segments with a random level, ReLU (many exact zeros)
        lvl = torch.randn((rows, -(-W // 2048)), device=DEV, generator=g).repeat_interleave(2048, 1)[:, :W]
        return torch.relu(lvl + 0.3 * x)
    if dist == "zsparse":  # 0 almost everywhere: fewer positives than k on long rows -> tie mode
        u = torch.rand((rows, W), device=DEV, generator=g)
        return torch.where(u < 0.0015, x.abs() + 0.1, torch.zeros_like(x))
    if dist == "quant":  # few distinct values: heavy ties at the k-th value
        return torch.round(x * 2) / 2
    if dist == "const":  # every element equal
        return torch.zeros_like(x)
    if dist == "hottail":  # the last 6K positions dominate: all candidates in the last segment
        x[:, max(0, W - 6144):] += 6.0
        return x
    if dist == "shared":  # rows of one request: one context row + small per-query noise
        base = torch.randn((1, W), device=DEV, generator=g)
        return base + 0.05 * x
    if dist == "neginf":  # mostly -inf (only ~1000 finite per row): -inf ties inside the top-k
        u = torch.rand((rows, W), device=DEV, generator=g)
        return torch.where(u < 1000.0 / W, x, torch.full_like(x, float("-inf")))
    raise ValueError(dist)


DISTS = ("randn", "ar", "segrelu", "zsparse", "quant", "const", "hottail", "shared", "neginf")


def bounds(kind: str, rows: int, W: int):
    """(starts or None, ends) int32 for a shape kind; every row ends <= W."""
    i = torch.arange(rows, device=DEV, dtype=torch.int64)
    if kind == "cached":  # one request with W - rows cached tokens: row i sees W - rows + 1 + i
        return None, (W - rows + 1 + i).clamp(min=0).to(torch.int32)
    if kind == "fresh":  # a new prompt: row i sees i + 1
        return None, (i + 1).clamp(max=W).to(torch.int32)
    if kind == "two":  # two requests in one chunk: contexts W and W // 4
        h = rows // 2
        e = torch.where(i < h, W - h + 1 + i, W // 4 - (rows - h) + 1 + (i - h))
        return None, e.clamp(min=1).to(torch.int32)
    if kind == "fp8":  # FP8 path: K of two requests concatenated, row i of request j sees [ks_j, ks_j + ctx_i)
        h = rows // 2
        la = W // 3
        lb = W - la
        ks = torch.where(i < h, 0, la)
        ctx = torch.where(i < h, la - h + 1 + i, lb - (rows - h) + 1 + (i - h)).clamp(min=1)
        return ks.to(torch.int32), (ks + ctx).to(torch.int32)
    if kind == "edge":  # lengths around k
        v = torch.tensor([1, 2047, 2048, 2049, 2050, 4095, 4096, 4097, 8191, 8192, 8193, 30000], device=DEV)
        return None, v[i % v.numel()].clamp(max=W).to(torch.int32)
    raise ValueError(kind)


def make_case(dist, kind, rows, W, seed):
    x = gen(dist, rows, W, seed)
    st, en = bounds(kind, rows, W)
    col = torch.arange(W, device=DEV)[None, :]
    lo = st.long()[:, None] if st is not None else torch.zeros((rows, 1), dtype=torch.long, device=DEV)
    inside = (col >= lo) & (col < en.long()[:, None])
    x = torch.where(inside, x, torch.full_like(x, GARBAGE))  # anything read outside a row would be selected
    return x, st, en


# ------------------------------------------------------------------------------------------------------------ exactness
FULL = [  # (kind, rows, W): rows 1 .. 4096, lengths 2K .. 1M
    ("cached", 1, 1048576),
    ("cached", 7, 2100),
    ("edge", 64, 32768),
    ("cached", 128, 1048576),
    ("two", 256, 524288),
    ("cached", 600, 200000),
    ("fp8", 1000, 131072),
    ("cached", 1024, 131072),
    ("two", 2048, 65536),
    ("fp8", 4096, 32768),
    ("fresh", 4096, 4096),
    ("cached", 4096, 32768),
]
QUICK = [("cached", 128, 1048576), ("fp8", 1000, 131072), ("fresh", 4096, 4096), ("edge", 64, 32768)]
QUICK_DISTS = ("randn", "zsparse", "quant", "hottail")


def exact(cases, dists) -> int:
    m = tkp()
    fails = 0
    tot = [0, 0, 0, 0]
    for kind, rows, W in cases:
        for di, dist in enumerate(dists):
            x, st, en = make_case(dist, kind, rows, W, seed=rows * 31 + di)
            out = torch.full((rows, 2 * K), -7, dtype=torch.int32, device=DEV)[:, :K]  # topk-buffer-like stride
            m.triton_top_k_per_row_prefill(x, st, en, out, K)
            torch.cuda.synchronize()
            res = m.check_call(x, st, en, out, K)
            for i, n in enumerate(m.MODE_NAMES):
                tot[i] += res[n]
            ok = res["bad"] == 0
            fails += not ok
            print(f"{'PASS' if ok else 'FAIL'} {kind:6s} rows {rows:5d} W {W:8d} {dist:8s} split/group {m.plan(rows)} "
                  f"normal {res['normal']} tie {res['tie']} whole_row {res['whole_row']} short {res['short']}"
                  + ("" if ok else f" {res}"), flush=True)
            del x, out
    print(f"rows by path: normal {tot[0]} tie {tot[1]} whole_row {tot[2]} short {tot[3]}; failed cases {fails}")
    return fails


# --------------------------------------------------------------------------------------------------------------- hooks
def load(name_or_path):
    """A patched module: by its vllm module name when the file is mounted over the image's (as on the prefill), else
    from a file path under a private name."""
    if not name_or_path.endswith(".py"):
        return importlib.import_module(name_or_path)
    spec = importlib.util.spec_from_file_location("vpd_pf_" + os.path.basename(name_or_path)[:-3], name_or_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def hooks(names, on: bool) -> int:
    fails = 0
    mods = [load(x) for x in names]
    for mod in mods:
        assert hasattr(mod, "_vpd_prefill_topk"), f"{mod.__name__} ({mod.__file__}) has no prefill hook"
        assert mod._VPD_TOPK_PREFILL is on, mod.__name__
        assert hasattr(mod, "_vpd_tkp") is on, mod.__name__
    rows, W = 300, 262144
    x, st, en = make_case("randn", "fp8", rows, W, 5)
    _, en_m = bounds("cached", rows, W)
    for mod in mods:
        for starts, ends, where in ((st, en, "fp8"), (None, en_m, "mxfp4")):
            buf = torch.full((rows, 2 * K), -7, dtype=torch.int32, device=DEV)
            out = buf[:, :K]
            took = mod._vpd_prefill_topk(x, starts, ends, out, K, 1, None, "model.layers.3.self_attn.indexer.k_cache", where)
            if took != on:
                print(f"FAIL {mod.__name__} {where}: hook returned {took} with the switch {'on' if on else 'off'}")
                fails += 1
                continue
            if on:
                res = tkp().check_call(x, starts, ends, out, K)
                ok = res["bad"] == 0 and bool((buf[:, K:] == -7).all())
                fails += not ok
                print(f"{'PASS' if ok else 'FAIL'} {mod.__name__} {where} hook exact on {rows} rows: {res}")
            else:
                ok = bool((buf == -7).all())
                fails += not ok
                print(f"{'PASS' if ok else 'FAIL'} {mod.__name__} {where} hook off: output untouched")
            if on:  # declines
                dec = [
                    ("compressed", dict(compress_ratio=4)),
                    ("candidates", dict(candidates=torch.zeros(1, device=DEV))),
                    ("int64 ends", dict(row_ends=ends.long())),
                    ("out width", dict(out=buf[:, : K // 2])),
                    ("k 3000", dict(topk_tokens=3000)),
                    ("bf16 logits", dict(logits=x[:4].bfloat16())),
                ]
                for name, kw in dec:
                    a = dict(logits=x, row_starts=starts, row_ends=ends, out=out, topk_tokens=K, compress_ratio=1,
                             candidates=None, prefix="model.layers.3.self_attn.indexer.k_cache", where=where)
                    a.update(kw)
                    if mod._vpd_prefill_topk(**a):
                        print(f"FAIL {mod.__name__} {where}: hook took a {name} call")
                        fails += 1
    print(f"hooks ({'on' if on else 'off'}): {fails} failures")
    return fails


# ---------------------------------------------------------------------------------------------------------------- time
def timed(fn, iters, flush):
    ts = []
    for _ in range(iters):
        flush.add_(1.0)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def time_main(iters=10) -> int:
    from aiter.ops.topk import top_k_per_row_decode, top_k_per_row_prefill

    m = tkp()
    flush = torch.zeros(2**28, device=DEV)  # 1 GiB
    shapes = [("cached", 128, 1048576), ("cached", 256, 524288), ("cached", 512, 262144), ("two", 1024, 131072),
              ("cached", 2048, 65536), ("cached", 4096, 32768), ("fp8", 1024, 131072), ("fresh", 4096, 4096)]
    fails = 0
    print("shape | rows | W | dist | triton us | aiter decode (one-block) us | aiter prefill us | speedup vs aiter decode "
          "| triton paths n/t/w/s | aiter decode exact")
    for kind, rows, W in shapes:
        for dist in ("randn", "ar", "zsparse", "shared"):
            x, st, en = make_case(dist, kind, rows, W, 11)
            out = torch.full((rows, K), -1, dtype=torch.int32, device=DEV)
            ref_out = torch.full((rows, K), -1, dtype=torch.int32, device=DEV)
            stt = st if st is not None else torch.zeros_like(en)
            t_tri = timed(lambda: m.triton_top_k_per_row_prefill(x, st, en, out, K), iters, flush)
            res = m.check_call(x, st, en, out, K)
            fails += res["bad"] != 0
            if st is None:  # what the MXFP4 path runs (lengths only)
                t_dec = timed(lambda: top_k_per_row_decode(x, 1, en, ref_out, rows, x.stride(0), x.stride(1), k=K),
                              iters, flush)
                dec_ok = check_ref(x, st, en, ref_out)
            else:
                t_dec, dec_ok = float("nan"), "-"
            pre_out = torch.full((rows, K), -1, dtype=torch.int32, device=DEV)
            t_pre = timed(lambda: top_k_per_row_prefill(x, stt, en, pre_out, None, rows, x.stride(0), x.stride(1), k=K),
                          iters, flush)
            base = t_dec if st is None else t_pre
            print(f"{kind} | {rows} | {W} | {dist} | {t_tri:.1f} | {t_dec:.1f} | {t_pre:.1f} | {base / t_tri:.2f}x "
                  f"| {res['normal']}/{res['tie']}/{res['whole_row']}/{res['short']} | {dec_ok}"
                  + ("" if res["bad"] == 0 else f" TRITON NOT EXACT {res}"), flush=True)
            del x, out, ref_out, pre_out
    # collect BLOCK / num_warps sweep on two shapes
    print("sweep: shape | BLOCK | NW | triton us")
    keep = (m._BLOCK, m._COLLECT_NW)
    for kind, rows, W in (("cached", 128, 1048576), ("cached", 1024, 131072)):
        x, st, en = make_case("randn", kind, rows, W, 12)
        out = torch.full((rows, K), -1, dtype=torch.int32, device=DEV)
        for blk, nw in ((2048, 4), (4096, 4), (4096, 8), (8192, 8), (8192, 16)):
            m._BLOCK, m._COLLECT_NW = blk, nw
            t = timed(lambda: m.triton_top_k_per_row_prefill(x, st, en, out, K), iters, flush)
            bad = m.check_call(x, st, en, out, K)["bad"]
            fails += bad != 0
            print(f"{kind} {rows}x{W} | {blk} | {nw} | {t:.1f}" + ("" if bad == 0 else f" NOT EXACT ({bad} rows)"))
        del x, out
    m._BLOCK, m._COLLECT_NW = keep
    return fails


def check_ref(x, st, en, out):
    """aiter's output in the same terms (sanity of the baseline): exact or not."""
    m = tkp()
    keep = m._SORT
    m._SORT = False  # aiter's order is not checked
    try:
        r = m.check_call(x, st, en, out, K)
    finally:
        m._SORT = keep
    return "yes" if r["bad"] == 0 else f"no ({r['bad']} rows)"


def main():
    mode = sys.argv[1]
    torch.cuda.init()
    print(f"device {torch.cuda.get_device_properties(0).gcnArchName}; mode {mode}", flush=True)
    if mode in ("hooks", "off"):
        fails = hooks(sys.argv[2:], on=mode == "hooks")
    elif mode == "exact":
        fails = exact(FULL, DISTS)
    elif mode == "quick":
        fails = exact(QUICK, QUICK_DISTS)
    elif mode == "time":
        fails = time_main(int(os.environ.get("ITERS", "10")))
    else:
        sys.exit(__doc__)
    if fails:
        print(f"prefill topk test FAILED ({fails})")
        sys.exit(1)
    print(f"prefill topk test ({mode}) passed")


if __name__ == "__main__":
    main()
