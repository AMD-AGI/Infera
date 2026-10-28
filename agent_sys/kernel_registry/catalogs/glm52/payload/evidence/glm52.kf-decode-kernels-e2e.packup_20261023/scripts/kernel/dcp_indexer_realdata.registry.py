"""DCP split indexer on REAL captured inputs (kdf1, compute-node-02, layer 10, DCP rank 0): prod pipeline vs KF candidate.
Per call: time (CUDA graph, 1 call per replay x 20), exactness of the merged slot sets, and KF selector stats."""
import glob, statistics, sys, torch
import drv as D
import prod_idx, kf_idx
from graph_harness import cuda_graph_bench
kf_idx._SEL_DEBUG = True
dump = sys.argv[1]
pool = torch.load(glob.glob(f"{dump}/kvpool.L*.pt")[0]).cuda()
kv = pool.view(-1, D.PS, 1, D.D + 4) if pool.dim() != 4 else pool
print("pool", tuple(pool.shape), pool.dtype, flush=True)
for f in sorted(glob.glob(f"{dump}/call.*.pt")):
    c = torch.load(f)
    q, w, lens, bt = c["q"].cuda(), c["w"].cuda(), c["lens"].cuda().to(torch.int32), c["bt"].cuda().to(torch.int32)
    if w.dim() == 3:
        w = w.squeeze(-1)
    T = q.shape[0]
    bt = bt[:T].contiguous(); lens = lens[:T].contiguous()
    P = (bt.shape[1] + 3) // 4 * 4
    if P != bt.shape[1]:
        bt = torch.nn.functional.pad(bt, (0, P - bt.shape[1])).contiguous()
    packs = []
    for r in range(4):
        tab, ln = D.owned_page_table(bt, 4, r), D.owned_lens(lens, D.PS, 4, r)
        packs.append(D.ref_pack(D.aiter_logits(q, kv, w.float().contiguous(), ln, tab, tab.shape[1] * D.PS), ln, r))
    ref = D.ref_merge_slots(torch.cat(packs, 0), bt)
    other = torch.cat(packs[1:], 0).contiguous()
    tab0, len0 = D.owned_page_table(bt, 4, 0), D.owned_lens(lens, D.PS, 4, 0)
    res = {}
    for name, mod in (("prod", prod_idx), ("kf", kf_idx)):
        fn = lambda mod=mod: mod.dcp_indexer_topk(q, kv, w.float().contiguous(), tab0, len0, bt, other, 2048)
        out = fn()
        s, same, rec = D.recall_db(out, ref)
        r = cuda_graph_bench(lambda: fn(), warmup=5, iters=20)
        res[name] = (statistics.median(r["times_ms"]), same)
    st = kf_idx.last_stats() if hasattr(kf_idx, "last_stats") else ""
    sib = int(sum(int(not torch.equal(bt[6*b:6*b+6], bt[6*b].expand(6, -1))) for b in range(T // 6)))
    print(f"{f.split('/')[-1]} rows {T} maxlen {int(lens.max())} meanlen {int(lens.float().mean())} | prod {res['prod'][0]:.4f} ms "
          f"kf {res['kf'][0]:.4f} ms x{res['prod'][0]/res['kf'][0]:.2f} | exact prod {res['prod'][1]}/{T} kf {res['kf'][1]}/{T} | "
          f"groups with non-identical sibling tables {sib}/{T//6} | {st}", flush=True)
