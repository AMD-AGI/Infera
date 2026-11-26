#!/usr/bin/env python3
"""Tests / microbenchmark of the packed FP4 dispatch (scripts/make_fp4dispatch_pack_overlay.py, item 8), in the image:
  test_fp4dispatch_pack.py layout PATCHED_NAIVE_DP_EP_PY   switch off = base FP4 overlay module state; packed row layout
                                                           and receive views vs the 4 unpacked gathers (CPU tensors, 4
                                                           ranks simulated by torch.cat = what the all-gather produces)
  test_fp4dispatch_pack.py gpu    PATCHED_NAIVE_DP_EP_PY   1 GPU: pack kernel with the default quant=hip vs aiter HIP
                                                           per_1x32 quant + the raw ids / weights, bit-exact, M = 1..256
                                                           (+ edge rows); 4 ranks simulated (quant=hip, recv split and
                                                           view); quant=triton printed as "info" only (not bit-identical,
                                                           job 313003); graph-replayed kernel times
  test_fp4dispatch_pack.py moe    PATCHED_NAIVE_DP_EP_PY   1 GPU: aiter fused_moe (random MXFP4 experts) on the strided recv
                                                           views vs contiguous copies (job 313003: it is not; kept for the
                                                           record, not in the job any more)
  test_fp4dispatch_pack.py dist   PATCHED_NAIVE_DP_EP_PY   4 GPUs, set up as vLLM does (NCCL default group, gloo cpu group,
                                                           vLLM's AiterCustomAllreduce on it): today's FP4 path (HIP quant
                                                           + 4 AGs) vs packed (HIP quant + pack copy + 1 AG + split),
                                                           bit-exact per rank, per-layer time eager and in graph replay.
                                                           Progress lines "rank r: <phase>"; a phase over
                                                           VPD_PHASE_TIMEOUT s (default 900) prints "TIMEOUT in phase",
                                                           all thread stacks, and exits 3
Exit code != 0 on any correctness failure ("FAIL" lines); timings are informational.
"""
import importlib.util
import os
import sys

import torch

K, ROWS = 6144, (1, 2, 3, 4, 7, 8, 15, 16, 33, 56, 64, 100, 128, 129, 200, 255, 256)
FAILS = []


def load(name, path, pack):
    os.environ["VPD_FP4_DISPATCH"] = "1"
    if pack:
        os.environ["VPD_FP4_DISPATCH_PACK"] = "1"
    else:
        os.environ.pop("VPD_FP4_DISPATCH_PACK", None)
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def check(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        FAILS.append(msg)


def same(a, b):
    a, b = a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    return a.shape == b.shape and torch.equal(a.cpu(), b.cpu())


def inputs(m, t, dev, seed, experts=257):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(m, K, generator=g) * torch.exp(torch.randn(m, 1, generator=g) * 2)
    if m >= 4:   # edge rows: zeros, huge, tiny, a block of one value next to zeros
        x[0] = 0
        x[1] *= 1e4
        x[2] *= 1e-6
        x[3, :32] = 3.0
        x[3, 32:64] = 0
    ids = torch.stack([torch.randperm(experts, generator=g)[:t] for _ in range(m)]).int() if m else torch.empty(0, t, dtype=torch.int32)
    w = torch.rand(m, t, generator=g)
    return x.to(torch.bfloat16).to(dev), ids.to(dev), w.to(dev)


def ref_pack(q, s, ids, w, lay):
    k, t, qb, sb, ioff, woff, row = lay
    buf = torch.zeros(q.shape[0], row, dtype=torch.uint8, device=q.device)
    buf[:, :qb] = q.view(torch.uint8)
    buf[:, qb:qb + sb] = s.view(torch.uint8)
    buf[:, ioff:ioff + 4 * t] = ids.contiguous().view(torch.uint8)
    buf[:, woff:woff + 4 * t] = w.contiguous().view(torch.uint8)
    return buf


def t_layout(path):
    off = load("vpd_naive_off", path, pack=False)
    check(off._VPD_FP4_DISPATCH_PACK is False and "triton" not in vars(off) and "_vpd_fp4_pack_kernel" not in vars(off),
          "switch off: no Triton import, no kernels defined (base FP4 path only)")
    m = load("vpd_naive_pack", path, pack=True)
    check(m._VPD_FP4_DISPATCH_PACK is True and m._VPD_PACK_QUANT in ("triton", "hip") and m._VPD_PACK_RECV == "view",
          f"switch on: quant {m._VPD_PACK_QUANT}, recv {m._VPD_PACK_RECV}")
    x = torch.zeros(5, K, dtype=torch.bfloat16)
    lay8 = m._vpd_pack_layout(x, torch.zeros(5, 8, dtype=torch.int32), torch.zeros(5, 8))
    lay9 = m._vpd_pack_layout(x, torch.zeros(5, 9, dtype=torch.int32), torch.zeros(5, 9))
    check(lay8 == (6144, 8, 3072, 192, 3264, 3296, 3328) and lay9 == (6144, 9, 3072, 192, 3264, 3300, 3344),
          f"layout topk 8 {lay8}, topk 9 {lay9}")
    check(m._vpd_pack_layout(x, torch.zeros(5, 8, dtype=torch.int64), torch.zeros(5, 8)) is None
          and m._vpd_pack_layout(x.float(), torch.zeros(5, 8, dtype=torch.int32), torch.zeros(5, 8)) is None,
          "int64 ids / fp32 hidden -> unpacked FP4 path")
    g = torch.Generator().manual_seed(0)
    for t in (8, 9):
        for M in ROWS:
            ranks = []
            for r in range(4):
                q = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, generator=g).view(torch.float4_e2m1fn_x2)
                s = torch.randint(0, 256, (M, K // 32), dtype=torch.uint8, generator=g).view(torch.float8_e8m0fnu)
                ids = torch.randint(0, 257, (M, t), dtype=torch.int32, generator=g)
                w = torch.rand(M, t, generator=g)
                ranks.append((q, s, ids, w))
            lay = m._vpd_pack_layout(torch.zeros(M, K, dtype=torch.bfloat16), ranks[0][2], ranks[0][3])
            gathered = torch.cat([ref_pack(*rk, lay) for rk in ranks]).view(torch.bfloat16)   # = the 1 all-gather
            hq, hs, hi, hw = m._vpd_unpack(gathered, lay, recv="view")
            want = [torch.cat([rk[i] for rk in ranks]) for i in range(4)]                     # = the 4 all-gathers
            ok = all(same(a, b) for a, b in zip((hq, hs, hi, hw), want))
            ok &= (hq.dtype, hs.dtype, hi.dtype, hw.dtype) == (want[0].dtype, want[1].dtype, torch.int32, torch.float32)
            ok &= hq.shape == (4 * M, K // 2) and hi.shape == (4 * M, t) and hq.data_ptr() == gathered.data_ptr()
            if not ok or M in (1, 256):
                check(ok, f"layout views topk {t} M {M}/rank x 4: hidden {tuple(hq.shape)} stride {hq.stride()}, "
                          f"scales {tuple(hs.shape)}, ids {tuple(hi.shape)} stride {hi.stride()}, weights {tuple(hw.shape)}")
    print("layout test done")


def graph_time(fn, reps=20, iters=30):
    """us per call of fn, replayed in a CUDA graph (reps calls per graph)."""
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / (iters * reps)


def t_gpu(path):
    m = load("vpd_naive_pack", path, pack=True)
    from aiter import QuantType, get_hip_quant

    quant = get_hip_quant(QuantType.per_1x32)
    dev = "cuda"
    for t in (8, 9):
        for M in ROWS:
            x, ids, w = inputs(M, t, dev, seed=M * 10 + t)
            q, s = quant(x, shuffle=False)
            s = s[:M].contiguous()
            lay = m._vpd_pack_layout(x, ids, w)
            want = ref_pack(q, s, ids, w, lay)
            k_, t_, qb, sb, ioff, woff, row = lay
            for mode in ("triton", "hip"):
                buf = m._vpd_pack(x, ids, w, lay, quant=mode)
                fields = [(0, qb), (qb, qb + sb), (ioff, ioff + 4 * t), (woff, woff + 4 * t)]
                bad = [n for n, (a, b) in zip(("fp4", "scales", "ids", "weights"), fields) if not torch.equal(buf[:, a:b], want[:, a:b])]
                msg = f"pack quant={mode} topk {t} M {M}: " + ("bit-identical to aiter HIP quant + ids / weights" if not bad else f"MISMATCH in {bad}")
                if bad and mode == "triton" and ("fp4" in bad or "scales" in bad):
                    d = (buf[:, :qb] != want[:, :qb]).sum().item(), (buf[:, qb:qb + sb] != want[:, qb:qb + sb]).sum().item()
                    alt = m._vpd_pack(x, ids, w, lay, quant=mode, scaling=1)
                    msg += f" (fp4 bytes differing {d[0]}, scale bytes {d[1]}; SCALING_MODE=1 identical: {torch.equal(alt[:, :qb + sb], want[:, :qb + sb])})"
                if mode == "triton":   # opt-in only, known not bit-identical (job 313003): informational
                    if M in (1, 56, 256):
                        print("info " + msg, flush=True)
                elif bad or M in (1, 56, 256):
                    check(not bad, msg)
            # 4 ranks simulated: unpacked path = cat of each tensor, packed = cat of rows then views / split
            rk = [inputs(M, t, dev, seed=1000 + 4 * M + r) for r in range(4)]
            qs = [quant(xr, shuffle=False) for xr, _, _ in rk]
            want4 = [torch.cat([q_ for q_, _ in qs]), torch.cat([s_[:M] for _, s_ in qs]),
                     torch.cat([r[1] for r in rk]), torch.cat([r[2] for r in rk])]
            if M == 1 and t == 8:
                check(m._VPD_PACK_QUANT == "hip" and m._VPD_PACK_RECV == "split", "defaults: quant hip, recv split")
            gathered = torch.cat([m._vpd_pack(*r, lay, quant="hip") for r in rk]).view(torch.bfloat16)
            for recv in ("split", "view"):   # view: layout check only (fused_moe cannot consume it)
                got = m._vpd_unpack(gathered, lay, recv=recv)
                ok = all(same(a, b) for a, b in zip(got, want4)) and got[0].dtype == want4[0].dtype and got[1].dtype == want4[1].dtype
                if not ok or M in (1, 56, 256):
                    check(ok, f"4 ranks x M {M} topk {t} recv={recv}: hidden / scales / ids / weights bit-identical to the 4 gathers"
                              f" (dtypes {got[0].dtype}, {got[1].dtype}; contiguous {got[0].is_contiguous()})")
    print("-- kernel times, us per call (CUDA graph, 20 calls / graph):")
    for M in (16, 56, 64, 128, 256):
        x, ids, w = inputs(M, 8, dev, seed=7)
        lay = m._vpd_pack_layout(x, ids, w)
        g4 = torch.cat([m._vpd_pack(x, ids, w, lay)] * 4).view(torch.bfloat16)
        th = graph_time(lambda: quant(x, shuffle=False))
        tt = graph_time(lambda: m._vpd_pack(x, ids, w, lay, quant="triton"))
        tp = graph_time(lambda: m._vpd_pack(x, ids, w, lay, quant="hip"))
        ts = graph_time(lambda: m._vpd_unpack(g4, lay, recv="split"))
        print(f"M {M:4d}/rank: aiter HIP quant {th:6.2f} | pack quant=triton {tt:6.2f} | pack quant=hip (quant + copy) {tp:6.2f}"
              f" | recv split (N={4 * M}) {ts:6.2f}", flush=True)


def t_moe(path, experts=64, inter=2048, t=8, M=64):
    """Does aiter fused_moe read the recv=view tensors (row stride ROW, not contiguous) correctly? Compared with the
    same values as contiguous copies; atomics may reorder fp32 adds, so contiguous-vs-contiguous is the noise floor."""
    m = load("vpd_naive_pack", path, pack=True)
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe, moe_sorting
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    dev = "cuda"
    g = torch.Generator(device="cpu").manual_seed(1)
    w13 = torch.randint(0, 256, (experts, 2 * inter, K // 2), dtype=torch.uint8, generator=g).to(dev)
    w2 = torch.randint(0, 256, (experts, K, inter // 2), dtype=torch.uint8, generator=g).to(dev)
    w13s = torch.randint(118, 124, (experts * 2 * inter, K // 32), dtype=torch.uint8, generator=g).to(dev)
    w2s = torch.randint(118, 124, (experts * K, inter // 32), dtype=torch.uint8, generator=g).to(dev)
    w13 = shuffle_weight(w13.view(torch.float4_e2m1fn_x2), layout=(16, 16))
    w2 = shuffle_weight(w2.view(torch.float4_e2m1fn_x2), layout=(16, 16))
    w13.is_shuffled = w2.is_shuffled = True
    w13s = e8m0_shuffle(w13s.view(torch.float8_e8m0fnu)).view(experts, 2 * inter, -1)
    w2s = e8m0_shuffle(w2s.view(torch.float8_e8m0fnu)).view(experts, K, -1)
    rk = [inputs(M, t, dev, seed=50 + r, experts=experts) for r in range(4)]
    lay = m._vpd_pack_layout(*rk[0])
    gathered = torch.cat([m._vpd_pack(*r, lay) for r in rk]).view(torch.bfloat16)
    hq, hs, hi, hw = m._vpd_unpack(gathered, lay, recv="view")
    print(f"recv=view: hidden {tuple(hq.shape)} stride {hq.stride()} contiguous {hq.is_contiguous()}; ids stride {hi.stride()}")
    cq, cs, ci, cw = (v.contiguous() for v in (hq, hs, hi, hw))

    def run(a, s, i, w):
        return fused_moe(a, w13, w2, w, i, quant_type=QuantType.per_1x32, activation=ActivationType.Silu,
                         w1_scale=w13s, w2_scale=w2s, a1_scale=s, dtype=torch.bfloat16)

    try:
        so_v, so_c = moe_sorting(hi, hw, experts, K, torch.bfloat16), moe_sorting(ci, cw, experts, K, torch.bfloat16)
        print("moe_sorting on strided ids / weights: " + ", ".join(
            f"{n} {'same' if torch.equal(a, b) else 'DIFFERENT'}" for n, a, b in
            zip(("sorted_ids", "sorted_weights", "sorted_expert_ids", "num_valid_ids"), so_v[:4], so_c[:4])))
    except Exception as e:  # informational only
        print(f"moe_sorting on strided inputs raised {type(e).__name__}: {e}")
    try:
        ref, ref2 = run(cq, cs, ci, cw), run(cq, cs, ci, cw)
        torch.cuda.synchronize()
    except Exception as e:
        check(False, f"fused_moe on contiguous FP4 inputs raised {type(e).__name__}: {e} (test setup, not the overlay)")
        return
    noise = (ref.float() - ref2.float()).abs().max().item()
    scale = ref.float().abs().max().item()
    for name, args in (("all four strided (recv=view)", (hq, hs, hi, hw)), ("hidden strided only", (hq, cs, ci, cw)),
                       ("scales strided only", (cq, hs, ci, cw)), ("ids / weights strided only", (cq, cs, hi, hw))):
        try:
            out = run(*args)
            torch.cuda.synchronize()
            d = (out.float() - ref.float()).abs().max().item()
            ok = torch.equal(out, ref) or d <= max(4 * noise, 1e-3 * scale)
            check(ok, f"fused_moe {name}: max |diff| {d:.4g} (contiguous rerun {noise:.4g}, |out| max {scale:.4g})"
                      + ("" if ok else " -> use VPD_FP4_DISPATCH_PACK_RECV=split"))
        except Exception as e:
            check(False, f"fused_moe {name} raised {type(e).__name__}: {e} -> use VPD_FP4_DISPATCH_PACK_RECV=split")


class Watchdog:
    """Per-phase timeout: prints the phase, every thread's stack, and exits 3 (so a hang says where it is)."""

    def __init__(self, rank, timeout):
        self.rank, self.timeout, self.timer, self.name = rank, timeout, None, "start"

    def phase(self, name):
        import threading
        import time

        if self.timer is not None:
            self.timer.cancel()
        self.name = name
        print(f"rank {self.rank}: {time.strftime('%H:%M:%S')} {name}", flush=True)
        self.timer = threading.Timer(self.timeout, self._fire)
        self.timer.daemon = True
        self.timer.start()

    def _fire(self):
        import faulthandler

        print(f"rank {self.rank}: TIMEOUT in phase '{self.name}' after {self.timeout} s; stacks:", flush=True)
        faulthandler.dump_traceback(file=sys.stdout, all_threads=True)
        sys.stdout.flush()
        os._exit(3)

    def done(self):
        if self.timer is not None:
            self.timer.cancel()


DIST_M = (8, 32, 64, 128, 256)


def _dist_worker(rank, path, port, out_q):
    import torch.distributed as dist

    wd = Watchdog(rank, float(os.environ.get("VPD_PHASE_TIMEOUT", "900")))
    wd.phase("init: process groups (nccl default + gloo cpu group, as vLLM)")
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    dist.init_process_group("nccl", rank=rank, world_size=4, device_id=dev)
    cpu = dist.new_group(list(range(4)), backend="gloo")
    dist.barrier(group=cpu)
    wd.phase("init: overlay module + aiter quant")
    m = load("vpd_naive_pack", path, pack=True)
    from aiter import QuantType, get_hip_quant

    quant = get_hip_quant(QuantType.per_1x32)
    wd.phase("init: AiterCustomAllreduce on the gloo group (vLLM's wrapper, its MAX_SIZE)")
    from vllm.distributed.device_communicators.aiter_custom_all_reduce import AiterCustomAllreduce

    ca = AiterCustomAllreduce(group=cpu, device=dev)
    assert not ca.disabled, "aiter custom all-reduce disabled"
    dist.barrier(group=cpu)
    wd.phase("init done")
    lines, L = [], 30   # L MoE layers per timed batch / captured graph

    def today(x, ids, w):   # base FP4 overlay: quant, then AgRsAll2AllManager.dispatch's list [hidden, weights, ids, scales]
        q, s = quant(x, shuffle=False)
        s = s[: x.shape[0]].contiguous()
        outs = [ca.custom_all_gather(v, dim=0) for v in (q.view(torch.bfloat16), w, ids, s.view(torch.bfloat16))]
        return outs[0].view(q.dtype), outs[3].view(s.dtype), outs[2], outs[1]

    def packed(x, ids, w, lay):   # the overlay's defaults: HIP quant + pack copy, 1 AG, split
        buf = m._vpd_pack(x, ids, w, lay, quant="hip")
        return m._vpd_unpack(ca.custom_all_gather(buf.view(torch.bfloat16), dim=0), lay, recv="split")

    def timed(fn):
        torch.cuda.synchronize()
        dist.barrier(group=cpu)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(L):
            fn()
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b) * 1000 / L

    for t, Ms in ((8, DIST_M), (9, (64,))):
        for M in Ms:
            tag = f"topk {t} M {M}"
            x, ids, w = inputs(M, t, dev, seed=100 * M + 10 * rank + t)
            lay = m._vpd_pack_layout(x, ids, w)
            fns = {"today": lambda: today(x, ids, w), "packed": lambda: packed(x, ids, w, lay)}
            if not (ca.should_custom_ag(w) and ca.should_custom_ag(ids)):
                print(f"rank {rank}: {tag}: today's ids / weights not aiter-AG eligible (bytes % 16)", flush=True)
            res, times = {}, {}
            for name, fn in fns.items():
                wd.phase(f"{tag} {name}: eager (first call compiles)")
                eager = fn()
                torch.cuda.synchronize()
                te = timed(fn)
                wd.phase(f"{tag} {name}: graph capture")
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    fn()
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                dist.barrier(group=cpu)
                g = torch.cuda.CUDAGraph()
                with ca.aiter_ca.capture():
                    with torch.cuda.graph(g, stream=s):
                        for _ in range(L):
                            last = fn()
                wd.phase(f"{tag} {name}: graph replay")
                dist.barrier(group=cpu)
                g.replay()
                torch.cuda.synchronize()
                tg = timed(g.replay) / L   # timed() gives us per replay; one replay = L layers
                res[name] = (eager, last)
                times[name] = (te, tg)
                del g
            wd.phase(f"{tag}: compare")
            q, s_ = quant(x, shuffle=False)
            loc = [q.view(torch.uint8).cpu(), s_[:M].contiguous().view(torch.uint8).cpu(), ids.cpu(), w.cpu()]
            allr = [None] * 4
            dist.all_gather_object(allr, loc, group=cpu)
            want = [torch.cat([r[i] for r in allr]) for i in range(4)]
            for name, (eager, last) in res.items():
                ok = all(same(a_, b_) for a_, b_ in zip(eager, want)) and all(same(a_, b_) for a_, b_ in zip(last, want))
                ok &= all(v.is_contiguous() for v in last) if name == "packed" else True
                if not ok or rank == 0:
                    check(ok, f"rank {rank} {name} {tag}/rank: gathered hidden / scales / ids / weights bit-identical to "
                              f"each rank's HIP quant + ids / weights (eager and graph replay)")
            tt = [None] * 4
            dist.all_gather_object(tt, times, group=cpu)
            if rank == 0:
                mx = {n: (max(r[n][0] for r in tt), max(r[n][1] for r in tt)) for n in times}
                lines.append(f"{tag:12s}/rank: today (HIP quant + 4 AG) eager {mx['today'][0]:7.2f} graph {mx['today'][1]:6.2f}"
                             f" | packed (HIP quant + pack + 1 AG + split) eager {mx['packed'][0]:7.2f} graph {mx['packed'][1]:6.2f}"
                             f" | graph saving {mx['today'][1] - mx['packed'][1]:6.2f}")
                print(lines[-1], flush=True)
    wd.phase("done")
    out_q.put(lines + [f"FAIL {f}" for f in FAILS])
    dist.barrier(group=cpu)
    wd.done()
    dist.destroy_process_group()


def t_dist(path):
    import time

    import torch.multiprocessing as mp

    assert torch.cuda.device_count() >= 4, f"needs 4 GPUs, sees {torch.cuda.device_count()}"
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    ps = [ctx.Process(target=_dist_worker, args=(r, path, 29561, q)) for r in range(4)]
    for p in ps:
        p.start()
    outs = []
    while any(p.is_alive() for p in ps):   # drain while waiting (a full pipe would block a worker's put)
        while not q.empty():
            outs += q.get()
        if any(p.exitcode not in (None, 0) for p in ps):   # one rank died / timed out: the others would wait forever
            time.sleep(5)
            for p in ps:
                if p.is_alive():
                    p.terminate()
            break
        time.sleep(1)
    for p in ps:
        p.join()
    while not q.empty():
        outs += q.get()
    print("-- per MoE layer, us (max over ranks; 30 layers per timed batch / graph; aiter custom AG):")
    for line in outs:
        print(line)
        if line.startswith("FAIL"):
            FAILS.append(line)
    codes = [p.exitcode for p in ps]
    if any(c != 0 for c in codes):
        FAILS.append(f"worker exit codes {codes} (3 = phase timeout, see 'TIMEOUT in phase'; negative = terminated)")


def main():
    mode, path = sys.argv[1], sys.argv[2]
    {"layout": t_layout, "gpu": t_gpu, "moe": t_moe, "dist": t_dist}[mode](path)
    if FAILS:
        print(f"fp4 pack test {mode} failed: {len(FAILS)} failure(s)")
        sys.exit(1)
    print(f"fp4 pack test {mode} passed")


if __name__ == "__main__":
    main()
