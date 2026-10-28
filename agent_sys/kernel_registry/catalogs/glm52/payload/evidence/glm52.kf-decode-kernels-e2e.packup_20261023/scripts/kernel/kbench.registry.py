#!/usr/bin/env python3
"""Kernel-level A/B of the KF overlay inside a target sglang stack (no server).

For each kernel, the SAME patched sglang entry point is timed twice under CUDA-graph replay with the KF task
drivers' workset cases: infera_kernels.FORCE_ORIG=True (the stack's own production kernel) vs False (KF candidate).
Also checks KF vs production output (sparse/MoE: SNR; indexer: identical top-k index sets).
Usage (in a container of the target image, sglang under test importable):
  INFERA_KERNEL_SPARSE=1 INFERA_KERNEL_INDEXER=1 INFERA_KERNEL_MOE=1 SGLANG_OPT_USE_TOPK_V2=false AITER_USE_FLYDSL_MOE_SORTING=1 \
  python3 kbench.registry.py <tasks_dir> <overlay_dir> [sparse,indexer,moe]
"""
import importlib.util
import math
import statistics
import sys
import types

import torch

TASKS, OVL = sys.argv[1], sys.argv[2]
WHICH = (sys.argv[3] if len(sys.argv) > 3 else "sparse,indexer,moe").split(",")
sys.path.insert(0, OVL)
import infera_kernels  # noqa: E402

infera_kernels.install()
infera_kernels._mode.sparse = infera_kernels._mode.indexer = True   # the overlay only engages on TARGET_VERIFY
dev = "cuda"


def load_driver(task):
    d = f"{TASKS}/{task}"
    sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(f"drv_{task}", f"{d}/driver.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    sys.path.remove(d)
    return m


def snr(ref, x):
    ref, x = ref.float(), x.float()
    n = torch.mean((x - ref) ** 2).item()
    return 100.0 if n == 0 else 10 * math.log10(torch.mean(ref ** 2).item() / n)


def ab(name, step_for, n_calls, check):
    """step_for(force_orig) -> zero-arg closure doing n_calls calls; returns (orig_ms, kf_ms) per call."""
    from graph_harness import cuda_graph_bench
    res = {}
    for force in (True, False):
        infera_kernels.FORCE_ORIG = force
        r = cuda_graph_bench(step_for(force), warmup=10, iters=50)
        res[force] = statistics.median(t / n_calls for t in r["times_ms"])
    infera_kernels.FORCE_ORIG = False
    print(f"{name}: prod {res[True]:.4f} ms  KF {res[False]:.4f} ms  speedup {res[True] / res[False]:.3f}x "
          f"({r['mode']}) | {check()}", flush=True)


if "sparse" in WHICH:
    import sglang.kernels.ops.attention.dsa.triton_sparse_mla as tsm
    D = load_driver("sparse_mla")
    sys.path.insert(0, f"{TASKS}/sparse_mla")
    pools = D.make_kv(D.N_LAYERS, dev)
    for bs in (8, 10, 12, 14):
        qn, qr, idx = D.make_case(bs, 0, dev)

        def call():
            return tsm.triton_sparse_mla_fwd(qn, qr, pools[0], idx, D.SM_SCALE, d_v=D.D_V, topk_length=None,
                                             max_topk_length=D.TOPK)

        def step_for(force):
            def step():
                for kv in pools:
                    tsm.triton_sparse_mla_fwd(qn, qr, kv, idx, D.SM_SCALE, d_v=D.D_V, topk_length=None,
                                              max_topk_length=D.TOPK)
            return step

        def check():
            infera_kernels.FORCE_ORIG = True
            ref = call().clone()
            infera_kernels.FORCE_ORIG = False
            r = D.reference(qn, qr, pools[0], idx)
            out = call()
            return (f"snr KF/prod {snr(ref, out):.2f} dB, prod/fp32 {snr(r, ref.reshape(r.shape)):.2f}, "
                    f"KF/fp32 {snr(r, out.reshape(r.shape)):.2f}")
        ab(f"sparse_mla T{6 * bs}_bs{bs}", step_for, len(pools), check)
    del pools
    sys.path.remove(f"{TASKS}/sparse_mla")
    sys.modules.pop("graph_harness", None)

if "indexer" in WHICH:
    import sglang.srt.layers.attention.dsa.dsa_indexer as dix
    import sglang.srt.layers.attention.dsa.dsa_topk_backend as dtb
    I = load_driver("indexer")
    sys.path.insert(0, f"{TASKS}/indexer")
    pools, _ = I.make_pools(I.N_LAYERS, dev)
    for bs in (8, 10, 12, 14):
        c = I.make_case(bs, 0, dev)
        msl = c["real_page_table"].shape[1] * 64
        md = types.SimpleNamespace(page_table_1=c["page_table_1"], token_to_batch_idx=None,
                                   real_page_table=c["real_page_table"])

        def one(pool):
            kvc = pool.view(pool.shape[0], 64, 1, 132)
            lg = dix.aiter_paged_mqa_logits(c["q"], kvc, c["weights"], c["seq_lens"], c["real_page_table"], msl,
                                            preshuffle=True, kv_block_size=64)
            return dtb.DSATopKBackend.SGL_KERNEL.topk_transform(lg, c["seq_lens"], 2048, dtb.TopkTransformMethod.PAGED,
                                                                md, cu_seqlens_q_topk=c["cu_seqlens_q"])

        def step_for(force):
            def step():
                for p in pools:
                    one(p)
            return step

        def check():
            infera_kernels.FORCE_ORIG = True
            ref = one(pools[0]).clone()
            infera_kernels.FORCE_ORIG = False
            out = one(pools[0])
            same = sum(int(torch.equal(torch.sort(ref[r]).values, torch.sort(out[r]).values)) for r in range(ref.shape[0]))
            return f"identical top-k sets {same}/{ref.shape[0]} rows"
        ab(f"indexer T{6 * bs}_bs{bs}", step_for, len(pools), check)
    del pools
    sys.path.remove(f"{TASKS}/indexer")
    sys.modules.pop("graph_harness", None)

if "moe" in WHICH:
    import sglang.srt.layers.moe.moe_runner.aiter as mra
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    M_ = load_driver("moe_mxfp4")
    sys.path.insert(0, f"{TASKS}/moe_mxfp4")
    stub = types.SimpleNamespace(config=MoeRunnerConfig(activation="silu", gate_up_interleaved=False))
    qis = []
    for k in range(M_.N_SETS):
        w13, w13_s, w2, w2_s = M_.make_weight_set(k, dev)
        pw13, pw2, pw13_s, pw2_s = M_.prepare_weights(w13, w13_s, w2, w2_s)
        del w13, w13_s, w2, w2_s
        qis.append(mra.AiterMoeQuantInfo(w13_weight=pw13, w2_weight=pw2, quant_type=mra.AiterQuantType.PER_1X32,
                                         w13_scale=pw13_s, w2_scale=pw2_s))
    for bs in (8, 10, 12, 14):
        x, ids, wts = M_.make_case(bs, 0, dev)
        ri = mra.AiterRunnerInput(hidden_states=x, topk_ids=ids, topk_weights=wts,
                                  quant_type=mra.AiterQuantType.PER_1X32)

        def step_for(force):
            def step():
                for qi in qis:
                    mra.AiterRunnerCore.run(stub, ri, qi, {})
            return step

        def check():
            infera_kernels.FORCE_ORIG = True
            ref = mra.AiterRunnerCore.run(stub, ri, qis[0], {}).hidden_states.clone()
            infera_kernels.FORCE_ORIG = False
            out = mra.AiterRunnerCore.run(stub, ri, qis[0], {}).hidden_states
            return f"snr KF/prod {snr(ref, out):.2f} dB"
        ab(f"moe M{24 * bs}_bs{bs}", step_for, len(qis), check)
print("KBENCH DONE", flush=True)
