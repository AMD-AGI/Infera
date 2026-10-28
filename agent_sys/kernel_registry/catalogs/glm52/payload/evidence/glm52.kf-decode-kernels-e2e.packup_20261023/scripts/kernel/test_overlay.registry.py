#!/usr/bin/env python3
"""Offline smoke test of e2e/overlay (no server): the patched sglang entry points, fed with the KF task
drivers' workset cases, must engage the KF kernels and match the originals (the overlay's own CHECK lines).
Run in the KF container:  INFERA_KERNEL_SPARSE=1 INFERA_KERNEL_INDEXER=1 INFERA_KERNEL_MOE=1 \
    PYTHONPATH=<overlay> python3 test_overlay.registry.py
"""
import importlib.util
import os
import sys
import types

import torch

KW = "REDACTED_LOCAL_PATH"
sys.path.insert(0, f"{KW}/e2e/overlay")
import infera_kernels  # noqa: E402

infera_kernels.install()
infera_kernels._mode.sparse = infera_kernels._mode.indexer = True   # the overlay only engages on TARGET_VERIFY


def load_driver(task):
    d = f"{KW}/tasks/{task}"
    sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(f"drv_{task}", f"{d}/driver.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    sys.path.remove(d)
    for k in ("graph_harness",):
        sys.modules.pop(k, None)
    return m


dev = "cuda"

# ---- sparse MLA
import sglang.kernels.ops.attention.dsa.triton_sparse_mla as tsm  # noqa: E402
D = load_driver("sparse_mla")
kv = D.make_kv(1, dev)[0]
for bs in (10, 12):
    qn, qr, idx = D.make_case(bs, 0, dev)
    out = tsm.triton_sparse_mla_fwd(qn, qr, kv, idx, D.SM_SCALE, d_v=D.D_V, topk_length=None, max_topk_length=D.TOPK)
    print("sparse", bs, tuple(out.shape), flush=True)
del kv

# ---- indexer logits + top-k
import sglang.srt.layers.attention.dsa.dsa_indexer as dix  # noqa: E402
import sglang.srt.layers.attention.dsa.dsa_topk_backend as dtb  # noqa: E402
I = load_driver("indexer")
pools, _ = I.make_pools(1, dev)
for bs in (10, 12):
    c = I.make_case(bs, 0, dev)
    max_seq_len = c["real_page_table"].shape[1] * 64
    kvc = pools[0].view(pools[0].shape[0], 64, 1, 132)
    logits = dix.aiter_paged_mqa_logits(c["q"], kvc, c["weights"], c["seq_lens"], c["real_page_table"], max_seq_len,
                                        preshuffle=True, kv_block_size=64)
    md = types.SimpleNamespace(page_table_1=c["page_table_1"], token_to_batch_idx=None, real_page_table=c["real_page_table"])
    topk = dtb.DSATopKBackend.SGL_KERNEL.topk_transform(logits, c["seq_lens"], 2048, dtb.TopkTransformMethod.PAGED, md,
                                                         cu_seqlens_q_topk=c["cu_seqlens_q"])
    print("indexer", bs, tuple(logits.shape), tuple(topk.shape), flush=True)
del pools

# ---- MoE
import sglang.srt.layers.moe.moe_runner.aiter as mra  # noqa: E402
M_ = load_driver("moe_mxfp4")
w13, w13_s, w2, w2_s = M_.make_weight_set(0, dev)
pw13, pw2, pw13_s, pw2_s = M_.prepare_weights(w13, w13_s, w2, w2_s)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig  # noqa: E402
stub = types.SimpleNamespace(config=MoeRunnerConfig(activation="silu", gate_up_interleaved=False))
qi = mra.AiterMoeQuantInfo(w13_weight=pw13, w2_weight=pw2, quant_type=mra.AiterQuantType.PER_1X32,
                           w13_scale=pw13_s, w2_scale=pw2_s)
for bs in (10, 12):
    x, ids, wts = M_.make_case(bs, 0, dev)
    ri = mra.AiterRunnerInput(hidden_states=x, topk_ids=ids, topk_weights=wts, quant_type=mra.AiterQuantType.PER_1X32)
    out = mra.AiterRunnerCore.run(stub, ri, qi, {})
    print("moe", bs, tuple(out.hidden_states.shape), flush=True)
print("SMOKE DONE")
