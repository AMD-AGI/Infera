"""Function-level check of the vLLM overlay: rocm_fp8_paged_mqa_logits with VLLM_ROCM_DSA_GLUON_PAGED_MQA_LOGITS on
(gfx950 gluon kernel, sglang #47897) against off (aiter), on GLM-5.2 decode shapes, with 1-D per-request and 2-D
per-row context lengths, eager and inside a CUDA graph. Run in the image with the overlay files bind-mounted.
"""

import inspect
import os
import sys

import torch

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from bench_paged_logits import MAX_MODEL_LEN, NEXT_N, make_inputs  # noqa: E402

from vllm.v1.attention.ops import rocm_aiter_mla_sparse as ops  # noqa: E402
from vllm.v1.worker.workspace import init_workspace_manager  # noqa: E402

dev = torch.device("cuda")
init_workspace_manager(dev)


os.environ["VLLM_ROCM_DSA_GLUON_LPT_GROUP_ROWS"] = str(NEXT_N)
# v3 overlay: the caller passes the flattened group (indexer metadata write_max_decode_len) instead of the env
V3 = "flat_group_rows" in inspect.signature(ops.rocm_fp8_paged_mqa_logits).parameters
KW = {"flat_group_rows": NEXT_N} if V3 else {}


def use_gluon(q, kv):
    if V3:
        return ops._use_gluon_paged_mqa_logits(q, kv, kv.shape[1], NEXT_N)
    return ops._use_gluon_paged_mqa_logits(q, kv, kv.shape[1])


def call(inp, lens, gluon: bool, flat: bool = False):
    os.environ["VLLM_ROCM_DSA_GLUON_PAGED_MQA_LOGITS"] = "1" if gluon else "0"
    ops._gluon_paged_mqa_logits_enabled.cache_clear()
    q, table = inp["q"], inp["table"]
    if flat:  # vLLM's flattened MTP decode on ROCm: one query per row, per-row lengths and block-table rows
        q = q.reshape(-1, 1, *q.shape[2:])
        table = table.repeat_interleave(NEXT_N, dim=0)
    out = ops.rocm_fp8_paged_mqa_logits(q, inp["kv"], inp["w"], lens, table, None, MAX_MODEL_LEN, **KW)
    return out.clone()


def visible_max_diff(a, b, row_lens):
    mask = torch.arange(a.shape[1], device=dev)[None] < row_lens[:, None]
    return float((a[mask] - b[mask]).abs().max()), float(a[mask].abs().max())


def main():
    assert ops._ON_GFX950, "gfx950 only"
    worst = 0.0
    for batch in (1, 4, 13, 26, 32, 40):
        inp = make_inputs(batch, 7 * batch)
        ctx = inp["ctx"]
        row_lens = (ctx[:, None] - (NEXT_N - 1) + torch.arange(NEXT_N, device=dev, dtype=torch.int32)).reshape(-1)
        ref = call(inp, ctx, False)
        for name, lens, flat in (("1-D", ctx, False), ("2-D", row_lens.view(batch, NEXT_N), False),
                                 ("flat", row_lens, True), ("flat2D", row_lens.view(-1, 1), True)):
            out = call(inp, lens, True, flat)
            d, m = visible_max_diff(ref, out, row_lens)
            q = inp["q"].reshape(-1, 1, *inp["q"].shape[2:]) if flat else inp["q"]
            used = use_gluon(q, inp["kv"])
            worst = max(worst, d / m)
            print(f"batch {batch:2d} rows {batch * NEXT_N:3d} lens {name}: gluon used {used}, max |diff| {d:.3g} / {m:.3g}",
                  flush=True)
    # CUDA graph: capture the gluon path once, replay with new contents in the same buffers.
    batch = 26
    inp = make_inputs(batch, 99)
    os.environ["VLLM_ROCM_DSA_GLUON_PAGED_MQA_LOGITS"] = "1"
    ops._gluon_paged_mqa_logits_enabled.cache_clear()
    static = {k: v.clone() for k, v in inp.items()}
    ops.rocm_fp8_paged_mqa_logits(static["q"], static["kv"], static["w"], static["ctx"], static["table"], None,
                                  MAX_MODEL_LEN, **KW)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        graph_out = ops.rocm_fp8_paged_mqa_logits(static["q"], static["kv"], static["w"], static["ctx"],
                                                  static["table"], None, MAX_MODEL_LEN, **KW)
    new = make_inputs(batch, 123)
    static["q"].copy_(new["q"]); static["w"].copy_(new["w"])
    static["ctx"].copy_(torch.minimum(new["ctx"], inp["ctx"]))  # stay within the captured block table
    g.replay()
    torch.cuda.synchronize()
    graph_out = graph_out.clone()  # the aiter call below reuses the same workspace
    ref = call(static, static["ctx"], False)
    row_lens = (static["ctx"][:, None] - (NEXT_N - 1) + torch.arange(NEXT_N, device=dev, dtype=torch.int32)).reshape(-1)
    d, m = visible_max_diff(ref, graph_out, row_lens)
    worst = max(worst, d / m)
    print(f"cuda graph replay batch {batch}: max |diff| {d:.3g} / {m:.3g}", flush=True)
    # The engine's form: flattened rows ([rows, 1] lengths, expanded block table) captured in a graph.
    inp = make_inputs(batch, 77)
    st = {"q": inp["q"].reshape(-1, 1, *inp["q"].shape[2:]).clone(), "kv": inp["kv"], "w": inp["w"].clone(),
          "table": inp["table"].repeat_interleave(NEXT_N, dim=0).contiguous()}
    base = (inp["ctx"][:, None] - (NEXT_N - 1) + torch.arange(NEXT_N, device=dev, dtype=torch.int32)).reshape(-1, 1)
    st["lens"] = base.clone()
    os.environ["VLLM_ROCM_DSA_GLUON_PAGED_MQA_LOGITS"] = "1"
    ops._gluon_paged_mqa_logits_enabled.cache_clear()
    ops.rocm_fp8_paged_mqa_logits(st["q"], st["kv"], st["w"], st["lens"], st["table"], None, MAX_MODEL_LEN, **KW)
    torch.cuda.synchronize()
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2):
        gout = ops.rocm_fp8_paged_mqa_logits(st["q"], st["kv"], st["w"], st["lens"], st["table"], None,
                                             MAX_MODEL_LEN, **KW)
    new = make_inputs(batch, 321)
    st["q"].copy_(new["q"].reshape(-1, 1, *new["q"].shape[2:])); st["w"].copy_(new["w"])
    st["lens"].copy_(torch.clamp(base - 1000, min=1))
    g2.replay()
    torch.cuda.synchronize()
    gout = gout.clone()
    used = use_gluon(st["q"], st["kv"])
    os.environ["VLLM_ROCM_DSA_GLUON_PAGED_MQA_LOGITS"] = "0"
    ops._gluon_paged_mqa_logits_enabled.cache_clear()
    ref = ops.rocm_fp8_paged_mqa_logits(st["q"], st["kv"], st["w"], st["lens"], st["table"], None,
                                        MAX_MODEL_LEN, **KW).clone()
    d, m = visible_max_diff(ref, gout, st["lens"].reshape(-1))
    worst = max(worst, d / m)
    print(f"cuda graph replay flattened rows {batch * NEXT_N} (v3 {V3}, gluon used "
          f"{used}): max |diff| {d:.3g} / {m:.3g}", flush=True)
    print("PASS" if worst < 1e-5 else f"FAIL (worst relative diff {worst:.3g})", flush=True)


if __name__ == "__main__":
    main()
