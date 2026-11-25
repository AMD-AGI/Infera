"""CPU-only tests of tool 2's pure logic (no GPU, no sglang import). Run: python3 -m pytest sglang_inproc/tool2_perop/tests -q"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

T2 = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(T2), str(T2 / "dsv41")]

import kernel_breakdown  # noqa: E402
import model_bench as mb  # noqa: E402
import shape_export  # noqa: E402
import sglang_bench as sb  # noqa: E402


def test_tree_residuals_and_leaves():
    assert mb.RESIDUALS["layer"] == ["input_layernorm", "attn", "post_attention_layernorm", "mlp"]
    assert "attn.indexer" in mb.RESIDUALS["attn"] and "attn.<rest>" not in mb.RESIDUALS["attn"]
    # leaves partition: every depth-2 row, the layer residual, and the engram outside the layer
    assert {"attn.<rest>", "mlp.<rest>", "layer.<rest>", "engram"} <= set(mb.LEAVES)
    assert "attn" not in mb.LEAVES and "layer" not in mb.LEAVES


def test_rollup_sums_leaves_exactly():
    counts = {"k1": 3, "k2": 2}
    times = {
        "k1|layer": 10.0, "k1|input_layernorm": 0.5, "k1|attn": 4.0, "k1|attn.wq_b": 1.0, "k1|mlp": 5.0,
        "k1|mlp.experts": 3.0, "k1|engram": 0.7,
        "k2|layer": 8.0, "k2|attn": 3.0, "k2|mlp": 4.0,
        "embed_tokens": 0.1, "hc_head": 0.05, "logits": 0.3,
    }
    mb.add_residuals(times, counts)
    assert times["k1|attn.<rest>"] == pytest.approx(3.0)
    assert times["k1|layer.<rest>"] == pytest.approx(10.0 - 0.5 - 4.0 - 5.0)
    totals = mb.leaf_totals(times, counts)
    layer_part = 3 * (10.0 + 0.7) + 2 * 8.0
    assert sum(totals.values()) == pytest.approx(layer_part + 0.1 + 0.05 + 0.3)
    assert totals["engram"] == pytest.approx(3 * 0.7)


def test_residual_owner_and_notes():
    assert mb.residual_owner("k|attn.wq_b") == "k|attn.<rest>"
    assert mb.residual_owner("k|input_layernorm") == "k|layer.<rest>"
    assert mb.residual_owner("k|engram") == ""
    assert mb.residual_owner("draft|layer") == ""
    note = mb.skipped_note({"k|attn.wo_a": "RuntimeError: x|y", "draft|mlp": "e", "logits": "e"})
    assert "`attn.<rest>`" in note and "the `draft model` row" in note and "nowhere" in note and "x\\|y" in note
    cov = mb.coverage_note({"k|input_layernorm": None, "norm": None, "k|attn": None}, {"k|attn": 1})
    assert "input_layernorm" in cov and "unattributed" in cov


def _layer(ratio, compressor=False, indexer=False, hash_=False, engram=False):
    attn = SimpleNamespace(compress_ratio=ratio, compressor=object() if compressor else None,
                           indexer=object() if indexer else None)
    return SimpleNamespace(self_attn=attn, mlp=SimpleNamespace(is_hash=hash_), engram=object() if engram else None)


def test_layer_kind_dsv41():
    assert mb.layer_kind(_layer(0, hash_=True), 0, 20) == "moe+hash+c0"
    assert mb.layer_kind(_layer(0, hash_=True, engram=True), 1, 20) == "moe+hash+c0+engram"
    assert mb.layer_kind(_layer(2, True, True), 2, 20) == "moe+c2+kvsrc+indexer"
    assert mb.layer_kind(_layer(2), 3, 20) == "moe+c2"
    assert mb.layer_kind(_layer(1, True, True), 20, 20) == "moe+c1+kvsrc+indexer+late"
    assert mb.layer_kind(_layer(1, False, True), 24, 20) == "moe+c1+indexer+late"
    # DSv4 compressors on c4/c128 are not tagged kvsrc
    assert mb.layer_kind(_layer(4, True, True), 5, None) == "moe+c4+indexer"


def test_ids_compact():
    assert mb._ids([0, 1, 2, 5, 7, 8]) == "0-2,5,7-8"


def test_rotation_plan():
    MB = 1 << 20
    assert mb.rotation_plan(300 * MB) == 0  # bigger than the MALL: cold anyway
    assert mb.rotation_plan(200 * MB) == 2
    assert mb.rotation_plan(150 * MB) == 3
    assert mb.rotation_plan(3 * MB) == 170
    # a ring capped at 256 slots that still fits in the MALL would stay cache-hot: no rotation (upstream rule)
    assert mb.rotation_plan(1 * MB) == 0


def test_snapshot_clones_tensors_and_structs():
    from typing import NamedTuple

    class P(NamedTuple):
        a: torch.Tensor
        b: torch.Tensor

    t = torch.ones(3)
    p = P(t, t)
    fb = object()
    out = mb.snapshot(((p, [t]), {"x": t, "fb": fb}))
    (sp, lst), d = out
    assert sp.a is not t and sp.a is sp.b is lst[0] is d["x"]  # one clone per id
    assert d["fb"] is fb
    t.add_(1)
    assert float(sp.a.sum()) == 3.0


def test_partitions_guard():
    assert sb._partitions({"k|attn": 1}, "k", "attn.wq_b")
    assert not sb._partitions({}, "k", "attn.wq_b")
    assert sb._partitions({}, "k", "layer")


def test_graph_reps_for():
    assert sb.graph_reps_for(1) == 20 and sb.graph_reps_for(16384) == 1 and sb.graph_reps_for(4096) == 4


def test_table_alignment():
    s = sb.table([("a", 1.23456), ("bb", 2.0)], ("k", "v"))
    lines = s.splitlines()
    assert len({len(x) for x in lines}) == 1 and "1.235" in s


def test_balanced_forward_keeps_shared_column():
    from typing import NamedTuple

    class Out(NamedTuple):
        topk_weights: torch.Tensor
        topk_ids: torch.Tensor

    seen = {}

    class Experts:
        num_experts = 17  # 16 routed + 1 fused shared
        num_fused_shared_experts = 1

        def forward(self, hidden, topk_output):
            seen["ids"] = topk_output.topk_ids
            return hidden

    e = Experts()
    e.forward = sb._balanced_forward(e)
    ids = torch.full((4, 7), 16, dtype=torch.int32)  # 6 routed + shared id 16
    e.forward(torch.zeros(4, 2), Out(torch.ones(4, 7), ids))
    got = seen["ids"]
    assert (got[:, 6] == 16).all() and int(got[:, :6].max()) < 16
    assert all(len(set(r.tolist())) == 6 for r in got[:, :6])
    first = got.clone()
    e.forward(torch.zeros(4, 2), Out(torch.ones(4, 7), ids))
    assert torch.equal(first, seen["ids"])  # seeded and cached


def test_balanced_forward_refuses_lazy_routing():
    class Experts:
        num_experts = 4

        def forward(self, hidden, topk_output):
            return hidden

    e = Experts()
    e.forward = sb._balanced_forward(e)
    with pytest.raises(RuntimeError):
        e.forward(torch.zeros(1), SimpleNamespace())


def _trace(kernels):
    """Minimal Kineto trace: one module scope, launches, and kernels."""
    ev = [{"ph": "X", "cat": "user_annotation", "name": kernel_breakdown.ROOT, "ts": 0, "dur": 1000, "pid": 1, "tid": 1}]
    for i, (name, dur, frame) in enumerate(kernels):
        ev.append({"ph": "X", "cat": "cuda_runtime", "name": "hipLaunchKernel", "ts": 10 + i * 10, "dur": 2,
                   "pid": 1, "tid": 1, "args": {"correlation": i}})
        ev.append({"ph": "X", "cat": "python_function", "name": frame, "ts": 5 + i * 10, "dur": 8, "pid": 1, "tid": 1})
        ev.append({"ph": "X", "cat": "kernel", "name": name, "ts": 100 + i * 10, "dur": dur, "args": {"correlation": i}})
    return ev


def test_kernel_breakdown_summarize_and_rank():
    ev = _trace([
        ("gemm_kernel", 300.0, "/sgl-workspace/aiter/aiter/ops/gemm.py(10): gemm_a8w8"),
        ("cross_device_reduce_1stage", 100.0, "/sgl-workspace/sglang/python/sglang/srt/x.py(3): all_reduce"),
    ])
    res = kernel_breakdown.summarize(ev)
    assert sum(v[0] for v in res.values()) == pytest.approx(0.4)
    ranking = kernel_breakdown.Ranking.of({"attn.wq_b": res}, {"attn.wq_b": 0.5}, frozenset())
    targets = ranking.targets()
    assert targets and not targets[0].collective
    assert "target" in ranking.section(sb.table)


def test_kernel_totals_multiplies_by_count():
    profiles = {"k|attn.wq_b": {("g", "api"): (0.1, 1, ())}, "embed_tokens": {("e", "api"): (0.01, 1, ())}}
    times = {"k|attn.wq_b": 0.1, "k|attn": 1.0, "embed_tokens": 0.01}
    tot = mb.kernel_totals(profiles, {"k": 4}, times)
    assert tot["attn.wq_b"][("g", "api")][0] == pytest.approx(0.4)
    assert tot["embed_tokens"][("e", "api")][1] == 1


def test_shape_export_recorder(tmp_path, monkeypatch):
    monkeypatch.setenv("AKLR_EXPORT", "1")
    monkeypatch.setattr(shape_export, "aklr_available", lambda: False)
    lin = torch.nn.Linear(8, 4)
    with shape_export.export_scope({"k|attn.wq_b": lin}, "t", str(tmp_path), 0) as rec:
        lin(torch.zeros(2, 8))
        lin(torch.zeros(2, 8))
        assert rec is not None
    files = list((tmp_path / "aklr_export" / "rank0").glob("calls.pid*.t.jsonl"))
    assert len(files) == 1
    rows = [json.loads(x) for x in files[0].read_text().splitlines()]
    assert rows[0]["op"] == "k|attn.wq_b" and rows[0]["count"] == 2
    assert rows[0]["args"][0]["shape"] == [2, 8] and rows[0]["weights"]["weight"]["shape"] == [4, 8]


def test_shape_export_disabled(monkeypatch):
    monkeypatch.delenv("AKLR_EXPORT", raising=False)
    with shape_export.export_scope({}, "t") as rec:
        assert rec is None


def test_capacity_math():
    mr = SimpleNamespace(
        server_args=None, max_total_num_tokens=100_000,
        token_to_kv_pool_allocator=SimpleNamespace(swa_attn_allocator=SimpleNamespace(size=50_000)),
        req_to_token_pool=SimpleNamespace(size=64),
    )
    assert mb.capacities(mr) == {"full KV": 100_000, "SWA": 50_000, "request slots": 64}
    assert mb._pages(1025, 256) == 1280


def test_coverage_points_backend_ops_at_sources_row():
    cov = mb.coverage_note({"k|attn.indexer": None, "k|attn.sources": None}, {"k|attn.sources": 1})
    assert "attn.sources (timed through the backend call site)" in cov


def test_layer_ops_adds_sources_site_only_for_low_ratio_owners():
    class M(torch.nn.Module):
        pass

    def layer(ratio, comp, idx):
        lay = M()
        lay.self_attn = M()
        lay.self_attn.compress_ratio = ratio
        lay.self_attn.compressor = M() if comp else None
        lay.self_attn.indexer = M() if idx else None
        lay.mlp = M()
        lay.engram = None
        return lay

    assert "attn.sources" in mb.layer_ops(layer(2, True, True))
    assert "attn.sources" in mb.layer_ops(layer(1, False, True))
    assert "attn.sources" not in mb.layer_ops(layer(1, False, False))
    ops4 = mb.layer_ops(layer(4, True, True))
    assert "attn.sources" not in ops4 and "attn.indexer" in ops4


def test_shape_export_dispatcher_ops(tmp_path, monkeypatch):
    monkeypatch.setenv("AKLR_EXPORT", "1")
    monkeypatch.setattr(shape_export, "aklr_available", lambda: False)
    lib = torch.library.Library("t2test", "DEF")
    lib.define("twice(Tensor x) -> Tensor")
    lib.impl("twice", lambda x: x * 2, "CPU")
    with shape_export.export_scope({}, "d", str(tmp_path), 0):
        torch.ops.t2test.twice(torch.ones(3, 2))
        torch.ones(2) + 1  # aten: not recorded
    rows = [json.loads(x) for x in next((tmp_path / "aklr_export" / "rank0").glob("calls*.jsonl")).read_text().splitlines()]
    ops = [r for r in rows if r["kind"] == "op"]
    assert len(ops) == 1 and "t2test" in ops[0]["api"] and ops[0]["args"][0]["shape"] == [3, 2]
