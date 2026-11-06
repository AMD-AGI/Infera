###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""One replay stands in for many: its step mix re-derives another deployment's kernels.

The kernels a step launches follow from the step's composition and the
deployment alone, so a recorded mix under a new split must report what a fresh
replay of that split reports, and re-pricing it must charge what that replay
spent. Measured kernel speedups fed back scale each step by its own kernels;
the kernel tasks hand an agent each kernel's shapes, tensors and reference.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from infera.projection.core.projection.inference_projection import des as des_mod
from infera.projection.core.projection.inference_projection import gemm_shapes as gs
from infera.projection.core.projection.inference_projection import kernel_shapes as ks
from infera.projection.core.projection.inference_projection import kernel_speedups as ksp
from infera.projection.core.projection.inference_projection import kernel_tasks as kt
from infera.projection.core.projection.inference_projection import step_mix as sm
from infera.projection.core.projection.training_config import ModelParallelConfig

from .test_des_gemm_shapes import _gqa, _moe, _Projector, _Req
from .test_des_kernel_shapes import _Backend, _kernels
from .test_des_kernel_shapes import _cfg as _rank_cfg


class _Timed(_Projector):
    """A cost model whose kernels have times, scaled by ``scale``."""

    def __init__(self, cfg, scale=1.0):
        super().__init__(cfg)
        self._gemm = self._sdpa = _Backend()
        self._comm = None
        self.scale = scale

    def decode_step_latency_ms(self, batch, ctx, q_len):
        return self.scale * super().decode_step_latency_ms(batch, ctx, q_len)

    def mixed_step_latency_ms(self, *a):
        return self.scale * super().mixed_step_latency_ms(*a)


def _cfg(tp=8, ep=8, mc=None, **req):
    return SimpleNamespace(
        request_config=_Req(max_concurrency=4, cudagraph_mode=None, **req),
        model_config=mc or _moe(),
        model_parallel_config=ModelParallelConfig(
            tensor_model_parallel_size=tp, expert_model_parallel_size=ep
        ),
    )


def _replay(proj, trace=None, requests=12):
    trace = trace if trace is not None else ks.KernelTrace()
    with gs.recording(trace):
        point = des_mod.simulate_once(
            proj.cfg,
            proj,
            rate_per_s=0.0,
            arrival_model="closed",
            num_requests=requests,
            warmup_frac=0.0,
            closed_loop_clients=4,
        )
    return trace, point


def _rows(trace):
    return sorted(json.dumps(r, sort_keys=True) for r in trace.kernels())


def _mix(trace, proj, point=None, **sched):
    return sm.from_trace(
        trace,
        {"engine": proj},
        {"arrival": "closed", "think_ms": 0.0, **sched},
        sm.deployment_of(proj.cfg),
        sm.des_summary(point) if point is not None else {},
    )


# ── reuse ───────────────────────────────────────────────────────────────────


def test_a_saved_mix_reports_another_split_s_kernels_as_its_own_replay_would(tmp_path):
    p8 = _Timed(_cfg(tp=8, ep=8))
    t8, _ = _replay(p8)
    sm.dump(_mix(t8, p8), str(tmp_path / "mix.json"))
    mix = sm.load(str(tmp_path / "mix.json"))
    assert mix.num_steps == t8.num_steps
    assert _rows(sm.to_trace(mix, {"engine": p8})) == _rows(t8)

    p4 = _Timed(_cfg(tp=4, ep=2, linear_weight_dtype="fp8"))
    fresh, _ = _replay(p4)
    reused = sm.to_trace(mix, {"engine": p4})
    assert _rows(reused) == _rows(fresh)
    assert {r["dtype"] for r in reused.kernels() if r["op"] == "attn_qkv"} == {"fp8"}


def test_re_pricing_a_mix_charges_what_the_new_deployment_s_replay_spent():
    p = _Timed(_cfg())
    trace, point = _replay(p)
    mix = _mix(trace, p, point)
    faster = _Timed(_cfg(tp=4, ep=4), scale=0.5)
    _, fresh = _replay(faster)
    est = sm.reprice(mix, sm.to_trace(mix, {"engine": faster}), {"engine": faster})
    engine = est.pools["engine"]
    assert engine["busy_ms"] == pytest.approx(0.5 * engine["recorded_busy_ms"], rel=1e-6)
    assert engine["ratio"] == pytest.approx(0.5, rel=1e-6)
    assert engine["busy_ms_with_speedups"] is None
    # A closed loop with no think time is busy end to end: the makespan moves with the steps.
    assert est.des["makespan_ms"] == pytest.approx(fresh.makespan_ms, rel=1e-3)
    assert est.des["system_throughput_tps"] == pytest.approx(fresh.system_throughput_tps, rel=1e-3)


def _pool(*steps):
    p = sm.Pool()
    for decode, cost in steps:
        p.priced[(ks.Step(decode.q, decode.n, decode, None, None), cost)] += 1
    return p


def test_reuse_is_refused_when_the_schedule_changed_and_flagged_when_it_may_have():
    p = _pool((ks.Seqs(8, 8, 1, 2048, 1024), ("decode", 8, 1024)))
    rec = {"arrival": "closed", "think_ms": 0.0, "max_concurrency": 8, "input_len": 1024}
    mix = sm.Mix({"engine": p}, rec, {}, {})
    ok = sm.check(mix, dict(rec))
    assert (ok.exact, ok.allowed, ok.reasons) == (True, True, [])

    moved = sm.check(mix, {**rec, "max_concurrency": 16})
    assert not moved.allowed and "max_concurrency" in moved.reasons[0]

    open_loop = sm.check(
        sm.Mix({"engine": p}, {**rec, "arrival": "poisson"}, {}, {}), rec | {"arrival": "poisson"}
    )
    assert open_loop.allowed and not open_loop.exact
    thinking = sm.check(
        sm.Mix({"engine": p}, {**rec, "think_ms": 50.0}, {}, {}), rec | {"think_ms": 50.0}
    )
    assert thinking.allowed and not thinking.exact

    small = sm.check(mix, rec, kv_capacity={"engine": 4096})
    assert small.allowed and not small.exact and "8192" in small.reasons[0]
    assert sm.check(mix, rec, kv_capacity={"engine": 0}).exact  # unbounded


def test_a_shorter_decode_window_caps_the_reused_spans():
    mc = _gqa(
        sink_sliding_window=128, sink_window_even_layers_only=True, sink_window_decode_capped=True
    )
    step = ks.Step(4, 4, ks.Seqs(4, 4, 1, 8192, 4096), None, None)
    capped = sm._rewindow(step, SimpleNamespace(model_config=mc))
    assert (capped.decode_window.max_kv, capped.decode_window.mean_kv) == (256, 256)
    assert capped.decode == step.decode


# ── speedups ────────────────────────────────────────────────────────────────


def test_the_most_specific_speedup_wins_and_a_wildcard_only_fills_in():
    sp = ksp.Speedups.parse(
        {
            "schema": ksp.SCHEMA,
            "entries": [
                {"op": "*", "speedup": 1.1},
                {"op": "attn_qkv", "speedup": 1.5},
                {"op": "attn_qkv", "ranges": {"m": [1, 64]}, "baseline_us": 30, "tuned_us": 10},
                {"op": "attn_o", "dims": {"m": 8}, "us": 2.0},
            ],
        }
    )
    assert sp.factor({"op": "attn_qkv", "m": 64}, 9.0) == 3.0
    assert sp.factor({"op": "attn_qkv", "m": 65}, 9.0) == 1.5
    assert sp.factor({"op": "attn_o", "m": 8}, 9.0) == 4.5  # simulated over measured
    assert sp.factor({"op": "mlp_down", "m": 8}, 9.0) == 1.1
    assert (
        ksp.Speedups.parse([{"op": "x", "pool": "decode", "speedup": 2}]).factor(
            {"op": "x", "pool": "prefill"}, 1.0
        )
        == 1.0
    )
    for bad in ({"speedup": 2}, {"op": "x"}, {"op": "x", "speedup": 0}):
        with pytest.raises(ValueError):
            ksp.Speedups.parse([bad])
    with pytest.raises(ValueError):
        ksp.Speedups.parse({"schema": "something.else", "entries": []})


def test_a_speedup_rewrites_the_row_and_scales_a_step_by_its_own_kernels():
    sp = ksp.Speedups.parse([{"op": "a", "speedup": 2.0, "source": "gemm-tune"}])
    row = {"op": "a", "calls": 3, "est_us": 10.0, "est_total_ms": 0.03, "est_source": "origami"}
    sp.annotate(row)
    assert (row["est_us_base"], row["speedup"], row["est_us"]) == (10.0, 2.0, 5.0)
    assert (row["est_total_ms"], row["est_source"]) == (0.015, "origami+gemm-tune")
    # 10 us of `a` halves; 30 us of `b` stays: Amdahl's (5 + 30) / 40.
    timed = [({"op": "a"}, 10.0, 1), ({"op": "b"}, 15.0, 2), ({"op": "c"}, None, 4)]
    assert sp.step_factor(timed) == pytest.approx(35.0 / 40.0)


def test_speedups_shorten_the_replay_and_the_re_pricing_alike():
    p = _Timed(_cfg())
    trace, base = _replay(p)
    twice = ksp.Speedups.parse([{"op": "*", "speedup": 2.0}])
    with ksp.applying(twice):
        _, sped = _replay(_Timed(_cfg()))
    assert ksp.active() is None
    # Every kernel with a time halves (collectives have none here), so every step does.
    assert sped.makespan_ms == pytest.approx(0.5 * base.makespan_ms, rel=1e-6)
    mix = _mix(trace, p, base)
    est = sm.reprice(mix, sm.to_trace(mix, {"engine": p}), {"engine": p}, twice)
    assert est.pools["engine"]["ratio"] == pytest.approx(0.5, rel=1e-6)
    assert est.des["makespan_ms"] == pytest.approx(sped.makespan_ms, rel=1e-3)
    trace.speedups = twice
    qkv = [r for r in trace.kernels() if r["op"] == "attn_qkv"]
    assert all(r["speedup"] == 2.0 and r["est_us"] == r["est_us_base"] / 2 for r in qkv)


# ── kernel tasks ────────────────────────────────────────────────────────────


def _tasks():
    p = _Timed(_cfg(linear_weight_dtype="fp8"))
    trace, _ = _replay(p)
    timer = SimpleNamespace(bw=8000.0, peak=2500.0)
    return kt.build(trace.kernels(), p.cfg, timer=timer)


def _strings(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from _strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from _strings(v)
    elif isinstance(x, str):
        yield x


def test_each_task_is_a_kernel_an_agent_can_write_without_the_trace():
    doc = _tasks()
    tasks = doc["tasks"]
    assert [t["task_id"] for t in tasks] == [f"t{i:03d}" for i in range(1, len(tasks) + 1)]
    times = [t["est_total_ms"] for t in tasks]
    assert times == sorted(times, reverse=True)
    assert sum(t["share_pct"] for t in tasks) == pytest.approx(100.0, abs=0.1)
    qkv = next(t for t in tasks if t["op"] == "attn_qkv")
    assert qkv["family"] == "gemm" and qkv["dtype"] == "fp8"
    act = doc["deployment"]["dtypes"]["activations"]
    assert [x["dtype"] for x in qkv["inputs"]] == [act, "fp8"]
    assert {"m", "n", "k"} <= set(qkv["cases"][0]["dims"])
    c01 = qkv["cases"][0]
    assert c01["case_id"] == "c01"
    d = c01["dims"]
    assert [x["shape"] for x in c01["inputs"]] == [[d["m"], d["k"]], [d["n"], d["k"]]]
    decode = next(t for t in tasks if t["op"] == "attn_decode")["cases"][0]
    table = next(x for x in decode["inputs"] if x["name"] == "block_table")
    # One row per sequence, one entry per 16-token block (vLLM's default) of its longest span.
    assert table["shape"] == [decode["dims"]["batch"], -(-decode["dims"]["max_kv"] // 16)]
    for t in tasks:
        signature = [t["inputs"], t["outputs"], t["tuning_keys"], t["cases"]]
        assert not [s for s in _strings(signature) if "$" in s], t["op"]
        assert t["cases_coverage_pct"] is None or (
            t["cases_coverage_pct"] >= 100 * kt.CASE_COVERAGE or len(t["cases"]) == kt.MAX_CASES
        )
        if t["reference"]:
            compile(t["reference"], t["op"], "exec")
        if t["speed_of_light_ms"]:
            # The fake backend charges every GEMM 1 us, some under speed of light.
            assert (t["warning"] is not None) == (t["est_total_ms"] < t["speed_of_light_ms"])
    assert doc["deployment"]["parallel"]["tp"] == 8
    assert doc["deployment"]["cudagraph_capture_sizes"][:4] == [1, 2, 4, 8]


def test_the_gemm_reference_runs():
    torch = pytest.importorskip("torch")
    ref = next(t for t in _tasks()["tasks"] if t["family"] == "gemm")["reference"]
    scope = {"torch": torch}
    exec(ref, scope)
    a, b = torch.randn(4, 8), torch.randn(16, 8)
    assert scope["ref"](a, b).shape == (4, 16)


def test_the_tasks_render_for_a_reader(tmp_path):
    doc = _tasks()
    kt.write(doc, str(tmp_path / "tasks.json"))
    assert json.loads((tmp_path / "tasks.json").read_text())["schema"] == kt.SCHEMA
    md = (tmp_path / "tasks.md").read_text()
    assert "## t001:" in md and "| c01 |" in md and "```python" in md


# ── what the replay launches under graphs ───────────────────────────────────


def test_engines_capture_their_own_graph_ladders_and_eager_captures_none():
    def sizes(**req):
        return gs.capture_sizes(SimpleNamespace(request_config=SimpleNamespace(**req)))

    vllm = sizes()
    assert vllm[:5] == (1, 2, 4, 8, 16) and vllm[-2:] == (496, 512) and 264 not in vllm
    assert sizes(serving_engine="sglang")[:6] == (1, 2, 4, 8, 12, 16)
    assert sizes(cudagraph_mode="none") == () == sizes(cudagraph_mode="EAGER")
    assert sizes(cudagraph_capture_sizes=[8, 1, 8]) == (1, 8)
    cfg = SimpleNamespace(request_config=SimpleNamespace())
    assert (gs.graph_tokens(cfg, 3, True), gs.graph_tokens(cfg, 3, False)) == (4, 3)
    assert gs.graph_tokens(cfg, 600, True) == 600  # past the largest graph: eager


def test_a_graphed_decode_pads_the_linears_but_not_attention_or_sampling():
    ops = _kernels(_rank_cfg(_moe(), tp=8, ep=8), decode=[100] * 3)
    assert ops["attn_qkv"][0][1]["m"] == 4
    assert ops["moe_topk"][0][1]["tokens"] == 4
    assert ops["attn_decode"][0][1]["batch"] == 3
    assert ops["lm_head"][0][1]["m"] == 3
    mixed = _kernels(_rank_cfg(_moe(), tp=8, ep=8), decode=[100] * 3, prefill=[(10, 0)])
    assert mixed["attn_qkv"][0][1]["m"] == 13


def _mla():
    return _gqa(
        multi_latent_attention=True,
        num_attention_heads=128,
        qk_head_dim=128,
        qk_pos_emb_head_dim=64,
        v_head_dim=128,
        q_lora_rank=1536,
        kv_lora_rank=512,
        hidden_size=7168,
    )


def test_mla_decode_runs_absorbed_and_prefill_up_projects_its_keys():
    ops = _kernels(_rank_cfg(_mla(), tp=8), decode=[100] * 3)
    assert "attn_kv_b" not in ops
    q, o = ops["attn_q_absorb"][0][1], ops["attn_o_absorb"][0][1]
    # 16 local heads, each over the 4 graph-padded decode tokens.
    assert (q["groups"], q["m"], q["n"], q["k"]) == (16, 4, 512, 128)
    assert (o["groups"], o["m"], o["n"], o["k"]) == (16, 4, 128, 512)
    ops = _kernels(_rank_cfg(_mla(), tp=8), decode=[100] * 3, prefill=[(64, 1000)])
    kv_b = ops["attn_kv_b"][0][1]
    assert kv_b["m"] == ks._up(1064, ks.KV_BUCKET)  # the new tokens and the cached context
    assert ops["attn_q_absorb"][0][1]["m"] == 3


def test_a_step_s_cost_key_stands_for_its_composition():
    cfg = _rank_cfg(_gqa(), tp=8)
    dec = sm.step_of_cost(cfg, ("decode", 6, 1000))
    assert (dec.decode.n, dec.decode.max_kv, dec.prefill) == (6, 1024, None)
    mixed = sm.step_of_cost(cfg, ("mixed", 2, 300, 1000, 4000, 2))
    assert (mixed.tokens, mixed.prefill.n, mixed.prefill.q, mixed.decode.n) == (
        302,
        2,
        320,
        2,
    )  # q bucketed as the report buckets it
