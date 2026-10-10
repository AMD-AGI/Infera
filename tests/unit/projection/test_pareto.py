"""Pareto sweep: spec parsing, anchor choice, the frontier and the writers."""

from __future__ import annotations

import csv
import json
import os

import pytest

from infera.projection.core.projection.inference_projection import pareto as P


def _pt(x: float, y: float, **kw) -> P.ParetoPoint:
    base = dict(
        model="m",
        checkpoint="M",
        engine="sglang",
        precision="fp8",
        gpu_arch="mi355x",
        scenario="8k/1k",
        isl=8192,
        osl=1024,
        mode="simulate",
        kv_config="hbm",
        parallelism="TP8",
        tp=8,
        ep=1,
        dp_attn=1,
        pp=1,
        gpus=8,
        concurrency=1,
    )
    base.update(kw)
    return P.ParetoPoint(**base, interactivity=x, total_tput_per_gpu=y)


def test_parallelism_spec():
    p = P.Parallelism.parse("tp8ep8dp8")
    assert (p.tp, p.ep, p.dp_attn, p.pp, p.gpus) == (8, 8, 8, 1, 8)
    assert p.label == "TP8 EP8 DPA8"
    assert P.Parallelism.parse("TP4pp2").gpus == 8
    with pytest.raises(ValueError):
        P.Parallelism.parse("tp2dp4")
    with pytest.raises(ValueError):
        P.Parallelism.parse("8")


def test_model_scenario_kv_specs(tmp_path):
    assert P.ModelSpec.parse("glm5@GLM-5.2-MXFP4") == P.ModelSpec("glm5", "GLM-5.2-MXFP4")
    assert P.ModelSpec.parse("llama3_8b").checkpoint == "llama3_8b"
    s = P.Scenario.parse("8k/1k")
    assert (s.isl, s.osl, s.trace) == (8192, 1024, None)
    tr = tmp_path / "t.jsonl"
    tr.write_text(
        "".join(json.dumps(dict(input_length=n, output_length=10)) + "\n" for n in (100, 300))
    )
    s = P.Scenario.parse(f"trace:{tr}")
    assert (s.name, s.isl, s.osl) == ("trace:t", 200, 10)
    assert P.Scenario.parse(f"trace:{tr}", trace_max_requests=1).isl == 100
    assert P.KvConfig.parse("hbm") == P.KvConfig("hbm")
    kv = P.KvConfig.parse("dram+ssd=dram:512,ssd:4000")
    assert kv.label == "dram+ssd" and kv.tiers == ("dram:512", "ssd:4000")
    assert P.KvConfig.parse("ssd:100").label == "ssd"


def test_trace_per_concurrency(tmp_path):
    for c, n in ((4, 100), (8, 300)):
        (tmp_path / f"lanes_c{c}.jsonl").write_text(
            json.dumps(dict(timestamp=0, input_length=n, output_length=5, hash_ids=[1])) + "\n"
        )
    s = P.Scenario.parse(f"trace:{tmp_path}/lanes_c{{c}}.jsonl")
    assert s.name == "trace:lanes" and s.isl == 100
    tasks, uncovered = P.build_tasks(
        models=[P.ModelSpec("glm5", "GLM")],
        engines=["sglang"],
        precisions=["fp8"],
        parallelisms=[P.Parallelism(8)],
        concurrencies=[4, 8, 16],
        scenarios=[s],
        modes=["simulate"],
        kv_configs=[P.KvConfig("hbm")],
        anchors=[],
        workdir=str(tmp_path / "w"),
    )
    assert [t["point"]["isl"] for t in tasks] == [100, 300]
    assert [t["argv"][t["argv"].index("--des-mooncake-trace") + 1][-14:] for t in tasks] == [
        "lanes_c4.jsonl",
        "lanes_c8.jsonl",
    ]
    assert len(uncovered) == 1 and "no trace" in uncovered[0].reason
    # An attention-DP rank replays its own share of the lanes.
    tasks, _ = P.build_tasks(
        models=[P.ModelSpec("glm5", "GLM")],
        engines=["sglang"],
        precisions=["fp8"],
        parallelisms=[P.Parallelism(8, 8, 2)],
        concurrencies=[16],
        scenarios=[s],
        modes=["simulate"],
        kv_configs=[P.KvConfig("hbm")],
        anchors=[],
        workdir=str(tmp_path / "w"),
    )
    assert tasks[0]["argv"][tasks[0]["argv"].index("--des-mooncake-trace") + 1].endswith(
        "_c8.jsonl"
    )


def _anchor(tp, ep=1, isl=8192, ckpt="GLM-5.2-MXFP4", backend="sglang", mtime=0.0, name=""):
    return dict(
        path=name or f"a_tp{tp}_ep{ep}_{isl}_{backend}",
        backend=backend,
        checkpoint=ckpt,
        tp=tp,
        ep=ep,
        isl=isl,
        gpus=tp,
        mtime=mtime,
    )


def test_pick_anchor_rules():
    pool = [
        _anchor(2),
        _anchor(4),
        _anchor(4, ep=4),
        _anchor(8),
        _anchor(4, backend="vllm"),
        _anchor(4, isl=1024, mtime=9.0),
    ]
    pick = lambda **kw: P.pick_anchor(
        pool,
        checkpoint="GLM-5.2-MXFP4",
        engine="sglang",  # noqa: E731
        isl=8192,
        **kw,
    )
    # Narrow targets restore at their own width; wide ones from TP4, never TP8.
    assert pick(par=P.Parallelism(2))["tp"] == 2
    a = pick(par=P.Parallelism(8))
    assert (a["tp"], a["ep"], a["isl"]) == (4, 1, 8192)
    assert pick(par=P.Parallelism(8, ep=8))["ep"] == 4
    # No anchor wider than the target, nor from another engine or checkpoint.
    assert (
        P.pick_anchor(
            [_anchor(4)],
            checkpoint="GLM-5.2-MXFP4",
            engine="sglang",
            par=P.Parallelism(2),
            isl=8192,
        )
        is None
    )
    assert (
        P.pick_anchor(
            pool, checkpoint="GLM-5.2-MXFP4", engine="atom", par=P.Parallelism(4), isl=8192
        )
        is None
    )
    assert (
        P.pick_anchor(pool, checkpoint="Other", engine="sglang", par=P.Parallelism(4), isl=8192)
        is None
    )
    # A checkpoint that names its precision does not cover another one.
    assert pick(par=P.Parallelism(4), precision="fp8") is None
    assert pick(par=P.Parallelism(4), precision="fp4")["tp"] == 4


def test_pareto_front_and_series_grouping():
    pts = [
        _pt(10, 100),
        _pt(20, 90),
        _pt(15, 80),
        _pt(30, 10),
        _pt(5, 50),
        _pt(40, 500, feasible=False),
    ]
    front = P.pareto_front(pts, "interactivity", "total_tput_per_gpu")
    assert [(p.interactivity, p.total_tput_per_gpu) for p in front] == [
        (10, 100),
        (20, 90),
        (30, 10),
    ]
    # Lower-is-better axes flip the dominance.
    for p, t in zip(pts, (50, 40, 30, 60, 70, 1)):
        p.ttft_ms = t
    front = P.pareto_front(pts, "ttft_ms", "total_tput_per_gpu")
    assert {(p.ttft_ms, p.total_tput_per_gpu) for p in front} == {(50, 100), (40, 90), (30, 80)}
    # Each (chart, series) group has its own frontier.
    b = _pt(1, 1, mode="benchmark")
    P.mark_frontiers(pts + [b], "interactivity", "total_tput_per_gpu")
    assert b.on_frontier
    assert [p.on_frontier for p in pts] == [True, True, False, True, False, False]
    assert _pt(1, 1, kv_config="dram").series() == "MI355X sglang KV:dram (simulate)"


def test_merge_replaces_reprojected_points():
    old = [_pt(1, 1, concurrency=1), _pt(2, 2, concurrency=2)]
    new = [_pt(9, 9, concurrency=2), _pt(3, 3, concurrency=2, model="other")]
    merged = P.merge_points(old, new)
    assert sorted((p.model, p.concurrency, p.interactivity) for p in merged) == [
        ("m", 1, 1),
        ("m", 2, 9),
        ("other", 2, 3),
    ]


def test_resume_skips_points_a_killed_run_streamed_out(tmp_path, monkeypatch):
    seen = {}

    def fake_run(tasks, jobs=1, sink=None):
        seen["n"] = len(tasks)
        return [P.ParetoPoint(**t["point"], interactivity=1.0, tpot_ms=1.0) for t in tasks]

    monkeypatch.setattr(P, "run", fake_run)
    argv = [
        "--model",
        "glm5",
        "--engine",
        "sglang",
        "--precision",
        "fp8",
        "--parallelism",
        "tp8",
        "--concurrency",
        "1",
        "2",
        "4",
        "--scenario",
        "1k/1k",
        "--mode",
        "simulate",
        "--out-dir",
        str(tmp_path),
    ]
    P.main(argv + ["--concurrency", "1"])
    with open(tmp_path / "pareto_points.partial.jsonl", "w") as f:
        done = P.load_points(str(tmp_path / "pareto_points.json"))[0]
        done.concurrency = 2
        f.write(json.dumps(P.asdict(done)) + "\n")
        f.write('{"truncated')
    P.main(argv + ["--resume"])
    assert seen["n"] == 1  # c1 merged before, c2 salvaged from the partial file
    pts = P.load_points(str(tmp_path / "pareto_points.json"))
    assert sorted(p.concurrency for p in pts) == [1, 2, 4]
    assert not (tmp_path / "pareto_points.partial.jsonl").exists()


def _die_on_c2(task):
    if task["point"]["concurrency"] == 2:
        os.kill(os.getpid(), 9)
    return P.ParetoPoint(**task["point"], interactivity=1.0)


def test_a_killed_worker_loses_only_its_point(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "project_point", _die_on_c2)
    base = _pt(0, 0)
    tasks = [dict(point={**P.asdict(base), "concurrency": c}) for c in (1, 2, 4)]
    for t in tasks:
        for k in ("feasible", "reason", "on_frontier", "interactivity", "total_tput_per_gpu"):
            t["point"].pop(k)
    sink = tmp_path / "s.jsonl"
    pts = P.run(tasks, jobs=2, progress=False, sink=str(sink))
    by_c = {p.concurrency: p for p in pts}
    assert by_c[1].feasible and by_c[4].feasible
    assert not by_c[2].feasible and by_c[2].reason.startswith("worker failed")
    assert len(P.load_partial(str(sink))) == 3


def test_point_argv_attention_dp_and_tiers():
    kw = dict(
        scenario=P.Scenario("8k/1k", 8192, 1024),
        trace_path=None,
        precision="mxfp4",
        engine="sglang",
        par=P.Parallelism(8, 8, 8),
        concurrency=64,
        gpu_arch="mi355x",
        hbm_gb=288.0,
        num_requests=256,
        checkpoint="GLM",
    )
    s = " ".join(
        P.point_argv(**kw, anchor=None, kv=P.KvConfig("dram", ("dram:256",)), pool_from_memory=True)
    )
    assert "--max-concurrency 8 " in s  # 64 clients over 8 attention ranks
    assert "--des-num-requests 32 " in s
    assert "--attention-dp-size 8" in s
    assert "--max-num-batched-tokens 2048" in s
    assert "--kv-tier dram:256" in s and "--des-kv-pool-from-memory" in s
    assert "--load-benchmark" not in s
    s = " ".join(P.point_argv(**kw, anchor="a.json"))
    assert "--load-benchmark a.json --bench-model GLM" in s and "--kv-tier" not in s


def test_writers_round_trip(tmp_path):
    pts = [
        _pt(10, 100),
        _pt(20, 90, mode="benchmark", anchor="a.json"),
        _pt(1, 1, feasible=False, reason="weights exceed HBM"),
    ]
    P.mark_frontiers(pts, "interactivity", "total_tput_per_gpu")
    res = P.ParetoResult(points=pts)
    out = P.write_outputs(res, str(tmp_path), meta={"argv": []})
    rows = list(csv.DictReader(open(out["csv"][0])))
    assert len(rows) == 3 and rows[2]["reason"] == "weights exceed HBM"
    assert len(P.load_points(out["json"][0])) == 3
    html = open(out["html"][0]).read()
    assert "<script" in html and '"8k/1k"' in html
    assert len(out["svg"]) == 1 and open(out["svg"][0]).read().startswith("<svg")
    assert all(os.path.exists(p) for p in out["png"])
