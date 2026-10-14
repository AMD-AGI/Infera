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


def _prefill(lengths_ms, fixed, a, b=0.0, packed=None):
    pa = dict(
        points=[dict(input_len=n, mean_ttft_ms=t) for n, t in lengths_ms],
        curve_fit=dict(fixed_ms=fixed, ms_per_token=a, ms_per_token_sq=b),
    )
    if packed:
        pa["packed"] = packed
    return pa


def test_pick_anchor_prefers_the_richer_prefill_probe():
    # A short ladder whose fit has a negative rate against a long ladder with a
    # packed probe: the second prices a step of several prompts, so it wins
    # even though the first was harvested later.
    short = dict(
        _anchor(4, isl=1024, mtime=9.0, ckpt="R1", name="plain"),
        prefill=_prefill(
            [(128, 68.6), (427, 67.1), (725, 70.5), (1024, 72.1)], 68.8, -0.005, 8.7e-6
        ),
        has_prefill=True,
        decode_ok=True,
    )
    packed = dict(
        method="p99",
        ms_per_token=0.041,
        points=[
            dict(step_tokens=1024, last_ttft_ms=79.9),
            dict(step_tokens=8192, last_ttft_ms=367.0),
        ],
    )
    long = dict(
        _anchor(4, isl=1024, mtime=1.0, ckpt="R1", name="pk"),
        prefill=_prefill(
            [(256, 66.4), (2240, 104.3), (4224, 185.2), (6208, 264.3), (8192, 401.1)],
            63.2,
            0.0117,
            3.56e-6,
            packed,
        ),
        has_prefill=True,
        decode_ok=True,
    )
    assert P._curve_is_physical(long["prefill"], 1024)
    assert not P._curve_is_physical(short["prefill"], 1024)
    assert P._packed_is_usable(long["prefill"])
    a = P.pick_anchor(
        [short, long], checkpoint="R1", engine="sglang", par=P.Parallelism(8), isl=1024
    )
    assert a["path"] == "pk"


def test_decode_context_grid_from_sibling_anchors(tmp_path):
    def art(name, isl, tp=4, backend="sglang", ckpt="M"):
        p = tmp_path / name
        p.write_text(
            json.dumps(
                dict(
                    backend=backend,
                    meta=dict(model=f"/models/{ckpt}", tp=tp, input_len=isl),
                    sweep=[
                        dict(batch=1, decode_ms=10.0 + isl / 1024),
                        dict(batch=8, decode_ms=12.0 + isl / 1024),
                    ],
                )
            )
        )
        return P._anchor_meta(str(p))

    long, short = art("a8k.json", 8192), art("a1k.json", 1024)
    others = [
        art("tp2.json", 1024, tp=2),
        art("vllm.json", 1024, backend="vllm"),
        art("other.json", 1024, ckpt="N"),
    ]
    out = tmp_path / "derived"
    dest = P.with_decode_context(long, [long, short, *others], str(out))
    doc = json.loads(open(dest).read())
    assert dest.startswith(str(out))
    assert doc["decode_ctx"] == [
        dict(batch=1, context=1024.0, decode_ms=11.0),
        dict(batch=8, context=1024.0, decode_ms=13.0),
    ]
    assert doc["meta"]["decode_ctx_donors"] == ["a1k.json"]
    # No sibling at another length: the anchor is used as harvested.
    assert P.with_decode_context(long, [long, *others], str(out)) == long["path"]


def test_harness_warmup_convention():
    sc = P.Scenario("1k1k", 1024, 1024)
    kw = dict(
        scenario=sc,
        trace_path=None,
        precision="fp8",
        engine="sglang",
        par=P.Parallelism(8),
        concurrency=16,
        gpu_arch="mi355x",
        hbm_gb=288.0,
        num_requests=160,
        anchor=None,
        checkpoint="X",
    )
    argv = P.point_argv(**kw)
    assert argv[argv.index("--des-warmup-requests") + 1] == "0"
    argv = P.point_argv(**kw, warmup_requests=16)
    assert argv[argv.index("--des-warmup-requests") + 1] == "16"


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
    # A name that says nothing defers to the dtype the anchor recorded.
    bf = dict(_anchor(2, ckpt="Qwen3-8B"), precision="bf16")
    unknown = dict(_anchor(2, ckpt="Qwen3-8B", name="u"), precision=None)
    kw = dict(checkpoint="Qwen3-8B", engine="sglang", par=P.Parallelism(2), isl=8192)
    assert P.pick_anchor([bf], **kw, precision="fp8") is None
    assert P.pick_anchor([bf], **kw, precision="bf16") is bf
    assert P.pick_anchor([bf, unknown], **kw, precision="fp8") is unknown
    # ... unless the caller says what the checkpoint is.
    assert P.pick_anchor([unknown], **kw, precision="fp8", native="bf16") is None
    assert P.pick_anchor([unknown], **kw, precision="bf16", native="bf16") is unknown
    assert P.ModelSpec.parse("qwen3_8B@Qwen3-8B=bf16").precision == "bf16"
    with pytest.raises(ValueError):
        P.ModelSpec.parse("qwen3_8B@Qwen3-8B=int3")


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


def test_spec_config():
    assert P.SpecConfig.parse("none") == P.SpecConfig()
    assert P.SpecConfig.parse("none").argv() == []
    mtp = P.SpecConfig.parse("mtp")
    assert (mtp.k, mtp.accept, mtp.draft_cost, mtp.label) == (3, 0.75, 0.05, "MTP3")
    s = P.SpecConfig.parse("mtp:2:0.8:0.1")
    assert s.label == "MTP2@0.8/d0.1"
    assert s.tokens_per_step == pytest.approx(1 + 0.8 + 0.64)
    assert " ".join(s.argv()) == (
        "--speculative-num-tokens 2 --speculative-acceptance-rate 0.8 "
        "--speculative-draft-cost-factor 0.1"
    )
    assert P.SpecConfig.parse("EAGLE3:4").label == "EAGLE3-4"
    for bad in ("medusa", "mtp:0", "mtp:3:1.5", "mtp:3:0.7:-1"):
        with pytest.raises(ValueError):
            P.SpecConfig.parse(bad)


def test_knob_axes():
    ax = P.KnobAxis.parse("sched=unified,exclusive:4")
    assert ax.argv("unified") == []
    assert ax.argv("exclusive:4") == ["--des-exclusive-prefill", "--des-max-prefill-seqs", "4"]
    raw = P.KnobAxis.parse("--prefix-cache-hit-rate=0,0.5")
    assert raw.argv("0.5") == ["--prefix-cache-hit-rate", "0.5"]
    assert P.KnobAxis("--enable-x", ("on", "off")).argv("on") == ["--enable-x"]
    assert P.KnobAxis("--enable-x", ("on", "off")).argv("off") == []
    for bad in ("sched", "nope=1", "sched=fifo", "chunk=big"):
        with pytest.raises(ValueError):
            P.KnobAxis.parse(bad)
    combos = P.knob_combos([ax, P.KnobAxis("budget", ("8192", "16384"))])
    assert len(combos) == 4 and combos[0] == (("sched", "unified"), ("budget", "8192"))
    assert P.knob_combos([]) == [()]
    a = P.build_parser().parse_args(
        [
            "--model",
            "m",
            "--sched",
            "unified",
            "--step-budget",
            "4096",
            "--hbm-capacity-gb",
            "192",
            "288",
            "--knob",
            "think=0,500",
        ]
    )
    axes, hbm = P.knob_axes_from_args(a)
    assert [x.name for x in axes] == ["sched", "budget", "hbm", "think"] and hbm == 288.0
    a = P.build_parser().parse_args(["--model", "m", "--hbm-capacity-gb", "192"])
    assert P.knob_axes_from_args(a) == ([], 192.0)
    a = P.build_parser().parse_args(
        ["--model", "m", "--sched", "unified", "--knob", "sched=exclusive"]
    )
    with pytest.raises(SystemExit):
        P.knob_axes_from_args(a)


def test_point_argv_spec_and_knobs():
    kw = dict(
        scenario=P.Scenario("8k/1k", 8192, 1024),
        trace_path=None,
        precision="fp8",
        engine="sglang",
        par=P.Parallelism(8, 8, 4),
        concurrency=16,
        gpu_arch="mi355x",
        hbm_gb=288.0,
        num_requests=64,
        anchor=None,
        checkpoint="GLM",
    )
    argv = P.point_argv(
        **kw,
        spec=P.SpecConfig.parse("mtp"),
        knobs=(("budget", "8192"), ("hbm", "192"), ("kvdtype", "bf16"), ("sched", "exclusive")),
    )
    s = " ".join(argv)
    assert "--max-num-batched-tokens 2048 " in s  # the swept budget, split over 4 ranks
    assert "--hbm-capacity-gb 192.0" in s and "--speculative-num-tokens 3" in s
    assert "--des-exclusive-prefill" in s
    # The swept KV dtype comes after the precision's, so argparse keeps it.
    assert argv[len(argv) - 1 - argv[::-1].index("--kv-cache-dtype") + 1] == "bf16"


def test_build_tasks_spec_and_knob_series(tmp_path):
    kw = dict(
        models=[P.ModelSpec("glm5", "GLM")],
        engines=["sglang"],
        precisions=["fp8"],
        parallelisms=[P.Parallelism(8)],
        concurrencies=[4],
        scenarios=[P.Scenario("8k/1k", 8192, 1024)],
        modes=["simulate"],
        anchors=[],
        workdir=str(tmp_path),
        specs=[P.SpecConfig(), P.SpecConfig.parse("mtp")],
        knob_axes=[P.KnobAxis("sched", ("unified", "exclusive"))],
    )
    tasks, _ = P.build_tasks(**kw)
    pts = [P.ParetoPoint(**t["point"]) for t in tasks]
    assert len(pts) == 4 and len({P._key(p) for p in pts}) == 4
    assert {p.series() for p in pts} == {
        "MI355X sglang (simulate)",
        "MI355X sglang MTP3 (simulate)",
    }
    assert {p.knobs for p in pts} == {"sched=unified", "sched=exclusive"}
    tasks, _ = P.build_tasks(**kw, series_by=["sched"])
    assert {P.ParetoPoint(**t["point"]).series() for t in tasks} == {
        "MI355X sglang sched=unified (simulate)",
        "MI355X sglang sched=exclusive (simulate)",
    }
    # Rows written before the axes existed still load, as the no-MTP default.
    assert _pt(1, 1).spec == "none" and _pt(1, 1, spec="MTP3").series().endswith("MTP3 (simulate)")


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


def test_chart_helpers():
    assert P._split_series("MI355X sglang MTP3 (simulate)") == ("MI355X sglang MTP3", "simulate")
    assert P._split_series("custom") == ("custom", "")
    assert P._nice_ticks(13400) == [0, 2500, 5000, 7500, 10000, 12500]
    assert P._nice_ticks(0) == [0.0, 1.0]
    assert [P._fmt_tick(v) for v in (0, 2.5, 250, 2500, 12000)] == [
        "0",
        "2.5",
        "250",
        "2.5k",
        "12k",
    ]
    assert P._compact(1.23456789) == 1.2346 and P._compact(7) == 7


def test_html_controls_and_measured_overlay(tmp_path):
    pts = [_pt(10, 100), _pt(20, 90, mode="benchmark", anchor="a.json"), _pt(30, 50)]
    P.mark_frontiers(pts, "interactivity", "total_tput_per_gpu")
    q = pts[0]
    meas = {
        "measured": [
            dict(
                model=q.model,
                scenario=q.scenario,
                precision=q.precision,
                engine=q.engine,
                spec="none",
                parallelism=q.parallelism,
                concurrency=q.concurrency,
                runs=2,
                interactivity=11.0,
                total_tput_per_gpu=95.0,
                output_tput_per_gpu=47.5,
                ttft_ms=50.0,
                tpot_ms=90.9,
                proj={"simulate": {"interactivity": 10.0, "tput_per_gpu": 100.0, "ttft_ms": 40.0}},
            )
        ],
        "accuracy": [
            dict(
                scope="all",
                mode="simulate",
                metric="tput_per_gpu",
                n=1,
                r_log=1.0,
                rho=1.0,
                mape=5.0,
                w15=100.0,
                w25=100.0,
                bias=1.05,
            )
        ],
    }
    path = tmp_path / "m.json"
    path.write_text(json.dumps(meas))
    out = P.write_outputs(
        P.ParetoResult(points=pts),
        str(tmp_path / "o"),
        meta={},
        measured=P.load_measured(str(path)),
    )
    html = open(out["html"][0]).read()
    # Every control is a button group: no native <select> popups to fail in an embedded preview.
    assert "<select" not in html and 'class="chips" id="model"' in html
    assert '"measured": [{' in html and '"accuracy": [{' in html
    assert P.load_measured(None) == {}
    blank = P.write_outputs(P.ParetoResult(points=pts), str(tmp_path / "b"), meta={})
    assert '"measured": []' in open(blank["html"][0]).read()
