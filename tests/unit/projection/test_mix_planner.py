"""Deployment mix: tier qualification, the TTFT margin, tie-breaking and IO."""

from __future__ import annotations

import csv
import json

import pytest

from infera.projection import cli
from infera.projection.core.projection.inference_projection import mix_planner as M
from infera.projection.core.projection.inference_projection import pareto as P


def _pt(iv: float, ttft: float, tput: float, **kw) -> P.ParetoPoint:
    base = dict(
        model="m",
        checkpoint="M",
        engine="sglang",
        precision="fp8",
        gpu_arch="mi355x",
        scenario="trace",
        isl=8192,
        osl=1024,
        mode="benchmark",
        kv_config="hbm",
        parallelism="TP8",
        tp=8,
        ep=1,
        dp_attn=1,
        pp=1,
        gpus=8,
        concurrency=16,
    )
    base.update(kw)
    return P.ParetoPoint(**base, interactivity=iv, ttft_ms=ttft, total_tput_per_gpu=tput)


TIER = M.Tier("interactive", 70.0, 2_500.0)


def test_meets_checks_floor_limit_and_margin():
    assert M.meets(_pt(80, 2_000, 1), TIER)
    assert not M.meets(_pt(60, 500, 1), TIER)
    assert not M.meets(_pt(80, 3_000, 1), TIER)
    assert not M.meets(_pt(80, 2_000, 1), TIER, ttft_margin=0.5)
    assert M.meets(_pt(80, 1_200, 1), TIER, ttft_margin=0.5)
    assert M.meets(_pt(1, 1e9, 1), M.Tier("bulk"))


def test_margin_excludes_a_faster_point_close_to_the_limit():
    near_limit = _pt(80, 2_400, 1_000, concurrency=64)
    safe = _pt(90, 1_000, 800, concurrency=32)
    best, n_ok, applied = M.pick([near_limit, safe], TIER, ttft_margin=0.5)
    assert best is safe and n_ok == 1 and applied


def test_margin_falls_back_to_the_plain_limit():
    only = _pt(80, 2_400, 1_000)
    best, n_ok, applied = M.pick([only], TIER, ttft_margin=0.5)
    assert best is only and n_ok == 1 and not applied


def test_nothing_meets_the_tier():
    best, n_ok, applied = M.pick([_pt(10, 100, 1_000)], TIER)
    assert best is None and n_ok == 0 and not applied


def test_ties_break_on_slo_headroom():
    top = _pt(72, 1_000, 1_000, concurrency=64)
    roomy = _pt(140, 400, 970, concurrency=16)
    slow = _pt(200, 100, 800, concurrency=4)
    best, _, _ = M.pick([top, roomy, slow], TIER, ttft_margin=0.0, tie=0.05)
    assert best is roomy
    best, _, _ = M.pick([top, roomy, slow], TIER, ttft_margin=0.0, tie=0.0)
    assert best is top


def test_headroom_is_the_tighter_ratio():
    assert M.headroom(_pt(140, 1_250, 1), TIER) == pytest.approx(2.0)
    assert M.headroom(_pt(105, 500, 1), TIER) == pytest.approx(1.5)
    assert M.headroom(_pt(1, 1, 1), M.Tier("bulk")) == 1.0


def test_auto_mode_prefers_the_anchored_point_of_a_configuration():
    sim = _pt(100, 500, 2_000, mode="simulate")
    ben = _pt(90, 600, 1_000, mode="benchmark")
    other = _pt(80, 700, 500, mode="simulate", tp=4, parallelism="TP4", gpus=4)
    got = M.candidates([sim, ben, other])
    assert ben in got and other in got and sim not in got
    assert M.candidates([sim, ben, other], mode="simulate") == [sim, other]


def test_simulated_dcp_is_left_out_but_anchored_dcp_is_kept():
    sim = _pt(100, 500, 2_000, mode="simulate", dcp=4)
    ben = _pt(100, 500, 2_000, mode="benchmark", dcp=4, tp=4)
    assert M.candidates([sim, ben]) == [ben]


def test_infeasible_and_unpriced_points_are_dropped():
    bad = _pt(100, 500, 2_000, feasible=False, reason="OOM")
    zero = _pt(100, 500, 0.0, tp=4)
    assert M.candidates([bad, zero]) == []


def test_plan_mix_covers_every_model_scenario_and_tier():
    pts = [
        _pt(200, 300, 500, model="a"),
        _pt(30, 5_000, 3_000, model="a", concurrency=128),
        _pt(10, 100, 100, model="b", feasible=False),
    ]
    picks = M.plan_mix(pts)
    assert [(m.model, m.tier) for m in picks] == [
        (model, t.name) for model in ("a", "b") for t in M.DEFAULT_TIERS
    ]
    by = {(m.model, m.tier): m for m in picks}
    assert by[("a", "realtime")].point.concurrency == 16
    assert by[("a", "responsive")].point.concurrency == 128
    assert by[("a", "bulk")].point.concurrency == 128
    assert all(by[("b", t.name)].point is None for t in M.DEFAULT_TIERS)


def test_plan_mix_rejects_bad_parameters():
    with pytest.raises(ValueError):
        M.plan_mix([], ttft_margin=1.0)
    with pytest.raises(ValueError):
        M.plan_mix([], tie=-0.1)
    with pytest.raises(ValueError):
        M.candidates([], mode="measured")


def test_load_tiers_accepts_a_list_or_inferencex_mapping(tmp_path):
    lst = tmp_path / "tiers.json"
    lst.write_text(json.dumps([{"name": "fast", "min_interactivity": 100, "max_ttft_ms": 500}]))
    assert M.load_tiers(str(lst)) == (M.Tier("fast", 100.0, 500.0),)
    mp = tmp_path / "tiers.yaml"
    mp.write_text(
        "tiers:\n  chat: {intvty: 50, ttft_ms: 2000}\n  batch: {intvty: 0, ttft_ms: null}\n"
    )
    assert M.load_tiers(str(mp)) == (M.Tier("chat", 50.0, 2000.0), M.Tier("batch", 0.0, None))


def test_main_reads_a_pareto_out_dir_and_writes_the_mix(tmp_path, capsys):
    pts = [_pt(200, 300, 500), _pt(30, 5_000, 3_000, concurrency=128)]
    P.write_json(P.ParetoResult(points=pts), str(tmp_path / "pareto_points.json"), meta={})
    assert cli.main(["mix", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "realtime" in out and "confirm each pick" in out
    doc = json.loads((tmp_path / "mix.json").read_text())
    assert doc["meta"]["ttft_margin"] == 0.5
    assert [r["tier"] for r in doc["picks"]] == [t.name for t in M.DEFAULT_TIERS]
    assert doc["picks"][0]["picked"] and doc["picks"][0]["concurrency"] == 16
    with open(tmp_path / "mix.csv") as f:
        assert len(list(csv.DictReader(f))) == len(M.DEFAULT_TIERS)


def test_main_reports_a_missing_points_file(tmp_path, capsys):
    assert M.main([str(tmp_path)]) == 2
    assert "run `inferasim pareto` first" in capsys.readouterr().out
