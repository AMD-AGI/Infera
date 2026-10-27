"""The results ledger (ledger.py): result files read against the baseline, the best, the curves and the working
point, AgentX output turned into results, the check command, and the A/B judge of the infera-experiment skill."""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest
from conftest import get, new, ok, run_home, write_result
from fastapi.testclient import TestClient

from agent_sys_v2 import ledger, skills
from agent_sys_v2.scan import load_project
from agent_sys_v2.server import create_app

FIXTURES = Path(__file__).parent / "fixtures"


def result(value, **more):
    return {"metrics": {"tok_s_gpu": value, "p90_interactivity": 30}, **more}


# ---------------------------------------------------------------- results against the baseline


def test_ledger_reads_results_against_the_baseline(make_project, capsys):
    pdir = make_project(
        {"baseline/b0": "", "experiment/e1": "", "experiment/e2": ""},
        config="title: t\nledger: {primary: tok_s_gpu}\n",
    )
    write_result(
        pdir, "baseline/b0", "c64", result(10000.0, role="baseline", time="2026-10-01T00:00:00Z")
    )
    accept = {"decision": "accept", "notes": "[M] queue ↓"}
    write_result(
        pdir, "experiment/e1", "spf", result(11540.0, time="2026-10-02T00:00:00Z", verdict=accept)
    )
    void = {"status": "void", "notes": "foreign load"}
    write_result(
        pdir, "experiment/e1", "spf2", result(13000.0, time="2026-10-03T00:00:00Z", verdict=void)
    )
    reject = {"decision": "reject"}
    write_result(
        pdir, "experiment/e2", "dcp", result(12000.0, time="2026-10-04T00:00:00Z", verdict=reject)
    )
    write_result(pdir, "experiment/e2", "bad", {"label": "", "metrics": {}})
    write_result(pdir, "experiment/e2", "other", '{"some": "json of another tool"}')
    write_result(pdir, "experiment/e2", "broken", "{not json")
    full = ledger.ledger(load_project(pdir))
    assert full["primary"] == "tok_s_gpu" and full["higher_is_better"] and not full["curve_head"]
    rows = full["rows"]
    assert [r["label"] for r in rows] == ["c64", "spf", "spf2", "dcp"]
    assert full["baseline"]["label"] == "c64"
    # a void measurement has a delta, but no gain: it means nothing
    assert [(r["delta_pct"], r["gain"]) for r in rows] == [
        (0.0, 0.0),
        (15.4, 15.4),
        (30.0, None),
        (20.0, 20.0),
    ]
    # spf2 is void, dcp rejected: neither counts, for the best, the step line or a card
    assert full["best"]["label"] == "spf" and full["best"]["gain"] == 15.4
    assert [r["best_so_far"] for r in rows] == [10000.0, 11540.0, 11540.0, 11540.0]
    assert (full["count"], full["void"]) == (4, 1)
    assert sorted(e["file"] for e in full["errors"]) == ["results/bad.json", "results/broken.json"]
    briefs = ledger.task_briefs(full)
    assert (briefs["experiment/e1"]["best_delta_pct"], briefs["experiment/e1"]["best_gain"]) == (
        15.4,
        15.4,
    )
    assert (
        briefs["experiment/e2"]["best_delta_pct"] is None and briefs["experiment/e2"]["errors"] == 2
    )
    assert briefs["baseline/b0"]["baseline"] and not briefs["experiment/e1"]["baseline"]
    # a pinned baseline is the latest valid result there (not the void spf2), and a metric where lower is better
    (pdir / "project.yaml").write_text(
        "title: t\nledger: {primary: tok_s_gpu, higher_is_better: false, baseline: experiment/e1}\n"
    )
    full = ledger.ledger(load_project(pdir))
    assert full["baseline"]["label"] == "spf" and full["best"]["label"] == "c64"
    assert full["best"]["delta_pct"] == -13.34 and full["best"]["gain"] == 13.34
    # the check command: every result valid, a candidate with its decision
    ok_dir = pdir / "workspaces/experiment/e1/results"
    # an accept, and a void without decision
    assert ledger.main(["check", str(ok_dir), "--require-verdict"]) == 0
    write_result(pdir, "experiment/e1", "later", result(1.0))
    assert ledger.main(["check", str(ok_dir)]) == 0
    # a candidate nobody judged
    assert ledger.main(["check", str(ok_dir), "--require-verdict"]) == 1
    assert "verdict.decision is missing" in capsys.readouterr().out
    assert ledger.main(["check", str(pdir / "workspaces/experiment/e2/results")]) == 1
    assert ledger.main(["check", str(pdir / "workspaces/nothing")]) == 1


def test_only_an_accepted_result_or_a_baseline_counts_as_the_best(make_project):
    pdir = make_project(
        {"baseline/b0": "", "experiment/e1": ""}, config="title: t\nledger: {primary: tok_s_gpu}\n"
    )
    write_result(
        pdir, "baseline/b0", "b", result(100.0, role="baseline", time="2026-10-01T00:00:00Z")
    )
    for i, decision in enumerate(("noise", "rerun", "reject", "")):
        verdict = {"decision": decision} if decision else {}
        write_result(
            pdir,
            "experiment/e1",
            f"c{i}",
            result(200.0 + i, time=f"2026-10-0{i + 2}T00:00:00Z", verdict=verdict),
        )
    full = ledger.ledger(load_project(pdir))
    assert full["best"]["label"] == "b" and {r["best_so_far"] for r in full["rows"]} == {100.0}
    # a result inside the noise has no gain to show
    assert [r["gain"] is None for r in full["rows"]] == [False, True, False, False, False]
    assert ledger.task_briefs(full)["experiment/e1"]["best_delta_pct"] is None


def test_a_pinned_baseline_needs_the_primary_metric(make_project):
    pdir = make_project(
        {"baseline/b0": ""},
        config="title: t\nledger: {primary: tok_s_gpu, baseline: baseline/b0}\n",
    )
    write_result(pdir, "baseline/b0", "b", result(100.0, time="2026-10-01T00:00:00Z"))
    write_result(
        pdir, "baseline/b0", "other", {"metrics": {"ttft": 2.0}, "time": "2026-10-02T00:00:00Z"}
    )
    full = ledger.ledger(load_project(pdir))
    assert full["baseline"]["label"] == "b" and full["rows"][0]["delta_pct"] == 0.0
    assert full["pin_missing"] == ""
    # a pin that names nothing (a task deleted, say) is said, and the head takes the latest role baseline
    (pdir / "project.yaml").write_text("title: t\nledger: {primary: tok_s_gpu, baseline: x/b9}\n")
    write_result(pdir, "baseline/b0", "r", result(90.0, role="baseline", time="2026-10-03T00Z"))
    full = ledger.ledger(load_project(pdir))
    assert full["pin_missing"] == "x/b9" and full["baseline"]["label"] == "r"


def test_a_result_with_nan_is_wrong_and_the_console_still_answers(tmp_path):
    """NaN and Infinity parse as JSON in Python, but the console's responses cannot carry them."""
    home = run_home(tmp_path, "ledger: {primary: tok_s_gpu}\n")
    pdir = home / "projects/r"
    with TestClient(create_app(home)) as c:
        new(c, "a", mode="draft")
        write_result(pdir, "develop/a", "base", result(100.0, role="baseline"))
        nan = write_result(
            pdir,
            "develop/a",
            "nan",
            '{"schema": "infera.result/v1", "label": "x", "metrics": {"tok_s_gpu": NaN}}',
        )
        write_result(
            pdir,
            "develop/a",
            "inf",
            '{"schema": "infera.result/v1", "label": "y", "metrics": {"tok_s_gpu": 1, "z": -Infinity}}',
        )
        write_result(
            pdir,
            "develop/a",
            "huge",
            '{"schema": "infera.result/v1", "label": "z", "metrics": {"tok_s_gpu": 1e999}}',
        )
        c.app.state.store.sync(also=[("r", "develop/a")])
        for url in ("/api/home", "/api/projects/r/board", "/api/projects/r/ledger"):
            ok(c.get(url))
        full = ok(c.get("/api/projects/r/ledger"))
        assert [r["label"] for r in full["rows"]] == ["base"]
        assert sorted(e["file"] for e in full["errors"]) == [
            "results/huge.json",
            "results/inf.json",
            "results/nan.json",
        ]
        assert get(c, "a")["results"]["errors"] == 3
    passed, lines = ledger.check([nan], require_verdict=True)
    assert not passed and "NaN is not a number" in lines[0]
    assert not ledger.number(float("nan")) and not ledger.number(float("inf")) and ledger.number(1)
    # an int too large for a float: no number either, rather than an OverflowError that fails the scan
    assert not ledger.number(10**400)


def test_raw_data_under_results_is_not_read(make_project):
    """The rules put raw data in results/<label>/: a truncated export there is no result of the ledger."""
    pdir = make_project({"experiment/e1": ""}, config="title: t\nledger: {primary: tok_s_gpu}\n")
    write_result(pdir, "experiment/e1", "spf", result(1.0, verdict={"decision": "accept"}))
    raw = pdir / "workspaces/experiment/e1/results/spf/profile_export_aiperf.json"
    raw.parent.mkdir()
    raw.write_text('{"records": [1, 2', encoding="utf-8")
    full = ledger.ledger(load_project(pdir))
    assert [r["label"] for r in full["rows"]] == ["spf"] and full["errors"] == []
    assert ledger.main(["check", str(raw.parent.parent), "--require-verdict"]) == 0


def test_validate():
    base = {"schema": ledger.SCHEMA, "label": "x", "metrics": {"y": 1}}
    assert (
        ledger.validate(
            {**base, "role": "reference", "round": 0, "series": "ix:a", "workload": {"conc": 8}}
        )
        == []
    )
    assert ledger.validate(
        {**base, "role": "theirs", "round": True, "series": " ", "workload": {"conc": "32"}}
    ) == [
        "role should be one of baseline, candidate, reference",
        "round should be a whole number (the project's round)",
        "series should be a name (the curve the point belongs to)",
        "workload.conc should be a positive number (the concurrency)",
    ]
    assert ledger.validate({**base, "workload": {"conc": 0}, "verdict": []}) == [
        "workload.conc should be a positive number (the concurrency)",
        "verdict should be an object",
    ]
    # a reference needs no verdict
    assert ledger.validate({**base, "role": "reference"}, require_verdict=True) == []


def test_ledger_through_the_server(tmp_path):
    home = run_home(tmp_path, "ledger: {primary: tok_s_gpu}\n")
    pdir = home / "projects/r"
    with TestClient(create_app(home)) as c:
        new(c, "a", mode="draft")
        write_result(pdir, "develop/a", "base", result(100.0, role="baseline"))
        write_result(pdir, "develop/a", "try", result(110.0, verdict={"decision": "accept"}))
        c.app.state.store.sync(also=[("r", "develop/a")])
        full = ok(c.get("/api/projects/r/ledger"))
        assert [r["delta_pct"] for r in full["rows"]][-1] == 10.0
        board = ok(c.get("/api/projects/r/board"))
        assert board["project"]["ledger"]["best"]["label"] == "try"
        assert get(c, "a")["results"]["best_gain"] == 10.0


# ---------------------------------------------------------------- the curves


def point(series, conc, x, y, role="candidate", **more):
    return {
        "series": series,
        "role": role,
        "round": 1,
        "workload": {"conc": conc},
        "metrics": {"input_tput_per_gpu": y, "p90_intvty": x},
        **more,
    }


CURVES = "title: t\nledger: {primary: input_tput_per_gpu, x: p90_intvty, baseline_series: v0, working_point: %s}\n"
ACCEPT = {"verdict": {"decision": "accept"}}


def curve_project(make_project, wp="{x: 40}"):
    pdir = make_project(
        {"baseline/r1": "", "survey/r1": "", "experiment/r1-spf": "", "close/r1": ""},
        config=CURVES % wp,
    )
    for conc, x, y in ((16, 60.0, 10000.0), (32, 50.0, 16000.0), (64, 30.0, 20000.0)):
        write_result(pdir, "baseline/r1", f"v0-c{conc}", point("v0", conc, x, y, "baseline"))
    # an extra v0 point that the others beat on both axes: not on the frontier
    write_result(pdir, "baseline/r1", "v0-c48", point("v0", 48, 35.0, 15000.0, "baseline"))
    for conc, x, y in ((16, 70.0, 13000.0), (64, 40.0, 26000.0)):
        write_result(pdir, "survey/r1", f"ix-c{conc}", point("ix:atom", conc, x, y, "reference"))
    write_result(pdir, "experiment/r1-spf", "spf-c32", point("r1-spf", 32, 51.0, 17600.0, **ACCEPT))
    return pdir


def test_curves_frontier_and_the_working_point(make_project):
    pdir = curve_project(make_project)
    full = ledger.ledger(load_project(pdir))
    assert (full["x"], full["working_point"]) == ("p90_intvty", {"x": 40})
    series = {c["name"]: c for c in full["series"]}
    assert [c["name"] for c in full["series"]] == ["v0", "r1-spf", "ix:atom"]
    v0 = series["v0"]
    assert (v0["role"], v0["baseline"], v0["round"]) == ("baseline", True, 1)
    assert [(p["conc"], p["frontier"]) for p in v0["points"]] == [
        (64, True),
        (48, False),
        (32, True),
        (16, True),
    ]
    # Y at X = 40: between C64 (30, 20000) and C32 (50, 16000) on the frontier
    wp = {"x": 40, "y": 18000.0, "conc": None, "interpolated": True, "delta_pct": 0.0, "gain": 0.0}
    assert v0["wp"] == wp
    ix = series["ix:atom"]
    assert ix["role"] == "reference" and ix["wp"]["y"] == 26000.0 and ix["wp"]["delta_pct"] == 44.44
    # a series with a point past the working point has its best Y there
    assert series["r1-spf"]["wp"] == {
        "x": 51.0,
        "y": 17600.0,
        "conc": 32,
        "interpolated": False,
        "delta_pct": -2.22,
        "gain": -2.22,
    }
    # rows: compared with the baseline series at the same concurrency
    rows = {r["label"]: r for r in full["rows"]}
    assert rows["spf-c32"]["delta_pct"] == 10.0 and rows["ix-c64"]["delta_pct"] == 30.0
    # the head compares the series at the working point: the baseline series is the best while none beats it (the
    # reference is not ours), and the trajectory has no best across concurrencies
    assert full["curve_head"] and full["baseline"] == {"series": "v0", "value": 18000.0, **wp}
    assert full["best"]["series"] == "v0"
    assert all(r["best_so_far"] is None for r in full["rows"])
    assert {r["label"] for r in full["rows"] if r["baseline"]} == {
        "v0-c16",
        "v0-c32",
        "v0-c48",
        "v0-c64",
    }
    assert "series" not in ledger.head(full)
    briefs = ledger.task_briefs(full)
    assert briefs["survey/r1"]["best_delta_pct"] is None and briefs["baseline/r1"]["baseline"]
    # a series that beats it, and one that does better still but was rejected
    write_result(pdir, "experiment/r1-spf", "spf-c64", point("r1-spf", 64, 41.0, 21000.0, **ACCEPT))
    (pdir / "tasks/experiment/r1-bad.md").write_text("")
    bad = point("r1-bad", 64, 42.0, 30000.0, verdict={"decision": "reject"})
    write_result(pdir, "experiment/r1-bad", "bad-c64", bad)
    full = ledger.ledger(load_project(pdir))
    assert (full["best"]["series"], full["best"]["value"], full["best"]["gain"]) == (
        "r1-spf",
        21000.0,
        16.67,
    )


def test_without_a_working_point_values_at_different_concurrencies_do_not_compare(make_project):
    """The head is rows' then, and the best is the best gain over the baseline series at its own concurrency."""
    full = ledger.ledger(load_project(curve_project(make_project, wp="{}")))
    best = full["best"]
    assert not full["curve_head"] and (best["label"], best["gain"]) == ("spf-c32", 10.0)
    assert all(r["best_so_far"] is None for r in full["rows"])


def test_the_working_point_at_a_concurrency(make_project):
    pdir = curve_project(make_project, "{conc: 32}")
    write_result(pdir, "baseline/r1", "v0-c32-again", point("v0", 32, 52.0, 16400.0, "baseline"))
    series = {c["name"]: c for c in ledger.ledger(load_project(pdir))["series"]}
    assert series["v0"]["wp"] == {
        "x": 51.0,
        "y": 16200.0,
        "conc": 32,
        "n": 2,
        "interpolated": False,
        "delta_pct": 0.0,
        "gain": 0.0,
    }
    assert series["r1-spf"]["wp"]["delta_pct"] == 8.64
    assert series["ix:atom"]["wp"] is None  # it has no C32 point


def test_at_x_beyond_and_before_the_curve():
    front = [{"x": 30.0, "y": 20.0, "conc": 64}, {"x": 50.0, "y": 16.0, "conc": 32}]
    assert ledger.at_x(front, 60, 1, 1) is None
    assert ledger.at_x(front, 10, 1, 1)["y"] == 20.0
    assert ledger.at_x(front, 50, 1, 1)["conc"] == 32
    # lower X is better (a latency on X): the same rule the other way round
    assert ledger.at_x(front, 40, -1, 1)["y"] == 18.0


# ---------------------------------------------------------------- AgentX output


def test_agentx_output_becomes_a_result(tmp_path, capsys):
    src = FIXTURES / "agentx/agentx_conc40.json"
    agg = json.loads(src.read_text())
    doc = ledger.from_agentx(agg, src, "v0", 1, "baseline", time="2026-10-20T00:00:00Z")
    assert ledger.validate(doc) == []
    assert (doc["label"], doc["series"], doc["round"], doc["role"]) == (
        "v0-c40",
        "v0",
        1,
        "baseline",
    )
    m = doc["metrics"]
    # the InferenceX App's definitions: interactivity is 1 / p90 of the full-response ITL
    assert m["p90_intvty"] == round(1 / 0.01415, 6) and m["input_tput_per_gpu"] == 23873.55907
    assert m["p90_e2e_norm_intvty"] == 39.11896 and m["p90_ttft"] == 2.69414
    assert (m["requests"], m["errors"]) == (4561, 0)
    assert doc["workload"]["conc"] == 40 and doc["workload"]["num_gpus"] == 8
    assert doc["config"]["topology"] == "P TP1·PP4 → D TP4·DCP4"
    with pytest.raises(ValueError):
        ledger.from_agentx({"conc": 1}, src, "v0")
    # the command: a run directory in, result files out, checked like any
    run = tmp_path / "run/agentx"
    run.mkdir(parents=True)
    shutil.copy(src, run / src.name)
    (run / "agentx_conc8.json").write_text("{}")
    out = tmp_path / "results"
    argv = [
        "agentx",
        str(tmp_path / "run"),
        "--out",
        str(out),
        "--series",
        "r1-spf",
        "--round",
        "1",
    ]
    assert ledger.main(argv) == 1  # the broken one is reported
    printed = capsys.readouterr().out
    assert "wrote" in printed and "agentx_conc8.json: not an AgentX result" in printed
    assert ledger.main(["check", str(out)]) == 0
    written = json.loads((out / "r1-spf-c40.json").read_text())
    assert written["role"] == "candidate" and written["evidence"]["agentx"].endswith(
        "agentx_conc40.json"
    )
    assert (
        ledger.main(["agentx", str(tmp_path / "nothing"), "--out", str(out), "--series", "x"]) == 1
    )


def test_agentx_never_overwrites_a_result(tmp_path, capsys):
    """A repeat at the same concurrency (the baseline's same-node rerun) is a second result, not a replacement."""
    src = FIXTURES / "agentx/agentx_conc40.json"
    out = tmp_path / "results"
    argv = [
        "agentx",
        str(src),
        "--out",
        str(out),
        "--series",
        "v0",
        "--time",
        "2026-10-20T00:00:00Z",
    ]
    assert ledger.main(argv) == 0
    first = (out / "v0-c40.json").read_text()
    capsys.readouterr()
    assert ledger.main([*argv[:-1], "2026-10-21T00:00:00Z"]) == 1
    assert "v0-c40.json exists already" in capsys.readouterr().out
    assert (out / "v0-c40.json").read_text() == first
    assert ledger.main([*argv, "--label", "v0-rep1"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["v0-c40.json", "v0-rep1-c40.json"]
    # a label that is no file name is an error, not a traceback
    assert ledger.main([*argv, "--label", "no/such"]) == 1


# ---------------------------------------------------------------- the A/B judge of the infera-experiment skill


def load_judge():
    path = skills.SHIPPED / "infera-experiment/scripts/ab_judge.py"
    spec = importlib.util.spec_from_file_location("ab_judge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ab_judge_never_turns_noise_into_an_accept():
    j = load_judge()
    d = lambda base, cand, **kw: j.judge(base, cand, **kw)["decision"]  # noqa: E731
    assert d([100], [101.5], mechanism="agree") == "noise"  # inside the 2% band
    assert d([100], [97], mechanism="agree") == "reject"
    assert d([100], [102.5], mechanism="agree") == "rerun"  # between the band and the 3% bar
    assert d([100], [110]) == "rerun"  # a gain without the mechanism shown
    assert d([100], [110], mechanism="disagree") == "rerun"
    assert d([100], [110], mechanism="agree") == "accept"
    # a wide spread raises the floor: +6% on runs that scatter by 10% is noise
    assert d([90, 100, 110], [96, 106, 116], mechanism="agree") == "noise"
    # lower is better (a latency): -8% is the gain
    assert d([2.0], [1.84], higher_is_better=False, mechanism="agree") == "accept"
    assert d([2.0], [2.2], higher_is_better=False) == "reject"
    # Y up by 10% with X (interactivity) down by 20%: a trade, not an accept; X within its noise is fine
    assert d([100], [110], mechanism="agree", x=([50], [40], True)) == "rerun"
    assert d([100], [110], mechanism="agree", x=([50], [49.5], True)) == "accept"
    assert d([100], [110], mechanism="agree", x=([2.0], [2.5], False)) == "rerun"  # X a latency
    # numbers only: NaN or Infinity would pass every comparison as if it were a gain; a mean of 0 has no spread
    for cand in ([float("nan")], [float("inf")]):
        with pytest.raises(ValueError):
            j.judge([100], cand, mechanism="agree")
    assert d([100], [0, 0]) == "reject"
    with pytest.raises(ValueError):
        j.judge([100], [-1, 1])


def test_ab_judge_reads_and_writes_result_files(tmp_path, capsys):
    j = load_judge()
    base = write_result(tmp_path, "b", "base", point("v0", 64, 50.0, 10000.0, "baseline"))
    cands = [
        write_result(tmp_path, "c", f"cand{i}", point("r1-spf", 64, 50.0, y))
        for i, y in enumerate((11000.0, 11100.0))
    ]
    files = ["--baseline-files", str(base), "--candidate-files", *map(str, cands)]
    metric = ["--metric", "input_tput_per_gpu", "--x-metric", "p90_intvty"]
    assert j.main([*metric, *files, "--mechanism", "agree", "--write"]) == 0
    # every candidate file gets the verdict, so every one passes the experiment's check
    for c in cands:
        verdict = json.loads(c.read_text())["verdict"]
        assert (
            verdict["decision"] == "accept"
            and verdict["vs_baseline_pct"] == 10.5
            and verdict["x_gain_pct"] == 0.0
        )
    assert ledger.check(cands, require_verdict=True)[0]
    # a gain below the noise, left to the close
    capsys.readouterr()
    assert j.main(["--baseline", "100", "--candidate", "101", "--note", "待集成"]) == 0
    assert json.loads(capsys.readouterr().out)["notes"].endswith("; 待集成")
    assert (
        j.main(["--void", "foreign load on GPU 3", "--candidate-files", str(cands[0]), "--write"])
        == 0
    )
    assert json.loads(cands[0].read_text())["verdict"] == {
        "status": "void",
        "notes": "foreign load on GPU 3",
    }
    # wrong input: exit 2, no traceback
    nan = write_result(
        tmp_path, "c", "nan", '{"schema": "infera.result/v1", "label": "n", "metrics": {"y": NaN}}'
    )
    assert (
        j.main(["--metric", "y", "--baseline-files", str(base), "--candidate-files", str(nan)]) == 2
    )
    assert (
        j.main(
            [
                "--metric",
                "y",
                "--baseline-files",
                str(tmp_path / "missing.json"),
                "--candidate",
                "1",
            ]
        )
        == 2
    )
    assert j.main(["--baseline", "100", "--candidate", "nan"]) == 2
    # a noise floor or bar that is no percentage would let anything through
    for bad in ("nan", "-1"):
        with pytest.raises(SystemExit):
            j.main(["--baseline", "100", "--candidate", "110", "--noise-pct", bad])
    for argv in (
        ["--baseline-files", str(base)],
        ["--baseline", "1", "--candidate", "2", "--write"],
    ):
        # result files need --metric; --write needs the candidate files
        with pytest.raises(SystemExit):
            j.main(argv)
