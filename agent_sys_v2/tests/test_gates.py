"""The gates of an A/B (gates.py, verdict.judge_ab): what a run directory says of its measurement, formal measurements
only, the base arm reproducing the baseline series, the arms' timing, the noise floor with the version's retest noise,
and GSM8K; then a table of ways to spoil a judged A/B past them, each caught for its own reason."""

import importlib.util
import json
import math
import shutil
from datetime import timedelta
from pathlib import Path

import pytest
from measuring import AIPERF, NODE, T0, convert, gsm8k, measure, raw

from agent_sys_v2 import gates, ledger, procs, skills, verdict
from agent_sys_v2.verdict import MISMATCH, UNVERIFIABLE

CONFIG = """title: GLM-5.2 4P4D AgentX
round: 1
version: v0
ledger:
  primary: input_tput_per_gpu
  higher_is_better: true
  x: p90_intvty
  baseline_series: v0
  working_point: {conc: 40}
"""
EXP = "experiment/r1-spf"
CAND_WRONG = tuple(
    range(3, 200, 25)
)  # the candidate's GSM8K: 192 of 200, as v0 on the same questions


@pytest.fixture(autouse=True)
def no_agent_env(monkeypatch):
    for name in ("AGENT_SYS_PROJECT", "AGENT_SYS_WORKSPACE", procs.CHECK_MARK, verdict.RULES_ENV):
        monkeypatch.delenv(name, raising=False)
    verdict._cache.clear()


def load_judge():
    spec = importlib.util.spec_from_file_location(
        "ab_judge", skills.SHIPPED / "infera-experiment/scripts/ab_judge.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AB = load_judge()


def doc(path):
    return json.loads(path.read_text())


def edit(path, change):
    d = doc(path)
    change(d)
    path.write_text(json.dumps(d, indent=1))


# ---------------------------------------------------------------- what a run directory says


def test_the_measurement_of_a_real_point(tmp_path):
    """The real AIPerf export: its window as written (the client's local time), the node it measured, the duration it
    was set to; runtime.env as the ATOM kit writes it (no fast switch)."""
    run = tmp_path / "run"
    (run / "aiperf_artifacts").mkdir(parents=True)
    agg = raw(conc=96)
    (run / "agentx_conc96.json").write_text(json.dumps(agg))
    (run / "aiperf_artifacts/profile_export_aiperf.json").write_text(json.dumps(AIPERF))
    (run / "runtime.env").write_text("CONC=96\nDURATION=3600\n")
    m, ev = gates.read_point(run / "agentx_conc96.json", agg)
    assert m == {
        "duration_s": 3626.56748,
        "benchmark_duration_s": 3600.0,
        "fast": False,
        "window": {"start": "2026-10-26T09:25:25.190345", "end": "2026-10-26T10:26:05.191227"},
        "node": "10.0.0.1",
    }
    assert (
        ev["file"] == run / "aiperf_artifacts/profile_export_aiperf.json"
        and len(ev["sha256"]) == 64
    )
    assert gates.screening(m) == ""
    # InferenceX's fast mode forces a 1200 s window whatever DURATION says: AIPerf's own phase tells
    fast = json.loads(json.dumps(AIPERF))
    fast["input_config"]["phases"][0]["duration"] = 1200.0
    (run / "aiperf_artifacts/profile_export_aiperf.json").write_text(json.dumps(fast))
    (run / "runtime.env").write_text("CONC=96\nDURATION=3600\nAIPERF_EXPERIMENTAL_FAST=1\n")
    m, _ev = gates.read_point(run / "agentx_conc96.json", raw(conc=96, duration=1221.0))
    assert (m["benchmark_duration_s"], m["fast"]) == (1200.0, True)
    assert gates.screening(m) == "AgentX fast mode (AIPERF_EXPERIMENTAL_FAST=1)"
    # no AIPerf export: the window and node are missing, and say why; the duration from runtime.env
    shutil.rmtree(run / "aiperf_artifacts")
    (run / "runtime.env").write_text("CONC=96\nDURATION=3600\n")
    m, ev = gates.read_point(run / "agentx_conc96.json", agg)
    assert ev == {} and m["benchmark_duration_s"] == 3600 and m["fast"] is False
    assert m["missing"] == {
        "window": "no AIPerf export (profile_export_aiperf.json) of C96 below run/",
        "node": "neither AIPerf's export nor runtime.env names the server it measured",
    }
    # the client measured localhost: no node name
    m = gates.measurement(
        agg, {}, {**AIPERF, "input_config": {"endpoint": {"urls": ["http://localhost:8000"]}}}
    )
    assert (
        "node" not in m and m["missing"]["node"] == "the client measured localhost (no node name)"
    )


def test_formal_or_screening():
    s = lambda **m: gates.screening(m)  # noqa: E731
    assert s(duration_s=3626, benchmark_duration_s=3600, fast=False) == ""
    assert (
        s(fast=True, benchmark_duration_s=1200) == "AgentX fast mode (AIPERF_EXPERIMENTAL_FAST=1)"
    )
    assert s(benchmark_duration_s=1200, duration_s=1222) == "set to 1200 s, below the formal 3600 s"
    # AIPerf's own coverage bar: 95% of the set duration
    assert s(duration_s=3000).startswith("measured 3000 s, below 95%") and s(duration_s=3430) == ""
    assert s() == ""  # says nothing either way: an older result is not called screening
    assert gates.screening({"benchmark_duration_s": 3600}, min_duration_s=7200).startswith(
        "set to 3600"
    )


def test_timing_and_retest_and_reproduce_on_numbers():
    w = lambda m0, m1: {
        "start": (T0 + timedelta(minutes=m0)).isoformat(),
        "end": (T0 + timedelta(minutes=m1)).isoformat(),
    }  # noqa: E731
    one = lambda f, arm, a, b, node=NODE: {"file": f, "arm": arm, "window": w(a, b), "node": node}  # noqa: E731
    ok = [one("b1", "baseline", 0, 60), one("c1", "candidate", 85, 145)]
    assert gates.timing(ok, 60)["status"] == "ok" and gates.timing(ok, 60)["max_gap_min"] == 25
    over = [one("b1", "baseline", 0, 60), one("c1", "candidate", 30, 90)]
    assert gates.timing(over, 60)["why"] == f"b1 and c1 overlap on {NODE}"
    # on two nodes at once: no overlap of one node (and the noise of two nodes is the A/B's own spread)
    assert gates.timing([over[0], one("c1", "candidate", 30, 90, "10.0.0.2")], 60)["status"] == "ok"
    far = [one("b1", "baseline", 0, 60), one("c1", "candidate", 200, 260)]
    assert (
        gates.timing(far, 60)["why"] == "140 min pass between two of the measurements (at most 60)"
    )
    assert gates.timing([ok[0], {"file": "c1", "arm": "candidate"}], 60) == {
        "max_gap_min": 0.0,
        "nodes": [NODE],
        "status": "unchecked",
        "why": "no measurement window for c1",
    }
    # one window not known does not hide the others: an overlap among the known ones still fails
    assert gates.timing([*over, {"file": "c2", "arm": "candidate"}], 60)["status"] == "failed"
    # a node not known: two windows at once may share one
    blind = [one("b1", "baseline", 0, 60, ""), one("c1", "candidate", 30, 90, "")]
    assert "they may share it" in gates.timing(blind, 60)["why"]
    # the gap is from the latest end so far: a long window on another node still running is no gap
    long = [
        one("b1", "baseline", 0, 200, "10.0.0.2"),
        one("b2", "baseline", 10, 70),
        one("c1", "candidate", 95, 155),
    ]
    assert gates.timing(long, 60)["status"] == "ok"
    # Python 3.10 reads what 3.12 reads
    assert (
        gates.when("2026-10-27T01:00:00Z") == T0
        and gates.when("2026-10-27T01:00:00.12345") is not None
    )
    assert gates.retest([(100.0, "a"), (101.0, "a")]) == (1.95, 2)
    assert gates.retest([(100.0, "a"), (101.0, "b")]) is None
    assert gates.reproduce([100.0], [], 5)["status"] == "unchecked"
    assert (
        gates.reproduce([92.0], [100.0], 5)["why"]
        == "the base arm did not reproduce the baseline series (off by -8.00%, bar 5%)"
    )
    assert gates.reproduce([96.0], [100.0], 5)["status"] == "ok"
    # AIPerf writes local time without a zone; one with a zone is taken in UTC, and the two never fail to compare
    assert gates.when("2026-10-27T01:00:00+08:00") == T0 - timedelta(hours=8)
    window = {"start": "2026-10-27T01:00:00+08:00", "end": "2026-10-27T01:00:00"}
    assert ledger.workload_errors({"window": window}) == []
    assert ledger.workload_errors({"window": {"start": "x"}}) == [
        "workload.window should be {start, end}: AIPerf's ISO times"
    ]


# ---------------------------------------------------------------- an A/B, judged with its gates


def ab(
    tmp_path,
    ref=(1.0, 1.004),
    base=(1.0, 1.006),
    cand=(1.06, 1.066),
    starts=None,
    nodes=None,
    cand_wrong=CAND_WRONG,
    ref_nodes=None,
    write=True,
    extra=(),
):
    """A project: the baseline task's v0 at C40 (ref, on one node unless ref_nodes) and its full GSM8K, an
    experiment's base arms and candidates (starts: minutes after T0 of each, base arms first; nodes likewise), the
    candidate's GSM8K over 200 questions; judged with ab_judge --write."""
    pdir = tmp_path / "home/projects/glm"
    for t in ("tasks/baseline", "tasks/experiment"):
        (pdir / t).mkdir(parents=True)
    (pdir / "project.yaml").write_text(CONFIG)
    (pdir / "tasks/baseline/r1.md").write_text("基线\n")
    (pdir / "tasks/experiment/r1-spf.md").write_text("SPF\n")
    bws, ws = pdir / "workspaces/baseline/r1", pdir / "workspaces" / EXP
    for k, y in enumerate(ref):
        node = ref_nodes[k] if ref_nodes else NODE
        convert(bws, measure(bws, f"v0-{k}", raw(y), node=node), "v0", "baseline", f"v0-{k}")
    gsm8k(bws, "v0-full", "v0", 1319, range(0, 1319, 25))
    files = {}
    arms = [("b", k, y, "v0", "baseline") for k, y in enumerate(base)] + [
        ("c", k, y, "r1-spf", "candidate") for k, y in enumerate(cand)
    ]
    for i, (side, k, y, series, role) in enumerate(arms):
        start = T0 + timedelta(days=1, minutes=starts[i]) if starts else None
        node = nodes[i] if nodes else NODE
        run = measure(ws, f"{side}{k}", raw(y), node=node, start=start)
        files[f"{side}{k + 1}"] = convert(ws, run, series, role, f"{series}-{side}{k}")
    if cand_wrong is not None:
        gsm8k(ws, "r1-spf-200", "r1-spf", 200, cand_wrong)
    out = {"pdir": pdir, "ws": ws, "results": ws / "results", **files}
    out["argv"] = [
        "--baseline-files",
        *(str(f) for k, f in files.items() if k[0] == "b"),
        "--candidate-files",
        *(str(f) for k, f in files.items() if k[0] == "c"),
        *extra,
    ]
    if write:
        assert AB.main([*out["argv"], "--write"]) == 0
    return out


def v(e):
    return doc(e["c1"])["verdict"]


def findings(e):
    return verdict.verify(e["c1"], e["pdir"], current=verdict.project_rules(e["pdir"])[0])


def test_a_good_ab_passes_every_gate(tmp_path, capsys):
    e = ab(tmp_path)
    assert v(e)["decision"] == "accept" and gates.statuses(v(e)["gates"]) == {
        "reproduce": "ok",
        "timing": "ok",
        "noise": "ok",
        "accuracy": "ok",
    }
    assert findings(e) == []
    capsys.readouterr()
    assert ledger.main(["verify", str(e["results"])]) == 0


def test_only_formal_measurements_are_inputs(tmp_path, capsys):
    """A fast run and a short one are screening: in the ledger, marked; not peers of the A/B; refused as inputs."""
    e = ab(tmp_path)
    ws = e["ws"]
    fast = convert(
        ws, measure(ws, "fast", raw(1.2), fast=True), "r1-spf", "candidate", "r1-spf-fast"
    )
    short = convert(
        ws, measure(ws, "short", raw(1.2), duration=1200), "r1-spf", "candidate", "r1-spf-short"
    )
    assert (
        doc(fast)["workload"]["fast"] is True
        and doc(short)["workload"]["benchmark_duration_s"] == 1200
    )
    assert findings(e) == []  # neither is left out: they are no peers
    capsys.readouterr()
    assert AB.main([*e["argv"], str(fast), "--write"]) == 2
    err = capsys.readouterr().err
    assert (
        "a screening measurement (AgentX fast mode (AIPERF_EXPERIMENTAL_FAST=1)), never an input"
        in err
    )
    assert AB.main([*e["argv"], str(short), "--write"]) == 2
    assert "set to 1200 s, below the formal 3600 s" in capsys.readouterr().err
    from agent_sys_v2.scan import load_project

    rows = {r["label"]: r for r in ledger.ledger(load_project(e["pdir"]))["rows"]}
    assert (
        rows["r1-spf-fast-c40"]["screening"].startswith("AgentX fast mode")
        and not rows["r1-spf-fast-c40"]["repeat"]
    )
    assert rows["r1-spf-short-c40"]["screening"] == "set to 1200 s, below the formal 3600 s"
    assert rows["r1-spf-c0-c40"]["screening"] == ""


def test_a_base_arm_off_its_series_is_rerun(tmp_path):
    e = ab(tmp_path, base=(0.92, 0.921), cand=(0.98, 0.981))
    got = v(e)
    assert got["decision"] == "rerun" and got["gates"]["reproduce"]["status"] == "failed"
    assert got["notes"] == (
        "the base arm did not reproduce the baseline series (off by -8.13%, bar 5%); "
        "on the numbers alone accept: +6.52% past the 3% bar and the floor 2.00%"
    )
    assert (
        got["gates"]["accuracy"]["status"] == "unchecked"
    )  # no accept on these numbers: no GSM8K needed
    assert findings(e) == []
    # the command line can only make it stricter
    assert AB.main([*e["argv"], "--reproduce-pct", "10", "--write"]) == 2
    assert AB.main([*e["argv"], "--reproduce-pct", "3", "--write"]) == 0
    assert (
        v(e)["inputs"]["rules_from"]["reproduce_pct"]
        == "--reproduce-pct 3 (stricter than project.yaml: 5)"
    )


def test_arms_that_overlap_or_lie_apart_are_rerun(tmp_path):
    over = ab(
        tmp_path / "a", starts=(0, 85, 115, 200)
    )  # the first candidate starts while the second base runs
    assert v(over)["decision"] == "rerun" and "overlap on 10.0.0.1" in v(over)["notes"]
    assert findings(over) == []
    apart = ab(tmp_path / "b", starts=(0, 85, 300, 385))  # 155 min between the arms
    assert (
        v(apart)["decision"] == "rerun"
        and "155 min pass between two of the measurements (at most 60)" in v(apart)["notes"]
    )
    # on two nodes at once: allowed (the gate is about one node's GPUs)
    two = ab(tmp_path / "c", starts=(0, 85, 0, 85), nodes=(NODE, NODE, "10.0.0.2", "10.0.0.2"))
    assert v(two)["decision"] == "accept" and v(two)["gates"]["timing"]["nodes"] == [
        "10.0.0.1",
        "10.0.0.2",
    ]
    assert AB.main([*apart["argv"], "--max-arm-gap-min", "600", "--write"]) == 2


def test_a_noise_floor_above_the_bar_cannot_tell(tmp_path):
    """The baseline task's two measurements are 3% apart: a retest noise of 5.79% > the 3% bar. A +4% candidate is
    noise the measurement cannot resolve, left to the close; measuring another pair would not lower it."""
    e = ab(tmp_path, ref=(1.0, 1.03), base=(1.0, 1.01), cand=(1.045, 1.05))
    got = v(e)
    assert got["decision"] == "noise" and got["resolvable"] is False
    assert got["gates"]["noise"]["retest_pct"] == 5.79 and got["noise_floor_pct"] == 5.79
    assert (
        "cannot tell a change of the bar's size (noise floor 5.79% > accept bar 3%) · 待集成"
        in got["notes"]
    )
    assert findings(e) == []
    from agent_sys_v2.scan import load_project

    rows = {r["label"]: r for r in ledger.ledger(load_project(e["pdir"]))["rows"]}
    assert (
        rows["r1-spf-c0-c40"]["resolvable"] is False
        and rows["r1-spf-c0-c40"]["decision"] == "noise"
    )
    # a change far beyond the floor is still told apart
    big = ab(tmp_path / "big", ref=(1.0, 1.03), base=(1.0, 1.01), cand=(1.12, 1.125))
    assert v(big)["decision"] == "accept" and v(big)["resolvable"] is False
    # a stricter bar on the command line (6% > the floor) does not make the measurement able to tell the project's
    # 3%: still 判不了 · 待集成, judged against the project's bar, and verify recomputes the same
    assert AB.main([*e["argv"], "--accept-pct", "6", "--write"]) == 0
    got = v(e)
    assert (
        got["inputs"]["accept_pct"] == 6
        and got["decision"] == "noise"
        and got["resolvable"] is False
    )
    assert "(noise floor 5.79% > accept bar 3%) · 待集成" in got["notes"]
    assert findings(e) == []
    edit(e["c1"], lambda d: d["verdict"].update(resolvable=True))
    assert (
        "verdict.resolvable is True; recomputed from the raw AgentX output it is False"
        in " ".join(f.problem for f in findings(e))
    )


def test_retest_noise_falls_back_to_the_baseline(tmp_path):
    """No repeats on one node in the reference: the noise gate is unchecked and the band alone is the floor."""
    e = ab(tmp_path, ref_nodes=(NODE, "10.0.0.9"))
    assert (
        v(e)["gates"]["noise"]["status"] == "unchecked"
        and v(e)["gates"]["noise"]["floor_pct"] == 2.0
    )


def test_gsm8k_significantly_worse_is_a_reject_and_none_no_accept(tmp_path, capsys):
    e = ab(
        tmp_path, cand_wrong=range(0, 200, 8)
    )  # 175 of 200 against v0's 192 on the same questions
    got = v(e)
    assert got["decision"] == "reject" and got["gates"]["accuracy"]["status"] == "worse"
    assert "GSM8K is significantly worse: 175/200 against v0's 192" in got["notes"]
    assert findings(e) == []
    none = ab(tmp_path / "none", cand_wrong=None, write=False)
    capsys.readouterr()
    assert AB.main([*none["argv"], "--write"]) == 2
    err = capsys.readouterr().err
    assert (
        "an accept needs the candidate's GSM8K: no GSM8K of series r1-spf" in err
        and "ledger gsm8k" in err
    )


def test_a_result_from_before_windows_is_unchecked_not_refused(tmp_path):
    """No AIPerf export beside the AgentX output (a result converted before windows were recorded): timing is
    unchecked, says why, and the A/B is judged."""
    e = ab(tmp_path, write=False)
    for f in ("b1", "c2"):
        run = e["pdir"] / doc(e[f])["evidence"]["agentx"]
        shutil.rmtree(run.parent / "aiperf_artifacts")
        edit(
            e[f],
            lambda d: (
                d["evidence"].pop("aiperf"),
                d["evidence"].pop("aiperf_sha256"),
                d["workload"].pop("window"),
                d["workload"].pop("node"),
            ),
        )
    assert AB.main([*e["argv"], "--write"]) == 0
    got = v(e)
    assert got["decision"] == "accept" and got["gates"]["timing"]["status"] == "unchecked"
    assert "unchecked: timing (no measurement window for" in got["notes"]
    assert findings(e) == []


# ---------------------------------------------------------------- spoiled past the gates: each caught for its reason


def raw_of(e, key, part="agentx"):
    ev = doc(e[key])["evidence"]
    return e["pdir"] / (ev[part] if part != "runtime_env" else ev["runtime_env"]["file"])


def reference_8pct_higher(e):
    """The baseline task's v0 measured again 8.7% higher, replacing its points: the base arm no longer reproduces it."""
    bws = e["pdir"] / "workspaces/baseline/r1"
    for p in (bws / "results").glob("v0-*-c40.json"):
        p.unlink()
    shutil.rmtree(bws / "log/agentx")
    for k, y in enumerate((1.087, 1.09)):
        convert(bws, measure(bws, f"v0-{k}", raw(y)), "v0", "baseline", f"v0-{k}")


def accept_by_hand(e):
    edit(
        e["c1"],
        lambda d: (
            d["verdict"].update(decision="accept"),
            d["verdict"]["gates"]["reproduce"].update(status="ok"),
            d["verdict"]["gates"]["timing"].update(status="ok"),
        ),
    )


def reference_values(e):
    from agent_sys_v2.scan import load_project

    return sorted(
        r["value"]
        for r in ledger.ledger(load_project(e["pdir"]))["rows"]
        if r["task"] == "baseline/r1"
    )


def append_input(e, label, **kw):
    ws = e["ws"]
    f = convert(ws, measure(ws, label, raw(1.07), **kw), "r1-spf", "candidate", label)
    item = {
        "file": f"workspaces/{EXP}/results/{label}-c40.json",
        "agentx": doc(f)["evidence"]["agentx"],
        "agentx_sha256": doc(f)["evidence"]["agentx_sha256"],
        "value": doc(f)["metrics"]["input_tput_per_gpu"],
        "x": doc(f)["metrics"]["p90_intvty"],
    }
    edit(e["c1"], lambda d: d["verdict"]["inputs"]["candidate"].append(item))


# (name, scenario kwargs, how to read what verify reads, the spoil, kind, words)
SPOILS = [
    (
        "a fast result as an input",
        {},
        lambda e: len(v(e)["inputs"]["candidate"]),
        lambda e: append_input(e, "fast", fast=True),
        MISMATCH,
        "is a screening measurement (AgentX fast mode",
    ),
    (
        "a result too short as an input",
        {},
        lambda e: len(v(e)["inputs"]["candidate"]),
        lambda e: append_input(e, "short", duration=1200),
        MISMATCH,
        "set to 1200 s, below the formal 3600 s",
    ),
    (
        "the base arm 8% off the reference, the verdict kept",
        {},
        reference_values,
        reference_8pct_higher,
        MISMATCH,
        "the base arm did not reproduce the baseline series (off by -7.85%",
    ),
    (
        "a rerun for a base arm 8% off, made an accept by hand",
        {"base": (0.92, 0.921), "cand": (0.98, 0.981)},
        lambda e: v(e)["decision"],
        accept_by_hand,
        MISMATCH,
        "verdict.decision is 'accept'; recomputed from the raw AgentX output it is 'rerun'",
    ),
    (
        "a rerun for overlapping arms, made an accept by hand",
        {"starts": (0, 85, 115, 200)},
        lambda e: v(e)["decision"],
        accept_by_hand,
        MISMATCH,
        "overlap on 10.0.0.1",
    ),
    (
        "a rerun for arms far apart, made an accept by hand",
        {"starts": (0, 85, 300, 385)},
        lambda e: v(e)["decision"],
        accept_by_hand,
        MISMATCH,
        "155 min pass between two of the measurements",
    ),
    (
        "a measurement window edited in the result",
        {},
        lambda e: doc(e["b1"])["workload"]["window"],
        lambda e: edit(
            e["b1"], lambda d: d["workload"]["window"].update(start="2026-10-20T00:00:00")
        ),
        MISMATCH,
        "its workload differs from its run directory",
    ),
    (
        "AIPerf's export edited after the conversion",
        {},
        lambda e: raw_of(e, "c2", "aiperf").read_text(),
        lambda e: raw_of(e, "c2", "aiperf").write_text(
            raw_of(e, "c2", "aiperf").read_text().replace('"start_time": "', '"start_time": "2')
        ),
        MISMATCH,
        "its AIPerf export",
    ),
    (
        "runtime.env switched to fast after the conversion",
        {},
        lambda e: raw_of(e, "c2", "runtime_env").read_text(),
        lambda e: raw_of(e, "c2", "runtime_env").write_text(
            "CONC=40\nDURATION=3600\nAIPERF_EXPERIMENTAL_FAST=1\n"
        ),
        MISMATCH,
        "its runtime.env",
    ),
    (
        "an unresolvable noise made resolvable by hand",
        {"ref": (1.0, 1.03), "base": (1.0, 1.01), "cand": (1.045, 1.05)},
        lambda e: v(e)["resolvable"],
        lambda e: edit(e["c1"], lambda d: d["verdict"].update(resolvable=True)),
        MISMATCH,
        "verdict.resolvable is True; recomputed from the raw AgentX output it is False",
    ),
    (
        "a gate's status edited",
        {},
        lambda e: v(e)["gates"]["noise"]["status"],
        lambda e: edit(e["c1"], lambda d: d["verdict"]["gates"]["noise"].update(status="failed")),
        MISMATCH,
        "verdict.gates.noise is 'failed'; recomputed it is 'ok'",
    ),
    (
        "the reproduce bar loosened in the verdict",
        {},
        lambda e: v(e)["inputs"]["reproduce_pct"],
        lambda e: edit(e["c1"], lambda d: d["verdict"]["inputs"].update(reproduce_pct=20)),
        MISMATCH,
        "verdict.inputs.reproduce_pct is 20%, looser than the project's 5%",
    ),
    (
        "the arm gap loosened in the verdict",
        {},
        lambda e: v(e)["inputs"]["max_arm_gap_min"],
        lambda e: edit(e["c1"], lambda d: d["verdict"]["inputs"].update(max_arm_gap_min=600)),
        MISMATCH,
        "verdict.inputs.max_arm_gap_min is 600 min, looser than the project's 60 min",
    ),
    (
        "the formal duration lowered in the verdict",
        {},
        lambda e: v(e)["inputs"]["min_duration_s"],
        lambda e: edit(e["c1"], lambda d: d["verdict"]["inputs"].update(min_duration_s=600)),
        MISMATCH,
        "verdict.inputs.min_duration_s is 600 s, looser than the project's 3600 s",
    ),
    (
        "lm_eval's per-question samples edited",
        {},
        lambda e: next((e["ws"] / "log/gsm8k").rglob("samples_*.jsonl")).read_text(),
        lambda e: (s := next((e["ws"] / "log/gsm8k").rglob("samples_*.jsonl"))).write_text(
            s.read_text().replace('"exact_match": 0.0', '"exact_match": 1.0')
        ),
        UNVERIFIABLE,
        "an accept needs the candidate's GSM8K: workspaces/experiment/r1-spf/results/r1-spf-200.gsm8k.json: its lm_eval samples",
    ),
    (
        "the candidate's GSM8K deleted after the accept",
        {},
        lambda e: (e["results"] / "r1-spf-200.gsm8k.json").is_file(),
        lambda e: (e["results"] / "r1-spf-200.gsm8k.json").unlink(),
        UNVERIFIABLE,
        "an accept needs the candidate's GSM8K: no GSM8K of series r1-spf",
    ),
    (
        "GSM8K significantly worse, judged accept by hand",
        {"cand_wrong": range(0, 200, 8)},
        lambda e: v(e)["decision"],
        lambda e: edit(
            e["c1"],
            lambda d: (
                d["verdict"].update(decision="accept"),
                d["verdict"]["gates"]["accuracy"].update(status="ok"),
            ),
        ),
        MISMATCH,
        "GSM8K is significantly worse: 175/200",
    ),
]


@pytest.mark.parametrize("name,kw,read,spoil,kind,words", SPOILS, ids=[s[0] for s in SPOILS])
def test_a_spoiled_ab_is_refused_for_its_reason(
    tmp_path, capsys, name, kw, read, spoil, kind, words
):
    e = ab(tmp_path, **kw)
    assert findings(e) == []
    before = read(e)
    spoil(e)
    verdict._cache.clear()
    try:
        after = read(e)
    except (OSError, ValueError, KeyError, StopIteration):
        after = "<unreadable>"
    # the change landed on what verify reads
    assert after != before, f"{name}: the spoil did not change what verify reads"
    if isinstance(before, float) and isinstance(after, float):
        assert not math.isclose(before, after)
    found = findings(e)
    text = " | ".join(f"{f.problem} {f.context}" for f in found)
    assert kind in {f.kind for f in found} and words in text, (name, text)
    capsys.readouterr()
    assert ledger.main(["verify", str(e["results"])]) == 1
    out = capsys.readouterr().out
    assert f"[{kind}]" in out and "fix:" in out


def test_the_gate_spoils_cover_each_gate():
    names = " ".join(s[0] for s in SPOILS)
    for words in (
        "fast",
        "short",
        "8% off",
        "overlapping",
        "far apart",
        "window",
        "noise",
        "GSM8K",
        "samples",
    ):
        assert words in names


def test_files_named_relative_to_the_workspace(tmp_path, monkeypatch, capsys):
    """An agent runs ab_judge from its workspace with results/x.json: the gates and the no-picking rule read the same
    files as with absolute paths (not results/x.json under the project directory)."""
    e = ab(tmp_path, write=False)
    monkeypatch.chdir(e["ws"])
    rel = [a if a.startswith("--") else str(Path(a).relative_to(e["ws"])) for a in e["argv"]]
    assert AB.main([*rel, "--write"]) == 0 and v(e)["decision"] == "accept"
    assert findings(e) == []
    ws = e["ws"]
    convert(ws, measure(ws, "b9", raw(0.9)), "v0", "baseline", "v0-b9")
    capsys.readouterr()
    assert AB.main([*rel, "--write"]) == 2
    assert "the inputs leave out valid results at the same concurrency" in capsys.readouterr().err


def test_one_run_counts_once(tmp_path, capsys):
    """The same AgentX output converted under two labels is one measurement, not two inputs."""
    e = ab(tmp_path, write=False)
    ws = e["ws"]
    run = e["pdir"] / doc(e["c1"])["evidence"]["agentx"]
    assert (
        ledger.main(
            [
                "agentx",
                str(run),
                "--out",
                str(ws / "results"),
                "--series",
                "r1-spf",
                "--label",
                "dup",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert AB.main([*e["argv"], str(ws / "results/dup-c40.json"), "--write"]) == 2
    assert "one AgentX run is counted more than once" in capsys.readouterr().err


def test_a_screening_candidate_needs_no_decision(tmp_path, capsys):
    """A fast point converted into results/ as a candidate: no A/B decision is asked of it, and an experiment that
    keeps it passes its check. Its own file saying formal while the run directory says fast keeps it a peer."""
    e = ab(tmp_path)
    ws = e["ws"]
    fast = convert(
        ws, measure(ws, "fast", raw(1.2), fast=True), "r1-spf", "candidate", "r1-spf-fast"
    )
    capsys.readouterr()
    assert ledger.main(["verify", str(e["results"])]) == 0
    edit(fast, lambda d: d["workload"].update(fast=False))
    assert "r1-spf-fast-c40.json" in " ".join(f.problem for f in findings(e))
    # fields stripped from the file: the ledger does not take it for a formal point either
    edit(
        fast,
        lambda d: [d["workload"].pop(k) for k in ("fast", "benchmark_duration_s", "duration_s")],
    )
    assert gates.screening(ledger.measured_by(doc(fast)))


def test_a_verdict_from_before_the_gates_is_recomputed_not_refused(tmp_path):
    """A verdict ab_judge wrote before the gates (no gates, no resolvable, none of their thresholds): its decision is
    recomputed with them, and it holds when they agree."""
    e = ab(tmp_path)
    for f in ("c1", "c2"):
        edit(
            e[f],
            lambda d: (
                [d["verdict"].pop(k) for k in ("gates", "resolvable")],
                [d["verdict"]["inputs"].pop(k) for k in verdict.GATE_KEYS],
            ),
        )
    assert findings(e) == []
