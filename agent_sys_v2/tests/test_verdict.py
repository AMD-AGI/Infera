"""A/B verdicts that can be recomputed (verdict.py): the judge, ab_judge.py of the infera-experiment skill, the AgentX
source of a result, `ledger verify` on an experiment's workspace and a table of ways to spoil it, each of which must be
caught for its own reason."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest
from measuring import convert, gsm8k, measure, raw

from agent_sys_v2 import ledger, procs, skills, verdict
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


@pytest.fixture(autouse=True)
def no_agent_env(monkeypatch):
    for name in ("AGENT_SYS_PROJECT", "AGENT_SYS_WORKSPACE", procs.CHECK_MARK):
        monkeypatch.delenv(name, raising=False)
    verdict._cache.clear()


def load_judge():
    path = skills.SHIPPED / "infera-experiment/scripts/ab_judge.py"
    spec = importlib.util.spec_from_file_location("ab_judge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AB = load_judge()


# ---------------------------------------------------------------- the judgment


def test_the_numbers_alone_decide_and_noise_never_becomes_an_accept():
    d = lambda base, cand, **kw: verdict.judge(base, cand, **kw)["decision"]  # noqa: E731
    assert d([100], [101.5]) == "noise"  # inside the 2% band
    assert d([100], [97]) == "reject"
    assert d([100], [102.5]) == "rerun"  # between the band and the 3% bar
    assert d([100], [110]) == "accept"  # no mechanism needed: AgentX numbers decide
    # a wide spread raises the floor: +6% on runs that scatter by 10% is noise
    assert d([90, 100, 110], [96, 106, 116]) == "noise"
    # medians against means: a gain carried by a few outliers
    assert d([100] * 5, [99] * 6 + [140] * 5) == "rerun"
    # lower is better (a latency): -8% is the gain
    assert d([2.0], [1.84], higher_is_better=False) == "accept"
    assert d([2.0], [2.2], higher_is_better=False) == "reject"
    # Y up by 10% with X (interactivity) down by 20%: a trade, not an accept; X within its noise is fine
    assert d([100], [110], x=([50], [40], True)) == "rerun"
    assert d([100], [110], x=([50], [49.5], True)) == "accept"
    assert d([100], [110], x=([2.0], [2.5], False)) == "rerun"  # X a latency
    # a stricter bar or band
    assert d([100], [104], accept_pct=5) == "rerun"
    assert d([100], [104], noise_pct=5) == "noise"
    # numbers only: NaN or Infinity would pass every comparison as if it were a gain; a mean of 0 has no spread
    for cand in ([float("nan")], [float("inf")]):
        with pytest.raises(ValueError):
            verdict.judge([100], cand)
    assert d([100], [0, 0]) == "reject"
    with pytest.raises(ValueError):
        verdict.judge([100], [-1, 1])


def test_failed_requests_are_counted_from_agentx():
    assert verdict.failed_pct({"requests": 900, "errors": 100}) == 10.0
    assert verdict.failed_pct({"requests": 880, "errors": 120}) == 12.0
    assert verdict.failed_pct({"requests": 10}) is None


# ---------------------------------------------------------------- an experiment's workspace, judged


def start_run(ws, config, run_id="20261027-090000"):
    """What a run of the task leaves: launch.json with the project's rules as the run started."""
    from agent_sys_v2.model import load_yaml, plain, write_json

    write_json(
        ws / ".agent_sys/runs" / run_id / "launch.json",
        {"ledger": verdict.rules(plain(load_yaml(config)))},
    )


@pytest.fixture
def exp(tmp_path, capsys):
    """A project in its first round: the baseline task's v0 point at C40 with a repeat on the same node and v0's full
    GSM8K, and an experiment with two base arms (repeats of v0) and two candidates, each converted from its own AgentX
    output, the candidate's GSM8K over 200 questions, judged with ab_judge --write."""
    pdir = tmp_path / "home/projects/glm"
    (pdir / "tasks/baseline").mkdir(parents=True)
    (pdir / "tasks/experiment").mkdir(parents=True)
    (pdir / "project.yaml").write_text(CONFIG)
    (pdir / "tasks/baseline/r1.md").write_text("基线\n")
    (pdir / "tasks/experiment/r1-spf.md").write_text("SPF\n")
    bws = pdir / "workspaces/baseline/r1"
    convert(bws, measure(bws, "c40", raw(1.0)), "v0", "baseline", "v0")
    convert(bws, measure(bws, "c40-rep1", raw(1.004)), "v0", "baseline", "v0-rep1")
    gsm8k(bws, "v0-full", "v0", 1319, range(0, 1319, 25))  # 1266 of 1319 right: 0.960
    ws = pdir / "workspaces" / EXP
    start_run(ws, CONFIG)
    gsm8k(ws, "r1-spf-200", "r1-spf", 200, range(3, 200, 25))  # 192 of 200
    files = {
        "b1": convert(ws, measure(ws, "base1", raw(1.0)), "v0", "baseline", "r1-spf-base1"),
        "b2": convert(ws, measure(ws, "base2", raw(1.006)), "v0", "baseline", "r1-spf-base2"),
        "c1": convert(ws, measure(ws, "spf1", raw(1.060)), "r1-spf", "candidate", "r1-spf-1"),
        "c2": convert(ws, measure(ws, "spf2", raw(1.066)), "r1-spf", "candidate", "r1-spf-2"),
    }
    argv = ["--baseline-files", str(files["b1"]), str(files["b2"])]
    argv += ["--candidate-files", str(files["c1"]), str(files["c2"])]
    assert AB.main([*argv, "--mechanism", "prefill queue halves", "--write"]) == 0
    capsys.readouterr()
    return {"pdir": pdir, "ws": ws, "results": ws / "results", **files}


def doc(path):
    return json.loads(Path(path).read_text())


def put(path, d):
    Path(path).write_text(json.dumps(d, indent=1))


def kinds_and_text(path, pdir, current=True):
    findings = verdict.verify(
        path, pdir, current=verdict.project_rules(pdir)[0] if current else None
    )
    return {f.kind for f in findings}, " | ".join(f.problem for f in findings)


def test_the_experiment_as_judged_passes(exp, capsys):
    v = doc(exp["c1"])["verdict"]
    assert v["decision"] == "accept" and v["mechanism"] == "prefill queue halves"
    assert v == doc(exp["c2"])["verdict"]
    inputs = v["inputs"]
    assert (inputs["metric"], inputs["x_metric"], inputs["higher_is_better"]) == (
        "input_tput_per_gpu",
        "p90_intvty",
        True,
    )
    assert (inputs["noise_pct"], inputs["accept_pct"], inputs["baseline_series"]) == (
        2.0,
        3.0,
        "v0",
    )
    assert set(inputs["rules_from"].values()) == {"project.yaml"}
    assert (inputs["min_duration_s"], inputs["reproduce_pct"], inputs["max_arm_gap_min"]) == (
        3600,
        5,
        60,
    )
    # the gates: the base arms reproduce v0 (1.003 against the baseline task's 1.002), one after another on one node,
    # the retest noise from the baseline task's two measurements on that node, GSM8K not worse
    g = v["gates"]
    assert {k: g[k]["status"] for k in g} == {
        "reproduce": "ok",
        "timing": "ok",
        "noise": "ok",
        "accuracy": "ok",
    }
    assert g["reproduce"]["off_pct"] == 0.1 and g["reproduce"]["reference"] == [
        "workspaces/baseline/r1/results/v0-c40.json",
        "workspaces/baseline/r1/results/v0-rep1-c40.json",
    ]
    assert g["timing"]["nodes"] == ["10.0.0.1"] and g["timing"]["max_gap_min"] < 30
    assert (
        0 < g["noise"]["retest_pct"] < 2
        and g["noise"]["floor_pct"] == 2.0
        and v["resolvable"] is True
    )
    assert g["accuracy"]["files"][0]["candidate"] == [192, 200]
    b1 = inputs["baseline"][0]
    assert b1["file"] == f"workspaces/{EXP}/results/r1-spf-base1-c40.json"
    assert b1["agentx"] == f"workspaces/{EXP}/log/agentx/base1/agentx_conc40.json"
    assert (
        b1["agentx_sha256"] == hashlib.sha256((exp["pdir"] / b1["agentx"]).read_bytes()).hexdigest()
    )
    assert b1["value"] == doc(exp["b1"])["metrics"]["input_tput_per_gpu"] and "x" in b1
    for f in ("b1", "b2", "c1", "c2"):
        assert verdict.verify(exp[f], exp["pdir"]) == []
    assert ledger.main(["verify", str(exp["results"])]) == 0
    out = capsys.readouterr().out
    assert "r1-spf-1-c40.json: ok (r1-spf-1-c40, accept)" in out
    # from another directory, with $AGENT_SYS_PROJECT as an agent's run has it
    assert ledger.main(["verify", str(exp["c1"]), str(exp["c2"])]) == 0


def test_ledger_agentx_records_the_source(exp):
    ev = doc(exp["b2"])["evidence"]
    assert ev["agentx"] == f"workspaces/{EXP}/log/agentx/base2/agentx_conc40.json"
    raw_file = exp["pdir"] / ev["agentx"]
    assert ev["agentx_sha256"] == hashlib.sha256(raw_file.read_bytes()).hexdigest()
    assert ev["runtime_env"]["file"] == f"workspaces/{EXP}/log/agentx/base2/runtime.env"
    assert ev["runtime_env"]["vars"]["DURATION"] == "3600"
    # outside any project the path stays absolute
    out = exp["pdir"].parent.parent / "elsewhere"
    assert ledger.main(["agentx", str(raw_file), "--out", str(out), "--series", "x"]) == 0
    assert Path(doc(out / "x-c40.json")["evidence"]["agentx"]).is_absolute()


# ---------------------------------------------------------------- ab_judge: only AgentX numbers, only stricter rules


def test_ab_judge_takes_only_agentx_results(exp, capsys):
    pdir, res = exp["pdir"], exp["results"]
    hand = res / "hand-c40.json"
    put(
        hand,
        {
            **{k: v for k, v in doc(exp["b1"]).items() if k != "evidence"},
            "label": "hand-c40",
            "series": "v9",
        },
    )
    before = doc(exp["c1"])
    for argv, why in (
        (
            ["--baseline-files", str(hand), "--candidate-files", str(exp["c1"]), str(exp["c2"])],
            "no AgentX source",
        ),
        (
            ["--baseline", "100", "--candidate-files", str(exp["c1"]), str(exp["c2"])],
            "numbers typed in are a trial",
        ),
        (
            [
                "--baseline-files",
                str(exp["b1"]),
                "--candidate-files",
                str(exp["c1"]),
                str(exp["c2"]),
            ],
            "leave out",
        ),
        (
            [
                "--baseline-files",
                str(exp["c1"]),
                "--candidate-files",
                str(exp["b1"]),
                str(exp["b2"]),
            ],
            "the base arm is a repeat of the baseline series 'v0'",
        ),
        ([*arms(exp), "--accept-pct", "1"], "looser than the project's 3"),
        ([*arms(exp), "--metric", "tput_per_gpu"], "the project judges by 'input_tput_per_gpu'"),
        ([*arms(exp), "--lower-is-better"], "the project judges by True"),
    ):
        capsys.readouterr()
        assert AB.main([*argv, "--write"]) == 2, why
        assert why in capsys.readouterr().err
    assert doc(exp["c1"]) == before  # nothing written
    hand.unlink()  # a result at the A/B's concurrency would have to be an input
    # stricter is fine, and says where it came from
    assert AB.main([*arms(exp), "--accept-pct", "7", "--write"]) == 0
    v = doc(exp["c1"])["verdict"]
    assert v["decision"] == "rerun" and v["inputs"]["accept_pct"] == 7
    assert (
        v["inputs"]["rules_from"]["accept_pct"] == "--accept-pct 7 (stricter than project.yaml: 3)"
    )
    assert verdict.verify(exp["c1"], pdir) == []
    # a trial with numbers prints and writes nothing; the mechanism is a remark
    capsys.readouterr()
    assert AB.main(["--baseline", "100", "--candidate", "110", "--note", "待集成"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert (
        printed["decision"] == "accept"
        and printed["notes"].endswith("; 待集成")
        and "inputs" not in printed
    )
    # the noise band is the project's: a wider one would turn a reject into noise
    for bad in (["--noise-pct", "6"], ["--accept-pct", "nan"]):
        with pytest.raises(SystemExit):
            AB.main(["--baseline", "100", "--candidate", "110", *bad])


def arms(exp):
    return [
        "--baseline-files",
        str(exp["b1"]),
        str(exp["b2"]),
        "--candidate-files",
        str(exp["c1"]),
        str(exp["c2"]),
    ]


def test_ab_judge_voids_for_two_reasons_only(exp, capsys):
    ws = exp["ws"]
    bad = convert(ws, measure(ws, "spf3", raw(1.07, errors=600)), "r1-spf", "candidate", "r1-spf-3")
    # a third candidate run exists now: judging without it is picking
    capsys.readouterr()
    assert AB.main([*arms(exp), "--write"]) == 2
    assert "r1-spf-3-c40.json" in capsys.readouterr().err
    # failed requests: computed from AgentX, refused below the bar
    assert (
        AB.main(["--void", "failed-requests", "--candidate-files", str(exp["c1"]), "--write"]) == 2
    )
    assert "0.0% failed requests" in capsys.readouterr().err
    assert AB.main(["--void", "failed-requests", "--candidate-files", str(bad), "--write"]) == 0
    assert doc(bad)["verdict"]["void"] == {
        "kind": "failed_requests",
        "failed_pct": round(600 / (4561 + 600) * 100, 2),
    }
    # a void result is left out of the A/B, and the A/B holds
    assert AB.main([*arms(exp), "--write"]) == 0
    assert ledger.main(["verify", str(exp["results"])]) == 0
    # other: a reason and an evidence file
    capsys.readouterr()
    assert AB.main(["--void", "other", "--candidate-files", str(exp["c2"]), "--write"]) == 2
    err = capsys.readouterr().err
    assert "needs --reason" in err and "needs --evidence" in err
    log = ws / "log/rocm-smi.txt"
    log.write_text("GPU3 busy: pid 4242 (not ours)\n")
    argv = ["--void", "other", "--reason", "foreign load on GPU 3", "--evidence", str(log)]
    assert AB.main([*argv, "--candidate-files", str(exp["c2"]), "--write"]) == 0
    assert doc(exp["c2"])["verdict"]["void"] == {
        "kind": "other",
        "reason": "foreign load on GPU 3",
        "evidence": f"workspaces/{EXP}/log/rocm-smi.txt",
    }
    assert verdict.verify(exp["c2"], exp["pdir"]) == []


# ---------------------------------------------------------------- spoiled: each caught for its own reason


def first_input(exp, side="baseline"):
    return doc(exp["c1"])["verdict"]["inputs"][side][0]


def raw_of(exp, key):
    return exp["pdir"] / doc(exp[key])["evidence"]["agentx"]


def edit(path, change):
    d = doc(path)
    change(d)
    put(path, d)


def handwritten(exp):
    """A result written by hand (no AgentX source) put into the verdict's inputs."""
    hand = exp["results"] / "hand-c40.json"
    d = doc(exp["b2"])
    d.pop("evidence")
    put(hand, {**d, "label": "hand-c40", "series": "hand"})
    for c in ("c1", "c2"):
        edit(
            exp[c],
            lambda d: d["verdict"]["inputs"]["baseline"].append(
                {"file": f"workspaces/{EXP}/results/hand-c40.json", "value": 30000.0, "x": 70.0}
            ),
        )


def hidden_repeat(exp, change):
    """A worse base repeat measured and converted, then edited so as not to look like one."""
    run = measure(exp["ws"], "base3", raw(0.9))
    edit(convert(exp["ws"], run, "v0", "baseline", "r1-spf-base3"), change)


def results_of(exp):
    return sorted(
        (p.name, doc(p).get("role"), doc(p).get("series")) for p in exp["results"].glob("*.json")
    )


# (name, the file verify reads, how to read the field it reads, the change, the kind, words of the reason)
SPOILS = [
    (
        "decision edited, then role changed to baseline to hide it",
        "c1",
        lambda exp: (doc(exp["c1"])["role"], doc(exp["c1"])["verdict"]["decision"]),
        lambda exp: edit(
            exp["c1"], lambda d: (d.update(role="baseline"), d["verdict"].update(decision="reject"))
        ),
        MISMATCH,
        "role is 'baseline' but the file carries an A/B decision ('reject')",
    ),
    (
        "vs_baseline_pct edited, then role changed to reference to hide it",
        "c1",
        lambda exp: (doc(exp["c1"])["role"], doc(exp["c1"])["verdict"]["vs_baseline_pct"]),
        lambda exp: edit(
            exp["c1"],
            lambda d: (d.update(role="reference"), d["verdict"].update(vs_baseline_pct=50.0)),
        ),
        MISMATCH,
        "role is 'reference' but the file carries an A/B decision ('accept')",
    ),
    (
        "a worse repeat relabelled as a reference",
        "c1",
        results_of,
        lambda exp: hidden_repeat(exp, lambda d: d.update(role="reference")),
        MISMATCH,
        "r1-spf-base3-c40.json (series v0, C40)",
    ),
    (
        "a worse repeat moved to a series of its own",
        "c1",
        results_of,
        lambda exp: hidden_repeat(exp, lambda d: d.update(series="v0-old")),
        MISMATCH,
        "r1-spf-base3-c40.json (series v0-old, C40)",
    ),
    (
        "a worse repeat whose concurrency field was edited",
        "c1",
        results_of,
        lambda exp: hidden_repeat(exp, lambda d: d["workload"].update(conc=41)),
        MISMATCH,
        "r1-spf-base3-c40.json (series v0, C40)",
    ),
    (
        "a candidate's metric edited after the verdict",
        "c1",
        lambda exp: doc(exp["c1"])["metrics"]["input_tput_per_gpu"],
        lambda exp: edit(
            exp["c1"],
            lambda d: d["metrics"].update(
                input_tput_per_gpu=d["metrics"]["input_tput_per_gpu"] * 1.1
            ),
        ),
        MISMATCH,
        "its metrics differ from its raw AgentX output",
    ),
    (
        "a base arm's raw AgentX output edited",
        "c1",
        lambda exp: hashlib.sha256(raw_of(exp, "b1").read_bytes()).hexdigest(),
        lambda exp: raw_of(exp, "b1").write_text(json.dumps(raw(0.9))),
        MISMATCH,
        "changed after the conversion",
    ),
    (
        "a raw AgentX output deleted",
        "c1",
        lambda exp: raw_of(exp, "c2").is_file(),
        lambda exp: raw_of(exp, "c2").unlink(),
        UNVERIFIABLE,
        "is gone",
    ),
    (
        "a hand-written result as an input",
        "c1",
        lambda exp: len(doc(exp["c1"])["verdict"]["inputs"]["baseline"]),
        handwritten,
        UNVERIFIABLE,
        "it has no AgentX source",
    ),
    (
        "a base repeat dropped from the inputs",
        "c1",
        lambda exp: [i["file"] for i in doc(exp["c1"])["verdict"]["inputs"]["baseline"]],
        lambda exp: edit(exp["c1"], lambda d: d["verdict"]["inputs"]["baseline"].pop()),
        MISMATCH,
        "the inputs leave out valid results at the same concurrency",
    ),
    (
        "decision edited by hand",
        "c1",
        lambda exp: doc(exp["c1"])["verdict"]["decision"],
        lambda exp: edit(exp["c1"], lambda d: d["verdict"].update(decision="reject")),
        MISMATCH,
        "verdict.decision is 'reject'; recomputed from the raw AgentX output it is 'accept'",
    ),
    (
        "vs_baseline_pct edited by hand",
        "c1",
        lambda exp: doc(exp["c1"])["verdict"]["vs_baseline_pct"],
        lambda exp: edit(exp["c1"], lambda d: d["verdict"].update(vs_baseline_pct=9.5)),
        MISMATCH,
        "verdict.vs_baseline_pct is 9.5",
    ),
    (
        "noise band loosened in the verdict",
        "c1",
        lambda exp: doc(exp["c1"])["verdict"]["inputs"]["noise_pct"],
        lambda exp: edit(exp["c1"], lambda d: d["verdict"]["inputs"].update(noise_pct=0.5)),
        MISMATCH,
        "verdict.inputs.noise_pct is 0.5%, not the project's 2%",
    ),
    (
        "project.yaml's rules changed after the run started",
        "c1",
        lambda exp: verdict.project_rules(exp["pdir"])[0]["accept_pct"],
        lambda exp: (exp["pdir"] / "project.yaml").write_text(
            CONFIG.replace("baseline_series: v0", "baseline_series: v0\n  accept_pct: 1")
        ),
        MISMATCH,
        "project.yaml's ledger rules changed during the run: accept_pct 3.0 -> 1.0",
    ),
    (
        "a void without a reason",
        "c2",
        lambda exp: doc(exp["c2"])["verdict"].get("status"),
        lambda exp: edit(
            exp["c2"],
            lambda d: d.update(verdict={"status": "void", "void": {"kind": "other"}, "notes": ""}),
        ),
        MISMATCH,
        "a void of kind other has no reason",
    ),
    (
        "voided for failed requests below 10%",
        "c2",
        lambda exp: doc(exp["c2"])["verdict"].get("status"),
        lambda exp: edit(
            exp["c2"],
            lambda d: d.update(verdict={"status": "void", "void": {"kind": "failed_requests"}}),
        ),
        MISMATCH,
        "voided for failed requests, but AgentX counts 0.0% failed",
    ),
    (
        "a void the two kinds do not know",
        "c2",
        lambda exp: doc(exp["c2"])["verdict"].get("status"),
        lambda exp: edit(
            exp["c2"], lambda d: d.update(verdict={"status": "void", "notes": "looked slow"})
        ),
        MISMATCH,
        "verdict.void.kind is None",
    ),
    (
        "NaN in the verdict",
        "c1",
        lambda exp: "NaN" in exp["c1"].read_text(),
        lambda exp: exp["c1"].write_text(
            exp["c1"].read_text().replace('"vs_baseline_pct": ', '"vs_baseline_pct": NaN, "was": ')
        ),
        UNVERIFIABLE,
        "NaN is not a number",
    ),
    (
        "a verdict without inputs (written by hand, or before verify existed)",
        "c1",
        lambda exp: "inputs" in doc(exp["c1"])["verdict"],
        lambda exp: edit(exp["c1"], lambda d: d["verdict"].pop("inputs")),
        UNVERIFIABLE,
        "the verdict has no inputs",
    ),
    (
        "a verdict copied from another file",
        "c1",
        lambda exp: [i["file"] for i in doc(exp["c1"])["verdict"]["inputs"]["candidate"]],
        lambda exp: edit(
            exp["c1"],
            lambda d: d["verdict"]["inputs"].update(
                candidate=d["verdict"]["inputs"]["candidate"][1:]
            ),
        ),
        MISMATCH,
        "is not among the candidate inputs of its own verdict",
    ),
    (
        "judged on another metric",
        "c1",
        lambda exp: doc(exp["c1"])["verdict"]["inputs"]["metric"],
        lambda exp: edit(
            exp["c1"], lambda d: d["verdict"]["inputs"].update(metric="output_tput_per_gpu")
        ),
        MISMATCH,
        "verdict.inputs.metric is 'output_tput_per_gpu'; the project judges by 'input_tput_per_gpu'",
    ),
    (
        "an input's recorded value edited",
        "c1",
        lambda exp: first_input(exp)["value"],
        lambda exp: edit(
            exp["c1"], lambda d: d["verdict"]["inputs"]["baseline"][0].update(value=1.0)
        ),
        MISMATCH,
        "the verdict read value 1.0",
    ),
    (
        "a candidate converted again from a different raw file",
        "c2",
        lambda exp: doc(exp["c2"])["evidence"]["agentx_sha256"],
        lambda exp: put(exp["c2"], {**doc(exp["c2"]), "evidence": {**doc(exp["b1"])["evidence"]}}),
        MISMATCH,
        "the verdict recorded raw output",
    ),
]


@pytest.mark.parametrize("name,target,read,spoil,kind,words", SPOILS, ids=[s[0] for s in SPOILS])
def test_a_spoiled_experiment_is_refused_for_its_reason(
    exp, capsys, name, target, read, spoil, kind, words
):
    assert (
        verdict.verify(exp[target], exp["pdir"], current=verdict.project_rules(exp["pdir"])[0])
        == []
    )
    before = read(exp)
    spoil(exp)
    # the change landed on what verify reads (a spoil that changes nothing would pass for the wrong reason)
    try:
        after = read(exp)
    except (OSError, ValueError, KeyError):
        after = "<unreadable>"
    assert after != before, f"{name}: the spoil did not change what verify reads"
    if isinstance(before, float) and isinstance(after, float):
        assert not math.isclose(before, after)
    found, text = kinds_and_text(exp[target], exp["pdir"])
    assert kind in found and words in text, (name, found, text)
    # and the acceptance check fails on it, saying the same
    capsys.readouterr()
    assert ledger.main(["verify", str(exp["results"])]) == 1
    out = capsys.readouterr().out
    assert words in out and f"[{kind}]" in out and "fix:" in out


def test_the_spoils_cover_the_list():
    """At least ten ways, both kinds of failure."""
    assert len(SPOILS) >= 10 and {s[4] for s in SPOILS} == {MISMATCH, UNVERIFIABLE}


def test_the_snapshot_is_the_runs_not_project_yaml_now(exp):
    """The rules a verdict is checked against are the run's snapshot: an agent that loosens project.yaml and then
    judges is refused; the ledger's view does not compare with project.yaml now."""
    pdir = exp["pdir"]
    (pdir / "project.yaml").write_text(
        CONFIG.replace("baseline_series: v0", "baseline_series: v0\n  noise_pct: 0.5")
    )
    assert AB.main([*arms(exp), "--write"]) == 0  # judged by the loosened project.yaml
    assert doc(exp["c1"])["verdict"]["inputs"]["noise_pct"] == 0.5
    found, text = kinds_and_text(exp["c1"], pdir, current=False)
    assert "not the project's 2%" in text
    # the check of the run in flight ($AGENT_SYS_CHECK_RUN) is the one that counts; without a run, project.yaml
    later = exp["ws"] / ".agent_sys/runs/20261027-120000"
    start_run(
        exp["ws"],
        CONFIG.replace("baseline_series: v0", "baseline_series: v0\n  noise_pct: 0.5"),
        later.name,
    )
    assert verdict.snapshot(exp["ws"])[1] == later.name
    assert verdict.verify(exp["c1"], pdir) == []


def test_tools_of_the_checks_are_digested_and_compared(tmp_path, monkeypatch):
    from agent_sys_v2 import checks

    pdir = tmp_path / "p"
    script = pdir / "skills/own/scripts/check_more.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('ok')\n")
    (pdir / "tools").mkdir()
    (pdir / "tools/gate.sh").write_text("exit 0\n")
    ws = pdir / "workspaces/experiment/x"
    ws.mkdir(parents=True)
    (ws / "report.md").write_text("r\n")
    env = {
        "AGENT_SYS_PYTHON": "/usr/bin/python3",
        "AGENT_SYS_WORKSPACE": str(ws),
        "AGENT_SYS_PROJECT": str(pdir),
    }
    runs = [
        {"run": 'test -s "$AGENT_SYS_WORKSPACE/report.md"'},
        {"run": '"$AGENT_SYS_PYTHON" -m agent_sys_v2.ledger verify "$AGENT_SYS_WORKSPACE/results"'},
        {"run": "bash tools/gate.sh"},
    ]
    before = checks.toolchain(pdir, runs, env)
    names = {Path(f).name for f in before}
    # the package's modules the checks import, the skills' scripts, the files the commands name; not the workspace
    assert {
        "ledger.py",
        "verdict.py",
        "model.py",
        "perfcheck.py",
        "ab_judge.py",
        "check_more.py",
        "gate.sh",
    } <= names
    assert "report.md" not in names and "server.py" not in names
    assert checks.toolchain_check(before, checks.toolchain(pdir, runs, env)) is None
    (ws / "report.md").write_text("changed\n")  # the agent's own output is not a tool
    assert checks.toolchain_check(before, checks.toolchain(pdir, runs, env)) is None
    script.write_text("print('ok'); raise SystemExit(0)\n")
    (pdir / "tools/gate.sh").unlink()
    failed = checks.toolchain_check(before, checks.toolchain(pdir, runs, env))
    assert failed["ok"] is False and failed["name"] == checks.TOOLCHAIN_CHECK
    assert f"changed: {script.resolve()}" in failed["output"]
    assert f"removed: {(pdir / 'tools/gate.sh').resolve()}" in failed["output"]
    assert "a checkout of its own, or a wheel" in failed["output"]
    # a package of this name in the project directory would be imported by `python -m` run there
    shadow = pdir / "agent_sys_v2/ledger.py"
    shadow.parent.mkdir()
    shadow.write_text("print('ok')\n")
    failed = checks.toolchain_check(before, checks.toolchain(pdir, runs, env))
    assert f"added: {shadow.resolve()}" in failed["output"]
    assert checks.toolchain_check(None, {}) is None  # a run from before records nothing


# ---------------------------------------------------------------- the ledger: two deltas, the reference, unverified


def rows_of(pdir):
    from agent_sys_v2.scan import load_project

    full = ledger.ledger(load_project(pdir))
    return full, {r["label"]: r for r in full["rows"]}


def test_the_ledger_shows_both_deltas_and_whether_a_verdict_holds(exp, monkeypatch):
    full, rows = rows_of(exp["pdir"])
    assert full["errors"] == []
    c1, b1, v0 = rows["r1-spf-1-c40"], rows["r1-spf-base1-c40"], rows["v0-c40"]
    rep = rows["v0-rep1-c40"]
    # the ledger's delta is against the baseline task's v0 points; the verdict's against the experiment's base arm
    assert c1["delta_pct"] == ledger.pct(c1["value"], (v0["value"] + rep["value"]) / 2) == 5.79
    assert c1["verdict_delta_pct"] == doc(exp["c1"])["verdict"]["vs_baseline_pct"] == 5.98
    assert (c1["verified"], c1["unverified"], c1["judged_against"]) == (True, [], "")
    assert v0["verified"] is None and b1["verified"] is None  # nothing to recompute
    # the base arms are repeats of v0: drawn, never the reference
    assert (b1["repeat"], v0["repeat"], c1["repeat"]) == (True, False, False)
    curve = next(c for c in full["series"] if c["name"] == "v0")
    assert [p["repeat"] for p in curve["points"]].count(True) == 2
    assert [p["label"] for p in curve["points"] if p["frontier"]] == ["v0-rep1-c40"]
    assert curve["wp"]["n"] == 2 and not v0["repeat"] and not rep["repeat"]
    # a hand-edited decision: shown, marked, with the reason; cached until a file changes
    calls = []
    real = verdict.verify
    monkeypatch.setattr(verdict, "verify", lambda *a, **k: calls.append(a) or real(*a, **k))
    edit(exp["c1"], lambda d: d["verdict"].update(decision="reject"))
    _full, rows = rows_of(exp["pdir"])
    assert (
        rows["r1-spf-1-c40"]["verified"] is False and rows["r1-spf-1-c40"]["decision"] == "reject"
    )
    assert (
        "recomputed from the raw AgentX output it is 'accept'"
        in rows["r1-spf-1-c40"]["unverified"][0]
    )
    n = len(calls)
    rows_of(exp["pdir"])
    assert len(calls) == n  # nothing changed: nothing recomputed
    # a verdict from before (no inputs) is shown as it is, unverified, with no error
    edit(exp["c2"], lambda d: d["verdict"].pop("inputs"))
    full, rows = rows_of(exp["pdir"])
    assert full["errors"] == [] and rows["r1-spf-2-c40"]["decision"] == "accept"
    assert rows["r1-spf-2-c40"]["verified"] is False
    assert "no inputs" in rows["r1-spf-2-c40"]["unverified"][0]


def test_a_verdict_against_an_older_version_says_so(exp):
    """After close/r1 seals v1, the experiment's verdict was judged against v0: the ledger says it, and compares the
    row with v1, as measured by close/r1 only."""
    pdir = exp["pdir"]
    (pdir / "project.yaml").write_text(CONFIG.replace("baseline_series: v0", "baseline_series: v1"))
    (pdir / "tasks/close").mkdir()
    (pdir / "tasks/close/r1.md").write_text("收尾\n")
    cws = pdir / "workspaces/close/r1"
    v1 = convert(cws, measure(cws, "c40", raw(1.10)), "v1", "baseline", "v1")
    _full, rows = rows_of(pdir)
    c1 = rows["r1-spf-1-c40"]
    assert c1["judged_against"] == "v0" and c1["verified"] is True
    assert c1["delta_pct"] == ledger.pct(c1["value"], doc(v1)["metrics"]["input_tput_per_gpu"]) < 0
    assert c1["verdict_delta_pct"] == 5.98


def test_the_reference_is_what_the_sealing_task_measured():
    t = lambda task, typ, series, y, status="valid": {  # noqa: E731
        "task": task,
        "type": typ,
        "series": series,
        "status": status,
        "metrics": {"y": y},
    }
    table = [
        t("baseline/r1", "baseline", "v0", 100),
        t("experiment/r1-a", "experiment", "v0", 90),
        t("close/r1", "close", "v1", 120),
        t("close/r1", "close", "v0", 101),
        t("experiment/r2-b", "experiment", "v1", 110),
        t("close/r2", "close", "v1", 119),
        t("baseline/r1", "baseline", "v0", 0, status="void"),
        t("experiment/r1-a", "experiment", "r1-a", 105),
    ]
    ys = lambda series: [r["metrics"]["y"] for r in ledger.reference(table, series)]  # noqa: E731
    assert ys("v0") == [100]  # not the experiment's repeat, not close/r1's, not a void one
    assert ys("v1") == [120]  # close/r1 sealed v1
    assert ys("r1-a") == [105]  # not a version: all its points
    assert ys("v7") == []
    # a project without a sealing task: every valid point of the series
    bare = [t("develop/a", "develop", "v0", 100), t("develop/b", "develop", "v0", 104)]
    assert [r["metrics"]["y"] for r in ledger.reference(bare, "v0")] == [100, 104]


# ---------------------------------------------------------------- what an agent could reach: the snapshot, the arms


def test_the_snapshot_comes_from_the_runner_not_the_editable_launch_json(exp, monkeypatch):
    """launch.json is in the workspace: emptying its rules does not switch the rules off. The runner hands the check
    the snapshot it read before the agent started; without it, a missing key is project.yaml's."""
    from agent_sys_v2.model import write_json

    pdir, launch = exp["pdir"], exp["ws"] / ".agent_sys/runs/20261027-090000/launch.json"
    # a verdict hand-made with a bar of 1%
    edit(exp["c1"], lambda d: d["verdict"]["inputs"].update(accept_pct=1.0))
    write_json(launch, {"ledger": {}})
    assert "looser than the project's 3%" in kinds_and_text(exp["c1"], pdir)[1]
    write_json(launch, {"ledger": {**verdict.rules({}), "accept_pct": 1.0}})
    monkeypatch.setenv(verdict.RULES_ENV, json.dumps(verdict.rules({"ledger": {"accept_pct": 3}})))
    assert "looser than the project's 3%" in kinds_and_text(exp["c1"], pdir)[1]


def test_both_arms_at_one_concurrency_in_the_baseline_series(exp, capsys):
    ws = exp["ws"]
    c64 = [
        convert(ws, measure(ws, f"base64-{k}", raw(1.0, conc=64)), "v0", "baseline", f"b64-{k}", 64)
        for k in (1, 2)
    ]
    capsys.readouterr()
    assert (
        AB.main(
            [
                "--baseline-files",
                *map(str, c64),
                "--candidate-files",
                str(exp["c1"]),
                str(exp["c2"]),
                "--write",
            ]
        )
        == 2
    )
    assert "the inputs are at different concurrencies" in capsys.readouterr().err
    # by hand: verify says the same
    edit(
        exp["c1"],
        lambda d: d["verdict"]["inputs"]["baseline"].append(
            {
                **d["verdict"]["inputs"]["baseline"][0],
                "file": f"workspaces/{EXP}/results/b64-1-c64.json",
            }
        ),
    )
    assert "the inputs are at different concurrencies" in kinds_and_text(exp["c1"], exp["pdir"])[1]


def test_a_broken_project_yaml_or_a_wrong_file_is_an_error_not_a_traceback(exp, capsys):
    (exp["pdir"] / "project.yaml").write_text("ledger: [unclosed\n")
    capsys.readouterr()
    assert ledger.main(["verify", str(exp["results"])]) == 1
    assert "project.yaml cannot be read" in capsys.readouterr().out
    (exp["pdir"] / "project.yaml").write_text(CONFIG)
    log = exp["ws"] / "log/why.txt"
    log.write_text("x\n")
    argv = ["--void", "other", "--reason", "r", "--evidence", str(log), "--write"]
    assert AB.main([*argv, "--candidate-files", str(exp["ws"] / "results/missing.json")]) == 2
    assert AB.main([*argv, "--candidate-files", str(log)]) == 2


def test_the_ledgers_cache_sees_every_file_verify_read(exp):
    pdir = exp["pdir"]
    assert verdict.check_row(exp["c1"], pdir) == []
    # a change to the snapshot in place
    launch = exp["ws"] / ".agent_sys/runs/20261027-090000/launch.json"
    launch.write_text(launch.read_text().replace('"accept_pct": 3.0', '"accept_pct": 50.0'))
    assert verdict.check_row(exp["c1"], pdir) != []
    start_run(exp["ws"], CONFIG, "20261027-090000")
    assert verdict.check_row(exp["c1"], pdir) == []
    # the evidence of a void
    log = exp["ws"] / "log/why.txt"
    log.write_text("foreign load\n")
    argv = ["--void", "other", "--reason", "foreign load", "--evidence", str(log)]
    assert AB.main([*argv, "--candidate-files", str(exp["c2"]), "--write"]) == 0
    assert verdict.check_row(exp["c2"], pdir) == []
    log.unlink()
    assert "evidence file of the void" in verdict.check_row(exp["c2"], pdir)[0]["problem"]


def test_a_project_reached_through_a_symlink(exp, tmp_path):
    link = tmp_path / "link"
    link.symlink_to(exp["pdir"])
    assert verdict.verify(link / f"workspaces/{EXP}/results/r1-spf-1-c40.json", link) == []


def test_a_decision_moved_off_a_candidate_is_still_recomputed(exp, capsys):
    """Changing a judged file's role hides nothing: verify and the ledger still look at it, it is not the best, and
    an experiment with no verdict at all does not pass."""
    edit(
        exp["c1"], lambda d: (d.update(role="baseline"), d["verdict"].update(vs_baseline_pct=50.0))
    )
    full, rows = rows_of(exp["pdir"])
    r = rows["r1-spf-1-c40"]
    assert r["verified"] is False and "carries an A/B decision" in r["unverified"][0]
    assert not ledger.counts(r)
    # ab_judge writes into candidates only
    capsys.readouterr()
    assert AB.main([*arms(exp), "--write"]) == 2
    assert "a verdict is written into candidates only" in capsys.readouterr().err
    # results with no verdict at all: no A/B to accept
    for f in ("c1", "c2"):
        edit(exp[f], lambda d: (d.update(role="candidate"), d.pop("verdict")))
        edit(exp[f], lambda d: d.update(role="baseline"))
    capsys.readouterr()
    assert ledger.main(["verify", str(exp["results"])]) == 1
    assert "no A/B verdict under" in capsys.readouterr().out
