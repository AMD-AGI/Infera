"""The field checks of the infera-perf template's round (perfcheck.py): what intake, survey, plan and close leave
behind before they count as done."""

import pytest
from conftest import write_result
from measuring import gsm8k

from agent_sys_v2 import ops, perfcheck


def run_check(kind, pdir, tid):
    ok_, lines = perfcheck.run(kind, tid, pdir)
    return ok_, [x for x in lines if x.startswith("FAIL")]


def fill_intake(pdir):
    cfg = (pdir / "project.yaml").read_text(encoding="utf-8")
    cfg = cfg.replace('goal: ""', "goal: 工作点 Y +10%").replace(
        "machines: []", "machines: [{name: n1, gpus: 8}, n2]"
    )
    cfg = cfg.replace("working_point: {}", "working_point: {x: 30}")
    (pdir / "project.yaml").write_text(cfg, encoding="utf-8")
    rules = pdir / "rules/project.md"
    rules.write_text(rules.read_text().replace("<待 intake 填写>", "已填"), encoding="utf-8")
    (pdir / "workspaces/intake/r1/questions.md").write_text("| # | 问题 | 建议 | 你的意见 |\n")
    for tid in ("baseline/r1", "profile/r1"):
        ops.update_task(pdir, tid, {"machines": ["n1"]})


def test_intake_check(tmp_path):
    pdir = ops.new_project(tmp_path, "glm", "infera-perf")
    passed, failed = run_check("intake", pdir, "intake/r1")
    assert not passed and len(failed) == 7
    fill_intake(pdir)
    assert run_check("intake", pdir, "intake/r1") == (True, [])
    ops.update_task(pdir, "profile/r1", {"machines": ["nosuch"]})
    assert run_check("intake", pdir, "intake/r1")[1] == [
        "FAIL  profile/r1 has its machines: machines not in project.yaml: nosuch"
    ]
    # round 2: its plan has to be drafted, and project.yaml says the round
    (pdir / "tasks/intake/r2.md").write_text("---\ndepends_on: [intake/r1]\n---\n")
    (pdir / "workspaces/intake/r2").mkdir(parents=True)
    (pdir / "workspaces/intake/r2/questions.md").write_text("x\n")
    ops.update_task(pdir, "profile/r1", {"machines": ["n1"]})
    assert run_check("intake", pdir, "intake/r2")[1] == [
        "FAIL  project.yaml round is 2 (it is 1)",
        "FAIL  this round's plan is drafted (tasks/plan/r2.md), or report.md says 项目结束",
    ]
    # unless the questionnaire's last answer ended the project: no plan then
    (pdir / "workspaces/intake/r2/report.md").write_text("## 结论\n\n项目结束：目标已达成。\n")
    assert run_check("intake", pdir, "intake/r2")[1] == ["FAIL  project.yaml round is 2 (it is 1)"]
    # the command, as a check runs it
    assert perfcheck.main(["intake", "intake/nosuch"]) == 1
    assert perfcheck.main(["bogus"]) == 2


def test_survey_check(tmp_path):
    pdir = ops.new_project(tmp_path, "glm", "infera-perf")
    assert not run_check("survey", pdir, "survey/r1")[0]
    (pdir / "workspaces/survey/r1/report.md").write_text("## 结论\n\n无公开对标。\n")
    assert run_check("survey", pdir, "survey/r1")[0]


PLAN_CHILD = """---
parent: plan/r1
machines: [n1]
expect: {delta_pct: [2, 4], level: L2, falsify: 队列没有下降}
---
"""


def test_plan_check(tmp_path):
    pdir = ops.new_project(tmp_path, "glm", "infera-perf")
    fill_intake(pdir)
    passed, failed = run_check("plan", pdir, "plan/r1")
    assert not passed and failed == [
        "FAIL  a task waits for the plan's subtasks (depends_on {task: plan/r1, kind: children})"
    ]
    (pdir / "tasks/experiment/r1-spf.md").write_text(PLAN_CHILD, encoding="utf-8")
    (pdir / "tasks/experiment/r1-bad.md").write_text(
        "---\nparent: plan/r1\nmachines: [n9]\nexpect: {delta_pct: big, level: L9}\n---\n",
        encoding="utf-8",
    )
    (pdir / "tasks/close/r1.md").write_text(
        "---\ndepends_on: [{task: plan/r1, kind: children}]\nmachines: [n2]\n---\n",
        encoding="utf-8",
    )
    passed, failed = run_check("plan", pdir, "plan/r1")
    assert not passed and failed == [
        "FAIL  experiment/r1-bad has its machines: machines not in project.yaml: n9",
        "FAIL  experiment/r1-bad expect.delta_pct is a number or [low, high]",
        "FAIL  experiment/r1-bad expect.level is one of L0/L1/L2/L3/L4/L5 (the fast check it uses)",
        "FAIL  experiment/r1-bad expect.falsify says what result would refute it",
    ]
    (pdir / "tasks/experiment/r1-bad.md").unlink()
    assert run_check("plan", pdir, "plan/r1") == (True, [])


def test_close_check(tmp_path):
    pdir = ops.new_project(tmp_path, "glm", "infera-perf")
    fill_intake(pdir)
    (pdir / "tasks/close/r1.md").write_text(
        "---\ndepends_on: [{task: plan/r1, kind: children}]\n---\n", encoding="utf-8"
    )
    ws = pdir / "workspaces/close/r1"
    ws.mkdir(parents=True)
    passed, failed = run_check("close", pdir, "close/r1")
    assert failed == [
        'FAIL  stack.json is {"version": ..., "sealed": true|false, ...}',
        "FAIL  the next round's intake is drafted (tasks/intake/r2.md)",
    ]
    # not sealed: a retrospective, no curve needed
    (ws / "stack.json").write_text('{"version": "v0", "sealed": false}')
    (pdir / "tasks/intake/r2.md").write_text("---\ndepends_on: [close/r1]\n---\n")
    assert run_check("close", pdir, "close/r1") == (True, [])
    # sealed: the curve, and project.yaml moved on to it
    (ws / "stack.json").write_text('{"version": "v1", "sealed": true}')
    write_result(
        pdir,
        "close/r1",
        "v1-c64",
        {"series": "v1", "workload": {"conc": 64}, "metrics": {"y": 1.0}},
    )
    assert run_check("close", pdir, "close/r1")[1] == [
        "FAIL  results/ has the integration curve of series 'v1' (2+ concurrencies, it has 1)",
        "FAIL  project.yaml version is 'v1'",
        "FAIL  project.yaml ledger.baseline_series is 'v1'",
        "FAIL  results/ has v1's GSM8K over all 1319 questions (it has: none)",
    ]
    write_result(
        pdir,
        "close/r1",
        "v1-c32",
        {"series": "v1", "workload": {"conc": 32}, "metrics": {"y": 1.2}},
    )
    cfg = (pdir / "project.yaml").read_text(encoding="utf-8")
    cfg = cfg.replace("version: v0", "version: v1").replace(
        "baseline_series: v0", "baseline_series: v1"
    )
    (pdir / "project.yaml").write_text(cfg, encoding="utf-8")
    assert run_check("close", pdir, "close/r1")[1] == [
        "FAIL  results/ has v1's GSM8K over all 1319 questions (it has: none)"
    ]
    # the sealed version's GSM8K over the full set, against v0's
    gsm8k(ws, "v1-full", "v1", 1319, range(0, 1319, 25))
    assert run_check("close", pdir, "close/r1")[1] == [
        "FAIL  a version before v1 has a GSM8K to compare with (none has)"
    ]
    gsm8k(pdir / "workspaces/baseline/r1", "v0-full", "v0", 1319, range(0, 1319, 25))
    assert run_check("close", pdir, "close/r1") == (True, [])
    (ws / "stack.json").write_text("{broken")
    assert run_check("close", pdir, "close/r1")[1][0].startswith("FAIL  stack.json is JSON")


# ---------------------------------------------------------------- a whole round from the template, then spoiled


def good_round(tmp_path):
    """A first round of the infera-perf template as its tasks leave it, every field check passing: intake filled in
    (working point C40), a survey with a reference, the baseline's v0 curve with a repeat at the working point on one
    node and v0's full GSM8K, a plan with its experiment and close, a close that sealed v1 (curve, full GSM8K, project.yaml
    moved on) and drafted the next intake."""
    from measuring import convert, measure, raw

    pdir = ops.new_project(tmp_path, "glm", "infera-perf")
    fill_intake(pdir)
    cfg = (
        (pdir / "project.yaml")
        .read_text(encoding="utf-8")
        .replace("working_point: {x: 30}", "working_point: {conc: 40}")
    )
    (pdir / "project.yaml").write_text(cfg, encoding="utf-8")
    write_result(
        pdir,
        "survey/r1",
        "ix-c40",
        {
            "role": "reference",
            "series": "ix:a",
            "workload": {"conc": 40},
            "metrics": {"input_tput_per_gpu": 25000.0, "p90_intvty": 60.0},
        },
    )
    bws = pdir / "workspaces/baseline/r1"
    convert(bws, measure(bws, "v0-c40", raw(1.0)), "v0", "baseline", "v0")
    convert(bws, measure(bws, "v0-rep1-c40", raw(1.005)), "v0", "baseline", "v0-rep1")
    convert(bws, measure(bws, "v0-c64", raw(1.2, conc=64)), "v0", "baseline", "v0", conc=64)
    gsm8k(bws, "v0-full", "v0", 1319, range(0, 1319, 25))
    (bws / "stack.json").write_text('{"version": "v0", "sealed": true}')
    (pdir / "tasks/experiment/r1-spf.md").write_text(PLAN_CHILD, encoding="utf-8")
    (pdir / "tasks/close/r1.md").write_text(
        "---\ndepends_on: [{task: plan/r1, kind: children}]\nmachines: [n2]\n---\n",
        encoding="utf-8",
    )
    cws = pdir / "workspaces/close/r1"
    for conc, scale in ((40, 1.08), (64, 1.25)):
        convert(
            cws,
            measure(cws, f"v1-c{conc}", raw(scale, conc=conc)),
            "v1",
            "baseline",
            "v1",
            conc=conc,
        )
    gsm8k(cws, "v1-full", "v1", 1319, range(0, 1319, 25))
    (cws / "stack.json").write_text('{"version": "v1", "sealed": true}')
    cfg = cfg.replace("version: v0", "version: v1").replace(
        "baseline_series: v0", "baseline_series: v1"
    )
    (pdir / "project.yaml").write_text(cfg, encoding="utf-8")
    (pdir / "tasks/intake/r2.md").write_text("---\ndepends_on: [close/r1]\n---\n")
    return pdir


CHECKED = (
    ("intake", "intake/r1"),
    ("survey", "survey/r1"),
    ("baseline", "baseline/r1"),
    ("plan", "plan/r1"),
    ("close", "close/r1"),
)


def test_a_good_round_passes_every_check(tmp_path):
    pdir = good_round(tmp_path)
    for kind, tid in CHECKED:
        assert run_check(kind, pdir, tid) == (True, []), kind
    from agent_sys_v2 import ledger

    for tid in ("survey/r1", "baseline/r1", "close/r1"):
        assert ledger.check([pdir / "workspaces" / tid / "results"])[0], tid


def text(p):
    return p.read_text(encoding="utf-8")


def sub(path, old, new):
    t = text(path)
    assert old in t, (path, old)
    path.write_text(t.replace(old, new), encoding="utf-8")


def edit_json(path, change):
    import json

    d = json.loads(text(path))
    change(d)
    path.write_text(json.dumps(d), encoding="utf-8")


def remeasure(pdir, name, label, **kw):
    """Replace a baseline point by another measurement of it."""
    import shutil

    from measuring import convert, measure, raw

    bws = pdir / "workspaces/baseline/r1"
    (bws / f"results/{label}-c40.json").unlink()
    shutil.rmtree(bws / "log/agentx" / name)
    convert(bws, measure(bws, name, raw(kw.pop("scale", 1.005)), **kw), "v0", "baseline", label)


def regsm8k(pdir, tid, label, series, total, wrong):
    import shutil

    ws = pdir / "workspaces" / tid
    (ws / f"results/{label}.gsm8k.json").unlink()
    shutil.rmtree(ws / "log/gsm8k" / label)
    gsm8k(ws, label, series, total, wrong)


def samples(pdir, tid):
    return next((pdir / "workspaces" / tid / "log/gsm8k").rglob("samples_*.jsonl"))


Y = "project.yaml"
B = "workspaces/baseline/r1"
C = "workspaces/close/r1"
# (check, task, name, the file the check reads (relative to the project), the spoil, words of the FAIL line)
PERF_SPOILS = [
    (
        "intake",
        "intake/r1",
        "goal emptied",
        Y,
        lambda p: sub(p / Y, "goal: 工作点 Y +10%", 'goal: ""'),
        "project.yaml goal is set",
    ),
    (
        "intake",
        "intake/r1",
        "machines emptied",
        Y,
        lambda p: sub(p / Y, "machines: [{name: n1, gpus: 8}, n2]", "machines: []"),
        "project.yaml machines lists the nodes",
    ),
    (
        "intake",
        "intake/r1",
        "round moved on",
        Y,
        lambda p: sub(p / Y, "round: 1", "round: 2"),
        "project.yaml round is 1",
    ),
    (
        "intake",
        "intake/r1",
        "no working point",
        Y,
        lambda p: sub(p / Y, "working_point: {conc: 40}", "working_point: {}"),
        "ledger.working_point",
    ),
    (
        "intake",
        "intake/r1",
        "a placeholder left in the rules",
        "rules/project.md",
        lambda p: (p / "rules/project.md").write_text(
            text(p / "rules/project.md") + "- 交付：<待 intake 填写>\n"
        ),
        "rules/project.md is written",
    ),
    (
        "intake",
        "intake/r1",
        "the questionnaire deleted",
        "workspaces/intake/r1/questions.md",
        lambda p: (p / "workspaces/intake/r1/questions.md").unlink(),
        "the questionnaire was asked",
    ),
    (
        "intake",
        "intake/r1",
        "a task on an undeclared machine",
        "tasks/profile/r1.md",
        lambda p: ops.update_task(p, "profile/r1", {"machines": ["n9"]}),
        "profile/r1 has its machines: machines not in project.yaml: n9",
    ),
    (
        "survey",
        "survey/r1",
        "the reference deleted",
        "workspaces/survey/r1/results/ix-c40.json",
        lambda p: (p / "workspaces/survey/r1/results/ix-c40.json").unlink(),
        "results/ has reference curves (0 points)",
    ),
    (
        "baseline",
        "baseline/r1",
        "the working point measured once",
        f"{B}/results/v0-rep1-c40.json",
        lambda p: (p / B / "results/v0-rep1-c40.json").unlink(),
        "C40 has 2+ formal AgentX measurements of v0 on one node (ledger.min_repeats; it has 1 on 10.0.0.1",
    ),
    (
        "baseline",
        "baseline/r1",
        "the repeat in AgentX fast mode",
        f"{B}/results/v0-rep1-c40.json",
        lambda p: remeasure(p, "v0-rep1-c40", "v0-rep1", fast=True),
        "it has 1 on 10.0.0.1",
    ),
    (
        "baseline",
        "baseline/r1",
        "the repeat on another node",
        f"{B}/results/v0-rep1-c40.json",
        lambda p: remeasure(p, "v0-rep1-c40", "v0-rep1", node="10.0.0.2"),
        "it has 1 on",
    ),
    (
        "baseline",
        "baseline/r1",
        "repeats 3% apart",
        f"{B}/results/v0-rep1-c40.json",
        lambda p: remeasure(p, "v0-rep1-c40", "v0-rep1", scale=1.03),
        "their retest noise 5.79% (1.96 x sqrt(2) x RSD) is below the accept bar 3%",
    ),
    (
        "baseline",
        "baseline/r1",
        "a repeat's raw output edited",
        f"{B}/log/agentx/v0-rep1-c40/agentx_conc40.json",
        lambda p: sub(
            p / B / "log/agentx/v0-rep1-c40/agentx_conc40.json", '"conc": 40', '"conc": 40, "x": 1'
        ),
        "a v0 point holds against its run directory",
    ),
    (
        "baseline",
        "baseline/r1",
        "GSM8K over 200 questions only",
        f"{B}/results/v0-full.gsm8k.json",
        lambda p: regsm8k(p, "baseline/r1", "v0-full", "v0", 200, range(0, 200, 25)),
        "results/ has v0's GSM8K over all 1319 questions",
    ),
    (
        "baseline",
        "baseline/r1",
        "lm_eval's samples edited",
        f"{B}/log/gsm8k/v0-full",
        lambda p: sub(samples(p, "baseline/r1"), '"exact_match": 0.0', '"exact_match": 1.0'),
        "v0-full.gsm8k.json counts again from its lm_eval samples: its lm_eval samples",
    ),
    (
        "plan",
        "plan/r1",
        "an experiment's level removed",
        "tasks/experiment/r1-spf.md",
        lambda p: sub(p / "tasks/experiment/r1-spf.md", " level: L2,", ""),
        "experiment/r1-spf expect.level is one of",
    ),
    (
        "plan",
        "plan/r1",
        "an experiment's machine undeclared",
        "tasks/experiment/r1-spf.md",
        lambda p: sub(p / "tasks/experiment/r1-spf.md", "machines: [n1]", "machines: [n9]"),
        "experiment/r1-spf has its machines: machines not in project.yaml: n9",
    ),
    (
        "plan",
        "plan/r1",
        "no close waiting for the subtasks",
        "tasks/close/r1.md",
        lambda p: (p / "tasks/close/r1.md").unlink(),
        "a task waits for the plan's subtasks",
    ),
    (
        "close",
        "close/r1",
        "stack.json without sealed",
        f"{C}/stack.json",
        lambda p: (p / C / "stack.json").write_text('{"version": "v1"}'),
        'stack.json is {"version": ..., "sealed": true|false, ...}',
    ),
    (
        "close",
        "close/r1",
        "the curve at one concurrency",
        f"{C}/results/v1-c64.json",
        lambda p: (p / C / "results/v1-c64.json").unlink(),
        "integration curve of series 'v1' (2+ concurrencies, it has 1)",
    ),
    (
        "close",
        "close/r1",
        "project.yaml not moved on",
        Y,
        lambda p: sub(p / Y, "baseline_series: v1", "baseline_series: v0"),
        "project.yaml ledger.baseline_series is 'v1'",
    ),
    (
        "close",
        "close/r1",
        "v1's GSM8K over 200 questions",
        f"{C}/results/v1-full.gsm8k.json",
        lambda p: regsm8k(p, "close/r1", "v1-full", "v1", 200, set()),
        "results/ has v1's GSM8K over all 1319 questions (it has: workspaces/close/r1/results/v1-full.gsm8k.json (200))",
    ),
    (
        "close",
        "close/r1",
        "v1 2.7 points worse on GSM8K",
        f"{C}/results/v1-full.gsm8k.json",
        lambda p: regsm8k(p, "close/r1", "v1-full", "v1", 1319, range(0, 1319, 15)),
        "is not worse than v0's by more than 0.02: 1231 vs 1266 of 1319 questions",
    ),
    (
        "close",
        "close/r1",
        "v1's samples edited",
        f"{C}/log/gsm8k/v1-full",
        lambda p: sub(samples(p, "close/r1"), '"exact_match": 0.0', '"exact_match": 1.0'),
        "counts again from its lm_eval samples: its lm_eval samples",
    ),
    (
        "close",
        "close/r1",
        "no next intake",
        "tasks/intake/r2.md",
        lambda p: (p / "tasks/intake/r2.md").unlink(),
        "the next round's intake is drafted",
    ),
]


def snapshot_of(path):
    """What a check reads at path: a file's bytes, a directory's files and their bytes, or None when it is gone."""
    if path.is_dir():
        return sorted((str(f), f.read_bytes()) for f in path.rglob("*") if f.is_file())
    return path.read_bytes() if path.exists() else None


@pytest.mark.parametrize(
    "kind,tid,name,reads,spoil,words", PERF_SPOILS, ids=[f"{s[0]}: {s[2]}" for s in PERF_SPOILS]
)
def test_a_spoiled_round_fails_its_check(tmp_path, kind, tid, name, reads, spoil, words):
    pdir = good_round(tmp_path)
    assert run_check(kind, pdir, tid) == (True, [])
    before = snapshot_of(pdir / reads)
    spoil(pdir)
    assert snapshot_of(pdir / reads) != before, (
        f"{name}: the spoil did not change what the check reads"
    )
    passed, failed = run_check(kind, pdir, tid)
    assert not passed and any(words in line for line in failed), (name, failed)


def test_the_spoils_cover_every_check():
    assert {s[0] for s in PERF_SPOILS} == {k for k, _t in CHECKED}


def test_a_baseline_that_cannot_judge_tells_the_agent_to_stop(tmp_path):
    pdir = good_round(tmp_path)
    remeasure(pdir, "v0-rep1-c40", "v0-rep1", scale=1.03)
    _ok, lines = perfcheck.run("baseline", "baseline/r1", pdir)
    assert any("set this task blocked (阻塞) and ask the user" in line for line in lines)
    # with a working point at X, the concurrency with the most repeats on one node is the one judged
    sub(pdir / Y, "working_point: {conc: 40}", "working_point: {x: 30}")
    assert any(
        "C40 has 2+ formal" in line for line in perfcheck.run("baseline", "baseline/r1", pdir)[1]
    )


# ---------------------------------------------------------------- ledger check, spoiled

R = f"{B}/results/v0-c40.json"
G = f"{B}/results/v0-full.gsm8k.json"
LEDGER_SPOILS = [
    ("no label", R, lambda d: d.update(label=" "), "label is missing"),
    (
        "metrics not numbers",
        R,
        lambda d: d.update(metrics={"input_tput_per_gpu": "fast"}),
        "metrics should hold at least one number",
    ),
    ("an unknown role", R, lambda d: d.update(role="winner"), "role should be one of"),
    ("a negative round", R, lambda d: d.update(round=-1), "round should be a whole number"),
    (
        "a concurrency of 0",
        R,
        lambda d: d["workload"].update(conc=0),
        "workload.conc should be a positive number",
    ),
    (
        "an unknown decision",
        R,
        lambda d: d.update(verdict={"decision": "great"}),
        "verdict.decision should be one of",
    ),
    (
        "a window that ends before it starts",
        R,
        lambda d: d["workload"]["window"].update(end="2026-10-01T00:00:00"),
        "workload.window ends before it starts",
    ),
    (
        "fast mode not a boolean",
        R,
        lambda d: d["workload"].update(fast="no"),
        "workload.fast should be true or false",
    ),
    (
        "a duration of 0",
        R,
        lambda d: d["workload"].update(benchmark_duration_s=0),
        "workload.benchmark_duration_s should be a positive number",
    ),
    (
        "GSM8K: more right than asked",
        G,
        lambda d: d.update(correct=2000),
        "correct is more than total",
    ),
    (
        "GSM8K: no evidence",
        G,
        lambda d: d.pop("evidence"),
        "evidence.samples and samples_sha256 are missing",
    ),
    ("GSM8K: no series", G, lambda d: d.pop("series"), "series is missing"),
]


@pytest.mark.parametrize("name,rel,change,words", LEDGER_SPOILS, ids=[s[0] for s in LEDGER_SPOILS])
def test_a_spoiled_result_fails_ledger_check(tmp_path, name, rel, change, words):
    from agent_sys_v2 import ledger

    pdir = good_round(tmp_path)
    results = pdir / B / "results"
    assert ledger.check([results])[0]
    before = text(pdir / rel)
    edit_json(pdir / rel, change)
    assert text(pdir / rel) != before
    ok, lines = ledger.check([results])
    assert not ok and any(words in line for line in lines), (name, lines)


def test_ledger_check_wants_a_decision_on_a_candidate_and_finite_numbers(tmp_path):
    from agent_sys_v2 import ledger

    pdir = good_round(tmp_path)
    results = pdir / B / "results"
    edit_json(results / "v0-c40.json", lambda d: d.update(role="candidate"))
    assert ledger.check([results])[0]
    ok, lines = ledger.check([results], require_verdict=True)
    assert not ok and any("verdict.decision is missing" in line for line in lines)
    sub(results / "v0-c40.json", '"role": "candidate"', '"role": "baseline", "x": NaN')
    ok, lines = ledger.check([results])
    assert not ok and any("NaN is not a number" in line for line in lines)
    (results / "v0-c40.json").write_text("{broken")
    assert any("not JSON" in line for line in ledger.check([results])[1])


def test_the_bars_are_the_runs_not_project_yaml_now(tmp_path, monkeypatch):
    """close may write project.yaml (version, baseline series): a margin or accept bar loosened there does not loosen
    its own check, nor baseline's; the run's snapshot ($AGENT_SYS_RULES) is what they judge by."""
    import json

    from agent_sys_v2 import verdict
    from agent_sys_v2.model import load_yaml

    pdir = good_round(tmp_path)
    snap = verdict.rules(load_yaml(text(pdir / Y)))
    monkeypatch.setenv(verdict.RULES_ENV, json.dumps(snap))
    regsm8k(pdir, "close/r1", "v1-full", "v1", 1319, range(0, 1319, 15))
    sub(pdir / Y, "gsm8k_margin: 0.02", "gsm8k_margin: 0.5")
    assert not run_check("close", pdir, "close/r1")[0]
    remeasure(pdir, "v0-rep1-c40", "v0-rep1", scale=1.03)
    sub(pdir / Y, "  accept_pct: 3", "  accept_pct: 6")
    assert not run_check("baseline", pdir, "baseline/r1")[0]
    monkeypatch.delenv(verdict.RULES_ENV)
    assert run_check("baseline", pdir, "baseline/r1")[0]  # without a run: project.yaml's
