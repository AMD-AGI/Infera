"""The field checks of the infera-perf template's round (perfcheck.py): what intake, survey, plan and close leave
behind before they count as done."""

from conftest import write_result

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
    assert run_check("close", pdir, "close/r1") == (True, [])
    (ws / "stack.json").write_text("{broken")
    assert run_check("close", pdir, "close/r1")[1][0].startswith("FAIL  stack.json is JSON")
