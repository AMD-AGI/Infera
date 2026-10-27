"""What a project that goes in rounds needs (the infera-perf template): the children dependency (a round's close
waits for its plan and all of the plan's subtasks), joining the drafts a plan wrote at once, the tasks a template
brings, the skeletons a round drafts its next tasks from, and the demo home."""

import importlib.util
import re
import shutil
from pathlib import Path

import pytest
from conftest import T, get, manage, new, ok, run_home, set_status
from fastapi.testclient import TestClient

from agent_sys_v2 import launch, ledger, ops, sched, skills
from agent_sys_v2.model import (
    AgentSysError,
    cycle_tasks,
    split_frontmatter,
)
from agent_sys_v2.scan import load_project
from agent_sys_v2.server import create_app

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- the children dependency


ROUND = {
    "plan/r1": "---\ntitle: 计划\n---\n",
    "experiment/a": "---\ntitle: 实验 a\nparent: plan/r1\n---\n",
    "experiment/b": "---\ntitle: 实验 b\nparent: plan/r1\n---\n",
    "close/r1": "---\ndepends_on: [{task: plan/r1, kind: children}]\n---\n收尾\n",
}


def test_children_waits_for_the_upstream_and_every_subtask(make_project):
    pdir = make_project(ROUND)
    manage(pdir, *ROUND)
    p = load_project(pdir)
    edge = next(e for e in p.edges if e.down == "close/r1")
    assert (edge.up, edge.kind, edge.blocking) == ("plan/r1", "children", True)
    # the plan itself first: an upstream like any other
    assert p.tasks["close/r1"].detail == {"kind": "upstream", "tasks": ["plan/r1"]}
    set_status(pdir, "plan/r1", "done")
    p = load_project(pdir)
    assert p.tasks["close/r1"].detail == {
        "kind": "children",
        "parents": ["plan/r1"],
        "tasks": ["experiment/a", "experiment/b"],
    }
    assert not p.edge_satisfied(edge) and p.satisfied("plan/r1")
    # a cancelled subtask counts as finished: cancelling the rest is how a round ends early
    set_status(pdir, "experiment/a", "done")
    set_status(pdir, "experiment/b", "cancelled")
    p = load_project(pdir)
    assert p.tasks["close/r1"].detail == {} and p.edge_satisfied(p.upstream("close/r1")[0])
    # a subtask added later (a draft too) holds the close back again
    (pdir / "tasks/experiment/c.md").write_text("---\nparent: plan/r1\n---\n后加的\n")
    p = load_project(pdir)
    assert p.tasks["close/r1"].detail["tasks"] == ["experiment/c"]
    # the scheduler leaves a task waiting for subtasks alone
    assert sched.plan([p], 10) == []


def test_children_leaves_out_the_waiting_task_and_its_downstream(make_project):
    pdir = make_project(
        {
            "plan/r1": "",
            "experiment/a": "---\nparent: plan/r1\n---\n",
            # the close is a subtask of the plan as well, and a follow-up subtask waits for the close
            "close/r1": "---\nparent: plan/r1\ndepends_on: [{task: plan/r1, kind: children}]\n---\n",
            "experiment/after": "---\nparent: plan/r1\ndepends_on: [close/r1]\n---\n",
        }
    )
    manage(pdir, "plan/r1", "experiment/a", "close/r1", "experiment/after")
    set_status(pdir, "plan/r1", "done")
    p = load_project(pdir)
    assert p.tasks["close/r1"].detail["tasks"] == ["experiment/a"]
    assert not p.tasks["close/r1"].warnings  # no cycle reported
    set_status(pdir, "experiment/a", "done")
    assert load_project(pdir).tasks["close/r1"].detail == {}


def test_children_edges_are_checked_for_cycles(make_project):
    pdir = make_project({"plan/r1": "", "close/r1": ""})
    ops.update_task(pdir, "close/r1", {"depends_on": [{"task": "plan/r1", "kind": "children"}]})
    fm = load_project(pdir).tasks["close/r1"].fm
    assert fm["depends_on"][0]["kind"] == "children"
    with pytest.raises(AgentSysError) as e:
        ops.update_task(pdir, "plan/r1", {"depends_on": ["close/r1"]})
    assert e.value.status == 409 and "循环" in str(e.value)


def test_the_close_task_is_told_about_the_plans_subtasks(make_project):
    pdir = make_project(ROUND)
    manage(pdir, *ROUND)
    (pdir / "workspaces/experiment/a/report.md").write_text("## 结论\n\n接受\n")
    p = load_project(pdir)
    assert launch.upstream_lines(p, p.tasks["close/r1"]) == [
        "- 计划（`plan/r1`，等它和它的子任务完成）：工作区 `workspaces/plan/r1/`，还没有 report.md",
        "  它的子任务：",
        "  - 实验 a（`experiment/a`，queued）：报告 `workspaces/experiment/a/report.md`",
        "  - 实验 b（`experiment/b`，queued）：还没有 report.md",
    ]
    en = "\n".join(launch.upstream_lines(p, p.tasks["close/r1"], "en"))
    assert "after it and its subtasks" in en and "Its subtasks:" in en


def test_children_through_the_server(tmp_path):
    home = run_home(tmp_path, "autorun: false\n")
    pdir = home / "projects/r"
    with TestClient(create_app(home)) as c:
        for name, text in (
            ("plan", "计划\n"),
            ("e1", "---\nparent: develop/plan\n---\n实验\n"),
            ("close", "---\ndepends_on: [{task: develop/plan, kind: children}]\n---\n收尾\n"),
        ):
            (pdir / f"tasks/develop/{name}.md").write_text(text)
        manage(pdir, "develop/plan", "develop/e1", "develop/close")
        set_status(pdir, "develop/plan", "done")
        c.app.state.store.sync(also=[("r", None)])
        t = get(c, "close")
        assert t["detail"] == {
            "kind": "children",
            "parents": ["develop/plan"],
            "tasks": ["develop/e1"],
        }
        up = t["upstream"][0]
        assert (up["kind"], up["blocking"], up["satisfied"]) == ("children", True, False)
        board = ok(c.get("/api/projects/r/board"))
        assert {"from": "develop/plan", "to": "develop/close", "kind": "children"}.items() <= next(
            e for e in board["edges"] if e["to"] == "develop/close"
        ).items()


# ---------------------------------------------------------------- joining what a plan drafted


def test_join_takes_the_subtasks_and_the_drafts_that_depend_on_the_plan(make_project):
    pdir = make_project(
        {
            **ROUND,
            "review/x": "---\ndepends_on: [{task: plan/r1, kind: review}]\n---\n评审\n",
            "experiment/other": "无关\n",
        },
        types=("experiment",),
    )
    (pdir / "task_types").mkdir()
    (pdir / "task_types/experiment.md").write_text(
        "---\nchecks:\n  - test -s report.md\n---\n## 任务描述\n"
    )
    (pdir / "task_types/close.md").write_text("---\nreview: true\n---\n## 任务描述\n")
    manage(pdir, "plan/r1")
    joined = ops.join_children(pdir, "plan/r1")
    assert sorted(joined) == ["close/r1", "experiment/a", "experiment/b"]
    p = load_project(pdir)
    assert p.tasks["review/x"].status == "draft" and p.tasks["experiment/other"].status == "draft"
    # a draft gets its type's review and checks when it has none of its own
    assert [c["run"] for c in p.tasks["experiment/a"].checks] == ["test -s report.md"]
    assert p.tasks["close/r1"].review and not p.tasks["close/r1"].checks
    assert ops.drafts_of(p, "plan/r1") == []
    with pytest.raises(AgentSysError) as e:
        ops.join_children(pdir, "plan/r1")
    assert e.value.status == 409


def test_join_keeps_what_a_draft_says_itself(make_project):
    pdir = make_project({"experiment/a": "---\nreview: false\nchecks: []\n---\n实验\n"})
    (pdir / "task_types").mkdir()
    (pdir / "task_types/experiment.md").write_text(
        "---\nreview: true\nchecks: [true]\n---\n## 任务描述\n"
    )
    ops.join(pdir, "experiment/a")
    t = load_project(pdir).tasks["experiment/a"]
    assert t.status == "queued" and not t.review and t.checks == []


def test_join_the_drafts_a_task_wrote(tmp_path):
    home = run_home(tmp_path, "autorun: false\n")
    pdir = home / "projects/r"
    with TestClient(create_app(home)) as c:
        new(c, "plan", mode="draft")
        for name in ("e1", "e2"):
            (pdir / f"tasks/develop/{name}.md").write_text(
                f"---\nparent: develop/plan\nsummary: 试 {name}\n---\n做 {name}\n"
            )
        (pdir / "tasks/develop/other.md").write_text("与 plan 无关\n")
        c.app.state.store.sync(also=[("r", None)])
        r = ok(c.post(f"{T}/plan/actions/join_children"))
        assert sorted(r["joined"]) == ["develop/e1", "develop/e2"]
        assert get(c, "e1")["status"] == "queued" and get(c, "other")["status"] == "draft"
        assert "没有待纳入托管" in c.post(f"{T}/plan/actions/join_children").json()["error"]
        with pytest.raises(AgentSysError) as e:
            ops.join_children(pdir, "develop/nope")
        assert e.value.status == 404


# ---------------------------------------------------------------- tasks a template brings


def test_a_template_brings_its_tasks_queued_for_review(tmp_path):
    templates = tmp_path / "templates"
    shutil.copytree(ops.TEMPLATES / "_common", templates / "_common")
    src = templates / "t"
    for path, text in {
        "project.yaml": "title: T\n",
        "task_types/develop.md": "## 任务描述\n",
        "tasks/develop/first.md": "---\ntitle: 第一步\n---\n做\n",
        "tasks/develop/second.md": "---\ndepends_on: [develop/first]\n---\n再做\n",
        "tasks/other/x.md": "没有骨架的类型\n",
        "tasks/develop/.hidden.md": "不是任务\n",
        "skills/own/SKILL.md": "---\ndescription: 模板自带的 skill\n---\n",
    }.items():
        (src / path).parent.mkdir(parents=True, exist_ok=True)
        (src / path).write_text(text, encoding="utf-8")
    assert ops.list_templates(templates)[0]["tasks"] == [
        "develop/first",
        "develop/second",
        "other/x",
    ]
    pdir = ops.new_project(tmp_path / "home", "c", "t", templates_dir=templates)
    p = load_project(pdir)
    assert {t.id: t.status for t in p.tasks.values()} == {
        "develop/first": "queued",
        "develop/second": "queued",
        "other/x": "queued",
    }
    assert p.tasks["develop/second"].detail == {"kind": "upstream", "tasks": ["develop/first"]}
    assert len({t.state["session_id"] for t in p.tasks.values()}) == 3
    assert len({t.created for t in p.tasks.values()}) == 1
    assert [s["name"] for s in skills.found(pdir / "skills")] == ["own"]
    assert not (pdir / "tasks/develop/.hidden.md").exists()
    # nothing starts before you have reviewed them: the first task waits for you, not for a slot
    assert p.template_review and p.to_json()["template_review"] is True
    assert p.tasks["develop/first"].detail == {"kind": "template_review"}
    assert sched.plan([p], 4) == []
    ops.start_project(pdir)
    p = load_project(pdir)
    assert not p.template_review and "template_review" not in (pdir / "project.yaml").read_text()
    assert p.tasks["develop/first"].detail == {}
    assert [t.id for _, t in sched.plan([p], 4)] == ["develop/first", "other/x"]
    # a second time: nothing to do
    ops.start_project(pdir)
    assert load_project(pdir).tasks["develop/first"].detail == {}
    # a template without tasks has nothing to review
    assert not load_project(ops.new_project(tmp_path / "home", "b", "blank")).template_review
    ops.set_archived(pdir, True)
    with pytest.raises(AgentSysError):
        ops.start_project(pdir)


def test_the_infera_perf_round_one(tmp_path):
    pdir = ops.new_project(tmp_path, "glm", "infera-perf")
    p = load_project(pdir)
    assert sorted(p.tasks) == ["baseline/r1", "intake/r1", "plan/r1", "profile/r1", "survey/r1"]
    assert all(t.managed and t.status == "queued" and not t.warnings for t in p.tasks.values())
    # only intake waits for nothing; nothing in a prompt names a later task (that would close a cycle)
    assert sorted((e.up, e.down) for e in p.edges if e.blocking) == [
        ("baseline/r1", "profile/r1"),
        ("intake/r1", "baseline/r1"),
        ("intake/r1", "plan/r1"),
        ("intake/r1", "survey/r1"),
        ("profile/r1", "plan/r1"),
        ("survey/r1", "plan/r1"),
    ]
    assert p.tasks["intake/r1"].detail == {"kind": "template_review"}
    assert {t.id for t in p.tasks.values() if t.review} == {"intake/r1", "profile/r1", "plan/r1"}
    assert p.tasks["intake/r1"].agent["model"] == "sonnet"
    # a preset task carries the checks of its type
    types = {ty.id: ty for ty in p.types}
    for t in p.tasks.values():
        assert t.checks == [dict(c, name=c["run"][:120]) for c in t.checks]
        assert [c["run"] for c in t.checks] == [str(x) for x in types[t.type].checks], t.id
    # the first round's questionnaire: 22 rows, the 8 that must be answered marked
    q = (pdir / "skills/infera-round/questionnaire_r1.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| ([A-D]\d)( ★)? \|", q, re.M)
    assert len(rows) == 22 and sum(bool(star) for _, star in rows) == 8
    assert "| # | 问题 | 建议 | 你的意见 |" in q


def test_tasks_drafted_from_the_skeletons_make_no_cycle(tmp_path):
    """A skeleton that names a later task by its path would make the task drafted from it wait for that task: close/r1
    drafts intake/r2 (which depends on it), plan/r2 drafts close/r2 (which waits for its subtasks)."""
    pdir = ops.new_project(tmp_path, "glm", "infera-perf")

    def draft(tid, fm, n):
        _fm, body = split_frontmatter(
            (pdir / "task_types" / f"{tid.split('/')[0]}.md").read_text(encoding="utf-8")
        )
        for mark, value in (("<N+1>", n + 1), ("<N−1>", n - 1), ("N−1", n - 1), ("<N>", n)):
            body = body.replace(mark, str(value))
        (pdir / "tasks" / f"{tid}.md").write_text(f"---\n{fm}---\n{body}", encoding="utf-8")

    draft("close/r1", "depends_on: [{task: plan/r1, kind: children}]\n", 1)
    draft("intake/r2", "depends_on: [close/r1]\n", 2)
    draft("plan/r2", "depends_on: [intake/r2]\n", 2)
    draft("close/r2", "depends_on: [{task: plan/r2, kind: children}]\n", 2)
    p = load_project(pdir)
    assert not cycle_tasks(p.edges)
    assert not [w for t in p.tasks.values() for w in t.warnings if "循环" in w]


def load_demo():
    spec = importlib.util.spec_from_file_location("demo", ROOT / "tools/demo.py")
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    return demo


def test_the_demo_project(tmp_path):
    """tools/demo.py: round one sealed, round two's close waiting for the plan's subtasks, the ledger's curves, verdicts
    that hold, one judged against v0, two that do not hold."""
    p = load_project(load_demo().main(tmp_path / "home"))
    assert p.tasks["close/r2"].detail["kind"] == "children"
    assert ops.drafts_of(p, "plan/r2") == ["experiment/r2-router"]
    full = ledger.ledger(p)
    assert full["errors"] == []
    assert [c["name"] for c in full["series"]][0] == "v1" and full["best"]["series"] == "r2-kvroute"
    rows = {r["label"]: r for r in full["rows"]}
    assert rows["r2-kvroute-1-c64"]["verified"] is True and rows["r2-kvroute-base1-c64"]["repeat"]
    assert rows["r1-spf-1-c64"]["judged_against"] == "v0" and rows["r1-spf-1-c64"]["verified"]
    assert (
        rows["r2-tokenizer-c64"]["verified"] is False and rows["r1-dcp4-c64"]["verified"] is False
    )
    assert rows["r2-hicache-c64"]["status"] == "void" and rows["r2-hicache-c64"]["verified"]
    # an AgentX fast run: screening, drawn, no input, and the A/B beside it holds
    fast = rows["r2-kvroute-fast-c64"]
    assert fast["screening"].startswith("AgentX fast mode") and not fast["repeat"]
    assert fast["delta_pct"] is not None and fast["gain"] is None  # shown, never colored as a gain
    assert rows["r2-kvroute-1-c64"]["decision"] == "accept"


def test_the_demo_replaces_only_a_demo_home(tmp_path):
    """A home of real projects given by mistake is left alone; a demo home made before is made again."""
    demo = load_demo()
    mine = tmp_path / "mine"
    (mine / "projects/real").mkdir(parents=True)
    with pytest.raises(SystemExit):
        demo.main(mine)
    assert (mine / "projects/real").is_dir()
    home = demo.main(tmp_path / "demo").parent.parent
    (home / "projects/glm52-agentx/tasks/develop").mkdir(parents=True, exist_ok=True)
    (home / "projects/glm52-agentx/tasks/develop/mine.md").write_text("x")
    demo.main(home)
    assert not (home / "projects/glm52-agentx/tasks/develop/mine.md").exists()
