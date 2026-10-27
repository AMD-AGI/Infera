import json
import os
import time

from conftest import BLOCKED, DONE, RUNNING, manage

from agent_sys_v2.model import local_time_iso, new_state, write_state
from agent_sys_v2.ops import update_task
from agent_sys_v2.scan import Watcher, load_project, project_ids


def edges_of(project):
    return {(e.up, e.down): e for e in project.edges}


# ---------------------------------------------------------------- the fixture


def test_scan_fixture(home, proj):
    assert project_ids(home) == ["llama70b"]
    p = load_project(proj)
    assert [t.id for t in p.types] == ["baseline", "experiment", "intake", "todo"]
    assert "notes" not in {t.type for t in p.tasks.values()}  # tasks/notes.md is not a task
    assert {tid: (t.status, t.detail) for tid, t in p.tasks.items()} == {
        # frontmatter, 已完成, review: true, unmanaged
        "baseline/c64": ("in_review", {"kind": "review"}),
        "experiment/fp8_kv": ("in_review", {"kind": "review"}),
        "experiment/tp4": ("running", {"kind": "unmanaged"}),
        "experiment/combined": ("draft", {}),
        "baseline/profile": ("needs_you", {"kind": "unmanaged"}),
        "intake/goal": ("done", {}),  # no frontmatter
        "todo/spec_decode": ("draft", {}),
    }
    assert not any(t.managed for t in p.tasks.values())

    base = p.tasks["baseline/c64"]
    assert base.title == "基线：C64 吞吐" and base.review and base.report == "report.md"
    assert base.finished == local_time_iso("2026-10-21 17:04") and base.started and base.created
    assert p.tasks["intake/goal"].title == "goal"
    assert p.tasks["experiment/tp4"].agent == {
        "model": "opus",
        "effort": "high",
        "ultracode": False,
        "budget_usd": 200,
        "extra_args": [],
    }

    e = edges_of(p)
    assert set(e) == {
        ("baseline/c64", "experiment/fp8_kv"),
        ("baseline/c64", "experiment/tp4"),
        ("experiment/fp8_kv", "experiment/combined"),
        ("experiment/tp4", "experiment/combined"),
    }
    # depends_on keeps the prompt line as evidence; Chinese right before the path does not hide it (ASCII names)
    fp8 = e["baseline/c64", "experiment/fp8_kv"]
    assert (fp8.kind, fp8.source) == ("input", "frontmatter")
    assert fp8.evidence == "先读workspaces/baseline/c64/report.md 的结论。"
    # the experiments wait for the baseline, which is unmanaged and waits for 审核通过
    assert p.waiting_on("experiment/tp4") == ["baseline/c64"]
    assert sorted(p.waiting_on("experiment/combined")) == ["experiment/fp8_kv", "experiment/tp4"]
    assert p.tasks["intake/goal"].warnings == ["已完成，但工作区中没有 report.md"]

    data = p.to_json()  # paths relative to AGENT_SYS_HOME
    assert (
        data["path"] == "projects/llama70b"
        and data["counts"]["in_review"] == 2
        and data["running"] == 0
    )
    detail = base.detail_json()
    assert detail["prompt"]["path"] == "projects/llama70b/tasks/baseline/c64.md"
    assert detail["workspace"] == "projects/llama70b/workspaces/baseline/c64"
    json.dumps(data, ensure_ascii=False)
    json.dumps([t.detail_json() for t in p.tasks.values()], ensure_ascii=False)


# ---------------------------------------------------------------- dependencies


def test_prompt_paths_parent_and_depends_on(make_project):
    pdir = make_project(
        {
            "develop/a": "## 任务描述\n",
            "develop/b": "## 任务描述\n",
            "develop/c": "- 读 `workspaces/develop/a/report.md` 的结论\n- 也看 `tasks/develop/b.md`\n"
            "- 别的项目的 projects/other/tasks/develop/a.md 不算\n- tasks/develop/missing.md 不存在\n",
            "develop/d": "---\nparent: develop/b\ndepends_on:\n  - {task: develop/a, kind: code, label: 基于 a 的分支}\n"
            "  - develop/missing\n---\n参考 projects/p/tasks/develop/a.md 和 tasks/develop/b.md\n",
            "review/r": "---\ndepends_on: {task: develop/d, kind: review}\n---\n",
        }
    )
    p = load_project(pdir)
    e = edges_of(p)
    assert set(e) == {
        ("develop/a", "develop/c"),
        ("develop/b", "develop/c"),
        ("develop/a", "develop/d"),
        ("develop/b", "develop/d"),
        ("develop/d", "review/r"),
    }
    assert (e["develop/a", "develop/c"].kind, e["develop/a", "develop/c"].source) == (
        "input",
        "prompt",
    )
    assert e["develop/a", "develop/c"].evidence == "读 `workspaces/develop/a/report.md` 的结论"
    # a parent is only the frontmatter's
    assert e["develop/b", "develop/c"].kind == "input" and p.tasks["develop/c"].parent is None
    ad = e["develop/a", "develop/d"]  # depends_on overrides the prompt edge, keeps its evidence
    assert (ad.kind, ad.label, ad.source) == (
        "code",
        "基于 a 的分支",
        "frontmatter",
    ) and "projects/p/" in ad.evidence
    assert (
        e["develop/b", "develop/d"].kind == "subtask" and not e["develop/b", "develop/d"].blocking
    )
    assert e["develop/d", "review/r"].kind == "review"
    assert p.tasks["develop/d"].warnings == ["depends_on 中的任务「develop/missing」不存在"]


def test_depends_on_kind_null_is_input(make_project):
    """A kind left empty is input, in the scan as in a write."""
    pdir = make_project(
        {
            "develop/a": "",
            "develop/b": "---\ndepends_on:\n  - {task: develop/a, kind: null, label: x}\n---\n",
        }
    )
    p = load_project(pdir)
    assert [(e.up, e.kind, e.label) for e in p.edges] == [("develop/a", "input", "x")]
    assert p.tasks["develop/b"].warnings == []
    update_task(pdir, "develop/b", {"depends_on": [{"task": "develop/a", "kind": None}]})
    assert load_project(pdir).tasks["develop/b"].fm["depends_on"] == ["develop/a"]


def test_at_mentions_are_dependencies(make_project):
    """@<type>/<name> (what the new-task form writes when you pick a task after @) counts as the task's path."""
    pdir = make_project(
        {
            "develop/a": "",
            "develop/b": "",
            "bug_fix/x": "",
            "develop/c": "参考@develop/a 的结论，以及 (@bug_fix/x)\n- 邮件 me@develop/b、git@github.com:develop/b 不算\n"
            "- @develop/missing 不存在\n",
        }
    )
    e = edges_of(load_project(pdir))
    assert set(e) == {("develop/a", "develop/c"), ("bug_fix/x", "develop/c")}
    assert (e["develop/a", "develop/c"].kind, e["develop/a", "develop/c"].source) == (
        "input",
        "prompt",
    )
    assert e["develop/a", "develop/c"].evidence == "参考@develop/a 的结论，以及 (@bug_fix/x)"


def test_odd_frontmatter_values_do_not_break_the_scan(make_project):
    pdir = make_project(
        {
            "develop/a": "",
            "develop/b": "---\ndepends_on: [[develop/a], 3, {kind: code}]\nparent: [x]\n"
            "title: 42\nagent: opus\n---\n",
            "develop/c": "---\ntitle: [unclosed\n---\n正文\n",
        }
    )
    p = load_project(pdir)
    b, c = p.tasks["develop/b"], p.tasks["develop/c"]
    assert p.edges == [] and b.title == "42" and b.parent is None
    assert len(b.warnings) == 4 and b.warnings[-1] == "parent「['x']」看不懂：应写成 类型/任务名"
    assert (
        c.status == "draft"
        and c.warnings[0].startswith("frontmatter 无法解析")
        and "正文" in c.body
    )


def test_more_odd_values(make_project):
    pdir = make_project(
        {
            "develop/a": "",
            "develop/b": "---\ndepends_on: 5\n---\n",
            "develop/c": "---\ndepends_on: 2026-10-22\nparent: {task: develop/a}\n---\n",
            "develop/d": "见 subtasks/develop/a 和 mytasks/develop/a\n",
        }
    )
    (pdir / "workspaces/develop/a/.agent_sys").mkdir(parents=True)
    (pdir / "workspaces/develop/a/.agent_sys/state.json").write_text("[]")
    p = load_project(pdir)
    assert p.edges == [] and p.tasks["develop/a"].status == "unknown"
    assert "看不懂" in p.tasks["develop/b"].warnings[0] and len(p.tasks["develop/c"].warnings) == 2
    assert p.tasks["develop/c"].parent is None
    from agent_sys_v2.scan import load_task

    # vanished during a scan
    t = load_task(pdir / "tasks/develop/gone.md", "p", "develop", "gone", {})
    assert t.status == "draft" and t.updated == ""


def test_empty_status_line_does_not_read_the_next_line(make_project):
    pdir = make_project({"develop/a": ""}, progress={"develop/a": "状态:\n下一步: 跑 C64\n"})
    t = load_project(pdir).tasks["develop/a"]
    assert (t.progress.status, t.progress.next, t.status) == ("", "跑 C64", "unknown")
    assert "progress.md 缺少「状态:」行" in t.warnings


def test_unmanaged_dependency_satisfaction(make_project):
    pdir = make_project(
        {
            "develop/done": "",
            "develop/reviewed": "---\nreview: true\n---\n",
            "develop/running": "",
            "develop/parent": "",
            "develop/down": "---\ndepends_on: [develop/done, develop/reviewed, develop/running]\nparent: develop/parent\n---\n",
        },
        progress={
            "develop/done": DONE,
            "develop/reviewed": DONE,
            "develop/running": RUNNING,
            "develop/parent": BLOCKED,
        },
    )
    p = load_project(pdir)
    assert p.tasks["develop/reviewed"].status == "in_review"
    assert p.tasks["develop/parent"].status == "needs_you"
    assert sorted(p.waiting_on("develop/down")) == ["develop/reviewed", "develop/running"]


def test_managed_status_comes_from_state(make_project):
    pdir = make_project({"develop/a": ""}, progress={"develop/a": DONE})
    write_state(
        pdir / "workspaces/develop/a", {**new_state("needs_you", {"kind": "asked"}), "started": "x"}
    )
    t = load_project(pdir).tasks["develop/a"]
    assert t.managed and (t.status, t.detail) == ("needs_you", {"kind": "asked"})
    (pdir / "workspaces/develop/a/.agent_sys/state.json").write_text("{")
    t = load_project(pdir).tasks["develop/a"]
    assert t.status == "unknown" and "state.json 无法解析" in t.warnings[0]


def test_derived_details(make_project):
    # state.json stores only facts; why a queued task waits and the sub-states are worked out on reading
    pdir = make_project(
        {
            "develop/up": "",
            "develop/a": "---\ndepends_on: [develop/up]\n---\n",
            "develop/b": "",
            "develop/c": "",
            "develop/r": "",
        },
        progress={"develop/up": RUNNING},
    )
    manage(pdir, "develop/a", "develop/b")
    manage(pdir, "develop/c", status="in_review")
    manage(pdir, "develop/r", status="running")
    d = {tid: t.detail for tid, t in load_project(pdir).tasks.items()}
    assert d["develop/a"] == {"kind": "upstream", "tasks": ["develop/up"]}
    assert d["develop/b"] == {}  # waits for nothing here; machines and slots are sched.plan's
    assert d["develop/c"] == {"kind": "review"} and d["develop/r"] == {"kind": "working"}
    (pdir / "project.yaml").write_text("title: t\nautorun: false\n")
    assert load_project(pdir).tasks["develop/b"].detail == {"kind": "manual"}
    (pdir / "project.yaml").write_text('title: t\narchived_at: "2026-10-22T00:00:00Z"\n')
    assert load_project(pdir).tasks["develop/a"].detail == {"kind": "archived"}


def test_unread(make_project):
    pdir = make_project({"develop/a": ""})
    ws = pdir / "workspaces/develop/a"
    write_state(
        ws, new_state("done", finished="2026-10-22T04:00:00Z", seen_at="2026-10-22T03:00:00Z")
    )
    assert load_project(pdir).tasks["develop/a"].unread
    write_state(
        ws, new_state("done", finished="2026-10-22T04:00:00Z", seen_at="2026-10-22T05:00:00Z")
    )
    assert not load_project(pdir).tasks["develop/a"].unread


# ---------------------------------------------------------------- record reminders


def test_lint(make_project):
    pdir = make_project(
        {
            "develop/nohead": "",
            "develop/nofinish": "",
            "develop/stale": "",
            "develop/noreport": "",
            "develop/otherreport": "",
            "develop/parent": "",
            "develop/child": "---\nparent: develop/parent\n---\n",
            "develop/x": "---\ndepends_on: [develop/y]\n---\n",
            "develop/y": "---\ndepends_on: [develop/x]\n---\n",
        },
        progress={
            "develop/nohead": "# 进展\n",
            "develop/nofinish": "状态: 已完成\n- [ ] 收尾\n",
            "develop/stale": RUNNING,
            "develop/noreport": DONE,
            "develop/otherreport": DONE,
            "develop/parent": RUNNING,
            "develop/child": RUNNING,
        },
    )
    ws = pdir / "workspaces/develop"
    (ws / "otherreport/pr-198-report.md").write_text("## 结论\n")
    old = time.time() - 30 * 3600
    os.utime(ws / "stale/progress.md", (old, old))
    os.utime(ws / "parent/progress.md", (old, old))
    p = load_project(pdir)
    w = {tid: t.warnings for tid, t in p.tasks.items()}
    h = {tid: t.hints for tid, t in p.tasks.items()}
    assert w["develop/nohead"] == ["progress.md 缺少「状态:」行"]
    assert w["develop/nofinish"] == [
        "已完成，但 progress.md 缺少「完成时间:」行",
        "已完成，但工作区中没有 report.md",
    ]
    assert h["develop/nofinish"] == ["已完成，但 progress.md 仍有 1 个未勾选项"]
    assert w["develop/stale"] == ["progress.md 已 30 小时未更新"]
    assert w["develop/noreport"] == ["已完成，但工作区中没有 report.md"]
    assert w["develop/otherreport"] == [] and h["develop/otherreport"] == [
        "没有 report.md，报告为 pr-198-report.md",
        '报告中没有交接清单（"## 交接清单"一节），下游任务只能自己翻工作区',
    ]
    assert p.tasks["develop/otherreport"].report == "pr-198-report.md"
    assert "子任务「child」的进展晚于本任务 progress.md" in w["develop/parent"][-1]
    assert "progress.md 已 30 小时未更新" in w["develop/parent"][0]
    assert all("形成了循环" in w[t][0] for t in ("develop/x", "develop/y"))


def test_handoff_list(make_project):
    pdir = make_project(
        {"develop/a": "", "develop/b": "", "develop/c": ""},
        progress={"develop/a": DONE, "develop/b": DONE, "develop/c": RUNNING},
    )
    ws = pdir / "workspaces/develop"
    (ws / "a/log").mkdir()
    (ws / "a/log/run.log").write_text("x")
    (pdir / "repos").mkdir()
    (ws / "a/report.md").write_text(
        "## 结论\n\n好了\n\n## 交接清单\n\n- `log/run.log` — 测试日志\n- `log/` — 所有日志\n"
        "* `repos` — 代码仓库（相对项目目录）\n- `shots/` — 截图（没有生成）\n- 分支 `dev/x`，PR #5\n"
        "- `https://github.com/x/y/pull/5` — PR\n- `~nobody_here_xyz/x` — 不存在的用户\n"
        f"- `{'x' * 300}` — 名字太长\n- `uv run pytest -q` — 命令不是路径\n\n## 替你做的决定\n\n- 不算\n",
        encoding="utf-8",
    )
    (ws / "b/report.md").write_text(
        "## Conclusion\n\nok\n\n## Handoff\n\n- `/nonexistent/x` — gone\n"
    )
    (ws / "c/report.md").write_text("## 结论\n\n## 交接清单\n- `nope` — 还没做完\n")
    p = load_project(pdir)
    a = p.tasks["develop/a"]
    assert [(h["path"], h["exists"], h["kind"], h["rel"]) for h in a.handoff] == [
        ("log/run.log", True, "file", "log/run.log"),
        ("log/", True, "dir", "log"),
        ("repos", True, "dir", ""),
        ("shots/", False, None, ""),
        ("", False, None, ""),
        ("", False, None, ""),
        ("~nobody_here_xyz/x", False, None, ""),
        ("x" * 300, False, None, ""),
        ("", False, None, ""),
    ]
    assert a.handoff[4]["text"] == "分支 `dev/x`，PR #5"
    assert (
        a.warnings == ["交接清单中的路径不存在：shots/、~nobody_here_xyz/x、" + "x" * 300]
        and a.hints == []
    )
    assert p.tasks["develop/b"].warnings == ["交接清单中的路径不存在：/nonexistent/x"]
    # not done: not checked
    assert p.tasks["develop/c"].handoff is None and p.tasks["develop/c"].warnings == []
    assert a.detail_json()["handoff"] == a.handoff


def test_long_finished_tasks_get_hints_not_reminders(make_project):
    pdir = make_project(
        {"develop/old": "", "develop/new": ""},
        progress={"develop/old": "状态: 已完成\n", "develop/new": "状态: 已完成\n"},
    )
    old = time.time() - 8 * 86400
    os.utime(pdir / "workspaces/develop/old/progress.md", (old, old))
    p = load_project(pdir)
    expected = ["已完成，但 progress.md 缺少「完成时间:」行", "已完成，但工作区中没有 report.md"]
    assert (p.tasks["develop/old"].warnings, p.tasks["develop/old"].hints) == ([], expected)
    assert (p.tasks["develop/new"].warnings, p.tasks["develop/new"].hints) == (expected, [])


# ---------------------------------------------------------------- change detection


def test_watcher(home, proj):
    w = Watcher(home)
    assert w.poll() == []

    def bump(path):
        t = time.time() + 5
        os.utime(path, (t, t))

    bump(proj / "workspaces/baseline/c64/progress.md")
    assert w.poll() == [("llama70b", "baseline/c64")]
    (proj / "tasks/experiment/new.md").write_text("## 任务描述\n")
    assert w.poll() == [("llama70b", None), ("llama70b", "experiment/new")]
    write_state(proj / "workspaces/experiment/new", new_state("queued"))
    assert w.poll() == [("llama70b", "experiment/new")]
    (proj / "workspaces/experiment/new/results").mkdir()
    assert w.poll() == [("llama70b", "experiment/new")]
    (proj / "workspaces/experiment/new/results/c64.json").write_text("{}")
    # a later second: NFS keeps a directory's mtime in seconds
    bump(proj / "workspaces/experiment/new/results")
    assert w.poll() == [("llama70b", "experiment/new")]  # a result file added: the ledger changed
    # rewritten in place (ab_judge --write): the ledger changed too
    (proj / "workspaces/experiment/new/results/c64.json").write_text('{"verdict": {}}')
    assert w.poll() == [("llama70b", "experiment/new")]
    (proj / "workspaces/experiment/new/results/r1").mkdir()
    assert w.poll() == [("llama70b", "experiment/new")]
    # raw data one level down is no result: a benchmark writing there rescans nothing
    (proj / "workspaces/experiment/new/results/r1/c96.json").write_text("{}")
    bump(proj / "workspaces/experiment/new/results/r1")
    assert w.poll() == []
    (proj / "workspaces/experiment/tp4/report.md").write_text("## 结论\n")
    assert w.poll() == [("llama70b", "experiment/tp4")]
    (home / "agent_sys.yaml").write_text("max_running: 2\n")  # AGENT_SYS_HOME's own settings
    assert w.poll() == [("", None)]
    bump(proj / "project.yaml")
    assert w.poll() == [("llama70b", None)]
    (proj / "task_types").mkdir()
    (proj / "task_types/experiment.md").write_text("## 任务描述\n")
    assert w.poll() == [("llama70b", None)]
    (proj / "task_types/experiment.md").write_text("---\nreview: true\n---\n## 任务描述\n")
    assert w.poll() == [("llama70b", None)]  # a skeleton edited in place
    (proj / "tasks/experiment/new.md").unlink()
    assert w.poll() == [("llama70b", None), ("llama70b", "experiment/new")]
    other = home / "projects/other"
    (other / "tasks/develop").mkdir(parents=True)
    (other / "tasks/develop/a.md").write_text("")
    assert w.poll() == []  # no project.yaml yet: not a project
    (other / "project.yaml").write_text("title: other\n")
    assert w.poll() == [("other", None), ("other", "develop/a")]
    # files deeper in a workspace are not watched (results/ is, for the ledger)
    (proj / "workspaces/baseline/c64/scratch").mkdir()
    assert w.poll() == [("llama70b", "baseline/c64")]
    (proj / "workspaces/baseline/c64/scratch/x.log").write_text("x")
    assert w.poll() == []


def test_project_yaml_values_are_checked(make_project):
    import pytest

    from agent_sys_v2.model import AgentSysError

    pdir = make_project({"develop/a": ""}, config="title: x\nagent: opus\n")
    with pytest.raises(AgentSysError, match="agent 应为键值表"):
        load_project(pdir)
    # YAML 1.2: the string "off"
    (pdir / "project.yaml").write_text("title: x\nautorun: off\n", encoding="utf-8")
    with pytest.raises(AgentSysError, match="autorun 应为 true 或 false"):
        load_project(pdir)


def test_a_name_with_a_dot_is_reported(make_project):
    p = load_project(make_project({"develop/release_v1.0": "", "develop/ok": ""}))
    assert any("名称只能包含" in w for w in p.tasks["develop/release_v1.0"].warnings)
    assert not p.tasks["develop/ok"].warnings


def test_a_task_that_cannot_be_read_says_why(make_project):
    """Checks or a frontmatter that cannot be read: the task says why (invalid), so that it is not run without them."""
    pdir = make_project(
        {
            "develop/a": '---\nchecks: [{run: "true", timeout: "60"}]\n---\n',
            "develop/b": "---\ntitle: [unclosed\n---\n",
            "develop/c": '---\nchecks: ["test -s report.md"]\n---\n',
        }
    )
    tasks = load_project(pdir).tasks
    a, b, c = (tasks[f"develop/{n}"] for n in "abc")
    assert "timeout" in a.invalid and a.warnings == [a.invalid]
    assert b.invalid.startswith("frontmatter 无法解析") and c.invalid == ""


def test_a_finish_time_that_is_no_time_does_not_fail_the_scan(make_project):
    pdir = make_project(
        {"develop/a": "", "develop/b": ""},
        progress={"develop/a": "状态: 已完成\n完成时间: 2026-09-31 10:00\n", "develop/b": DONE},
    )
    a = load_project(pdir).tasks["develop/a"]  # a ValueError failed the whole project
    assert a.status == "done" and a.finished  # the file's mtime instead
    assert "不是 YYYY-MM-DD HH:MM 格式的时间" in a.warnings[0]


def test_stale_progress_is_for_unmanaged_tasks(make_project):
    """A managed task may wait in the background for hours (its run says so): no reminder for it."""
    pdir = make_project(
        {"develop/m": "", "develop/u": ""}, progress={"develop/m": RUNNING, "develop/u": RUNNING}
    )
    write_state(pdir / "workspaces/develop/m", new_state("running"))
    old = time.time() - 30 * 3600
    for tid in ("m", "u"):
        os.utime(pdir / f"workspaces/develop/{tid}/progress.md", (old, old))
    p = load_project(pdir)
    assert p.tasks["develop/m"].warnings == []
    assert p.tasks["develop/u"].warnings == ["progress.md 已 30 小时未更新"]


def test_children_edges_count_what_they_wait_for(make_project):
    """The board's count of a children edge follows pending_children: the waiter's own downstream is left out."""
    pdir = make_project(
        {
            "plan/r1": "",
            "close/r1": "---\ndepends_on: [{task: plan/r1, kind: children}]\n---\n",
            "experiment/a": "---\nparent: plan/r1\n---\n",
            "experiment/after": "---\nparent: plan/r1\ndepends_on: [close/r1]\n---\n",
        },
        progress={"plan/r1": DONE, "experiment/a": DONE},
    )
    p = load_project(pdir)
    e = next(e for e in p.edges if e.kind == "children")
    assert p.edge_json(e)["children"] == {"done": 1, "total": 1} and p.edge_satisfied(e)
    assert p.edge_json(e)["blocking"] and not e.to_json().get("children")
