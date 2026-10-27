import os
import re
import time

import pytest
from conftest import BLOCKED, DONE, RUNNING

from agent_sys_v2 import ops
from agent_sys_v2.model import (
    AgentSysError,
    iso,
    mtime,
    new_state,
    read_state,
    split_frontmatter,
    write_state,
)
from agent_sys_v2.scan import load_project


def fm_of(pdir, tid):
    return split_frontmatter((pdir / "tasks" / f"{tid}.md").read_text(encoding="utf-8"))[0]


def text_of(pdir, tid):
    """Task file text; the gap before a comment is squeezed (ruamel keeps a comment's column, not its gap)."""
    return re.sub(r" +#", " #", (pdir / "tasks" / f"{tid}.md").read_text(encoding="utf-8"))


def task(pdir, tid):
    """A task as a fresh scan reads it: a write returns no scan."""
    return load_project(pdir).tasks[tid]


def refused(status, fn, *args, **kw):
    with pytest.raises(AgentSysError) as e:
        fn(*args, **kw)
    assert e.value.status == status
    return str(e.value)


# ---------------------------------------------------------------- types and tasks


def test_new_type(make_project):
    pdir = make_project({"develop/a": ""})
    ops.new_type(pdir, "codereview", review=True)
    ty = next(ty for ty in load_project(pdir).types if ty.id == "codereview")
    assert (ty.id, ty.review, ty.count) == ("codereview", True, 0) and "## 任务描述" in ty.skeleton
    assert (pdir / "tasks/codereview").is_dir()
    refused(409, ops.new_type, pdir, "codereview")
    refused(400, ops.new_type, pdir, "a/b")


def test_new_task_queue_and_draft(make_project):
    pdir = make_project({"develop/a": ""}, progress={"develop/a": DONE})
    ops.new_task(
        pdir,
        "develop",
        "b",
        "## 任务描述\n\n- 读 workspaces/develop/a/report.md\n",
        title="B",
        depends_on=[{"task": "develop/a", "kind": "code", "label": "基于 a"}],
        review=True,
        agent={"budget_usd": 150},
    )
    t = task(pdir, "develop/b")
    assert t.managed and t.status == "queued" and t.review
    # project defaults
    assert t.agent == {
        "model": "opus",
        "effort": "high",
        "ultracode": False,
        "budget_usd": 150,
        "extra_args": [],
    }
    state = read_state(t.workspace)
    assert state["session_id"] and state["runs"] == 0 and state["created"]
    text = t.path.read_text(encoding="utf-8")
    assert text == (
        "---\ntitle: B\ndepends_on:\n  - {task: develop/a, kind: code, label: 基于 a}\nreview: true\n"
        "agent: {budget_usd: 150}\n---\n## 任务描述\n\n- 读 workspaces/develop/a/report.md\n"
    )

    ops.new_task(pdir, "develop", "c", "草稿\n", queue=False)
    d = task(pdir, "develop/c")
    assert not d.managed and d.status == "draft" and not d.workspace.exists()
    assert d.path.read_text(encoding="utf-8") == "草稿\n"

    refused(409, ops.new_task, pdir, "develop", "c", "")
    refused(404, ops.new_task, pdir, "nosuchtype", "x", "")
    refused(404, ops.new_task, pdir, "develop", "x", "", depends_on=["develop/missing"])
    refused(400, ops.new_task, pdir, "develop", "bad name", "")
    refused(400, ops.new_task, pdir, "develop", "中文任务", "")
    refused(400, ops.new_task, pdir, "develop", "restest_c96.v2", "")


def test_new_task_cycle_through_existing_prompt_path(make_project):
    # a already names tasks/develop/b.md in its prompt; creating b depending on a would close a cycle
    pdir = make_project({"develop/a": "按 tasks/develop/b.md 的结果继续\n"})
    msg = refused(409, ops.new_task, pdir, "develop", "b", "", depends_on=["develop/a"])
    assert "循环依赖" in msg and not (pdir / "tasks/develop/b.md").exists()


def test_archived_project_refuses_new_tasks_and_join(make_project):
    pdir = make_project({"develop/a": ""}, config='title: t\narchived_at: "2026-10-22T00:00:00Z"\n')
    refused(409, ops.new_task, pdir, "develop", "b", "")
    refused(409, ops.join, pdir, "develop/a")


# ---------------------------------------------------------------- edges


# ---------------------------------------------------------------- 修改任务配置


def test_update_task(make_project):
    pdir = make_project(
        {
            "develop/a": "",
            "develop/b": "---\ntitle: B   # 标题\nreview: true\nowner: x\n---\n正文\n",
        }
    )
    path = pdir / "tasks/develop/b.md"
    os.utime(path, (time.time() - 60,) * 2)  # mtimes compare to the second, as the API shows them
    base = iso(mtime(path))
    ops.update_task(
        pdir,
        "develop/b",
        {"title": "新标题", "review": None, "summary": "一句话", "agent": {"budget_usd": 300}},
        "新正文\n",
        base,
    )
    t = task(pdir, "develop/b")
    assert (t.title, t.review, t.summary, t.agent["budget_usd"], t.body) == (
        "新标题",
        False,
        "一句话",
        300,
        "新正文\n",
    )
    assert text_of(pdir, "develop/b") == (
        "---\ntitle: 新标题 # 标题\nowner: x\nsummary: 一句话\n"
        "agent: {budget_usd: 300}\n---\n新正文\n"
    )
    ops.update_task(pdir, "develop/b", {"agent": {}, "summary": ""})
    assert dict(fm_of(pdir, "develop/b")) == {"title": "新标题", "owner": "x"}
    # the file changed after the editor was opened: refused, nothing written
    assert "已被修改" in refused(
        409, ops.update_task, pdir, "develop/b", {"title": "x"}, None, base
    )
    assert fm_of(pdir, "develop/b")["title"] == "新标题"
    # a prompt that names a task depending on this one closes a cycle
    ops.update_task(pdir, "develop/a", None, "用 workspaces/develop/b/report.md\n")
    assert "循环依赖" in refused(
        409, ops.update_task, pdir, "develop/b", None, "读 tasks/develop/a.md\n"
    )
    assert text_of(pdir, "develop/b").endswith("---\n新正文\n")
    refused(404, ops.update_task, pdir, "develop/x", {"title": "x"})


# ---------------------------------------------------------------- 加入 / 审核通过


def test_join_three_cases(make_project):
    pdir = make_project(
        {
            "develop/draft": "",
            "develop/done": "",
            "develop/reviewed": "---\nreview: true\n---\n",
            "develop/running": "",
            "develop/blocked": "",
            "develop/paused": "",
        },
        progress={
            "develop/done": DONE,
            "develop/reviewed": DONE,
            "develop/running": RUNNING,
            "develop/blocked": BLOCKED,
            "develop/paused": "状态: 暂停\n",
        },
    )
    for tid in load_project(pdir).tasks:
        ops.join(pdir, tid)
    joined = load_project(pdir).tasks
    st = {tid: (t.status, t.detail) for tid, t in joined.items()}
    assert st["develop/draft"] == ("queued", {})
    assert st["develop/done"] == ("done", {})  # a finished task never goes to 待开始
    assert st["develop/reviewed"] == ("in_review", {"kind": "review"})
    for tid in ("develop/running", "develop/blocked", "develop/paused"):
        assert st[tid] == ("needs_you", {"kind": "joined"})
    assert all(t.managed and t.state["session_id"] for t in joined.values())
    assert not joined["develop/done"].unread and joined["develop/done"].finished
    assert joined["develop/running"].started  # when work began in the terminal
    refused(409, ops.join, pdir, "develop/draft")


def test_join_fixture_baseline(proj):
    # the baseline is done, review: true and unmanaged: joining it must not rerun it
    ops.join(proj, "baseline/c64")
    t = task(proj, "baseline/c64")
    assert (t.status, t.managed) == ("in_review", True)
    ops.approve(proj, "baseline/c64")
    t = task(proj, "baseline/c64")
    assert t.status == "done" and not t.unread
    p = load_project(proj)
    assert p.waiting_on("experiment/tp4") == []


def test_approve(make_project):
    pdir = make_project(
        {
            "develop/up": "---\nreview: true\n---\n",
            "develop/down": "---\ndepends_on: [develop/up]\n---\n",
            "develop/running": "",
        },
        progress={"develop/up": DONE, "develop/running": RUNNING},
    )
    assert load_project(pdir).waiting_on("develop/down") == ["develop/up"]
    # unmanaged 待审核 -> 已完成, still unmanaged: approved_at in frontmatter
    ops.approve(pdir, "develop/up")
    t = task(pdir, "develop/up")
    assert (t.status, t.managed) == ("done", False) and fm_of(pdir, "develop/up")[
        "approved_at"
    ].endswith("Z")
    assert load_project(pdir).waiting_on("develop/down") == []
    assert "「已完成」" in refused(409, ops.approve, pdir, "develop/up")
    refused(409, ops.approve, pdir, "develop/running")
    # a 完成时间 later than now (another time zone, a guess): approving it still approves it
    later = pdir / "workspaces/develop/later/progress.md"
    later.parent.mkdir(parents=True)
    later.write_text("状态: 已完成\n下一步: 无\n完成时间: 2099-01-01 08:00\n", encoding="utf-8")
    (pdir / "tasks/develop/later.md").write_text("---\nreview: true\n---\n", encoding="utf-8")
    ops.approve(pdir, "develop/later")
    assert task(pdir, "develop/later").status == "done"
    # you went on in the terminal and it finished again after the approval: back to 待审核
    # DONE finished 10-21 17:10
    ops.update_task(pdir, "develop/up", {"approved_at": "2026-10-21T00:00:00Z"})
    assert load_project(pdir).tasks["develop/up"].status == "in_review"
    # a managed task
    ops.join(pdir, "develop/up")
    ops.approve(pdir, "develop/up")
    t = task(pdir, "develop/up")
    assert (t.status, t.managed, t.unread) == ("done", True, False)


def test_mark_read(make_project):
    pdir = make_project(
        {"develop/d": "", "develop/e": "", "develop/u": ""}, progress={"develop/u": RUNNING}
    )
    for tid in ("d", "e"):
        write_state(
            pdir / f"workspaces/develop/{tid}", new_state("done", finished="2026-10-22T04:00:00Z")
        )
    assert load_project(pdir).tasks["develop/d"].unread
    assert ops.mark_read(pdir, "develop/d") == ["develop/d"]
    assert not load_project(pdir).tasks["develop/d"].unread
    assert ops.mark_read(pdir, "develop/u") == []  # nothing to do
    assert ops.mark_read(pdir, "develop/d", "develop/e") == ["develop/e"]  # several at once


def test_lane_order(make_project):
    pdir = make_project({"develop/a": ""}, types=["bug_fix", "perf"], config="title: t   # 标题\n")
    assert [ty.id for ty in load_project(pdir).types] == ["bug_fix", "develop", "perf"]  # by name
    ops.set_lanes(pdir, ["perf", "develop", "gone", "perf"])
    # the rest after, by name
    assert [ty.id for ty in load_project(pdir).types] == ["perf", "develop", "bug_fix"]
    assert (pdir / "project.yaml").read_text() == "title: t   # 标题\nlanes: [perf, develop]\n"
    ops.set_lanes(pdir, [])
    assert [ty.id for ty in load_project(pdir).types] == ["bug_fix", "develop", "perf"]
    assert "lanes" not in (pdir / "project.yaml").read_text()
    refused(400, ops.set_lanes, pdir, "perf")


def test_update_depends_on(make_project):
    pdir = make_project(
        {
            "develop/a": "",
            "develop/b": "",
            "develop/c": "读 tasks/develop/a.md\n",
            "develop/x": "---\ndepends_on:\n  - develop/gone   # 已删除的任务\n---\n",
        }
    )
    ops.update_task(
        pdir,
        "develop/c",
        {
            "depends_on": [
                "develop/b",
                {"task": "develop/a", "kind": "code", "label": "L"},
                {"task": "develop/b", "kind": "review"},
            ]
        },
    )
    assert (
        text_of(pdir, "develop/c")
        == "---\ndepends_on:\n  - develop/b\n  - {task: develop/a, kind: code, label: L}\n---\n读 tasks/develop/a.md\n"
    )
    assert {(e.up, e.kind) for e in load_project(pdir).upstream("develop/c")} == {
        ("develop/a", "code"),
        ("develop/b", "input"),
    }
    # back to the prompt's own edge
    ops.update_task(pdir, "develop/c", {"depends_on": []})
    assert fm_of(pdir, "develop/c") == {}
    assert [(e.up, e.kind, e.source) for e in load_project(pdir).upstream("develop/c")] == [
        ("develop/a", "input", "prompt")
    ]
    # an entry already written may name a task that is gone; a new one may not
    ops.update_task(pdir, "develop/x", {"depends_on": ["develop/gone", "develop/a"]})
    assert fm_of(pdir, "develop/x")["depends_on"] == ["develop/gone", "develop/a"]
    refused(404, ops.update_task, pdir, "develop/x", {"depends_on": ["develop/missing"]})
    refused(400, ops.update_task, pdir, "develop/x", {"depends_on": ["develop/x"]})
    refused(
        400,
        ops.update_task,
        pdir,
        "develop/x",
        {"depends_on": [{"task": "develop/a", "kind": "blocks"}]},
    )
    refused(400, ops.update_task, pdir, "develop/x", {"depends_on": [{"kind": "code"}]})
    # a cycle is refused and nothing is written
    ops.update_task(pdir, "develop/b", {"depends_on": ["develop/c"]})
    assert "循环依赖" in refused(
        409, ops.update_task, pdir, "develop/c", {"depends_on": ["develop/b"]}
    )
    assert fm_of(pdir, "develop/c") == {}


# ---------------------------------------------------------------- rename and delete


def test_rename_draft_rewrites_references(make_project):
    pdir = make_project(
        {
            "develop/draft": "",
            "develop/x": "---\ndepends_on:\n  - develop/draft   # 注释留着\n  - {task: develop/draft2, kind: code}\n---\n",
            "develop/y": "---\nparent: develop/draft\ndepends_on: [{task: develop/draft, kind: code, label: L}]\n---\n",
            "develop/draft2": "",
            "develop/started": "",
        },
        progress={"develop/started": RUNNING},
        types=["todo"],
    )
    assert ops.rename(pdir, "develop/draft", "todo", "renamed") == "todo/renamed"
    assert not (pdir / "tasks/develop/draft.md").exists()
    assert (
        text_of(pdir, "develop/x")
        == "---\ndepends_on:\n  - todo/renamed # 注释留着\n  - {task: develop/draft2, kind: code}\n---\n"
    )
    assert dict(fm_of(pdir, "develop/y")) == {
        "parent": "todo/renamed",
        "depends_on": [{"task": "todo/renamed", "kind": "code", "label": "L"}],
    }
    edges = {(e.up, e.down) for e in load_project(pdir).edges}
    assert {("todo/renamed", "develop/x"), ("todo/renamed", "develop/y")} <= edges

    # progress 进行中
    assert "正在运行" in refused(409, ops.rename, pdir, "develop/started", "develop", "z")
    refused(409, ops.rename, pdir, "develop/x", "develop", "y")  # target exists
    refused(404, ops.rename, pdir, "develop/x", "nosuchtype", "z")
    refused(400, ops.rename, pdir, "develop/x", "develop", "a b")
    assert ops.rename(pdir, "develop/x", "develop", "x") == "develop/x"  # nothing to do


def test_rename_a_managed_task_moves_its_workspace_and_prompt_paths(make_project):
    # agent_sys_hang_fix -> bug_fix: the task has run; others name it in their prompt and frontmatter
    pdir = make_project(
        {
            "develop/fix": "读 `workspaces/develop/fix/progress.md`，见 tasks/develop/fix.md；tasks/develop/fix_more 不是它\n",
            "develop/next": "---\ndepends_on: [{task: develop/fix, kind: code}]\n---\n基于 workspaces/develop/fix/ 的分支\n",
            "develop/doc": "参考 projects/p/workspaces/develop/fix/report.md 和 projects/other/tasks/develop/fix.md\n",
            "develop/fix_more": "",
            "develop/ref": "看 @develop/fix 的结论，不是 @develop/fix_more 或 me@develop/fix\n",
        },
        progress={"develop/fix": DONE},
        types=["bug_fix"],
    )
    state = {**new_state("done", finished="2026-10-22T04:00:00Z"), "runs": 2}
    write_state(pdir / "workspaces/develop/fix", state)
    (pdir / "workspaces/develop/fix/.agent_sys/runs/20261022-040000").mkdir(parents=True)
    before = {(e.up, e.down, e.kind) for e in load_project(pdir).edges}

    t = task(pdir, ops.rename(pdir, "develop/fix", "bug_fix", "fix"))
    assert t.id == "bug_fix/fix" and t.managed and t.status == "done" and t.report is None
    assert (
        not (pdir / "workspaces/develop/fix").exists()
        and (t.workspace / ".agent_sys/runs/20261022-040000").is_dir()
    )
    new = read_state(t.workspace)
    assert new["session_id"] == state["session_id"] and new["moved_from"] == "develop/fix"
    assert (
        t.body
        == "读 `workspaces/bug_fix/fix/progress.md`，见 tasks/bug_fix/fix.md；tasks/develop/fix_more 不是它\n"
    )
    assert (
        text_of(pdir, "develop/next")
        == "---\ndepends_on: [{task: bug_fix/fix, kind: code}]\n---\n基于 workspaces/bug_fix/fix/ 的分支\n"
    )
    assert (
        text_of(pdir, "develop/doc")
        == "参考 projects/p/workspaces/bug_fix/fix/report.md 和 projects/other/tasks/develop/fix.md\n"
    )
    assert (
        text_of(pdir, "develop/ref")
        == "看 @bug_fix/fix 的结论，不是 @develop/fix_more 或 me@develop/fix\n"
    )
    rename = lambda tid: tid.replace("develop/fix", "bug_fix/fix") if tid == "develop/fix" else tid
    assert {(e.up, e.down, e.kind) for e in load_project(pdir).edges} == {
        (rename(u), rename(d), k) for u, d, k in before
    }

    # moved again before the session heard of it: it still knows the first id; moved back: nothing to tell
    ops.rename(pdir, "bug_fix/fix", "develop", "fix2")
    assert read_state(pdir / "workspaces/develop/fix2")["moved_from"] == "develop/fix"
    ops.rename(pdir, "develop/fix2", "develop", "fix")
    assert "moved_from" not in read_state(pdir / "workspaces/develop/fix")

    write_state(pdir / "workspaces/develop/fix", {**state, "status": "running"})
    assert "请先停止" in refused(409, ops.rename, pdir, "develop/fix", "bug_fix", "fix")
    write_state(
        pdir / "workspaces/develop/fix",
        {**state, "status": "needs_you", "detail": {"kind": "joined"}},
    )
    assert "终端" in refused(409, ops.rename, pdir, "develop/fix", "bug_fix", "fix")


def test_rename_keeps_a_cycle_it_was_on(make_project):
    pdir = make_project(
        {
            "develop/a": "读 tasks/develop/b.md\n",
            "develop/b": "读 tasks/develop/a.md\n",
            "develop/c": "",
        }
    )
    # a and b already wait for each other: not a new cycle
    assert ops.rename(pdir, "develop/a", "develop", "a2") == "develop/a2"
    assert load_project(pdir).tasks["develop/b"].body == "读 tasks/develop/a2.md\n"


def test_rename_refuses_a_cycle_through_prompt_paths(make_project):
    pdir = make_project(
        {
            "develop/y": "按 workspaces/develop/z/report.md 继续\n",
            "develop/d": "---\ndepends_on: [develop/y]\n---\n",
            "develop/w": "---\ndepends_on: [develop/d]\n---\n",
        }
    )
    assert "循环依赖" in refused(409, ops.rename, pdir, "develop/d", "develop", "z")
    p = load_project(pdir)  # undone
    assert "develop/d" in p.tasks and "develop/z" not in p.tasks
    assert fm_of(pdir, "develop/w")["depends_on"] == ["develop/d"]


def test_write_to_a_vanished_task_file_is_refused(make_project):
    pdir = make_project({"develop/a": "", "develop/b": ""})
    path = pdir / "tasks/develop/b.md"
    path.unlink()
    refused(404, ops.update_frontmatter, path, lambda fm: fm.update(title="x"))
    assert not path.exists()


def test_delete_moves_to_trash(make_project):
    pdir = make_project(
        {
            "develop/a": "",
            "develop/b": "---\ndepends_on: [develop/a]\nparent: develop/a\n---\n",
            "develop/r": "",
        },
        progress={"develop/a": DONE, "develop/r": RUNNING},
    )
    (pdir / "workspaces/develop/a/log").mkdir()
    (pdir / "workspaces/develop/a/log/x.log").write_text("x")
    trash = ops.delete(pdir, "develop/a")
    assert trash.parent == pdir / ".trash"
    assert (trash / "tasks/develop/a.md").is_file() and (
        trash / "workspaces/develop/a/log/x.log"
    ).is_file()
    assert (
        not (pdir / "tasks/develop/a.md").exists() and not (pdir / "workspaces/develop/a").exists()
    )
    p = load_project(pdir)
    # other tasks are not rewritten: moving the files back restores everything; meanwhile a reminder
    assert "develop/a" not in p.tasks and p.edges == []
    assert p.tasks["develop/b"].warnings == [
        "depends_on 中的任务「develop/a」不存在",
        "parent「develop/a」不存在",
    ]
    assert dict(fm_of(pdir, "develop/b")) == {"depends_on": ["develop/a"], "parent": "develop/a"}

    # progress 进行中, or just joined: its terminal session may still work in the workspace, as for rename
    assert "正在运行" in refused(409, ops.delete, pdir, "develop/r")
    ops.join(pdir, "develop/r")
    assert "终端" in refused(409, ops.delete, pdir, "develop/r")
    write_state(pdir / "workspaces/develop/r", new_state("needs_you", {"kind": "stopped"}))
    assert ops.delete(pdir, "develop/r").is_dir()
    refused(404, ops.delete, pdir, "develop/a")


def test_delete_refuses_a_running_managed_task(make_project):
    from agent_sys_v2.model import new_state, write_state

    pdir = make_project({"develop/a": ""})
    write_state(pdir / "workspaces/develop/a", new_state("running", {"kind": "working"}))
    assert "请先停止" in refused(409, ops.delete, pdir, "develop/a")


def test_delete_moves_the_workspace_first_and_undoes_a_failure(make_project, monkeypatch):
    pdir = make_project({"develop/a": ""}, progress={"develop/a": DONE})
    real = ops.shutil.move

    def fail_on_the_prompt(src, dst):
        if str(src).endswith(".md"):
            raise OSError("no room")
        return real(src, dst)

    monkeypatch.setattr(ops.shutil, "move", fail_on_the_prompt)
    with pytest.raises(OSError):
        ops.delete(pdir, "develop/a")
    # the workspace came back: the task is as it was
    assert (pdir / "workspaces/develop/a/progress.md").is_file() and task(
        pdir, "develop/a"
    ).status == "done"


def test_rename_moves_the_ledger_baseline_along(make_project):
    """project.yaml's ledger.baseline names a task (or a file of it): renamed with it, or the ledger quietly took
    another baseline."""
    for pinned, moved in (
        ("develop/b0", "experiment/b0"),
        ("develop/b0/results/c64.json", "experiment/b0/results/c64.json"),
        ("workspaces/develop/b0/results/c64.json", "workspaces/experiment/b0/results/c64.json"),
        ("develop/b01", "develop/b01"),  # another task
    ):
        pdir = make_project(
            {"develop/b0": "", "develop/b01": ""},
            name=f"p{len(pinned)}",
            config=f"title: t  # 标题\nledger: {{primary: tok_s, baseline: {pinned}}}\n",
            types=["experiment"],
        )
        ops.rename(pdir, "develop/b0", "experiment", "b0")
        assert (pdir / "project.yaml").read_text() == (
            f"title: t  # 标题\nledger: {{primary: tok_s, baseline: {moved}}}\n"
        )


def test_join_children_scans_once(make_project, monkeypatch):
    """全部纳入托管 of many drafts: one scan, not one per draft and per write."""
    pdir = make_project(
        {"plan/r1": "", **{f"experiment/e{i}": "---\nparent: plan/r1\n---\n" for i in range(5)}}
    )
    scans = []
    real = ops.load_project
    monkeypatch.setattr(ops, "load_project", lambda d: scans.append(d) or real(d))
    assert len(ops.join_children(pdir, "plan/r1")) == 5 and len(scans) == 1
    assert all(t.status == "queued" for t in load_project(pdir).tasks.values() if t.parent)


def test_settings_that_would_break_the_project_are_not_written(make_project):
    pdir = make_project({"develop/a": ""}, config="title: t\n")
    home = pdir.parent.parent
    for values in ({"autorun": "yes"}, {"agent": "opus"}):
        assert "project.yaml" in refused(422, ops.save_settings, home, "project", values, "p")
        assert (pdir / "project.yaml").read_text() == "title: t\n" and load_project(pdir)
    assert "正整数" in refused(400, ops.save_settings, home, "project", {"max_running": True}, "p")


def test_a_template_is_a_name(tmp_path):
    for template in ("/etc", "blank/../infera-perf", "_common", ".."):
        assert "不存在" in refused(404, ops.new_project, tmp_path, "x", template)
    assert not (tmp_path / "projects/x").exists()
