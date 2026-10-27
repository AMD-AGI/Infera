"""upstream.py: 上游已更新. A run records its input and code upstream tasks as they were (launch.json); a task done or in
review whose upstream changed after says so on its card and in its detail, with a message to send. Nothing is
reopened."""

import json
import os
import time

from conftest import DONE, manage, ok
from fastapi.testclient import TestClient
from test_launch import project_with_rules

from agent_sys_v2 import launch, upstream
from agent_sys_v2.model import now_iso, read_json, write_json
from agent_sys_v2.ops import new_task
from agent_sys_v2.scan import load_project
from agent_sys_v2.server import create_app

REPORT = "## 结论\n\n好\n\n## 交接清单\n\n- `log/notes.md` — 笔记\n- `results/` — 结果\n- `report.md` — 本报告\n- 分支 `x`\n"


def project(make_project, extra=None):
    """a (done, with a report, results and a handoff list) → b (input), c (code), d (review), e (subtask of a)."""
    pdir = make_project(
        {
            "develop/a": "",
            "develop/b": "---\ndepends_on: [develop/a]\n---\n做 B\n",
            "develop/c": "---\ndepends_on: [{task: develop/a, kind: code}]\n---\n做 C\n",
            "develop/d": "---\ndepends_on: [{task: develop/a, kind: review}]\n---\n做 D\n",
            "develop/e": "---\nparent: develop/a\n---\n做 E\n",
            **(extra or {}),
        },
        progress={"develop/a": DONE},
    )
    ws = pdir / "workspaces/develop/a"
    (ws / "report.md").write_text(REPORT, encoding="utf-8")
    (ws / "log").mkdir()
    (ws / "log/notes.md").write_text("n\n", encoding="utf-8")
    (ws / "results/raw").mkdir(parents=True)
    (ws / "results/x.json").write_text("{}", encoding="utf-8")
    return pdir


def start_round(pdir, tid, status="done"):
    """A round of tid as launch.prepare records it, then the task in status."""
    p = load_project(pdir)
    t = p.tasks[tid]
    run_id = f"20261027-{len(list((t.workspace / '.agent_sys/runs').glob('*'))):06d}"
    write_json(
        t.workspace / ".agent_sys/runs" / run_id / "launch.json",
        {"upstream": upstream.snapshot(p, t), "started": now_iso()},
    )
    manage(pdir, tid, status=status, run_id=run_id)


def later(path, text=None):
    """Write path (or only touch it) with an mtime a second later: the change shows whatever the clock's grain."""
    if text is not None:
        path.write_text(text, encoding="utf-8")
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))


def changed(pdir, tid):
    return load_project(pdir).tasks[tid].upstream_changed


def test_the_snapshot(make_project):
    """input and code upstream tasks only: status, the report's sha256, results/'s own files and the handoff list's
    paths (not the report again, not results/ again, not a branch)."""
    pdir = project(make_project)
    p = load_project(pdir)
    assert set(upstream.snapshot(p, p.tasks["develop/b"])) == {"develop/a"}
    assert set(upstream.snapshot(p, p.tasks["develop/c"])) == {"develop/a"}
    assert (
        upstream.snapshot(p, p.tasks["develop/d"])
        == upstream.snapshot(p, p.tasks["develop/e"])
        == {}
    )
    snap = upstream.snapshot(p, p.tasks["develop/b"])["develop/a"]
    assert snap["status"] == "done" and snap["report"] == "report.md"
    assert len(snap["report_sha256"]) == 64
    assert sorted(snap["files"]) == ["log/notes.md", "results/x.json"]
    assert all(len(v) == 2 for v in snap["files"].values())


def test_a_run_records_its_upstream(tmp_path, make_project):
    home = tmp_path / "home"
    pdir = project_with_rules(home, make_project)
    new_task(pdir, "develop", "c", "做 C", queue=True, depends_on=["develop/a"])
    p = load_project(pdir)
    t = p.tasks["develop/c"]
    parts, text = launch.system(home, p, t)
    cfg = launch.prepare(
        home,
        p,
        t,
        run_dir=t.workspace / ".agent_sys/runs/20261027-100000",
        kind="sdk",
        parts=parts,
        system_text=text,
        message="m",
        resume=False,
        max_budget_usd=1,
        cost_start_usd=0.0,
        window="c",
        cli_path="/x/claude",
    )
    assert cfg["upstream"] == upstream.snapshot(p, t) and cfg["upstream"]["develop/a"]["report"]
    assert read_json(t.workspace / ".agent_sys/runs/20261027-100000/launch.json")["upstream"]


def test_nothing_changed_nothing_shown(make_project):
    pdir = project(make_project)
    for tid in ("develop/b", "develop/c"):
        start_round(pdir, tid)
    p = load_project(pdir)
    assert p.tasks["develop/b"].upstream_changed == p.tasks["develop/c"].upstream_changed == []
    # touched, the same bytes: the report is compared by its content
    later(pdir / "workspaces/develop/a/report.md")
    assert changed(pdir, "develop/b") == []
    # raw data under results/: not a result
    (pdir / "workspaces/develop/a/results/raw/r.json").write_text("{}", encoding="utf-8")
    assert changed(pdir, "develop/b") == []


def test_what_changed(make_project):
    pdir = project(make_project)
    start_round(pdir, "develop/b")
    ws = pdir / "workspaces/develop/a"
    later(ws / "report.md", REPORT + "\n补充\n")
    (ws / "results/y.json").write_text("{}", encoding="utf-8")
    later(ws / "results/x.json", '{"a": 1}')
    later(ws / "log/notes.md", "n2\n")
    [c] = changed(pdir, "develop/b")
    assert (c["task"], c["title"], c["status"], c["report"]) == ("develop/a", "a", None, True)
    assert (c["added"], c["removed"], c["changed"]) == (
        ["results/y.json"],
        [],
        ["log/notes.md", "results/x.json"],
    )
    assert c["since"]
    assert c["text"] == (
        "上游「a」在本任务开始后更新了：报告有改动；新增 results/y.json；log/notes.md、results/x.json 有改动"
    )
    # removed, and run again: the status says so
    (ws / "results/y.json").unlink()
    (ws / "results/x.json").unlink()
    start_round(pdir, "develop/b")  # a new round: compared with this one from now on
    assert changed(pdir, "develop/b") == []
    (ws / "results/x.json").write_text("{}", encoding="utf-8")
    (ws / "log/notes.md").unlink()
    manage(pdir, "develop/a", status="running")
    [c] = changed(pdir, "develop/b")
    assert c["status"] == ["done", "running"]
    assert (c["added"], c["removed"]) == (["results/x.json"], ["log/notes.md"])
    assert c["text"] == (
        "上游「a」在本任务开始后更新了：状态从「已完成」变为「运行中」；新增 results/x.json；删除了 log/notes.md"
    )


def test_only_tasks_done_or_in_review_say_so(make_project):
    pdir = project(make_project, {"develop/f": "---\ndepends_on: [develop/a]\n---\n做 F\n"})
    start_round(pdir, "develop/b", "in_review")
    start_round(pdir, "develop/c", "running")
    start_round(pdir, "develop/f", "needs_you")
    later(pdir / "workspaces/develop/a/report.md", "## 结论\n\n变了\n")
    p = load_project(pdir)
    assert [t.id for t in p.tasks.values() if t.upstream_changed] == ["develop/b"]
    # nothing else follows: the task's status, its card's reminders
    assert p.tasks["develop/b"].status == "in_review" and not p.tasks["develop/b"].warnings
    assert p.tasks["develop/b"].to_json()["upstream_changed"][0]["report"] is True


def test_a_run_from_before_or_an_upstream_gone(make_project):
    pdir = project(make_project)
    manage(pdir, "develop/b", status="done", run_id="20261020-000000")
    run = pdir / "workspaces/develop/b/.agent_sys/runs/20261020-000000"
    write_json(run / "launch.json", {"started": now_iso()})  # no upstream: a run from before
    later(pdir / "workspaces/develop/a/report.md", "x\n")
    assert changed(pdir, "develop/b") == []
    write_json(
        run / "launch.json", {"upstream": {"develop/gone": {"status": "done"}}, "started": ""}
    )
    assert changed(pdir, "develop/b") == []
    manage(pdir, "develop/c", status="done")  # no run at all
    assert changed(pdir, "develop/c") == []


def test_the_message_in_the_agents_language(make_project):
    pdir = project(make_project)
    start_round(pdir, "develop/b")
    later(pdir / "workspaces/develop/a/report.md", "x\n")
    for n in range(7):
        (pdir / f"workspaces/develop/a/results/n{n}.json").write_text("{}", encoding="utf-8")
    t = load_project(pdir).tasks["develop/b"]
    zh = upstream.message(t, "zh")
    assert zh.startswith(
        "你开始本任务之后，上游任务更新了：\n- a（`workspaces/develop/a/`）：报告有改动；新增 "
    )
    assert "n4.json等 7 个" in zh and "n5.json" not in zh and "复核本任务的结论" in zh
    en = upstream.message(t, "en")
    assert en.startswith(
        "Upstream tasks changed after you started this task:\n- a (`workspaces/develop/a/`): its report changed; "
    )
    assert "n4.json (7 in all) added" in en and "check this task's conclusion" in en
    with TestClient(create_app(pdir.parent.parent)) as client:
        detail = ok(client.get("/api/projects/p/tasks/develop/b"))
        assert detail["upstream_message"] == zh  # protocol_language: zh by default
        assert detail["upstream_changed"][0]["task"] == "develop/a"
        board = ok(client.get("/api/projects/p/board"))
        assert [t["id"] for t in board["tasks"] if t["upstream_changed"]] == ["develop/b"]
        assert ok(client.get("/api/projects/p/tasks/develop/c"))["upstream_message"] == ""


def test_digests_are_cached(make_project, monkeypatch):
    """A scan reads a report again only when it changed, and a launch.json only when it did."""
    pdir = project(make_project)
    start_round(pdir, "develop/b")
    load_project(pdir)
    reads = []
    real_sha, real_read = upstream.hashlib.sha256, upstream.read_json
    monkeypatch.setattr(upstream.hashlib, "sha256", lambda b: reads.append("sha") or real_sha(b))
    monkeypatch.setattr(upstream, "read_json", lambda p: reads.append("json") or real_read(p))
    load_project(pdir)
    assert reads == []
    later(pdir / "workspaces/develop/a/report.md", "y\n")
    load_project(pdir)
    assert reads == ["sha"]
    time.sleep(0.01)
    run = next((pdir / "workspaces/develop/b/.agent_sys/runs").glob("*/launch.json"))
    later(run, json.dumps(read_json(run)))
    load_project(pdir)
    assert reads == ["sha", "json"]


def test_what_launch_json_or_results_hold_cannot_break_a_scan(make_project):
    """launch.json is in the workspace: unreadable, or anything in it; a dotfile in results/ (an editor's, NFS's) is
    no result."""
    pdir = project(make_project)
    start_round(pdir, "develop/b")
    run = next((pdir / "workspaces/develop/b/.agent_sys/runs").glob("*/launch.json"))
    (pdir / "workspaces/develop/a/results/.nfs0001").write_text("x", encoding="utf-8")
    assert changed(pdir, "develop/b") == []
    for bad in (
        {"upstream": {"develop/a": {"status": "done", "files": ["x"]}}},
        {"upstream": {"develop/a": {"status": "done", "files": {"x": 3}}}},
        {"upstream": {"develop/a": {"files": {}}}},
        {"upstream": {"develop/a": "done"}},
        {"upstream": ["develop/a"]},
    ):
        later(run, json.dumps(bad))
        assert changed(pdir, "develop/b") == []
    run.chmod(0)
    try:
        later(run)
        assert changed(pdir, "develop/b") == []
    finally:
        run.chmod(0o644)


def test_handoff_files_outside_the_workspace_and_not_directories(make_project, tmp_path):
    """A handoff item may link out of the workspace (scratch on another disk): its target is digested. A directory is
    not (a code dependency's checkout changes with what the downstream builds in it)."""
    pdir = project(make_project)
    ws = pdir / "workspaces/develop/a"
    outside = tmp_path / "scratch/data.csv"
    outside.parent.mkdir()
    outside.write_text("1\n", encoding="utf-8")
    (ws / "data.csv").symlink_to(outside)
    (ws / "repo").mkdir()
    (ws / "report.md").write_text(
        REPORT + "- `data.csv` — 数据\n- `repo/` — 代码\n", encoding="utf-8"
    )
    p = load_project(pdir)
    assert sorted(upstream.snapshot(p, p.tasks["develop/b"])["develop/a"]["files"]) == [
        "data.csv",
        "log/notes.md",
        "results/x.json",
    ]
    start_round(pdir, "develop/b")
    (ws / "repo/build").mkdir()
    assert changed(pdir, "develop/b") == []
    later(outside, "2\n")
    assert changed(pdir, "develop/b")[0]["changed"] == ["data.csv"]
