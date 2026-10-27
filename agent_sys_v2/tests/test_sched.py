"""The scheduler (sched.py), questions.md answered as a form (questions.py), 继续全部中断任务, 全部标为已查看 and
the review round. The reasons and slots are checked on scans alone; the rest runs real runner processes in replay
mode (project r of conftest, autorun on), with the server's own loop doing the scheduling."""

import os
import subprocess
import time

import pytest
from conftest import DONE, RUNNING, T, get, launch_json, manage, new, ok, replay, run_home, wait, ws
from fastapi.testclient import TestClient

from agent_sys_v2 import questions, runs, sched, tmux
from agent_sys_v2.model import AgentSysError, new_state, read_state, write_state
from agent_sys_v2.scan import load_project
from agent_sys_v2.server import create_app

ASK = "状态: 阻塞\n下一步: 选方案 A 还是 B？\n"
QUESTIONS = (
    "# 需要你决定\n\n两个问题。\n\n| # | 问题 | 建议 | 你的意见 |\n|---|---|---|---|\n"
    "| Q1 | 用哪个镜像？ | 用 a\\|b 里的 a | |\n| Q2 | 跑几轮？ | 3 轮 | |\n\n其他说明。\n"
)


def started(home, name):
    """When the task's first run started (its launch.json), as a float."""
    return min(
        p.stat().st_mtime for p in (ws(home, name) / ".agent_sys/runs").glob("*/launch.json")
    )


# ---------------------------------------------------------------- why a queued task waits, and slots


def test_why_a_queued_task_waits_first_reason_wins(make_project):
    pdir = make_project(
        {
            "develop/up": "x",
            "develop/r1": "x",
            "develop/r2": "x",
            "develop/a": "---\ndepends_on: [develop/up]\n---\n做 A\n",
        }
    )
    for tid in ("develop/up", "develop/r1", "develop/r2"):
        manage(pdir, tid, status="running")
    manage(pdir, "develop/a", status="queued")

    def reason(config, limit=4):
        (pdir / "project.yaml").write_text(f"title: p\n{config}", encoding="utf-8")
        p = load_project(pdir)
        go = sched.plan([p], limit)
        return p.tasks["develop/a"].detail, [t.id for _, t in go]

    assert reason("autorun: false\nmax_running: 2\narchived_at: 2026-10-23T00:00:00Z\n") == (
        {"kind": "archived"},
        [],
    )
    assert reason("autorun: false\nmax_running: 2\n") == (
        {"kind": "upstream", "tasks": ["develop/up"]},
        [],
    )
    manage(pdir, "develop/up", status="done")
    assert reason("autorun: false\nmax_running: 2\n") == ({"kind": "manual"}, [])
    # r1, r2
    assert reason("max_running: 2\n") == (
        {"kind": "concurrency", "scope": "project", "limit": 2},
        [],
    )
    assert reason("max_running: 3\n", limit=2) == (
        {"kind": "concurrency", "scope": "global", "limit": 2},
        [],
    )
    assert reason("max_running: 3\n", limit=3) == ({}, ["develop/a"])
    # checks that cannot be read: it would not run, so it is not started
    (pdir / "tasks/develop/a.md").write_text('---\nchecks: [{run: "false", timeout: "1"}]\n---\n')
    assert reason("max_running: 3\n", limit=3) == (
        {"kind": "invalid", "error": "checks 的第 1 项的 timeout 应为正数（秒）"},
        [],
    )


def test_slots_and_order(make_project):
    p1 = make_project(
        {f"develop/{n}": "x" for n in ("w", "t", "u", "q1", "q2", "q3")},
        progress={"develop/u": RUNNING},
        name="p1",
        config="title: p1\nmax_running: 3\n",
    )
    p2 = make_project(
        {f"develop/{n}": "x" for n in ("r", "x", "y")}, name="p2", config="title: p2\n"
    )
    manage(p1, "develop/w", status="running")
    manage(p1, "develop/t", status="running")
    for tid, since in (("q1", "03"), ("q2", "01"), ("q3", "02")):
        manage(p1, f"develop/{tid}", status="queued", since=f"2026-10-23T{since}:00:00Z")
    manage(p2, "develop/r", status="running")
    manage(p2, "develop/x", status="queued", since="2026-10-23T00:30:00Z")
    manage(p2, "develop/y", status="queued", since="2026-10-23T02:30:00Z")

    def plan(limit):
        a, b = load_project(p1), load_project(p2)
        # waiting in the background and terminal takeovers hold a slot; the unmanaged u (progress 进行中) does not
        a.tasks["develop/w"].detail = {"kind": "waiting"}
        a.tasks["develop/t"].detail = {"kind": "terminal"}
        go = [f"{p.id}:{t.id}" for p, t in sched.plan([a, b], limit)]
        return go, {
            f"{p.id}:{t.id}": t.detail
            for p in (a, b)
            for t in p.tasks.values()
            if t.status == "queued"
        }

    go, details = plan(10)
    assert go == ["p2:develop/x", "p1:develop/q2", "p2:develop/y"]  # the longest waiting first
    assert (
        details["p1:develop/q3"]
        == details["p1:develop/q1"]
        == {"kind": "concurrency", "scope": "project", "limit": 3}
    )
    go, details = plan(4)  # 3 running: one global slot, the oldest across the projects gets it
    assert go == ["p2:develop/x"]
    assert {k: d.get("scope") for k, d in details.items() if k != "p2:develop/x"} == dict.fromkeys(
        ["p1:develop/q1", "p1:develop/q2", "p1:develop/q3", "p2:develop/y"], "global"
    )


def test_replay_writes_stay_in_the_workspace(make_project):
    pdir = make_project(
        {
            "develop/a": "---\nagent: {replay: {events: x.jsonl, writes: [{progress.md: a}, {report.md: b}]}}\n"
            "---\n做 A\n",
            "develop/b": "---\nagent: {replay: {events: x.jsonl, "
            "writes: {../a/progress.md: x}}}\n---\n",
        }
    )
    manage(pdir, "develop/a", status="queued")
    p = load_project(pdir)
    a = p.tasks["develop/a"]
    assert runs.replay_spec(a, 0)["writes"] == {str(a.workspace / "progress.md"): "a"}
    # the last one after that
    assert runs.replay_spec(a, 5)["writes"] == {str(a.workspace / "report.md"): "b"}
    with pytest.raises(AgentSysError, match="不在任务工作区内"):
        runs.replay_spec(p.tasks["develop/b"], 0)


# ---------------------------------------------------------------- questions.md


def test_questions_table():
    text = "| 说明 | 值 |\n|---|---|\n| 不是这张 | 1 |\n\n" + QUESTIONS.replace(
        "| Q2 | 跑几轮？ | 3 轮 | |", "| Q2 | 跑几轮？ | 3 轮 | 已答<br>两行 |"
    )
    q = questions.parse(text)
    assert q["columns"] == ["#", "问题", "建议", "你的意见"] and q["answer_column"] == 3
    assert [(r["key"], r["cells"][2], r["answer"]) for r in q["rows"]] == [
        ("Q1", "用 a\\|b 里的 a", ""),
        ("Q2", "3 轮", "已答\n两行"),
    ]
    assert q["before"].endswith("两个问题。") and q["after"] == "其他说明。"
    new, filled, total = questions.fill(text, {"Q1": "用 b | 不用 a\n原因见 log"})
    assert (filled, total) == (2, 2)
    assert "| Q1 | 用哪个镜像？ | 用 a\\|b 里的 a | 用 b \\| 不用 a<br>原因见 log |" in new
    assert questions.parse(new)["rows"][0]["answer"] == "用 b | 不用 a\n原因见 log"
    # nothing else changed
    assert (
        new.replace(
            "| Q1 | 用哪个镜像？ | 用 a\\|b 里的 a | 用 b \\| 不用 a<br>原因见 log |",
            "| Q1 | 用哪个镜像？ | 用 a\\|b 里的 a | |",
        )
        == text
    )
    with pytest.raises(AgentSysError, match="没有「Q9」"):
        questions.fill(text, {"Q9": "x"})
    assert questions.parse("| # | 问题 |\n|---|---|\n| 1 | x |\n") is None
    rows = questions.parse("| # | 你的意见 |\n|---|---|\n| | a |\n| 1 | |\n| 1 | |\n")["rows"]
    # the English protocol's table
    q = questions.parse(
        "| # | Question | Suggestion | Your answer |\n|---|---|---|---|\n| 1 | a? | yes | no |\n"
    )
    assert q["answer_column"] == 3 and q["rows"][0]["answer"] == "no"
    assert [r["key"] for r in rows] == ["#1", "1", "#3"]  # empty or repeated first cells


# ---------------------------------------------------------------- through the server, with the scheduler running


@pytest.fixture
def home(tmp_path):
    return run_home(tmp_path)


def test_a_chain_runs_through(client, home):
    """调研 -> 开发 -> 评审 (review: true) -> 收尾: each starts by itself when its upstream is done. On the way: a
    reply to 待答复, questions.md answered as a form, and a review sent back once with 修改意见."""
    new(
        client,
        "research",
        "调研。",
        agent=replay(
            ["message_mid_command", "message_mid_command"],
            speed=40,
            writes=[
                {"progress.md": ASK},
                {"progress.md": DONE, "report.md": "## 结论\n\n用方案 A。\n"},
            ],
        ),
    )
    new(
        client,
        "dev",
        "按 workspaces/develop/research/report.md 开发。",
        agent=replay(
            ["message_mid_command", "message_mid_command"],
            speed=40,
            writes=[
                {
                    "progress.md": "状态: 阻塞\n下一步: 回答 questions.md\n",
                    "questions.md": QUESTIONS,
                },
                {"progress.md": DONE, "report.md": "## 结论\n\n做完了。\n"},
            ],
        ),
    )
    new(
        client,
        "review",
        "评审 tasks/develop/dev.md 的改动。",
        review=True,
        agent=replay(
            ["message_mid_command", "message_mid_command"],
            speed=40,
            writes=[
                {"progress.md": DONE, "report.md": "## 结论\n\n第一版。\n"},
                {"progress.md": DONE, "report.md": "## 结论\n\n第二版。\n"},
            ],
        ),
    )
    new(
        client,
        "wrapup",
        "按 workspaces/develop/review/report.md 收尾。",
        agent=replay(
            "message_mid_command", speed=40, writes={"progress.md": DONE, "report.md": "## 结论\n"}
        ),
    )
    assert get(client, "dev")["detail"] == {"kind": "upstream", "tasks": ["develop/research"]}

    # research starts by itself and asks; a reply from the home page's card resumes it
    t = wait(client, "research", lambda t: t["status"] == "needs_you")
    assert t["detail"] == {"kind": "asked"} and get(client, "dev")["status"] == "queued"
    assert ok(client.post(f"{T}/research/messages", json={"text": "选 A"}))["delivery"] == "resume"
    wait(client, "research", lambda t: t["status"] == "done")

    # dev starts by itself and writes questions.md; answered as a form, it is no longer this round's
    t = wait(client, "dev", lambda t: t["status"] == "needs_you")
    assert t["detail"] == {"kind": "questions", "count": 2, "path": "questions.md"}
    q = ok(client.get(f"{T}/dev/questions"))
    assert [r["key"] for r in q["rows"]] == ["Q1", "Q2"]
    r = ok(
        client.post(
            f"{T}/dev/questions", json={"answers": {"Q1": "用 b"}, "base_mtime": q["mtime"]}
        )
    )
    assert (r["saved"], r["delivery"], r["task"]["status"]) == ("questions.md", "resume", "running")
    t = get(client, "dev")
    cfg = launch_json(home, t)
    assert (
        cfg["message"]
        == "已在 questions.md 中回答（已填写 1 / 2 项），未填写的项请按你的建议处理。"
    )
    assert t["questions"] is None and client.get(f"{T}/dev/questions").status_code == 404
    assert (
        "| Q1 | 用哪个镜像？ | 用 a\\|b 里的 a | 用 b |"
        in (ws(home, "dev") / "questions.md").read_text()
    )
    wait(client, "dev", lambda t: t["status"] == "done")

    # the review waits for you; 修改意见 sends it back, the downstream waits all along; then 审核通过
    t = wait(client, "review", lambda t: t["status"] == "in_review")
    summary = ok(client.get("/api/home"))["in_review"][0]
    assert summary["id"] == "develop/review" and summary["round_cost_usd"] == pytest.approx(
        0.01712, abs=1e-4
    )
    assert (
        ok(client.post(f"{T}/review/messages", json={"text": "结论再写清楚一点"}))["task"]["status"]
        == "running"
    )
    assert get(client, "wrapup")["detail"] == {"kind": "upstream", "tasks": ["develop/review"]}
    t = wait(client, "review", lambda t: t["status"] == "in_review")
    assert "第二版" in (ws(home, "review") / "report.md").read_text() and len(t["runs"]) == 2
    assert get(client, "wrapup")["status"] == "queued"
    ok(client.post(f"{T}/review/actions/approve"))
    t = wait(client, "wrapup", lambda t: t["status"] == "done")
    order = [started(home, n) for n in ("research", "dev", "review", "wrapup")]
    assert order == sorted(order)


def test_slots_through_the_server_and_a_manual_start(home):
    (home / "projects/r/project.yaml").write_text(
        (home / "projects/r/project.yaml")
        .read_text()
        .replace("title: r\n", "title: r\nmax_running: 1\n")
    )
    with TestClient(create_app(home)) as client:
        new(client, "a", agent=replay("message_mid_command", speed=1))  # 30 s
        wait(client, "a", lambda t: t["status"] == "running")
        new(client, "b")
        t = wait(client, "b", lambda t: t["detail"].get("kind") == "concurrency", 5)
        assert t["detail"] == {"kind": "concurrency", "scope": "project", "limit": 1}
        board = ok(client.get("/api/projects/r/board"))["running"]
        assert (board["count"], board["limit"]) == (1, 1)
        # 开始 ignores the limit
        assert ok(client.post(f"{T}/b/actions/start"))["task"]["status"] == "running"
        assert ok(client.get("/api/projects/r/board"))["running"]["count"] == 2
        ok(client.post(f"{T}/a/actions/stop"))
        wait(client, "b", lambda t: t["status"] != "running")


def test_a_templates_tasks_wait_for_your_review(tmp_path):
    home = run_home(tmp_path, "template_review: true\n")
    with TestClient(create_app(home)) as client:
        new(client, "a")
        new(client, "b", depends_on=["develop/a"])
        client.app.state.store.poll()  # a round of the scheduler starts neither
        assert (get(client, "a")["status"], get(client, "a")["detail"]) == (
            "queued",
            {"kind": "template_review"},
        )
        assert get(client, "b")["detail"]["kind"] == "upstream"
        assert ok(client.get("/api/home"))["projects"][0]["template_review"] is True
        # 开始运行: the project runs as any other, a starts by itself
        assert ok(client.post("/api/projects/r/start"))["template_review"] is False
        wait(client, "a", lambda t: t["status"] != "queued", 5)


def test_starting_a_task_by_hand_ends_the_template_review(tmp_path):
    home = run_home(tmp_path, "template_review: true\n")
    with TestClient(create_app(home)) as client:
        new(client, "a")
        new(client, "b")
        assert ok(client.post(f"{T}/a/actions/start"))["task"]["status"] == "running"
        assert ok(client.get("/api/home"))["projects"][0]["template_review"] is False
        # the others start by themselves from now on
        wait(client, "b", lambda t: t["status"] != "queued", 5)


def test_archived_and_manual_projects_are_not_scheduled(client, home):
    ok(client.post("/api/projects/r/archive"))
    # queued in an archived project (new tasks are refused)
    write_state(ws(home, "a"), new_state("queued"))
    (home / "projects/r/tasks/develop/a.md").write_text("做 A\n", encoding="utf-8")
    client.app.state.store.poll()
    assert get(client, "a")["detail"] == {"kind": "archived"}
    ok(
        client.put(
            "/api/settings", json={"scope": "project", "project": "r", "values": {"autorun": False}}
        )
    )
    ok(client.post("/api/projects/r/unarchive"))
    client.app.state.store.poll()
    assert (get(client, "a")["status"], get(client, "a")["detail"]) == (
        "queued",
        {"kind": "manual"},
    )
    ok(
        client.put(
            "/api/settings", json={"scope": "project", "project": "r", "values": {"autorun": True}}
        )
    )
    wait(client, "a", lambda t: t["status"] != "queued", 5)


def test_a_start_that_fails_is_not_retried(client, home, monkeypatch):
    calls = []

    def broken(name, cmd, cwd):
        calls.append(name)
        raise AgentSysError("无法在 tmux 中启动运行：no server", 500)

    monkeypatch.setattr(tmux, "open_window", broken)
    new(client, "a")
    t = wait(client, "a", lambda t: t["status"] == "needs_you", 5)
    assert t["detail"] == {
        "kind": "interrupted",
        "error": "自动开始失败：无法在 tmux 中启动运行：no server",
    }
    assert t["runs"] == [] and read_state(ws(home, "a"))["runs"] == 0  # nothing half made
    for _ in range(2):  # more rounds of the scheduler
        client.app.state.store.poll()
    assert calls == ["a"] and get(client, "a")["status"] == "needs_you"
    monkeypatch.undo()
    t = ok(client.post(f"{T}/a/actions/resume", json={"text": "继续"}))["task"]  # 继续 tries again
    assert t["status"] == "running"
    wait(client, "a", lambda t: t["status"] != "running")


def test_questions_refusals(client, home):
    new(
        client,
        "a",
        agent=replay(
            "message_mid_command", speed=40, writes={"progress.md": ASK, "questions.md": QUESTIONS}
        ),
    )
    assert client.get(f"{T}/a/questions").status_code == 404  # no run yet
    wait(client, "a", lambda t: t["detail"].get("kind") == "questions")
    q = ok(client.get(f"{T}/a/questions"))
    path = ws(home, "a") / "questions.md"
    # the prompt broken while the task waited: the run is refused before the answers are written, and once the
    # prompt is fixed the same form can be sent again
    prompt = home / "projects/r/tasks/develop/a.md"
    good = prompt.read_text(encoding="utf-8")
    prompt.write_text(good.replace("---\n", '---\nchecks: [{run: "true", timeout: "60"}]\n', 1))
    r = client.post(f"{T}/a/questions", json={"answers": {"Q1": "x"}, "base_mtime": q["mtime"]})
    assert r.status_code == 409 and "timeout" in r.json()["error"]
    assert path.read_text() == QUESTIONS and ok(client.get(f"{T}/a/questions")) == q
    prompt.write_text(good, encoding="utf-8")
    # the agent changed it while you filled it in
    path.write_text(QUESTIONS.replace("3 轮", "5 轮"))
    os.utime(path, (time.time() + 2,) * 2)
    r = client.post(f"{T}/a/questions", json={"answers": {"Q1": "x"}, "base_mtime": q["mtime"]})
    assert (
        r.status_code == 409 and "已被修改" in r.json()["error"] and "| x |" not in path.read_text()
    )
    path.write_text("没有表格\n")
    r = client.get(f"{T}/a/questions")
    assert r.status_code == 422 and "你的意见" in r.json()["error"]


def test_resume_all_only_the_interrupted(client, home, own_tmux):
    """The machine went down (the runners' tmux server is gone); 继续全部中断任务 brings back only those."""
    # waits in the background
    new(client, "w", agent=replay(["wait_in_background", "message_mid_command"], speed=40))
    # a 10 s command
    new(client, "k", agent=replay(["message_mid_command", "message_mid_command"], speed=2))
    task = home / "projects/r/tasks/develop/k.md"
    new(client, "asked", agent=replay("message_mid_command", speed=40))
    wait(client, "w", lambda t: t["detail"].get("kind") == "waiting")
    wait(client, "k", lambda t: t["detail"].get("current"))
    wait(client, "asked", lambda t: t["status"] == "needs_you")
    subprocess.run(["tmux", "-L", own_tmux, "kill-server"], check=True)
    task.write_text(task.read_text().replace("speed: 2", "speed: 40"))  # k's next run, in no time
    w = wait(client, "w", lambda t: t["status"] != "running", 10)
    k = wait(client, "k", lambda t: t["status"] != "running", 10)
    assert w["detail"]["kind"] == "wait_interrupted" and k["detail"] == {"kind": "interrupted"}
    assert {t["id"] for t in ok(client.get("/api/home"))["needs_you"]} == {
        "develop/asked",
        "develop/w",
        "develop/k",
    }
    r = ok(
        client.post(
            "/api/resume",
            json={
                "tasks": [
                    {"project": "r", "task": f"develop/{n}"} for n in ("w", "k", "asked", "nope")
                ]
            },
        )
    )
    assert r["resumed"] == ["r/develop/w", "r/develop/k"]
    assert r["errors"] == [{"task": "r/develop/nope", "error": "任务「develop/nope」不存在"}]
    cfg = launch_json(home, get(client, "w"))
    assert "后台等待的命令已经结束" in cfg["message"] and cfg["message"].endswith("继续")
    assert get(client, "asked")["status"] == "needs_you"
    for n in ("w", "k"):
        assert wait(client, n, lambda t: t["status"] != "running", 60)["runs"][0]["exit"] == "idle"


def test_mark_read(client, home):
    for n in ("a", "b"):
        new(
            client,
            n,
            agent=replay(
                "message_mid_command",
                speed=40,
                writes={"progress.md": DONE, "report.md": "## 结论\n"},
            ),
        )
    for n in ("a", "b"):
        wait(client, n, lambda t: t["status"] == "done")
    assert [t["id"] for t in ok(client.get("/api/home"))["unread"]] in (
        ["develop/a", "develop/b"],
        ["develop/b", "develop/a"],
    )
    ok(client.post(f"{T}/a/actions/mark_read"))
    r = ok(
        client.post(
            "/api/mark_read",
            json={
                "tasks": [
                    {"project": "r", "task": "develop/a"},
                    {"project": "r", "task": "develop/b"},
                ]
            },
        )
    )
    assert r == {"marked": ["r/develop/b"], "errors": []}
    assert ok(client.get("/api/home"))["unread"] == []
