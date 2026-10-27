"""Runs through the server (runs.py + server.py): real runner processes in replay mode in a tmux of their own, no
model. Start -> running -> done / in_review / needs_you, messages, 停止, a killed runner, a restarted server,
terminal takeover, budget, cleaning up processes, and the run's events as the run view gets them."""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import (
    DONE,
    EVENTS,
    T,
    get,
    launch_json,
    new,
    ok,
    replay,
    run_dir_of,
    run_home,
    wait,
    wait_for,
    ws,
)
from fastapi.testclient import TestClient

from agent_sys_v2 import events, procs, runs, tmux
from agent_sys_v2.model import (
    AgentSysError,
    read_json,
    read_state,
    runs_dir,
    split_frontmatter,
    write_state,
)
from agent_sys_v2.scan import load_project
from agent_sys_v2.server import create_app


@pytest.fixture
def home(tmp_path):
    """Project r with autorun off: these tests start their tasks by hand (the scheduler is test_sched.py's)."""
    return run_home(tmp_path, "autorun: false\n")


def test_round_end_by_progress(client, home):
    for name in ("asked", "done", "review", "questions"):
        new(
            client,
            name,
            review=name == "review",
            **({"agent": replay("message_mid_command", speed=10)} if name == "questions" else {}),
        )
    for name in ("done", "review"):
        (ws(home, name) / "progress.md").write_text(DONE, encoding="utf-8")
        (ws(home, name) / "report.md").write_text("## 结论\n", encoding="utf-8")
    for name in ("asked", "done", "review", "questions"):
        t = ok(client.post(f"{T}/{name}/actions/start"))["task"]
        assert t["status"] == "running"
    time.sleep(1)  # the agent of `questions` writes questions.md during its round
    # a table of data first: only the rows of the form are questions
    (ws(home, "questions") / "questions.md").write_text(
        "| 并发 | tok/s |\n|---|---|\n| 64 | 1 |\n| 96 | 2 |\n| 128 | 3 |\n\n"
        "| # | 问题 | 建议 | 你的意见 |\n|---|---|---|---|\n| Q1 | a | b | |\n| Q2 | c | d | |\n",
        encoding="utf-8",
    )
    asked = wait(client, "asked", lambda t: t["status"] != "running")
    assert (asked["status"], asked["detail"]) == ("needs_you", {"kind": "asked"})
    assert asked["last_message"] == "got it" and asked["cost_usd"] == pytest.approx(
        0.01712215, abs=1e-5
    )
    run = asked["runs"][0]
    assert (run["kind"], run["exit"], run["live"], run["resume"]) == (
        "replay",
        "idle",
        False,
        False,
    )
    done = wait(client, "done", lambda t: t["status"] != "running")
    assert (done["status"], done["unread"], done["finished"] != "") == ("done", True, True)
    assert wait(client, "review", lambda t: t["status"] != "running")["status"] == "in_review"
    q = wait(client, "questions", lambda t: t["status"] != "running")
    assert q["detail"] == {"kind": "questions", "count": 2, "path": "questions.md"}
    assert q["questions"]["count"] == 2
    home_page = ok(client.get("/api/home"))
    assert home_page["cost_today_usd"] == pytest.approx(4 * 0.01712215, abs=1e-4)
    assert [t["id"] for t in home_page["needs_you"]] == ["develop/asked", "develop/questions"]


def test_resume_with_rules_update_and_a_done_task_that_stays_done(client, home):
    new(client, "a", agent=replay(["message_mid_command", "killed_runner_resume"], speed=40))
    ok(client.post(f"{T}/a/actions/start"))
    wait(client, "a", lambda t: t["status"] == "needs_you")
    (home / "rules").mkdir()
    (home / "rules/new.md").write_text("新规则", encoding="utf-8")
    r = ok(client.post(f"{T}/a/messages", json={"text": "继续"}))
    assert r["delivery"] == "resume" and r["task"]["status"] == "running"
    t = get(client, "a")
    cfg = launch_json(home, t)
    assert (
        cfg["resume"]
        and cfg["message"].startswith("（agent_sys 规则有更新，请先读 `")
        and cfg["message"].endswith("继续")
    )
    assert cfg["replay"]["events"].endswith("killed_runner_resume.jsonl")
    assert cfg["cost_start_usd"] == pytest.approx(0.01712215, abs=1e-5)
    (ws(home, "a") / "progress.md").write_text(DONE, encoding="utf-8")
    t = wait(client, "a", lambda t: t["status"] != "running")
    assert t["status"] == "done" and len(t["runs"]) == 2
    ok(client.post(f"{T}/a/actions/mark_read"))
    # a follow-up question: progress still done, no report change -> stays done, not unread again
    ok(client.post(f"{T}/a/messages", json={"text": "再问一句"}))
    t = wait(client, "a", lambda t: t["status"] != "running")
    assert (t["status"], t["unread"]) == ("done", False)
    cfg = launch_json(home, t)
    assert cfg["message"] == "再问一句"  # the rules are what the session saw last time: no hint


def test_a_moved_task_resumes_its_session_and_is_told_where_it_went(client, home):
    new(client, "a", agent=replay(["message_mid_command", "killed_runner_resume"], speed=40))
    ok(client.post(f"{T}/a/actions/start"))
    sid = wait(client, "a", lambda t: t["status"] == "needs_you")["session_id"]
    (home / "projects/r/tasks/bug_fix").mkdir()
    assert ok(client.post(f"{T}/a/rename", json={"type": "bug_fix"}))["id"] == "bug_fix/a"
    r = ok(client.post("/api/projects/r/tasks/bug_fix/a/messages", json={"text": "继续"}))
    assert r["delivery"] == "resume"
    t = ok(client.get("/api/projects/r/tasks/bug_fix/a"))
    assert t["session_id"] == sid and len(t["runs"]) == 2
    cfg = launch_json(home, t)
    assert cfg["resume"] and cfg["message"].startswith(
        "（这个任务已从 `tasks/develop/a.md` 移到 `tasks/bug_fix/a.md`"
    )
    # the new system.md names the new paths
    assert "请先读 `" in cfg["message"] and cfg["message"].endswith("继续")
    assert "moved_from" not in json.loads(
        (home / "projects/r/workspaces/bug_fix/a/.agent_sys/state.json").read_text()
    )


def test_messages_while_running_and_stop(client, home):
    new(client, "a", agent=replay("message_mid_command", speed=2))  # a 10 s command first
    ok(client.post(f"{T}/a/actions/start"))
    t = wait(client, "a", lambda t: t["detail"].get("current"))
    assert (
        t["detail"]["current"].startswith("Bash sleep 20") and t["runs"][0]["activity"] == "working"
    )
    assert ok(client.post(f"{T}/a/messages", json={"text": "只看 C96"}))["delivery"] == "after_tool"
    t = ok(client.post(f"{T}/a/actions/stop"))["task"]
    assert (t["status"], t["detail"]) == ("needs_you", {"kind": "stopped"})
    assert get(client, "a")["runs"][0]["exit"] == "stopped"

    # then waits 2.5 h in the background
    new(client, "w", agent=replay("wait_in_background", speed=40))
    ok(client.post(f"{T}/w/actions/start"))
    t = wait(client, "w", lambda t: t["detail"].get("kind") == "waiting")
    assert "wake_flag" in t["detail"]["waiting_for"] and t["runs"][0]["background"]
    assert ok(client.post(f"{T}/w/messages", json={"text": "进展如何？"}))["delivery"] == "now"
    run = t["runs"][0]["id"]
    wait_for(
        lambda: any(
            "进展如何" in json.dumps(e, ensure_ascii=False) and e["type"] == "assistant"
            for e in ok(client.get(f"{T}/w/runs/{run}/events"))["events"]
        ),
        10,
    )
    assert get(client, "w")["detail"]["kind"] == "waiting"  # the background command is still there
    ok(client.post(f"{T}/w/actions/stop"))
    assert get(client, "w")["detail"] == {"kind": "stopped"}


def test_killed_runner_and_resume(client, home):
    new(client, "w", agent=replay(["wait_in_background", "message_mid_command"], speed=40))
    ok(client.post(f"{T}/w/actions/start"))
    t = wait(client, "w", lambda t: t["detail"].get("kind") == "waiting")
    rec = json.loads(
        (ws(home, "w") / ".agent_sys/runs" / t["runs"][0]["id"] / "run.json").read_text()
    )
    os.kill(rec["pid"], signal.SIGKILL)  # the machine went down, say
    t = wait(client, "w", lambda t: t["status"] != "running")
    assert t["detail"]["kind"] == "wait_interrupted" and "wake_flag" in t["detail"]["waiting_for"]
    assert t["runs"][0]["exit"] == "interrupted"
    ok(client.post(f"{T}/w/actions/resume", json={"text": "继续"}))
    cfg = launch_json(home, get(client, "w"))
    assert "后台等待的命令已经结束" in cfg["message"]
    wait(client, "w", lambda t: t["status"] != "running")


def test_server_restart_picks_up_runs(home):
    with TestClient(create_app(home)) as c:
        new(c, "a", agent=replay("message_mid_command", speed=20))
        new(c, "b", agent=replay("message_mid_command", speed=6))
        ok(c.post(f"{T}/a/actions/start"))
        ok(c.post(f"{T}/b/actions/start"))
    # no server: a's runner ends meanwhile; b's is still running when the next server starts
    a_run = sorted((ws(home, "a") / ".agent_sys/runs").iterdir())[-1]
    wait_for(lambda: read_json(a_run / "run.json").get("ended"), 30)
    assert read_state(ws(home, "a"))["status"] == "running"  # nobody moved it on yet
    with TestClient(create_app(home)) as c:
        assert wait(c, "a", lambda t: t["status"] != "running", 5)["detail"] == {"kind": "asked"}
        # it waits for you since the run ended, not since this server found it over
        assert read_state(ws(home, "a"))["since"] == read_json(a_run / "run.json")["ended"]
        t = get(c, "b")
        assert t["status"] == "running" and t["runs"][0]["live"]
        assert wait(c, "b", lambda t: t["status"] != "running")["runs"][0]["exit"] == "idle"


def test_takeover_and_end_takeover(client, home):
    new(client, "a")
    ok(client.post(f"{T}/a/actions/start"))
    wait(client, "a", lambda t: t["status"] == "needs_you")
    r = ok(client.post(f"{T}/a/actions/takeover"))
    assert r["takeover"] == {
        "window": "agent_sys:a",
        "attach": f"tmux -L {tmux.TMUX} attach -t agent_sys:a",
    }
    t = r["task"]
    assert (t["status"], t["detail"]["kind"]) == ("running", "terminal")
    t = wait(client, "a", lambda t: t["runs"][0]["live"])
    assert t["runs"][0]["kind"] == "terminal" and t["runs"][0]["attach"] == r["takeover"]["attach"]
    assert "a" in tmux.window_names()
    msg = client.post(f"{T}/a/messages", json={"text": "x"})
    assert msg.status_code == 409 and "终端" in msg.json()["error"]
    # what you did in the terminal
    (ws(home, "a") / "progress.md").write_text(DONE, encoding="utf-8")
    t = ok(client.post(f"{T}/a/actions/end_takeover"))["task"]
    assert t["status"] == "done"
    assert get(client, "a")["runs"][0]["exit"] == "terminal_exit"


def test_budget(client, home):
    new(client, "a", agent={"budget_usd": 0.01, **replay("message_mid_command", speed=40)})
    ok(client.post(f"{T}/a/actions/start"))
    t = wait(client, "a", lambda t: t["status"] != "running")
    assert t["cost_usd"] > t["budget_usd"]
    state = read_state(ws(home, "a"))
    # as the CLI's budget stop leaves it
    write_state(ws(home, "a"), {**state, "detail": {"kind": "budget"}})
    client.app.state.store.sync(also=[("r", "develop/a")])
    r = client.post(f"{T}/a/messages", json={"text": "继续"})
    assert r.status_code == 409 and "追加预算" in r.json()["error"]
    r = ok(client.post(f"{T}/a/actions/add_budget", json={"usd": 1}))
    assert r["budget_usd"] == 1.01 and r["task"]["status"] == "running"
    fm, _ = split_frontmatter((home / "projects/r/tasks/develop/a.md").read_text())
    assert fm["agent"]["budget_usd"] == 1.01
    wait(client, "a", lambda t: t["status"] != "running")


def test_budget_that_is_not_a_number(client):
    """budget_usd: "50 USD" (written by hand) counts as no budget: refused (409, not 500) until you add some."""
    new(client, "a", agent={"budget_usd": "50 USD"})
    r = client.post(f"{T}/a/actions/start")
    assert r.status_code == 409 and "$0.00" in r.json()["error"]
    r = ok(client.post(f"{T}/a/actions/add_budget", json={"usd": 1}))
    assert r["budget_usd"] == 1 and r["task"]["status"] == "queued"
    ok(client.post(f"{T}/a/actions/start"))
    wait(client, "a", lambda t: t["status"] != "running")


def test_cancel_a_running_task(client, home):
    new(client, "a", agent=replay("message_mid_command", speed=1))
    ok(client.post(f"{T}/a/actions/start"))
    t = wait(client, "a", lambda t: t["runs"] and t["runs"][0]["live"])
    rec = json.loads(
        (ws(home, "a") / ".agent_sys/runs" / t["runs"][0]["id"] / "run.json").read_text()
    )
    assert ok(client.post(f"{T}/a/actions/cancel"))["task"]["status"] == "cancelled"
    assert procs.wait_gone([rec["pid"]], 5) == []


def test_clean_kills_only_what_it_should(home, tmp_path):
    pdir = home / "projects/r"
    (pdir / "tasks/develop/a.md").write_text("x", encoding="utf-8")
    write_state(
        ws(home, "a"),
        {
            **json.loads("{}"),
            "version": 1,
            "status": "needs_you",
            "detail": {"kind": "asked"},
            "session_id": "sid-a",
            "runs": 1,
            "run_id": "r1",
            "cost_usd": 0,
            "session_cost_usd": 0,
        },
    )
    t = load_project(pdir).tasks["develop/a"]
    sleep = [sys.executable, "-c", "import time; time.sleep(60)"]
    run1, other = runs_dir(t.workspace) / "r1", tmp_path / "other_run"
    runner = ["-m", "agent_sys_v2.runner"]
    procs = {
        "runner": subprocess.Popen(sleep + runner + [str(run1)]),  # this run's runner: KILL
        "claude": subprocess.Popen(sleep + ["--resume=sid-a"]),  # the session: TERM
        "claude2": subprocess.Popen(sleep + ["--session-id", "sid-a"]),
        "other_runner": subprocess.Popen(sleep + runner + [str(other)]),  # another run's runner
        "other_claude": subprocess.Popen(sleep + ["--resume=sid-b"]),  # another session
        "reused": subprocess.Popen(sleep),  # a reused pid
    }
    run1.mkdir(parents=True)
    run2 = runs_dir(t.workspace) / "r0"
    run2.mkdir()
    host = runs.HOST
    (run1 / "run.json").write_text(
        json.dumps({"pid": procs["runner"].pid, "host": host, "ended": None})
    )
    (run2 / "run.json").write_text(
        json.dumps({"pid": procs["reused"].pid, "host": host, "ended": None})
    )
    try:
        runs.clean(t)
        for name in ("runner", "claude", "claude2"):
            assert procs[name].wait(5) is not None, name
        for name in ("other_runner", "other_claude", "reused"):
            assert procs[name].poll() is None, name
    finally:
        for p in procs.values():
            p.kill()
            p.wait()


def test_run_events_come_in_pages_and_the_page_is_told_of_new_ones(client, home):
    told = []
    client.app.state.store.listeners.add(lambda c: "run" in c and told.append(c))
    new(client, "a")
    ok(client.post(f"{T}/a/actions/start"))
    t = wait(client, "a", lambda t: t["status"] != "running")
    run = t["runs"][0]["id"]
    lines = (ws(home, "a") / f".agent_sys/runs/{run}/events.jsonl").read_text().splitlines()
    # while it ran, /api/events said which run grew, up to its last line (one SSE per page, no stream per run)
    assert told and {(c["project"], c["task"], c["run"]) for c in told} == {("r", "develop/a", run)}
    assert [c["seq"] for c in told] == sorted({c["seq"] for c in told}) and told[-1]["seq"] <= len(
        lines
    ) - 1
    page = ok(client.get(f"{T}/a/runs/{run}/events"))
    shown = [s for s, line in enumerate(lines) if events.slim(json.loads(line)) is not None]
    assert ([e["seq"] for e in page["events"]], page["upto"], page["first"]) == (
        shown,
        len(lines) - 1,
        0,
    )
    assert [e["seq"] for e in ok(client.get(f"{T}/a/runs/{run}/events?after=5"))["events"]] == [
        s for s in shown if s > 5
    ]
    # the end first, then earlier pages; a page that starts in the middle carries the session's state
    end = ok(client.get(f"{T}/a/runs/{run}/events?tail=4"))
    assert [e["seq"] for e in end["events"]] == shown[-4:] and end["first"] == shown[-4]
    assert {e["subtype"] for e in end["carry"]} <= {
        "session_state_changed",
        "background_tasks_changed",
    }
    before = ok(client.get(f"{T}/a/runs/{run}/events?tail=4&before={end['first']}"))
    assert [e["seq"] for e in before["events"]] == shown[-8:-4]
    assert ok(client.get(f"{T}/a/runs/{run}/events?tail=1000"))["events"] == page["events"]
    # one event whole
    assert ok(client.get(f"{T}/a/runs/{run}/events/{shown[1]}")) == events.slim(
        {"seq": shown[1], **json.loads(lines[shown[1]])}, None
    )
    assert client.get(f"{T}/a/runs/{run}/events/{len(lines)}").status_code == 404
    assert ok(client.get(f"{T}/a/runs/{run}/launch"))["markdown"].startswith(f"# 运行 {run}")
    assert client.get(f"{T}/a/runs/nope/events").status_code == 404
    # packed: the browser asks for gzip
    r = client.get(f"{T}/a/runs/{run}/events", headers={"Accept-Encoding": "gzip"})
    assert r.headers.get("content-encoding") == "gzip" or len(r.content) < 1024


def test_slim_keeps_what_the_run_view_shows():
    big = "x" * (events.PREVIEW + 10)
    e = {
        "seq": 3,
        "type": "user",
        "ts": 1.0,
        "uuid": "u",
        "session_id": "s",
        "tool_use_result": {"stdout": big},
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [
                        {"type": "text", "text": big},
                        {"type": "image", "source": {"data": "AAAA"}},
                    ],
                },
                {"type": "tool_result", "tool_use_id": "t2", "content": "short", "is_error": True},
            ],
        },
    }
    s = events.slim(e)
    assert set(s) == {"seq", "type", "ts", "message"} and list(s["message"]) == ["content"]
    long, short = s["message"]["content"]
    assert (long["content"], long["clipped"], long["chars"]) == (
        big[: events.PREVIEW],
        True,
        len(big) + len("\n[image]"),
    )
    assert short == {
        "type": "tool_result",
        "tool_use_id": "t2",
        "is_error": True,
        "content": "short",
        "chars": 5,
    }
    assert events.slim(e, None)["message"]["content"][0]["content"] == big + "\n[image]"
    a = events.slim(
        {
            "type": "assistant",
            "ts": 2.0,
            "message": {
                "id": "m",
                "usage": {},
                "content": [
                    {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "Write",
                        "input": {"file_path": "a.py", "content": big},
                    },
                ],
            },
        }
    )
    assert a["message"]["content"] == [
        {"type": "thinking", "thinking": "", "chars": 3, "clipped": True},
        {
            "type": "tool_use",
            "id": "t1",
            "name": "Write",
            "input": {"file_path": "a.py", "content": big[: events.PREVIEW]},
            "clipped": True,
        },
    ]
    # tagged 回放 in the view
    result = {"type": "result", "subtype": "success", "replay": True}
    assert events.slim({**result, "usage": {}}) == result
    assert (
        events.slim({"type": "assistant", "replay": True, "message": {"content": []}})["replay"]
        is True
    )
    assert (
        events.slim(
            {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t"}]}}
        )["message"]["content"][0]["content"]
        == ""
    )
    assert events.slim({"type": "system", "subtype": "task_progress"}) is None
    assert events.slim(
        {"type": "system", "subtype": "init", "model": "m", "tools": ["Bash"], "cwd": "/"}
    ) == {"type": "system", "subtype": "init", "model": "m"}


@pytest.mark.skipif(not shutil.which("node"), reason="needs node")
def test_slim_events_fold_into_the_same_view(tmp_path):
    """web/run.js folds the slim events of the recordings into the items it folds of the whole ones: the same
    lines, and once opened (nothing cut) the same texts."""
    script = tmp_path / "check.mjs"
    script.write_text(f"""
import {{ fold }} from {json.dumps((Path(__file__).resolve().parents[1] / "src/agent_sys_v2/web/run.js").as_uri())};
import {{ setLang }} from {json.dumps((Path(__file__).resolve().parents[1] / "src/agent_sys_v2/web/i18n.js").as_uri())};
setLang('zh');
const data = JSON.parse(await new Response(process.stdin).text());
const strip = (items, cut) => items.map(({{ key, whole, wholeIn, wholeOut, chars, ...it }}) => cut
  ? {{ ...it, text: it.kind === 'thinking' ? '' : it.kind === 'note' ? it.text.slice(0, 20) : it.text, input: undefined,
      output: undefined, summary: undefined }} : it);  // a note may quote a cut command
const out = {{}};
for (const [name, r] of Object.entries(data)) {{
  const whole = fold(r.raw).items;
  out[name] = [JSON.stringify(strip(fold(r.full).items)) === JSON.stringify(strip(whole)),
               JSON.stringify(strip(fold(r.slim).items, true)) === JSON.stringify(strip(whole, true))];
}}
console.log(JSON.stringify(out));
""")
    data = {}
    for path in sorted(EVENTS.glob("*.jsonl")):
        raw = [
            {"seq": s, **json.loads(line)} for s, line in enumerate(path.read_text().splitlines())
        ]
        data[path.stem] = {
            "raw": raw,
            "full": [e for e in (events.slim(e, None) for e in raw) if e],
            "slim": [e for e in (events.slim(e, 20) for e in raw) if e],
        }
    out = subprocess.run(
        ["node", str(script)], input=json.dumps(data), capture_output=True, text=True, check=True
    )
    assert json.loads(out.stdout) == {name: [True, True] for name in data}


def test_a_run_on_another_host_is_only_shown(client, home):
    new(client, "a")
    state = read_state(ws(home, "a"))
    run_dir = ws(home, "a") / ".agent_sys/runs/20261023-090000"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "id": run_dir.name,
                "kind": "sdk",
                "pid": os.getpid(),
                "host": "elsewhere",
                "started": "2026-10-23T01:00:00Z",
                "ended": None,
            }
        )
    )
    write_state(ws(home, "a"), {**state, "status": "running", "run_id": run_dir.name, "runs": 1})
    client.app.state.store.sync(also=[("r", "develop/a")])
    for _ in range(3):  # a few rounds of the Monitor: it must not finish or kill anything
        client.app.state.store.monitor.poll()
    t = get(client, "a")
    assert (t["status"], t["detail"]["host"], t["runs"][0]["live"], t["runs"][0]["host"]) == (
        "running",
        "elsewhere",
        True,
        "elsewhere",
    )
    for action in ("stop", "takeover"):
        r = client.post(f"{T}/a/actions/{action}")
        assert r.status_code == 409 and "在 elsewhere 上运行" in r.json()["error"]


def test_prompt_change_sent_to_the_agent(client, home):
    new(client, "a")
    ok(client.post(f"{T}/a/actions/start"))
    t = wait(client, "a", lambda t: t["status"] == "needs_you")
    t = ok(
        client.put(
            f"{T}/a",
            json={"prompt": "新的要求", "base_mtime": t["prompt"]["mtime"], "send_to_agent": True},
        )
    )
    assert t["status"] == "running" and t["prompt"]["body"] == "新的要求"
    cfg = launch_json(home, t)
    assert (
        cfg["message"] == "任务 prompt 已更新（`tasks/develop/a.md`），请重新阅读并按新要求调整。"
    )
    wait(client, "a", lambda t: t["status"] != "running")


def test_a_message_with_line_separators_is_not_lost(tmp_path):
    from agent_sys_v2.runner import Inbox

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    inbox = Inbox(run_dir / "inbox.jsonl")
    texts = ["从网页粘贴的\u2028一段", "另一段\u2029", "NEL\x85", "普通"]
    for text in texts:
        runs.append_inbox(run_dir, {"type": "user", "text": text})
    assert [c["text"] for c in inbox.read()] == texts
    assert runs.unread_messages(run_dir, {}) == texts


def test_window_names_are_tmux_targets(monkeypatch):
    names = ("release_v1.0.0", "release_v1.0.0-p.1")  # a task's name, then with its project's
    monkeypatch.setattr(tmux, "window_names", lambda: set())
    assert tmux.window_name(*names) == "release_v1_0_0"
    monkeypatch.setattr(tmux, "window_names", lambda: {"release_v1_0_0", "release_v1_0_0-p_1"})
    assert tmux.window_name(*names) == "release_v1_0_0-p_1-2"


def test_a_new_session_counts_its_cost_from_zero(client, home):
    """A task whose session record is gone (cleaned up by Claude Code) starts a new session: its cost starts at 0,
    not at the old session's total."""
    new(client, "a", agent={"replay": False})  # not a replay: /bin/false for claude, no model
    w = ws(home, "a")
    state = {"status": "needs_you", "detail": {"kind": "asked"}, "runs": 2, "cost_usd": 0.5}
    write_state(w, {**read_state(w), **state, "session_cost_usd": 5.0})  # budget $1
    ok(client.post(f"{T}/a/messages", json={"text": "继续"}))
    run = sorted((w / ".agent_sys/runs").iterdir())[-1]
    cfg = json.loads((run / "launch.json").read_text())
    assert (cfg["resume"], cfg["cost_start_usd"]) == (False, 0.0)


# ---------------------------------------------------------------- the acceptance checks and the other actions


def slow_check(seconds, marker=None):
    run = f"sleep {seconds}" + (f"; touch {marker}" if marker else "")
    return [{"name": "slow", "run": run, "timeout": 120}]


def test_stop_and_a_dead_runner_end_the_check_in_flight(client, home, tmp_path):
    """停止 during the checks stops at once and kills the check; so does a runner that died. The stopped attempt does
    not count."""
    done = replay("message_mid_command", speed=40, writes={"progress.md": DONE})
    for name in ("a", "b"):
        new(client, name, agent=done, checks=slow_check(30, tmp_path / name))
        ok(client.post(f"{T}/{name}/actions/start"))
    in_check = lambda t: "Check slow" in (t["detail"].get("current") or "")  # noqa: E731
    a = wait(client, "a", in_check)
    assert procs.check_pids(run_dir_of(home, a))
    began = time.time()
    t = ok(client.post(f"{T}/a/actions/stop"))["task"]
    assert time.time() - began < 5  # not runs.STOP_WAIT and a KILL
    assert (t["status"], t["detail"], t["checks"]) == ("needs_you", {"kind": "stopped"}, None)
    assert get(client, "a")["runs"][0]["exit"] == "stopped"
    assert not procs.check_pids(run_dir_of(home, a))

    b = wait(client, "b", in_check)
    os.kill(read_json(run_dir_of(home, b) / "run.json")["pid"], signal.SIGKILL)
    assert wait(client, "b", lambda t: t["status"] != "running")["detail"] == {
        "kind": "interrupted"
    }
    assert not procs.check_pids(run_dir_of(home, b))
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


def test_checks_are_found_through_a_symlink(tmp_path):
    """The runner marks its checks with the run directory resolved; the server, which may reach the project through
    a symlink, finds them all the same (to kill them on 停止, or to leave them on 结束接管)."""
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real)
    check = subprocess.Popen(["sleep", "30"], env={**os.environ, procs.CHECK_MARK: str(real)})
    try:
        assert procs.check_pids(tmp_path / "link") == [check.pid]
    finally:
        check.kill()
        check.wait()


def test_end_takeover_leaves_the_checks_to_finish(client, home, monkeypatch):
    monkeypatch.setattr(runs, "END_TAKEOVER_WAIT", 1)
    new(client, "a", checks=slow_check(runs.END_TAKEOVER_WAIT + 3))
    ok(client.post(f"{T}/a/actions/start"))
    wait(client, "a", lambda t: t["status"] == "needs_you")
    ok(client.post(f"{T}/a/actions/takeover"))
    wait(client, "a", lambda t: t["runs"][0]["live"])
    (ws(home, "a") / "progress.md").write_text(DONE, encoding="utf-8")
    t = ok(client.post(f"{T}/a/actions/end_takeover"))["task"]
    # claude has exited; its wrapper still runs the check
    assert (t["status"], t["detail"]["kind"], t["detail"]["current"]) == (
        "running",
        "terminal",
        "Check slow",
    )
    t = wait(client, "a", lambda t: t["status"] != "running")
    assert (t["status"], t["checks"]["ok"], t["runs"][0]["exit"]) == ("done", True, "terminal_exit")


def test_a_follow_up_question_to_a_done_task_runs_no_checks(client, home):
    passes_once = 'test ! -e "$AGENT_SYS_WORKSPACE/.ran" && touch "$AGENT_SYS_WORKSPACE/.ran"'
    recording = ["message_mid_command", "message_mid_command"]
    new(
        client,
        "a",
        agent=replay(recording, speed=40, writes={"progress.md": DONE}),
        checks=[passes_once],
    )
    ok(client.post(f"{T}/a/actions/start"))
    assert wait(client, "a", lambda t: t["status"] != "running")["checks"]["ok"]
    ok(client.post(f"{T}/a/messages", json={"text": "C96 的 p90 是多少？"}))
    t = wait(client, "a", lambda t: t["status"] != "running")
    assert (t["status"], t["checks"], len(t["runs"])) == ("done", None, 2)


def test_no_run_starts_on_a_machine_another_task_holds(tmp_path):
    """A done task let its machine go; a message or a takeover would start it on the machine again."""
    home = run_home(tmp_path, "autorun: false\nmachines: [m1]\n")
    with TestClient(create_app(home)) as c:
        done = replay("message_mid_command", speed=40, writes={"progress.md": DONE})
        new(c, "a", agent=done, machines=["m1"])
        ok(c.post(f"{T}/a/actions/start"))
        wait(c, "a", lambda t: t["status"] == "done")
        new(c, "b", agent=replay("message_mid_command", speed=1), machines=["m1"])
        new(c, "c", machines=["m1"])
        ok(c.post(f"{T}/b/actions/start"))
        for name, action, body in [
            ("a", "messages", {"text": "再问一句"}),
            ("a", "actions/takeover", None),
            ("c", "actions/start", None),
        ]:
            r = c.post(f"{T}/{name}/{action}", json=body)
            assert r.status_code == 409 and "m1 正被 r/develop/b 占用" in r.json()["error"], r.text
        assert [get(c, x)["status"] for x in "abc"] == ["done", "running", "queued"]
        ok(c.post(f"{T}/b/actions/stop"))  # b waits for you: it still holds m1
        assert c.post(f"{T}/a/messages", json={"text": "再问一句"}).status_code == 409
        ok(c.post(f"{T}/b/actions/cancel"))
        assert ok(c.post(f"{T}/a/messages", json={"text": "再问一句"}))["delivery"] == "resume"
        wait(c, "a", lambda t: t["status"] != "running")


def test_a_task_whose_checks_cannot_be_read_does_not_run(client, home):
    new(client, "a")
    path = home / "projects/r/tasks/develop/a.md"
    path.write_text('---\nchecks: [{run: "false", timeout: "60"}]\n---\n做事\n', encoding="utf-8")
    r = client.post(f"{T}/a/actions/start")
    assert r.status_code == 409 and "改正后才能运行" in r.json()["error"]
    assert "timeout" in r.json()["error"]
    # YAML that cannot be read
    path.write_text("---\ntitle: [\n---\n做事\n", encoding="utf-8")
    r = client.post(f"{T}/a/actions/start")
    assert r.status_code == 409 and "改正后才能运行：frontmatter 无法解析" in r.json()["error"]
    # its card says why it waits
    t = wait(client, "a", lambda t: t["detail"].get("kind") == "invalid")
    assert t["status"] == "queued" and "frontmatter" in t["detail"]["error"]
    assert not runs_dir(ws(home, "a")).exists()


def test_a_takeover_whose_terminal_does_not_open_counts_the_cost_once(client, home, monkeypatch):
    new(client, "a", agent=replay("wait_in_background", speed=10))
    ok(client.post(f"{T}/a/actions/start"))
    wait(client, "a", lambda t: t["detail"].get("kind") == "waiting")

    def broken(*args):
        raise AgentSysError("tmux broken", 500)

    monkeypatch.setattr(tmux, "open_window", broken)
    assert client.post(f"{T}/a/actions/takeover").status_code == 500
    # a failed write is rescanned at the next poll
    t = wait(client, "a", lambda t: t["status"] != "running")
    assert (t["status"], t["detail"], len(t["runs"])) == ("needs_you", {"kind": "stopped"}, 1)
    assert t["cost_usd"] == pytest.approx(t["runs"][0]["cost_usd"]) and t["cost_usd"] > 0


def test_messages_too_late_for_a_run_go_with_the_next(client, home):
    """A message that came while the runner was exiting starts the next round; if that cannot start (no budget left)
    it waits in state.json for the next run."""
    new(client, "a", agent={"budget_usd": 0.01, **replay("message_mid_command", speed=40)})
    ok(client.post(f"{T}/a/actions/start"))
    t = wait(client, "a", lambda t: t["status"] != "running")
    # as if the message came in the moment the runner exited: after what it read, before the run was finished
    runs.append_inbox(run_dir_of(home, t), {"type": "user", "text": "还有一件事"})
    write_state(ws(home, "a"), {**read_state(ws(home, "a")), "status": "running"})
    client.app.state.store.sync(also=[("r", "develop/a")])
    wait(client, "a", lambda t: t["status"] != "running")
    assert read_state(ws(home, "a"))["unsent"] == ["还有一件事"]
    t = ok(client.post(f"{T}/a/actions/add_budget", json={"usd": 1}))["task"]
    cfg = launch_json(home, get(client, "a"))
    assert cfg["message"].startswith("还有一件事") and "预算已追加" in cfg["message"]
    assert "unsent" not in read_state(ws(home, "a"))
    wait(client, "a", lambda t: t["status"] != "running")


def test_the_agents_last_words_come_from_the_last_run_that_has_any(tmp_path):
    w = tmp_path / "ws"
    sdk, terminal = runs_dir(w) / "20261023-090000", runs_dir(w) / "20261023-100000"
    words = {"type": "assistant", "message": {"content": [{"type": "text", "text": "C96 跑完了"}]}}
    (sdk / "events.jsonl").parent.mkdir(parents=True)
    (sdk / "events.jsonl").write_text(json.dumps(words) + "\n")
    # a terminal takeover whose wrapper ran the checks: runner events only
    terminal.mkdir()
    (terminal / "events.jsonl").write_text(
        json.dumps({"type": "runner", "subtype": "checks"}) + "\n"
    )
    t = SimpleNamespace(workspace=w, state={"run_id": terminal.name})
    assert runs.last_message(t) == "C96 跑完了"


def test_a_live_run_is_read_once(tmp_path, monkeypatch):
    """A run's view reads its events once (the run view asks for it on each new event), not at all with the
    Monitor's Follow."""
    w = tmp_path / "ws"
    run_dir = runs_dir(w) / "r1"
    run_dir.mkdir(parents=True)
    runner = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
            "-m",
            "agent_sys_v2.runner",
            str(run_dir),
        ]
    )
    try:
        # until the child has exec'd, its command line is still pytest's
        deadline = time.time() + 5
        while not procs.is_runner(runner.pid, run_dir) and time.time() < deadline:
            time.sleep(0.01)
        rec = {"id": "r1", "kind": "sdk", "pid": runner.pid, "host": runs.HOST, "ended": None}
        t = SimpleNamespace(workspace=w, state={"run_id": "r1"})
        polls = []
        poll = events.Follow.poll
        monkeypatch.setattr(events.Follow, "poll", lambda f: polls.append(f) or poll(f))
        assert runs.run_json(t, rec)["live"] and len(polls) == 1
        follow = events.Follow(run_dir)
        assert runs.run_json(t, rec, follow)["activity"] == "working" and len(polls) == 1
    finally:
        runner.kill()
        runner.wait()
