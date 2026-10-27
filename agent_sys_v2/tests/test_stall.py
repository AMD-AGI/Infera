"""疑似卡住 (events.Follow.stall, runs.Monitor, the home page's list): judged from a run's events only, never from
how long it has gone on. A recorded run with a foreground subagent (fg_subagent.jsonl, real), made-up runs for the
rest: a model call that hangs, a foreground command past its timeout, a long one under it, a subagent that goes on
and one that goes quiet, background waits, the checks, a takeover, and the reminder going away when events come."""

import json
import time
from types import SimpleNamespace

import pytest
from conftest import EVENTS, T, get, new, ok, run_home, wait
from fastapi.testclient import TestClient

from agent_sys_v2 import events, runs
from agent_sys_v2.model import iso, runs_dir
from agent_sys_v2.server import create_app

T0 = 1_793_000_000.0
QUIET = 900  # stall_minutes 15


def start(ts=T0):
    return [
        {"ts": ts, "type": "runner", "subtype": "connected"},
        {"ts": ts, "type": "system", "subtype": "session_state_changed", "state": "running"},
        {"ts": ts, "type": "system", "subtype": "init"},
    ]


def say(ts, text="…", parent=None):
    return {
        "ts": ts,
        "type": "assistant",
        "parent_tool_use_id": parent,
        "message": {"content": [{"type": "text", "text": text}]},
    }


def think(ts, parent=None):
    return {
        "ts": ts,
        "type": "assistant",
        "parent_tool_use_id": parent,
        "message": {"content": [{"type": "thinking", "thinking": "…"}]},
    }


def use(ts, id, name, parent=None, **input):
    return {
        "ts": ts,
        "type": "assistant",
        "parent_tool_use_id": parent,
        "message": {"content": [{"type": "tool_use", "id": id, "name": name, "input": input}]},
    }


def done(ts, id, parent=None):
    return {
        "ts": ts,
        "type": "user",
        "parent_tool_use_id": parent,
        "message": {"content": [{"type": "tool_result", "tool_use_id": id, "content": "ok"}]},
    }


def beat(ts, id, name):
    """The CLI's 30 s heartbeat of a tool in flight: it comes whatever the tool does."""
    return {
        "ts": ts,
        "type": "tool_progress",
        "tool_use_id": f"{id}-heartbeat-0",
        "tool_name": name,
        "parent_tool_use_id": id,
        "elapsed_time_seconds": 30,
        "heartbeat": True,
    }


def state(ts, s):
    return {"ts": ts, "type": "system", "subtype": "session_state_changed", "state": s}


def follow(tmp_path, frames, env=None):
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    if env:
        (run_dir / "launch.json").write_text(json.dumps({"env": env}), encoding="utf-8")
    f = events.Follow(run_dir)
    for e in frames:
        f.feed(e)
    return f


def test_a_model_call_that_hangs(tmp_path):
    f = follow(tmp_path, [*start(), think(T0 + 10)])
    assert f.stall(T0 + 10 + QUIET, QUIET) is None
    assert f.stall(T0 + 11 + QUIET, QUIET) == {
        "kind": "silent",
        "tool": None,
        "since": iso(T0 + 10),
    }
    # your message is not an event of the session: sent to a session that hangs, it does not end the reminder
    f.feed(
        {"ts": T0 + 950, "type": "runner", "subtype": "inbox", "cmd": {"type": "user", "text": "?"}}
    )
    f.feed({"ts": T0 + 950, "type": "runner", "subtype": "send_user", "text": "?"})
    assert f.stall(T0 + 960, QUIET)["since"] == iso(T0 + 10)
    # the model answers: the reminder is gone, and the clock starts again
    f.feed(say(T0 + 2000))
    assert f.stall(T0 + 2001, QUIET) is None and f.stall(T0 + 2000 + QUIET, QUIET) is None
    assert f.stall(T0 + 2001 + QUIET, QUIET)["since"] == iso(T0 + 2000)
    # stall_minutes: 0, off
    assert f.stall(T0 + 99999, 0) is None


def test_a_runner_whose_cli_never_answers(tmp_path):
    f = follow(tmp_path, [{"ts": T0, "type": "runner", "subtype": "connected"}])
    assert f.stall(T0 + QUIET + 1, QUIET) == {"kind": "silent", "tool": None, "since": iso(T0)}


def test_a_foreground_command_past_its_timeout(tmp_path):
    beats = [beat(T0 + s, "b1", "Bash") for s in range(30, 1200, 30)]
    f = follow(tmp_path, [*start(), use(T0, "b1", "Bash", command="make", timeout=600000), *beats])
    # the heartbeats are no sign that the command goes on: its timeout says when it is late
    assert f.stall(T0 + 600 + events.GRACE, QUIET) is None
    assert f.stall(T0 + 601 + events.GRACE, QUIET) == {
        "kind": "timeout",
        "tool": "Bash",
        "since": iso(T0),
        "timeout": 600,
    }
    f.feed(done(T0 + 1300, "b1"))  # it returns at last
    assert f.stall(T0 + 1301, QUIET) is None
    # no timeout written: the tool's default, 2 minutes, unless the run's environment sets another
    f = follow(tmp_path, [*start(), use(T0, "b2", "Bash", command="make")])
    assert f.current["timeout"] == 120 and f.stall(T0 + 120 + events.GRACE, QUIET) is None
    assert f.stall(T0 + 121 + events.GRACE, QUIET)["kind"] == "timeout"
    f = follow(
        tmp_path,
        [*start(), use(T0, "b3", "Bash", command="make")],
        env={"BASH_DEFAULT_TIMEOUT_MS": "1800000"},
    )
    assert f.current["timeout"] == 1800 and f.stall(T0 + 1800 + events.GRACE, QUIET) is None
    f = follow(tmp_path, [*start(), use(T0, "o1", "TaskOutput", task_id="x", block=True)])
    assert f.stall(T0 + 31 + events.GRACE, QUIET)["timeout"] == 30


def test_a_long_command_under_its_timeout_is_not_stuck(tmp_path):
    """Not even with no event at all for longer than stall_minutes: under its timeout a command may run."""
    f = follow(tmp_path, [*start(), use(T0, "b1", "Bash", command="sleep 590", timeout=600000)])
    assert f.stall(T0 + 590, 60) is None
    assert f.current == {
        "tool": "Bash",
        "text": "sleep 590",
        "since": iso(T0),
        "timeout": 600,
    }


def test_a_recorded_foreground_subagent(tmp_path):
    """A real run: the agent runs a subagent in the foreground, which runs two 70 s commands. The subagent's own
    frames are in events.jsonl (with parent_tool_use_id), and at no point of the recording is the run stuck, even
    with stall_minutes at half a minute. (Its commands, timed, cover the quiet spells: that the subagent's frames
    count is the next test's.)"""
    frames = [json.loads(line) for line in (EVENTS / "fg_subagent.jsonl").read_text().splitlines()]
    agent = next(
        c
        for e in frames
        if e.get("type") == "assistant"
        for c in e["message"]["content"]
        if c.get("type") == "tool_use"
    )
    assert agent["name"] == "Agent" and agent["input"]["run_in_background"] is False
    own = [
        e
        for e in frames
        if e.get("parent_tool_use_id") == agent["id"] and e["type"] != "tool_progress"
    ]
    assert {e["type"] for e in own} == {"assistant", "user"}
    f, inside = follow(tmp_path, []), set()
    for e in frames:
        assert f.stall(e["ts"] - 1e-3, 31) is None, e
        f.feed(e)
        if (
            len(f.fg) > 1
        ):  # the subagent's command, in flight within it: the card names the subagent
            inside.add(
                (f.current["tool"], f.current["timeout"], *[x["tool"] for x in f.fg.values()])
            )
    assert inside == {("Agent", None, "Agent", "Bash")}


def test_a_foreground_subagent_that_goes_on_and_one_that_goes_quiet(tmp_path):
    frames = [*start(), use(T0, "a1", "Agent", description="survey", prompt="…")]
    for m in range(10, 60, 10):  # a frame of its own every 10 minutes, heartbeats between
        frames += [beat(T0 + 60 * m - 30 * k, "a1", "Agent") for k in range(1, 20)]
        frames += [think(T0 + 60 * m, parent="a1")]
    f = follow(tmp_path, [])
    for e in sorted(frames, key=lambda e: e["ts"]):
        assert f.stall(e["ts"] - 1e-3, QUIET) is None
        f.feed(e)
    # then only the heartbeats for 16 minutes: the subagent is as stuck as a model call would be
    for s in range(30, 16 * 60, 30):
        f.feed(beat(T0 + 3000 + s, "a1", "Agent"))
    assert f.stall(T0 + 3000 + 16 * 60, QUIET) == {
        "kind": "silent",
        "tool": "Agent",
        "since": iso(T0 + 3000),
    }
    f.feed(use(T0 + 4000, "s1", "Read", parent="a1", file_path="x"))  # it moves again
    assert f.stall(T0 + 4001, QUIET) is None


def test_background_work_is_not_stuck(tmp_path):
    # waiting in the background: the session is idle
    bg = {
        "ts": T0 + 1,
        "type": "system",
        "subtype": "background_tasks_changed",
        "tasks": [{"id": "x"}],
    }
    f = follow(
        tmp_path,
        [*start(), use(T0, "b1", "Bash", command="sleep 9999", run_in_background=True), bg],
    )
    f.feed(state(T0 + 2, "idle"))
    assert f.stall(T0 + 5 * 3600, QUIET) is None and f.activity == "waiting"
    # a turn that ends with subagents in the background: the session stays running until they end, their frames
    # come all the while, and none of it is a model call that hangs
    frames = [
        *start(),
        use(T0, "a1", "Agent", description="bg", prompt="…"),
        {
            "ts": T0,
            "type": "system",
            "subtype": "task_started",
            "tool_use_id": "a1",
            "is_backgrounded": True,
        },
        say(T0 + 5, "started it"),
        {"ts": T0 + 5, "type": "result", "subtype": "success"},
        {"ts": T0 + 6, "type": "system", "subtype": "vcs_state_changed"},
    ]
    for s in range(60, 7200, 60):
        frames += [
            use(T0 + s, f"r{s}", "Bash", parent="a1", command="ls"),
            done(T0 + s + 1, f"r{s}", parent="a1"),
            {"ts": T0 + s, "type": "system", "subtype": "task_progress", "tool_use_id": "a1"},
        ]
    (tmp_path / "2").mkdir()
    f = follow(tmp_path / "2", [])
    for e in frames:
        assert f.stall(e["ts"] - 1e-3, QUIET) is None
        f.feed(e)
    assert f.current is None  # the background subagent's commands are not in the foreground
    # the next turn's model call hangs: the background subagent's frames do not hide it
    f.feed(think(T0 + 7200))
    for s in range(60, 1200, 60):
        f.feed(use(T0 + 7200 + s, f"q{s}", "Bash", parent="a1", command="ls"))
        f.feed(
            {"ts": T0 + 7200 + s, "type": "system", "subtype": "task_progress", "tool_use_id": "a1"}
        )
    assert f.stall(T0 + 7200 + QUIET + 1, QUIET) == {
        "kind": "silent",
        "tool": None,
        "since": iso(T0 + 7200),
    }


def test_frames_that_may_be_a_background_subagents(tmp_path):
    """Frames with nothing that says whose they are, or that a background subagent caused, neither start a turn nor
    show the main session going on (recorded: a subagent's api_retry with no agent id, a denied command of one, a
    commit it made)."""
    bg = [
        *start(),
        use(T0, "a1", "Agent", description="bg", prompt="…"),
        {
            "ts": T0,
            "type": "system",
            "subtype": "task_started",
            "tool_use_id": "a1",
            "is_backgrounded": True,
        },
        say(T0 + 5, "started it"),
        {"ts": T0 + 5, "type": "result", "subtype": "success"},
        use(T0 + 60, "r1", "Bash", parent="a1", command="rm -rf out"),
    ]
    retry = {
        "ts": T0 + 61,
        "type": "system",
        "subtype": "api_retry",
        "attempt": 1,
        "error_status": 502,
    }
    f = follow(tmp_path, [*bg, retry])
    assert f.stall(T0 + 3600, QUIET) is None  # the turn is still over
    # the main session's model call hangs while the subagent commits and is denied a command
    f.feed(think(T0 + 4000))
    for s in range(60, 3600, 600):
        f.feed(
            {"ts": T0 + 4000 + s, "type": "system", "subtype": "vcs_state_changed", "branch": "x"}
        )
        f.feed(
            {"ts": T0 + 4001 + s, "type": "system", "subtype": "code_change_published", "url": "u"}
        )
        f.feed(use(T0 + 4002 + s, f"d{s}", "Bash", parent="a1", command="rm x"))
        f.feed(
            {
                "ts": T0 + 4002 + s,
                "type": "system",
                "subtype": "permission_denied",
                "tool_use_id": f"d{s}",
            }
        )
    assert f.stall(T0 + 4000 + QUIET + 1, QUIET)["since"] == iso(T0 + 4000)


def test_a_default_timeout_set_where_the_cli_reads_it(tmp_path, monkeypatch):
    """BASH_DEFAULT_TIMEOUT_MS in the settings the CLI reads (your ~/.claude when it inherits them, its own
    directory's): the largest one found counts, so a healthy command is not late."""
    from agent_sys_v2 import claude_config

    own, operator = tmp_path / "own", tmp_path / "operator"
    for d, ms in ((own, 900000), (operator, 1800000)):
        d.mkdir()
        (d / "settings.json").write_text(json.dumps({"env": {"BASH_DEFAULT_TIMEOUT_MS": str(ms)}}))
    monkeypatch.setattr(claude_config, "OPERATOR_DIR", operator)
    frames = [*start(), use(T0, "b1", "Bash", command="make")]
    f = follow(tmp_path, frames, env={"CLAUDE_CONFIG_DIR": str(own)})
    assert f.current["timeout"] == 1800 and f.stall(T0 + 1800 + events.GRACE, QUIET) is None
    monkeypatch.setattr(claude_config, "OPERATOR_DIR", tmp_path / "none")
    (tmp_path / "run2").mkdir()
    f = events.Follow(tmp_path / "run2")
    (tmp_path / "run2/launch.json").write_text(json.dumps({"env": {"CLAUDE_CONFIG_DIR": str(own)}}))
    for e in frames:
        f.feed(e)
    assert f.current["timeout"] == 900


def test_not_during_the_checks(tmp_path):
    f = follow(
        tmp_path,
        [
            *start(),
            say(T0 + 1),
            {"ts": T0 + 1, "type": "result", "subtype": "success"},
            state(T0 + 2, "idle"),
            {"ts": T0 + 3, "type": "runner", "subtype": "checks", "phase": "start", "total": 1},
            {
                "ts": T0 + 3,
                "type": "runner",
                "subtype": "check",
                "phase": "start",
                "n": 1,
                "name": "x",
            },
        ],
    )
    assert f.stall(T0 + 3 * 3600, QUIET) is None and f.current["tool"] == "Check"


def monitor_task(tmp_path, frames, kind="sdk", config=None):
    """A task whose current run has these events, as runs.Monitor sees it (the runner itself is not needed)."""
    home, w = tmp_path / "home", tmp_path / "ws"
    home.mkdir(parents=True, exist_ok=True)
    if config is not None:
        (home / "agent_sys.yaml").write_text(config, encoding="utf-8")
    run_dir = runs_dir(w) / "r1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(
        json.dumps({"id": "r1", "kind": kind, "host": runs.HOST, "ended": None, "window": "w"})
    )
    (run_dir / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in frames))
    m = runs.Monitor(SimpleNamespace(home=home))
    return m, SimpleNamespace(workspace=w, state={"run_id": "r1"}, detail={}, cost_running=None)


def test_the_monitor_marks_the_task_and_reads_stall_minutes(tmp_path):
    ago = time.time() - 20 * 60
    m, t = monitor_task(tmp_path, [*start(ago), think(ago + 1)])
    m.annotate_task(t)
    assert t.detail == {
        "kind": "working",
        "stall": {"kind": "silent", "tool": None, "since": iso(ago + 1)},
    }
    # stall_minutes from agent_sys.yaml, read again when it changes
    (tmp_path / "home/agent_sys.yaml").write_text("stall_minutes: 30\n", encoding="utf-8")
    m.annotate_task(t)
    assert "stall" not in t.detail
    (tmp_path / "home/agent_sys.yaml").write_text("stall_minutes: 0\n", encoding="utf-8")
    m.annotate_task(t)
    assert "stall" not in t.detail
    # a long command under its timeout: how long it has run and its timeout, for 已运行 N 分钟（超时 M 分钟）
    m, t = monitor_task(
        tmp_path / "2", [*start(ago), use(ago + 1, "b1", "Bash", command="make", timeout=1800000)]
    )
    m.annotate_task(t)
    assert t.detail == {
        "kind": "working",
        "current": "Bash make",
        "current_since": iso(ago + 1),
        "current_timeout": 1800,
    }


def test_not_for_a_run_on_another_host(tmp_path):
    ago = time.time() - 3 * 3600
    m, t = monitor_task(tmp_path, [*start(ago), think(ago + 1)])
    run = runs_dir(tmp_path / "ws") / "r1" / "run.json"
    run.write_text(json.dumps({**json.loads(run.read_text()), "host": "elsewhere"}))
    m.annotate_task(t)
    assert t.detail == {"kind": "working", "host": "elsewhere"}


def test_not_during_a_takeover(tmp_path):
    ago = time.time() - 3 * 3600
    m, t = monitor_task(tmp_path, [*start(ago), think(ago + 1)], kind="terminal")
    m.annotate_task(t)
    assert t.detail["kind"] == "terminal" and "stall" not in t.detail


def test_the_task_needs_you_and_stays_running(tmp_path, monkeypatch):
    """Through the server, with a real runner replaying a run whose model call takes 4 s and stall_minutes at
    a second and a half: the task is listed on the home page with what needs you, stays running, and leaves the
    list when the answer comes."""
    from agent_sys_v2 import server

    monkeypatch.setattr(server, "POLL_SECONDS", 0.2)
    monkeypatch.setattr(server, "RUN_POLL", 0.2)
    home = run_home(tmp_path, "autorun: false\n")
    (home / "agent_sys.yaml").write_text("stall_minutes: 0.025\n", encoding="utf-8")
    rec = tmp_path / "slow_answer.jsonl"
    rec.write_text(
        "".join(
            json.dumps(e) + "\n"
            for e in [
                *start(T0),
                think(T0 + 0.5),
                say(T0 + 4.5, "done"),
                {"ts": T0 + 4.5, "type": "result", "subtype": "success", "total_cost_usd": 0.01},
                state(T0 + 4.6, "idle"),
            ]
        )
    )
    with TestClient(create_app(home)) as client:
        new(client, "slow", agent={"replay": {"events": str(rec)}})
        ok(client.post(f"{T}/slow/actions/start"))
        t = wait(client, "slow", lambda t: "stall" in t["detail"], 10)
        assert t["status"] == "running" and t["detail"]["stall"]["kind"] == "silent"
        home_page = ok(client.get("/api/home"))
        assert [x["id"] for x in home_page["stalled"]] == ["develop/slow"]
        assert home_page["needs_you"] == []
        # the answer comes: the reminder goes, nothing was stopped or sent
        t = wait(
            client, "slow", lambda t: t["status"] != "running" or "stall" not in t["detail"], 10
        )
        assert ok(client.get("/api/home"))["stalled"] == []
        t = wait(client, "slow", lambda t: t["status"] != "running", 10)
        assert t["status"] == "needs_you" and t["detail"] == {"kind": "asked"}
        assert t["runs"][0]["exit"] == "idle"
        assert get(client, "slow")["last_message"] == "done"


@pytest.mark.parametrize("value", [-1, "x", True, None])
def test_a_bad_stall_minutes_in_the_file_takes_the_default(tmp_path, value):
    m, _ = monitor_task(tmp_path, [], config=f"stall_minutes: {json.dumps(value)}\n")
    assert m.quiet() == 15 * 60
