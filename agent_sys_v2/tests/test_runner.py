"""runner.py: when a round is over, and a whole run in replay mode (a real runner process, no model)."""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from conftest import wait_for

from agent_sys_v2 import procs, runner
from agent_sys_v2.events import DROP

EVENTS = Path(__file__).parent / "fixtures" / "events"


def test_round_over_waits_for_the_round_the_cli_owes(monkeypatch):
    s = runner.Session()
    state = lambda x: {"type": "system", "subtype": "session_state_changed", "state": x}  # noqa: E731
    bg = lambda *ids: {  # noqa: E731
        "type": "system",
        "subtype": "background_tasks_changed",
        "tasks": [{"task_id": i, "description": i} for i in ids],
    }
    s.sent()  # the first message, before the CLI said anything
    assert not s.round_over()
    s.feed(state("running"))
    s.feed(bg("b1"))
    s.feed(state("idle"))
    assert not s.round_over()  # waiting in the background: the runner stays
    # the command ended while idle: Claude Code wakes the agent (recording wait_in_background)
    s.feed(bg())
    assert not s.round_over()
    s.feed(state("running"))
    s.feed({"type": "result", "subtype": "success", "total_cost_usd": 0.5})
    s.feed(state("idle"))
    assert s.round_over() and s.cost == 0.5 and not s.budget
    s.sent()  # a message while idle: owed until the CLI starts the round
    assert not s.round_over()
    later = time.time() + runner.OWED_SECONDS + 1
    monkeypatch.setattr(runner.time, "time", lambda: later)
    assert s.round_over()  # it never came: do not wait forever
    s.feed({"type": "result", "subtype": "error_max_budget_usd"})
    assert s.budget


def start(run_dir, recording, message="开始", **replay):
    run_dir.mkdir(parents=True)
    cfg = {
        "id": run_dir.name,
        "kind": "replay",
        "session_id": "s-1",
        "resume": False,
        "model": "",
        "message": message,
        "window": "",
        "cost_start_usd": 0.5,
        "language": "zh",
        "replay": {"events": str(EVENTS / f"{recording}.jsonl"), **replay},
    }
    (run_dir / "launch.json").write_text(json.dumps(cfg))
    return subprocess.Popen(
        [sys.executable, "-m", "agent_sys_v2.runner", str(run_dir)],
        stderr=subprocess.PIPE,
        text=True,
    )


def events(run_dir):
    return [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]


def text_of(e):
    return "".join(
        c.get("text", "")
        for c in (e.get("message") or {}).get("content") or []
        if isinstance(c, dict)
    )


def test_replay_whole_run(tmp_path):
    run_dir = tmp_path / "r1"
    proc = start(run_dir, "message_mid_command", speed=40)
    assert proc.wait(30) == 0, proc.stderr.read()
    ev = events(run_dir)
    assert ev[0]["subtype"] == "connected" and ev[1] == {
        **ev[1],
        "subtype": "send_user",
        "text": "开始",
    }
    assert not any(e.get("subtype") in DROP for e in ev)
    # the recording's opening prompt gave way to this run's message; its mid-run message stayed
    assert [e["text"][:15] for e in ev if e.get("subtype") == "send_user"] == [
        "开始",
        "NEW INSTRUCTION",
    ]
    assert ev[-1]["subtype"] == "exit" and ev[-1]["reason"] == "idle"
    rec = json.loads((run_dir / "run.json").read_text())
    assert (rec["exit"], rec["kind"], rec["cost_start_usd"], rec["cost_end_usd"]) == (
        "idle",
        "replay",
        0.5,
        0.01712215,
    )
    assert rec["pid"] == proc.pid and rec["ended"] and rec["inbox_read"] == 0


def test_replay_message_while_waiting_then_stop(tmp_path):
    run_dir = tmp_path / "r2"
    # waits 2.5 h in the background after a few seconds
    proc = start(run_dir, "wait_in_background", speed=10)
    inbox = run_dir / "inbox.jsonl"
    wait_for(
        lambda: (
            (run_dir / "events.jsonl").exists()
            and any(e.get("state") == "idle" for e in events(run_dir))
        )
    )
    rec = json.loads((run_dir / "run.json").read_text())
    assert rec["ended"] is None  # still running: waiting in the background
    with open(inbox, "a") as f:
        f.write(json.dumps({"type": "user", "text": "进展如何？"}) + "\n")
    wait_for(lambda: any("收到你的消息：「进展如何？」" in text_of(e) for e in events(run_dir)))
    states = [e["state"] for e in events(run_dir) if e.get("subtype") == "session_state_changed"]
    assert states[-2:] == ["running", "idle"]  # answered at once in a round of its own
    with open(inbox, "a") as f:
        f.write(json.dumps({"type": "stop"}) + "\n")
    assert proc.wait(20) == 0
    rec = json.loads((run_dir / "run.json").read_text())
    assert rec["exit"] == "stopped" and rec["inbox_read"] == inbox.stat().st_size
    ev = events(run_dir)
    assert [e["cmd"]["type"] for e in ev if e.get("subtype") == "inbox"] == ["user", "stop"]
    assert ev[-1]["bg"]  # the background command was still running


def test_replay_stop_mid_command(tmp_path):
    run_dir = tmp_path / "r3"
    proc = start(run_dir, "message_mid_command", speed=2)  # the first command sleeps 10 s
    wait_for(
        lambda: (
            (run_dir / "events.jsonl").exists()
            and any(e.get("subtype") == "task_started" for e in events(run_dir))
        )
    )
    with open(run_dir / "inbox.jsonl", "a") as f:
        # read when the command ends...
        f.write(json.dumps({"type": "user", "text": "只看 C96"}) + "\n")
        f.write(json.dumps({"type": "stop"}) + "\n")  # ...which it never does
    assert proc.wait(20) == 0
    ev = events(run_dir)
    result = next(e for e in ev if e.get("type") == "result")
    assert (result["subtype"], result["terminal_reason"]) == (
        "error_during_execution",
        "aborted_tools",
    )
    assert any(e.get("status") == "stopped" for e in ev)
    assert json.loads((run_dir / "run.json").read_text())["exit"] == "stopped"
    assert not any("收到你的消息" in text_of(e) for e in ev)


def test_stop_during_the_checks(tmp_path):
    """停止 while the checks run: the check in flight is killed at once, without an interrupt() of the idle session
    (the SDK may take a minute to answer it, and the server would kill the runner first), and it is not a failed
    check: no end event for it, the attempt ends as stopped."""
    run_dir, workspace = tmp_path / "run", tmp_path / "ws"
    run_dir.mkdir()
    workspace.mkdir()
    (workspace / "progress.md").write_text("状态: 已完成\n", encoding="utf-8")
    cfg = {
        "kind": "sdk",
        "session_id": "s-1",
        "resume": False,
        "message": "",
        "cwd": str(tmp_path),
        "env": {"AGENT_SYS_WORKSPACE": str(workspace)},
        "checks": [{"name": "bench", "run": "sleep 30", "timeout": 60}],
    }

    class Done:
        """A session that is idle at once, its progress.md 已完成; 停止 comes once the check runs."""

        async def start(self):
            return {}

        async def frames(self):
            yield {"type": "system", "subtype": "session_state_changed", "state": "idle"}
            while not procs.check_pids(run_dir):
                await asyncio.sleep(0.1)
            with open(run_dir / "inbox.jsonl", "a") as f:
                f.write(json.dumps({"type": "stop"}) + "\n")
            await asyncio.sleep(60)

        async def interrupt(self):
            await asyncio.sleep(60)

        async def close(self):
            pass

    began = time.time()
    assert asyncio.run(runner.drive(Done, cfg, run_dir)) == "stopped"
    assert time.time() - began < 5 and not procs.check_pids(run_dir)
    ev = [(e["subtype"], e.get("phase")) for e in events(run_dir) if e["type"] == "runner"]
    assert ev[-4:] == [("check", "start"), ("inbox", None), ("checks", "end"), ("exit", None)]
    assert events(run_dir)[-2]["stopped"] and not (run_dir / "checks.json").exists()


@pytest.mark.parametrize("recording", ["message_mid_text", "message_while_waiting"])
def test_replay_recordings_end_as_recorded(tmp_path, recording):
    run_dir = tmp_path / recording
    proc = start(run_dir, recording, speed=200, max_gap=0.2)
    assert proc.wait(30) == 0
    assert json.loads((run_dir / "run.json").read_text())["exit"] == "idle"


def test_runner_environment_keeps_the_shell_and_drops_the_parent_session(tmp_path):
    """The agent's Bash tool needs PATH and HOME; CLAUDE* / ANTHROPIC* of a parent session must not reach claude."""
    run_dir = tmp_path / "env"
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "CLAUDECODE": "1",
        "ANTHROPIC_BASE_URL": "x",
        "CLAUDE_CODE_ENTRYPOINT": "cli",
        # the package under test, also when it is not the installed one
        **{k: v for k, v in os.environ.items() if k == "PYTHONPATH"},
    }
    code = (
        "import os, json, sys; from agent_sys_v2 import runner, replay; replay.Replay.__init__ = lambda *a, **k: sys.exit("
        "json.dumps(dict(os.environ)))"
    )
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            code + '; runner.main([sys.argv[1], "--replay", "x"])',
            str(run_dir),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    seen = json.loads(out.stderr.strip().splitlines()[-1])
    assert seen["PATH"] == "/usr/bin:/bin" and seen["HOME"] == str(tmp_path)
    assert not [k for k in seen if k.startswith(("CLAUDE", "ANTHROPIC"))]
