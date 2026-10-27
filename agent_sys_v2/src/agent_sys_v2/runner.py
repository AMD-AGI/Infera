"""agent_sys runner: one run of the agent, a process of its own in a tmux window.

  python -m agent_sys_v2.runner <run_dir>                the agent, through the Claude Agent SDK
  python -m agent_sys_v2.runner <run_dir> --replay <events.jsonl> [--speed N] [--max-gap S]
                                                  a recorded events.jsonl played back, no model (replay.py);
                                                  run_dir is created if it does not exist
  python -m agent_sys_v2.runner <run_dir> --terminal     interactive claude in this terminal (在终端中接管)

The runner and the server only share files in run_dir. The runner reads launch.json (the server writes it, see
launch.py) and the new lines of inbox.jsonl: {"type": "user", "text"} is your message, sent into the session at
once; {"type": "stop", "reason"} is 停止 = interrupt(), after which the runner exits. It writes events.jsonl, one
JSON per line: every raw frame of the CLI except events.DROP, plus its own `runner` events (connected, send_user,
inbox, checks and check with a phase, exit); and run.json: the record of its start, then of its end (model.py).

A round is over when the session is idle, the latest background_tasks_changed is empty, the inbox is read and the
CLI owes no round: a message sent while the session was idle, or a background command that ended while it was idle,
starts one a moment later (recording wait_in_background: 0.1 s between the empty list and `running`). With a
background command still running the runner stays (运行中 · 等待中) and Claude Code starts the next round itself.
A round that ends with progress.md 已完成 runs the task's acceptance checks (checks.py) before the runner exits; the
inbox is read meanwhile, and 停止 kills the check in flight (no end event for it: it did not fail).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

from . import checks, i18n, procs
from .events import DROP
from .i18n import _
from .launch import session_cost, terminal_argv
from .model import Progress, iso, read_text, write_json
from .tmux import clean_own_env
from .verdict import RULES_ENV

# how long a round the CLI owes may take to start before the runner stops waiting for it
OWED_SECONDS = 10
# after interrupt(), how long the round may take to end before the runner exits anyway
STOP_SECONDS = 10
POLL = 0.3  # inbox and exit checks
CUT = 20000  # longer strings (tool output, file contents) are cut in events.jsonl


def cut(x):
    if isinstance(x, str):
        return x if len(x) <= CUT else x[:CUT] + _("…[共 {n} 字符]", n=len(x))
    if isinstance(x, list):
        return [cut(v) for v in x]
    if isinstance(x, dict):
        return {k: cut(v) for k, v in x.items()}
    return x


class Events:
    """events.jsonl, appended one whole line at a time."""

    def __init__(self, path):
        self.file = open(path, "a", encoding="utf-8")
        self.lock = threading.Lock()  # the checks write from a thread of their own

    def emit(self, frame):
        if frame.get("subtype") in DROP:
            return
        with self.lock:
            self._write(frame)

    def _write(self, frame):
        self.file.write(json.dumps({"ts": time.time(), **cut(frame)}, ensure_ascii=False) + "\n")
        self.file.flush()

    def note(self, subtype, **fields):
        self.emit({"type": "runner", "subtype": subtype, **fields})


class Record:
    """run.json: written when the run starts and when it ends."""

    def __init__(self, run_dir, cfg):
        self.path = run_dir / "run.json"
        self.data = {
            "version": 1,
            "id": run_dir.name,
            "kind": cfg["kind"],
            "resume": cfg["resume"],
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "session_id": cfg["session_id"],
            "model": cfg.get("model") or "",
            "window": cfg.get("window") or "",
            "started": iso(time.time()),
            "ended": None,
            "cost_start_usd": cfg.get("cost_start_usd") or 0.0,
            "cost_end_usd": None,
            "exit": None,
            "exit_detail": "",
            "inbox_read": 0,
        }
        self.save()

    def save(self):
        write_json(self.path, self.data)

    def end(self, exit, cost_end, detail="", inbox_read=0):
        self.data.update(
            ended=iso(time.time()),
            exit=exit,
            exit_detail=detail,
            inbox_read=inbox_read,
            cost_end_usd=self.data["cost_start_usd"] if cost_end is None else cost_end,
        )
        self.save()


class Inbox:
    """New complete lines of inbox.jsonl since the last read."""

    def __init__(self, path):
        self.path, self.pos = path, 0
        path.touch()

    def read(self):
        with open(self.path, "rb") as f:
            f.seek(self.pos)
            data = f.read()
        end = data.rfind(b"\n") + 1  # a line the server is still writing waits for the next read
        self.pos += end
        cmds = []
        for line in data[:end].decode("utf-8", "replace").splitlines():
            try:
                cmds.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return cmds


class Session:
    """What the runner follows in the frames: session state, background commands, cost, rounds the CLI owes."""

    def __init__(self):
        self.state = None  # running | idle | requires_action (session_state_changed)
        self.bg = []  # latest background_tasks_changed
        self.owed_until = None  # the CLI owes a round that should start before this time
        self.cost = None  # latest total_cost_usd: the session's total, across resumes
        self.budget = False  # the last round ended on the budget

    def feed(self, frame):
        t, sub = frame.get("type"), frame.get("subtype")
        if t == "system" and sub == "session_state_changed":
            self.state = frame.get("state")
            if self.state == "running":
                self.owed_until = None
        elif t == "system" and sub == "background_tasks_changed":
            was, self.bg = self.bg, frame.get("tasks") or []
            if was and not self.bg and self.state == "idle":
                self.owe()  # a background command ended: Claude Code wakes the agent up
        elif t == "result":
            if isinstance(frame.get("total_cost_usd"), (int, float)):
                self.cost = frame["total_cost_usd"]
            self.budget = sub == "error_max_budget_usd"

    def owe(self):
        self.owed_until = time.time() + OWED_SECONDS

    def sent(self):
        # a message sent mid-round is taken in the same round, without an idle between
        if self.state != "running":
            self.owe()

    def round_over(self):
        return (
            self.state == "idle"
            and not self.bg
            and (self.owed_until is None or time.time() > self.owed_until)
        )


# ---------------------------------------------------------------- where the frames come from


class Agent:
    """The CLI through the SDK. Frames are read raw (client._query.receive_messages(), a private interface): the
    typed one drops frames it does not know, such as tool_progress."""

    def __init__(self, cfg, run_dir):
        from claude_agent_sdk import ClaudeSDKClient

        from .launch import sdk_options

        log = open(run_dir / "stderr.log", "a", encoding="utf-8")
        self.stderr = []

        def on_stderr(line):
            log.write(line.rstrip() + "\n")
            log.flush()
            self.stderr = (self.stderr + [line.rstrip()])[-5:]

        self.client = ClaudeSDKClient(sdk_options(cfg, stderr=on_stderr))

    async def start(self):
        await self.client.connect()
        return {"claude_pid": self.client._transport._process.pid}

    def frames(self):
        return self.client._query.receive_messages()

    async def send(self, text):
        await self.client.query(text)

    async def interrupt(self):
        await self.client.interrupt()

    async def close(self):
        await self.client.disconnect()


# ---------------------------------------------------------------- one run


async def drive(make_source, cfg, run_dir):
    events = Events(run_dir / "events.jsonl")
    record = Record(run_dir, cfg)
    inbox = Inbox(run_dir / "inbox.jsonl")
    s = Session()
    reason, detail, source = None, "", None
    checking, halt = None, threading.Event()  # the checks in flight (a task), and what stops them
    try:
        source = make_source()
        info = await source.start()
        events.note("connected", **info)

        async def read():
            async for frame in source.frames():
                events.emit(frame)
                s.feed(frame)

        reader = asyncio.create_task(read())

        async def send(text):
            events.note("send_user", text=text)
            s.sent()
            await source.send(text)

        if cfg.get("message"):
            await send(cfg["message"])
        stopping, stop_by = None, 0
        while reason is None:
            await asyncio.sleep(POLL)
            for cmd in inbox.read():  # also while the checks run
                events.note("inbox", cmd=cmd)
                text = str(cmd.get("text") or "").strip()
                if cmd.get("type") == "user" and text and not stopping:
                    await send(cmd["text"])
                elif cmd.get("type") == "stop" and not stopping:
                    stopping, stop_by = cmd.get("reason") or "stopped", time.time() + STOP_SECONDS
                    if checking and not checking.done():
                        stop_checks(halt, run_dir)
                    # an idle session (running the checks, waiting in the background) has no round to interrupt,
                    # and the SDK may take a minute to answer an interrupt then
                    if s.state != "idle":
                        await source.interrupt()
            if stopping and checking and not checking.done():
                stop_checks(halt, run_dir)
            if reader.done():
                reason = "error"
                err = reader.exception()
                detail = _("claude 已退出：{error}", error=err) if err else _("claude 进程已退出")
                stderr = getattr(source, "stderr", [])
                detail += _("（stderr：{stderr}）", stderr=" / ".join(stderr)) if stderr else ""
            elif stopping:
                checks_over = not checking or checking.done()
                if (s.state == "idle" and checks_over) or time.time() > stop_by:
                    reason = stopping
            elif checking:
                if checking.done():
                    attempt, checking = checking.result(), None
                    left = cfg.get("check_retries", checks.DEFAULT_RETRIES) + 1 - attempt["n"]
                    if not attempt["ok"] and left > 0:
                        # the agent fixes it; the checks run again when it says done again
                        lang = cfg.get("agent_language") or "zh"
                        await send(checks.feedback(attempt, left, lang))
                    elif s.round_over():  # else a message came meanwhile: its round goes on
                        reason = "idle"
            elif s.budget and s.state == "idle":
                # no more rounds: a background command still running is ended with the session
                reason = "budget"
            elif s.round_over():
                if checks_due(cfg):
                    halt.clear()
                    checking = asyncio.create_task(
                        asyncio.to_thread(run_checks, cfg, run_dir, events, halt)
                    )
                else:
                    reason = "idle"
        reader.cancel()
    except Exception as e:  # noqa: BLE001 - whatever went wrong goes into the run's end record
        reason, detail = "error", f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        if checking and not checking.done():
            stop_checks(halt, run_dir)
        try:
            if source:
                await asyncio.wait_for(source.close(), 15)
        except Exception:  # noqa: BLE001 - the session's processes go away with this one anyway
            traceback.print_exc()
        reason = reason or "error"
        events.note(
            "exit",
            reason=reason,
            detail=detail,
            bg={b.get("task_id"): b.get("description") for b in s.bg},
        )
        record.end(reason, s.cost, detail, inbox.pos)
    return reason


def checks_due(cfg):
    """The round ended with progress.md 已完成 and the task has checks: time to run them, unless a done task only
    answered a follow-up question."""
    if not cfg.get("checks"):
        return False
    workspace = Path(cfg["env"]["AGENT_SYS_WORKSPACE"])
    path = workspace / "progress.md"
    return (
        path.is_file()
        and Progress.parse(read_text(path), 0).done
        and not checks.followup_only(cfg.get("before") or {}, workspace)
    )


def run_checks(cfg, run_dir, events, halt=None):
    """One attempt at the task's checks, its steps written as runner events and the outcome into checks.json. None if
    halt was set meanwhile (停止): the check in flight was killed and the attempt is not recorded."""
    # the checks' own files are as they were when the run started, or a failure says what changed
    tampered = checks.toolchain_check(
        cfg.get("toolchain"), checks.toolchain(cfg["cwd"], cfg["checks"], cfg["env"])
    )
    first = [tampered] if tampered else []
    total = len(cfg["checks"]) + len(first)
    events.note("checks", phase="start", total=total)
    for n, r in enumerate(first, 1):
        events.note("check", phase="end", n=n, total=total, **r)
    attempt = checks.run_all(
        cfg["checks"],
        cfg["cwd"],
        {**cfg["env"], **check_env(cfg), procs.CHECK_MARK: str(run_dir)},
        first=first,
        on_start=lambda n, c: events.note(
            "check", phase="start", n=n, total=total, name=c["name"], run=c["run"]
        ),
        on_end=lambda n, r: events.note("check", phase="end", n=n, total=total, **r),
        halt=halt,
    )
    if attempt is None:
        events.note("checks", phase="end", stopped=True, total=total)
        return None
    attempt["n"] = checks.record(run_dir, attempt)
    passed = sum(r["ok"] for r in attempt["results"])
    events.note(
        "checks", phase="end", ok=attempt["ok"], passed=passed, total=total, attempt=attempt["n"]
    )
    return attempt


def check_env(cfg):
    """What a check gets besides the agent's variables: the rules the run snapshotted, from this process (it read
    launch.json before the agent started; the file is in the agent's reach), and PYTHONSAFEPATH, so that
    `python -m agent_sys_v2...` run in the project directory does not import a package of that name lying there."""
    out = {"PYTHONSAFEPATH": "1"}
    if isinstance(cfg.get("ledger"), dict):
        out[RULES_ENV] = json.dumps(cfg["ledger"])
    return out


def stop_checks(halt, run_dir):
    """No other check starts; the one in flight, and what it started, is killed."""
    halt.set()
    procs.end_checks(run_dir)


def terminal(cfg, run_dir):
    """在终端中接管: interactive claude on the session, here in the tmux window. When it exits, the checks run once
    (no retries without the runner's session; 结束接管 waits for them), then the end record gets the session's total
    cost from its record in ~/.claude/projects (the CLI writes cost-state there)."""
    record = Record(run_dir, cfg)
    if cfg.get("replay"):  # a replay task has no claude session: a shell stands in for it
        argv = ["bash", "--norc", "-i"]
        print(
            _(
                "回放任务：这里是一个 shell，代替交互式 claude（不调用模型）。\n"
                "可以编辑 workspaces/{task}/progress.md 模拟 agent 的进展，然后 exit 结束接管。\n",
                task=cfg["task"],
            ),
            flush=True,
        )
    else:
        argv = terminal_argv(cfg)
    code = subprocess.call(argv, cwd=cfg["cwd"], env={**os.environ, **cfg["env"]})
    if checks_due(cfg):
        print(_("运行验收检查……"), flush=True)
        attempt = run_checks(cfg, run_dir, Events(run_dir / "events.jsonl"))
        for r in attempt["results"]:
            print(f"{'✓' if r['ok'] else '✗'} {r['name']}\n{r['output']}", flush=True)
    record.end("terminal_exit", session_cost(cfg["session_id"]), _("退出码 {code}", code=code))


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m agent_sys_v2.runner", description=__doc__.split("\n\n")[0]
    )
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--replay", type=Path, help="replay a recorded events.jsonl, no model")
    ap.add_argument(
        "--speed", type=float, help="replay speed factor (default 1, the recorded pace)"
    )
    ap.add_argument("--max-gap", type=float, help="longest gap between replayed events, in seconds")
    ap.add_argument(
        "--terminal", action="store_true", help="run interactive claude in this terminal (takeover)"
    )
    args = ap.parse_args(argv)
    # The parent Claude Code session's variables would override the gateway settings in ~/.claude; the server
    # cleans its own environment, but tmux may hand a new window an older one.
    clean_own_env()
    # an agent's bare `tmux kill-server` would otherwise reach agent_sys's tmux, every run in it
    if not args.terminal:
        for name in ("TMUX", "TMUX_PANE"):
            os.environ.pop(name, None)
    run_dir = args.run_dir.resolve()
    if (run_dir / "launch.json").is_file():
        cfg = json.loads(read_text(run_dir / "launch.json"))
    else:  # a replay by hand: python -m agent_sys_v2.runner /tmp/r1 --replay events.jsonl
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = {
            "kind": "replay",
            "session_id": str(uuid.uuid4()),
            "resume": False,
            "message": "",
            "replay": {},
        }
    i18n.set_language(cfg.get("language"))  # what the server spoke when it wrote launch.json
    if args.terminal:
        return terminal(cfg, run_dir)
    if args.replay or cfg["kind"] == "replay":
        from .replay import Replay

        r = cfg.get("replay") or {}

        def source():
            return Replay(
                args.replay or r["events"],
                args.speed or r.get("speed") or 1,
                args.max_gap or r.get("max_gap"),
                cfg["session_id"],
                r.get("writes"),
            )
    else:
        # tracebacks outlive the window
        sys.stderr = open(run_dir / "stderr.log", "a", encoding="utf-8")

        def source():
            return Agent(cfg, run_dir)

    asyncio.run(drive(source, cfg, run_dir))


if __name__ == "__main__":
    main()
