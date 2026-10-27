"""A recorded events.jsonl played back as if the CLI sent it: runs without a model, for tests, demos and UI work
(agent.replay in project.yaml or a task's frontmatter; runner.py --replay)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from .events import DROP
from .i18n import _
from .model import read_text, write_text


class Replay:
    """A recorded events.jsonl played back as if the CLI sent it, at the recorded pace divided by `speed`, each gap
    at most `max_gap` seconds. The recording's own runner events are left out (this runner writes its own), and so
    is the message it opened with (this run's message takes its place); messages recorded later stay. Messages you
    send get a short reply marked as a replay, where the CLI would read them: after the command in
    flight, at the end of the round, or at once while the session waits in the background. stop() ends the round
    the way interrupt() does (recording stop_mid_command). When the recording is over the source stays open, as the CLI
    would: the runner then exits by its usual rule, or waits (a recording cut mid-round, like killed_runner). `writes`
    ({absolute path: text}) are written just before the recording's last frame, as the agent would have written
    progress.md, report.md or questions.md by the end of its round; a stopped replay writes nothing."""

    def __init__(self, path, speed=1.0, max_gap=None, session_id=None, writes=None):
        if not Path(path).is_file():
            raise FileNotFoundError(_("回放文件不存在：{path}", path=path))
        self.frames_left, opened = [], False
        for line in read_text(Path(path)).splitlines():
            e = json.loads(line)
            if e.get("subtype") in DROP:
                continue
            if e.get("type") == "runner":
                if e.get("subtype") not in ("send_user", "inbox") or not opened:
                    continue
            else:
                opened = True
            if session_id and "session_id" in e:
                e["session_id"] = session_id
            self.frames_left.append(e)
        self.speed, self.max_gap, self.writes = speed or 1.0, max_gap, writes or {}
        self.queue = asyncio.Queue()  # frames to give out now (replies, the interrupt)
        self.pending = []  # your messages: [(text, when)]
        self.fg = set()  # foreground tool_use ids in flight
        self.state, self.bg, self.started, self.stopped = None, [], False, False

    async def start(self):
        return {"replay": True}

    async def frames(self):
        last = None
        while True:
            if not self.queue.empty():
                yield self.follow(self.queue.get_nowait())
                continue
            if self.stopped or not self.frames_left:
                yield self.follow(await self.queue.get())
                continue
            e = self.frames_left[0]
            if last is not None:
                gap = max(0.0, e["ts"] - last) / self.speed
                gap = min(gap, self.max_gap) if self.max_gap else gap
                try:  # a reply or stop() may come during the gap
                    yield self.follow(await asyncio.wait_for(self.queue.get(), gap))
                    continue
                except asyncio.TimeoutError:
                    pass
            if self.stopped:
                continue
            self.frames_left.pop(0)
            last = e["ts"]
            if not self.frames_left:
                for path, text in self.writes.items():
                    write_text(Path(path), text)
            if e.get("subtype") == "session_state_changed" and e.get("state") == "idle":
                # a message sent mid-round is answered before the round ends
                for frame in self.reply("next_turn"):
                    yield self.follow(frame)
            yield self.follow({k: v for k, v in e.items() if k != "ts"})
            if e.get("type") == "user" and not self.fg:
                for frame in self.reply("after_tool"):
                    yield self.follow(frame)

    def follow(self, e):
        self.started = True
        if e.get("type") == "assistant":
            for c in e.get("message", {}).get("content") or []:
                if c.get("type") == "tool_use" and not (c.get("input") or {}).get(
                    "run_in_background"
                ):
                    self.fg.add(c["id"])
        elif e.get("type") == "user":
            for c in e.get("message", {}).get("content") or []:
                if isinstance(c, dict) and c.get("type") == "tool_result":
                    self.fg.discard(c.get("tool_use_id"))
        elif e.get("subtype") == "session_state_changed":
            self.state = e.get("state")
        elif e.get("subtype") == "background_tasks_changed":
            self.bg = e.get("tasks") or []
        return e

    def reply(self, when):
        take = ("next_turn", "after_tool") if when == "next_turn" else (when,)
        ready = [t for t, w in self.pending if w in take]
        self.pending = [(t, w) for t, w in self.pending if w not in take]
        if not ready:
            return []
        text = "\n\n".join(
            _("（回放模式，没有调用模型）收到你的消息：「{text}」。", text=t) for t in ready
        )
        frames = [
            {
                "type": "assistant",
                "replay": True,
                "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
            }
        ]
        if when != "after_tool":
            frames.append(
                {"type": "result", "subtype": "success", "replay": True, "is_error": False}
            )
        if when == "now":
            frames = [
                {"type": "system", "subtype": "session_state_changed", "state": "running"},
                *frames,
                {"type": "system", "subtype": "session_state_changed", "state": "idle"},
            ]
        return frames

    async def send(self, text):
        if not self.started:
            return  # the recording answers this run's first message
        when = "after_tool" if self.fg else "next_turn" if self.state == "running" else "now"
        self.pending.append((text, when))
        if when == "now":
            for frame in self.reply("now"):
                self.queue.put_nowait(frame)

    async def interrupt(self):
        self.stopped, self.pending = True, []
        for tid in sorted(self.fg):
            self.queue.put_nowait(
                {
                    "type": "system",
                    "subtype": "task_notification",
                    "tool_use_id": tid,
                    "status": "stopped",
                    "replay": True,
                }
            )
            self.queue.put_nowait(
                {
                    "type": "user",
                    "replay": True,
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tid,
                                "is_error": True,
                                "content": "The user doesn't want to proceed with this tool use. The tool use was rejected.",
                            }
                        ],
                    },
                }
            )
        if self.state == "running":
            self.queue.put_nowait(
                {
                    "type": "user",
                    "replay": True,
                    "message": {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "[Request interrupted by user for tool use]"}
                        ],
                    },
                }
            )
            self.queue.put_nowait(
                {
                    "type": "result",
                    "subtype": "error_during_execution",
                    "is_error": True,
                    "terminal_reason": "aborted_tools",
                    "replay": True,
                }
            )
            self.queue.put_nowait(
                {"type": "system", "subtype": "session_state_changed", "state": "idle"}
            )

    async def close(self):
        pass
