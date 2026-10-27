"""The events of a run (events.jsonl, one JSON per line, written by the runner): as the run view gets them (slim, in
pages, one whole), and as the server follows a live run (Follow)."""

from __future__ import annotations

import json
import threading

from .model import iso

# frames the runner leaves out of events.jsonl: dozens per round, or a plugin's periodic status
DROP = ("thinking_tokens", "ui_status", "ui_panes")
PREVIEW = 2000  # characters of a tool's input or result that the run view gets at first; the rest when you open it
_UNREAD = {
    "uuid",
    "session_id",
    "timestamp",
    "usage",
    "tool_use_result",
    "tool_result_meta",
    "request_id",
    "prompt",
    "run_id",
}  # fields of the other events that the run view does not read


def result_text(content):
    """A tool result as the run view shows it: its text, an image as [image]."""
    if isinstance(content, str) or content is None:
        return content or ""
    if isinstance(content, list):
        return "\n".join(
            c.get("text", "") if c.get("type") != "image" else "[image]"
            for c in content
            if isinstance(c, dict)
        )
    return json.dumps(content, ensure_ascii=False)


def _clip(value, limit, cut):
    """value with every string longer than limit cut to it; cut[0] says whether one was."""
    if isinstance(value, str) and limit is not None and len(value) > limit:
        cut[0] = True
        return value[:limit]
    if isinstance(value, dict):
        return {k: _clip(v, limit, cut) for k, v in value.items()}
    if isinstance(value, list):
        return [_clip(v, limit, cut) for v in value]
    return value


def _block(c, limit):
    """One content block of a message, slim; the blocks keep their places, so a block is found again by index."""
    if not isinstance(c, dict):
        return c
    kind, cut = c.get("type"), [False]
    if kind == "thinking":  # folded in the view: only its length until you open it
        text = c.get("thinking") or ""
        out = {
            "type": kind,
            "thinking": _clip(text, 0 if limit is not None else None, cut),
            "chars": len(text),
        }
    elif kind == "tool_use":
        out = {
            "type": kind,
            "id": c.get("id"),
            "name": c.get("name"),
            "input": _clip(c.get("input") or {}, limit, cut),
        }
    elif kind == "tool_result":
        text = result_text(c.get("content"))
        out = {
            "type": kind,
            "tool_use_id": c.get("tool_use_id"),
            "is_error": bool(c.get("is_error")),
            "content": _clip(text, limit, cut),
            "chars": len(text),
        }
    elif kind in ("image", "redacted_thinking"):
        out = {"type": kind}
    else:
        out = c
    return {**out, "clipped": True} if cut[0] else out


def slim(e, limit=PREVIEW):
    """What the run view (web/run.js, Folder) reads of an event: no ids, usage or copies it does not show, long texts
    cut to `limit` characters (None: whole) and marked clipped; None for an event it ignores."""
    kind, sub = e.get("type"), e.get("subtype")

    def pick(*keys):
        return {k: e[k] for k in ("seq", "type", "subtype", "ts", "replay", *keys) if k in e}

    if kind == "system" and sub == "task_progress":
        return None
    if kind == "system" and sub == "init":
        return pick("model", "claude_code_version", "permissionMode")
    if kind == "result":
        return pick("total_cost_usd", "is_error", "terminal_reason", "origin")
    if kind in ("assistant", "user"):
        out = {
            k: e[k]
            for k in ("seq", "type", "ts", "replay", "isSynthetic", "parent_tool_use_id")
            if e.get(k) is not None
        }
        content = (e.get("message") or {}).get("content")
        out["message"] = {
            "content": [_block(c, limit) for c in content] if isinstance(content, list) else content
        }
        return out
    return {k: v for k, v in e.items() if k not in _UNREAD}


class Lines:
    """Where each whole line of an events.jsonl starts (a line's seq is its number from 0), kept for the files read
    lately and extended as a file grows: a page of events, or one, is read without going through the whole file."""

    KEEP = 32

    def __init__(self):
        self.lock = threading.Lock()
        self.files = {}  # path -> [start of line 0, of line 1, ..., end of the last whole line]

    def starts(self, path):
        with self.lock:
            starts = self.files.pop(path, None) or [0]
            try:
                size = path.stat().st_size
                if size < starts[-1]:  # the file was written again
                    starts = [0]
                with open(path, "rb") as f:
                    f.seek(starts[-1])
                    data = f.read(size - starts[-1])
            except OSError:
                return [0]
            base, i = starts[-1], data.find(b"\n")
            while i >= 0:  # a line without its \n is still being written
                starts.append(base + i + 1)
                i = data.find(b"\n", i + 1)
            self.files[path] = starts
            while len(self.files) > self.KEEP:
                self.files.pop(next(iter(self.files)))
            return starts

    @staticmethod
    def read(path, starts, lo, hi):
        """Lines lo..hi-1 as (seq, bytes)."""
        if hi <= lo:
            return []
        with open(path, "rb") as f:
            f.seek(starts[lo])
            data = f.read(starts[hi] - starts[lo])
        return list(zip(range(lo, hi), data.splitlines()))


LINES = Lines()


def _parse(seq, line, limit=PREVIEW):
    try:
        e = json.loads(line)
    except json.JSONDecodeError:
        return None
    return slim({"seq": seq, **e}, limit) if isinstance(e, dict) else None


def read_events(run_dir, after=-1, before=None, tail=None):
    """The run view's events (slim): those after seq `after`; or with `tail`, the last `tail` before seq `before`
    (None: the end), with `carry`: the session's state and background commands as they were before the first
    (a view that starts there needs them). `upto` = the last seq in the file, `first` = where the page starts."""
    path = run_dir / "events.jsonl"
    try:
        starts = LINES.starts(path)
        n = len(starts) - 1
        if tail is None:
            lo = max(after + 1, 0)
            events = [e for s, line in Lines.read(path, starts, lo, n) if (e := _parse(s, line))]
            return {"events": events, "upto": n - 1, "first": lo, "carry": []}
        tail, hi = max(tail, 1), n if before is None else max(0, min(before, n))
        events, lo = [], hi
        # back a block at a time: the events it ignores do not count
        while lo > 0 and len(events) < tail:
            block = Lines.read(path, starts, max(0, lo - tail), lo)
            lo = block[0][0]
            events[:0] = [e for s, line in block if (e := _parse(s, line))]
        if len(events) > tail:
            events = events[-tail:]
            lo = events[0]["seq"]
        return {"events": events, "upto": n - 1, "first": lo, "carry": _carry(path, starts, lo)}
    except OSError:
        return {"events": [], "upto": -1, "first": 0, "carry": []}


def _carry(path, starts, lo):
    """The last session_state_changed and background_tasks_changed before seq lo."""
    want, found = [b'"session_state_changed"', b'"background_tasks_changed"'], []
    while lo > 0 and want:
        block = Lines.read(path, starts, max(0, lo - 500), lo)
        lo = block[0][0]
        for s, line in reversed(block):
            for w in [w for w in want if w in line]:
                if (e := _parse(s, line)) and e.get("subtype") == w.strip(b'"').decode():
                    want.remove(w)
                    found.append(e)
    return sorted(found, key=lambda e: e["seq"])


def read_event(run_dir, seq):
    """One event whole (slim, nothing cut): what a tool line or a thinking block shows when you open it."""
    path = run_dir / "events.jsonl"
    starts = LINES.starts(path)
    if not 0 <= seq < len(starts) - 1:
        return None
    return next((_parse(s, line, None) for s, line in Lines.read(path, starts, seq, seq + 1)), None)


class Follow:
    """A run as its events tell it, read incrementally: session state, background commands, the foreground command
    in flight, the agent's last words, the session's cost."""

    def __init__(self, run_dir):
        self.path = run_dir / "events.jsonl"
        self.pos, self.seq = 0, -1  # seq: of the last line read
        self.state, self.bg, self.fg = None, [], {}
        self.idle_since = None
        self.last_text, self.cost = "", None
        self.checking = None  # the acceptance check being run, as a foreground command

    def poll(self):
        try:
            with open(self.path, "rb") as f:
                f.seek(self.pos)
                data = f.read()
        except OSError:
            return self
        end = data.rfind(b"\n") + 1
        self.pos += end
        for line in data[:end].splitlines():
            self.seq += 1
            try:
                self.feed(json.loads(line))
            except json.JSONDecodeError:
                pass
        return self

    def feed(self, e):
        t, sub = e.get("type"), e.get("subtype")
        if t == "system" and sub == "session_state_changed":
            self.state = e.get("state")
            self.idle_since = e.get("ts") if self.state == "idle" else None
        elif t == "system" and sub == "background_tasks_changed":
            self.bg = e.get("tasks") or []
        elif t == "assistant":
            for c in (e.get("message") or {}).get("content") or []:
                if c.get("type") == "text" and c.get("text", "").strip():
                    self.last_text = c["text"].strip()
                elif c.get("type") == "tool_use" and not (c.get("input") or {}).get(
                    "run_in_background"
                ):
                    i = c.get("input") or {}
                    text = (
                        i.get("command")
                        or i.get("file_path")
                        or i.get("pattern")
                        or i.get("description")
                        or ""
                    )
                    self.fg[c.get("id")] = {
                        "tool": c.get("name"),
                        "text": str(text)[:200],
                        "since": iso(e.get("ts")),
                    }
        elif t == "system" and sub == "task_started" and e.get("is_backgrounded"):
            self.fg.pop(e.get("tool_use_id"), None)
        elif t == "user":
            content = (e.get("message") or {}).get("content")
            for c in content if isinstance(content, list) else []:
                if isinstance(c, dict) and c.get("type") == "tool_result":
                    self.fg.pop(c.get("tool_use_id"), None)
        elif t == "result" and isinstance(e.get("total_cost_usd"), (int, float)):
            self.cost = e["total_cost_usd"]
        elif t == "runner" and sub in ("check", "checks"):
            name, since = str(e.get("name") or "")[:200], iso(e.get("ts"))
            start = sub == "check" and e.get("phase") == "start"
            self.checking = {"tool": "Check", "text": name, "since": since} if start else None

    @property
    def activity(self):
        if self.checking:
            return "working"
        if self.state == "idle":
            return "waiting" if self.bg else "idle"
        return "working"

    @property
    def current(self):
        if self.checking:
            return self.checking
        return next(iter(self.fg.values()), None) if self.state != "idle" else None

    def delivery(self):
        """When a message sent now is read: during the checks the session is idle and takes it at once."""
        if self.checking:
            return "now"
        return (
            "after_tool" if self.current else "next_turn" if self.activity == "working" else "now"
        )
