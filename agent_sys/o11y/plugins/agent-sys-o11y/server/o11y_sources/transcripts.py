"""Claude Code transcript JSONL (`~/.claude/projects/<proj>/<sid>.jsonl` + `<sid>/subagents/*.jsonl`).

Format is internal to Claude Code and changes between versions (data.md §2), so only the fields below
are relied on, each checked:
- every line: `type`
- assistant usage: `message.id`, `message.usage.{input_tokens, cache_creation_input_tokens,
  cache_read_input_tokens, output_tokens}`; usage repeats on every content block of one message, so
  totals are summed per unique `message.id`; `message.model == "<synthetic>"` lines are error
  placeholders and carry no usage.
- compaction: `system/compact_boundary` with `compactMetadata.{trigger, preTokens, postTokens}`
- API errors: assistant lines with `isApiErrorMessage: true` (`apiErrorStatus`, `error`, text)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import Problems

USAGE_KEYS = {
    "input_tokens": "input",
    "cache_creation_input_tokens": "cache_creation",
    "cache_read_input_tokens": "cache_read",
    "output_tokens": "output",
}
NEW_KINDS = (
    "input",
    "cache_creation",
    "output",
)  # the milestone total (spec §6.2); cache reads kept apart
# Error kinds Claude Code writes on API error lines (`error`): fatal ones stop the session at once (spec §6.1).
FATAL_ERROR_KINDS = {"authentication_failed", "billing_error"}
QUOTA_WORDS = re.compile(r"quota|usage limit|spend limit|credit|exhausted", re.I)
RECENT_IDS = 64


def session_files(transcript_path: str) -> list[str]:
    """Main transcript plus subagent transcripts (session spend must include them)."""
    p = Path(transcript_path)
    files = [str(p)]
    sub = p.with_suffix("") / "subagents"
    if sub.is_dir():
        files += sorted(str(f) for f in sub.glob("*.jsonl"))
    return files


@dataclass
class Ingest:
    usage_delta: dict = field(default_factory=lambda: {v: 0 for v in USAGE_KEYS.values()})
    compactions: list[dict] = field(default_factory=list)
    api_errors: list[dict] = field(default_factory=list)
    api_error_streak: int = 0  # consecutive API error lines at the end of this chunk
    api_error_streak_max: int = 0
    last_ts: str | None = None  # last timestamp seen (activity)
    lines: int = 0


def ingest_lines(path: str, lines, extra: dict, problems: Problems) -> Ingest:
    """Consume parsed lines of one transcript file, updating `extra` (cursor state) in place."""
    usage = extra.setdefault("usage", {v: 0 for v in USAGE_KEYS.values()})
    recent = extra.setdefault("recent_ids", [])
    streak = extra.get("api_err_streak", 0)
    out = Ingest()
    bad_fields: set[str] = set()

    def bad(fld: str):
        if fld not in bad_fields:
            bad_fields.add(fld)
            problems.add("transcript", path, fld, "missing required field")

    for off, raw in lines:
        try:
            d = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(d, dict):
            continue
        out.lines += 1
        t = d.get("type")
        if t is None:
            bad("type")
            continue
        ts = d.get("timestamp")
        if ts:
            out.last_ts = ts
            if not extra.get("first_ts"):
                extra["first_ts"] = ts
        if t == "assistant":
            msg = d.get("message") if isinstance(d.get("message"), dict) else {}
            if d.get("isApiErrorMessage"):
                streak += 1
                text = ""
                content = msg.get("content")
                if isinstance(content, list) and content and isinstance(content[0], dict):
                    text = str(content[0].get("text", ""))[:300]
                kind = d.get("error")
                fatal = kind in FATAL_ERROR_KINDS or (
                    kind == "rate_limit" and bool(QUOTA_WORDS.search(text))
                )
                out.api_errors.append(
                    {
                        "ts": ts,
                        "status": d.get("apiErrorStatus"),
                        "error": kind,
                        "fatal": fatal,
                        "text": text,
                        "offset": off,
                        "uuid": d.get("uuid"),
                    }
                )
                out.api_error_streak_max = max(out.api_error_streak_max, streak)
                continue
            streak = 0
            if msg.get("model") == "<synthetic>":
                continue
            mid = msg.get("id")
            u = msg.get("usage")
            if mid is None:
                bad("message.id")
                continue
            if not isinstance(u, dict):
                bad("message.usage")
                continue
            if mid in recent:
                continue
            recent.append(mid)
            extra["messages"] = extra.get("messages", 0) + 1
            if len(recent) > RECENT_IDS:
                del recent[: len(recent) - RECENT_IDS]
            for k, name in USAGE_KEYS.items():
                v = u.get(k)
                if v is None:
                    bad(f"message.usage.{k}")
                    continue
                if isinstance(v, (int, float)):
                    usage[name] += int(v)
                    out.usage_delta[name] += int(v)
        elif t == "system" and d.get("subtype") == "compact_boundary":
            m = d.get("compactMetadata")
            if not isinstance(m, dict) or not all(
                k in m for k in ("trigger", "preTokens", "postTokens")
            ):
                bad("compactMetadata.{trigger,preTokens,postTokens}")
                m = m if isinstance(m, dict) else {}
            out.compactions.append(
                {
                    "uuid": d.get("uuid"),
                    "ts": ts,
                    "trigger": m.get("trigger"),
                    "pre_tokens": m.get("preTokens"),
                    "post_tokens": m.get("postTokens"),
                    "duration_ms": m.get("durationMs"),
                    "offset": off,
                }
            )
    extra["api_err_streak"] = streak
    out.api_error_streak = streak
    return out


def total(usage: dict) -> int:
    return sum(int(usage.get(k, 0)) for k in USAGE_KEYS.values())


def new_total(usage: dict) -> int:
    return sum(int(usage.get(k, 0)) for k in NEW_KINDS)


# ---------------------------------------------------------------- LLM slice condensation


def _short(s, n):
    s = str(s).replace("\n", " ")
    return s if len(s) <= n else s[: n - 3] + "..."


def condense(lines, max_chars: int) -> tuple[str, int]:
    """Render transcript lines as compact text for the LLM. Each entry is prefixed with its byte
    offset (`@<offset>`) so the model can cite evidence. Returns (text, offset after last used line)."""
    parts: list[str] = []
    used = 0
    end_off = None
    for off, raw in lines:
        try:
            d = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(d, dict):
            continue
        t = d.get("type")
        ts = (d.get("timestamp") or "")[:19]
        entry = None
        if t == "user" and not d.get("isMeta"):
            c = (d.get("message") or {}).get("content")
            if isinstance(c, str):
                entry = f"USER: {_short(c, 1500)}"
            elif isinstance(c, list):
                bits = []
                for b in c:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "text":
                        bits.append("USER: " + _short(b.get("text", ""), 1500))
                    elif b.get("type") == "tool_result":
                        body = b.get("content")
                        if isinstance(body, list):
                            body = " ".join(
                                str(x.get("text", "")) for x in body if isinstance(x, dict)
                            )
                        tag = "TOOL_ERROR" if b.get("is_error") else "TOOL_RESULT"
                        bits.append(f"{tag}: {_short(body, 400)}")
                entry = " | ".join(bits) if bits else None
        elif t == "assistant":
            msg = d.get("message") or {}
            if d.get("isApiErrorMessage"):
                entry = f"API_ERROR {d.get('apiErrorStatus')}"
            for b in msg.get("content") or []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text" and b.get("text"):
                    entry = "ASSISTANT: " + _short(b["text"], 1500)
                elif b.get("type") == "tool_use":
                    inp = b.get("input") or {}
                    arg = (
                        inp.get("command")
                        or inp.get("file_path")
                        or inp.get("prompt")
                        or inp.get("description")
                        or ""
                    )
                    entry = f"TOOL {b.get('name')}: {_short(arg, 300)}"
        elif t == "system":
            st = d.get("subtype")
            if st == "away_summary":
                entry = "RECAP: " + _short(d.get("content", ""), 1200)
            elif st == "compact_boundary":
                entry = "COMPACTION"
            elif st == "scheduled_task_fire":
                entry = "SCHEDULED: " + _short(d.get("prompt", ""), 200)
        if entry is None:
            end_off = off + len(raw) + 1
            continue
        line = f"@{off} {ts} {entry}"
        if used + len(line) + 1 > max_chars and parts:
            break
        parts.append(line)
        used += len(line) + 1
        end_off = off + len(raw) + 1
    return "\n".join(parts), (end_off if end_off is not None else (lines[0][0] if lines else 0))


# ---------------------------------------------------------------- usage over a fleet (shared pure rule)


@dataclass
class FileScan:
    path: str
    chunk: object  # cursor.Chunk
    ingest: Ingest


def scan_sessions(
    store, sessions, problems: Problems, prefix: str
) -> tuple[dict, dict, list[FileScan], list]:
    """Read new history of every session (main transcript + subagent files) with the caller's cursors.

    Returns (per-session usage, fleet usage, file scans, cursors to commit). Usage = sums over unique API
    messages: input, cache_creation, cache_read, output and `new` (input + cache_creation + output). The caller
    commits the returned cursors after its own outputs are written.
    """
    from .cursor import read_new

    per: dict = {}
    scans: list[FileScan] = []
    commit: list = []
    for s in sessions:
        if not s.transcript_path or not Path(s.transcript_path).exists():
            continue
        tokens = {k: 0 for k in USAGE_KEYS.values()}
        messages = 0
        first_ts = last_ts = None
        files = session_files(s.transcript_path)
        for path in files:
            key = f"{prefix}:{path}"
            chunk = read_new(store.get(key, path))
            extra = chunk.cursor.extra
            ing = ingest_lines(path, chunk.lines, extra, problems)
            if ing.last_ts:
                extra["last_ts"] = ing.last_ts
            commit.append((key, chunk.cursor))
            scans.append(FileScan(path, chunk, ing))
            for k, v in extra.get("usage", {}).items():
                tokens[k] = tokens.get(k, 0) + v
            messages += extra.get("messages", 0)
            if extra.get("first_ts") and (first_ts is None or extra["first_ts"] < first_ts):
                first_ts = extra["first_ts"]
            if extra.get("last_ts") and (last_ts is None or extra["last_ts"] > last_ts):
                last_ts = extra["last_ts"]
        per[s.session_id] = {
            "tokens": {**tokens, "new": new_total(tokens)},
            "messages": messages,
            "subagent_files": len(files) - 1,
            "name": s.name,
            "role": s.role,
            "first_at": first_ts,
            "last_at": last_ts,
            "files": files,
        }
    fleet = {k: 0 for k in [*USAGE_KEYS.values(), "new"]}
    msgs = 0
    for u in per.values():
        for k in fleet:
            fleet[k] += u["tokens"].get(k, 0)
        msgs += u["messages"]
    return per, {"tokens": fleet, "messages": msgs}, scans, commit
