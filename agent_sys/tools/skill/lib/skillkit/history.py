"""Read Claude Code's own history: list sessions, digest one into reviewable text.

Sessions live in `~/.claude/projects/<cwd, '/' and '.' -> '-'>/<sessionId>.jsonl`,
one record per line, chained by `parentUuid`. What is worth reading:

* `user` records whose content is a plain string and whose `origin.kind` is
  human — what was asked, including corrections;
* `user` records with `isCompactSummary` — the condensed past, best source
  once a session was compacted;
* `assistant` text blocks — conclusions and reports;
* `assistant` `tool_use` blocks — what was actually done (commands, files).

Skipped: sidechains (subagents), `attachment`, `api-request*`, `file-history*`,
and tool results (kept only when they are errors). Anything that looks like a
secret is redacted before it is written; the digest is meant to be read by a
model and may be pasted into a task, so a value must never reach it.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from skillkit.secretscan import redact  # noqa: E402  (re-exported for callers)

__all__ = ["redact"]


def projects_root() -> Path:
    base = os.environ.get("CLAUDE_CONFIG_DIR") or str(Path.home() / ".claude")
    return Path(base) / "projects"


def project_dir_for(cwd: str | Path) -> Path:
    """Claude Code's project dir: every non-alphanumeric character becomes `-`
    (measured: `.../agent_sys` -> `...-agent-sys`)."""
    slug = re.sub(r"[^A-Za-z0-9]", "-", str(Path(cwd).resolve()))
    path = projects_root() / slug
    if not path.is_dir():
        raise FileNotFoundError(f"no Claude Code history for {cwd} (looked for {path})")
    return path


@dataclass
class SessionInfo:
    session_id: str
    path: Path
    size: int
    title: str = ""
    first_prompt: str = ""
    start: str = ""
    end: str = ""
    cwd: str = ""
    branch: str = ""
    human_prompts: int = 0
    compactions: int = 0


def _records(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                yield rec


#: User-role records that a person did not type.
_NOT_HUMAN = (
    "<local-command",
    "<command-",
    "[Request interrupted by user",
    "Another Claude session sent a message",
    "<task-notification",
    "<system-reminder",
)
_NOT_HUMAN_INSIDE = ("<teammate-message", "<cross-session-message", "<task-notification")


def _is_human(rec: dict) -> bool:
    if rec.get("type") != "user" or rec.get("isSidechain") or rec.get("isMeta"):
        return False
    if rec.get("isCompactSummary"):
        return False
    origin = rec.get("origin") or {}
    if origin and origin.get("kind") not in (None, "human"):
        return False
    content = (rec.get("message") or {}).get("content")
    text = _text_of(content).lstrip()
    if text.startswith(_NOT_HUMAN) or any(m in text[:300] for m in _NOT_HUMAN_INSIDE):
        return False
    if isinstance(content, str):
        return True
    if isinstance(content, list):
        return any(b.get("type") == "text" for b in content if isinstance(b, dict)) and not any(
            b.get("type") == "tool_result" for b in content if isinstance(b, dict)
        )
    return False


def _text_of(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def session_info(path: Path) -> SessionInfo:
    info = SessionInfo(session_id=path.stem, path=path, size=path.stat().st_size)
    for rec in _records(path):
        ts = rec.get("timestamp")
        if ts:
            info.start = info.start or ts
            info.end = ts
        if rec.get("type") == "ai-title" and rec.get("aiTitle"):
            info.title = str(rec["aiTitle"])
        info.cwd = info.cwd or str(rec.get("cwd") or "")
        info.branch = str(rec.get("gitBranch") or info.branch)
        if rec.get("isCompactSummary"):
            info.compactions += 1
        elif _is_human(rec):
            info.human_prompts += 1
            if not info.first_prompt:
                info.first_prompt = redact(_text_of(rec["message"]["content"]))[:200]
    return info


def list_sessions(project: Path) -> list[SessionInfo]:
    files = sorted(project.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return [session_info(p) for p in files]


def resolve_sessions(project: Path, which: list[str]) -> list[Path]:
    files = sorted(project.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no sessions under {project}")
    out: list[Path] = []
    for w in which:
        if w == "latest":
            out.append(files[-1])
        elif w == "all":
            out.extend(files)
        else:
            hit = [p for p in files if p.stem.startswith(w)]
            if len(hit) != 1:
                raise FileNotFoundError(f"session {w!r} matches {len(hit)} files under {project}")
            out.append(hit[0])
    return list(dict.fromkeys(out))


@dataclass
class DigestOptions:
    since: str | None = None  # ISO prefix compare on `timestamp`
    until: str | None = None
    grep: str | None = None  # keep only entries matching (case-insensitive)
    max_text: int = 1500
    max_cmd: int = 400
    include_errors: bool = True
    tools: bool = True


@dataclass
class Digest:
    entries: list[str] = field(default_factory=list)
    files_written: dict[str, int] = field(default_factory=dict)
    commands: int = 0
    workflows: list[str] = field(default_factory=list)


def _clip(s: str, n: int) -> str:
    """Redact BEFORE cutting: a cut token prefix no longer matches a pattern."""
    s = redact(s).strip()
    return s if len(s) <= n else s[:n] + f" …[+{len(s) - n} chars]"


def digest(path: Path, opts: DigestOptions) -> Digest:
    d = Digest()
    pat = re.compile(opts.grep, re.I) if opts.grep else None

    def keep(ts: str) -> bool:
        # Prefix semantics: --until 2026-10-13 keeps all of the 13th.
        if opts.since and ts and ts[: len(opts.since)] < opts.since:
            return False
        return not (opts.until and ts and ts[: len(opts.until)] > opts.until)

    for rec in _records(path):
        if rec.get("isSidechain"):
            continue
        ts = str(rec.get("timestamp") or "")
        if not keep(ts):
            continue
        stamp = ts[:16].replace("T", " ")
        lines: list[str] = []
        if rec.get("type") == "user" and rec.get("isCompactSummary"):
            lines.append(
                f"### [{stamp}] COMPACTION SUMMARY\n{_text_of(rec['message']['content']).strip()}"
            )
        elif _is_human(rec):
            lines.append(
                f"### [{stamp}] USER\n{_clip(_text_of(rec['message']['content']), opts.max_text * 2)}"
            )
        elif rec.get("type") == "user" and opts.include_errors:
            for b in (rec.get("message") or {}).get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("is_error"):
                    body = b.get("content")
                    body = body if isinstance(body, str) else _text_of(body)
                    lines.append(f"- [{stamp}] tool error: {_clip(body, 300)}")
        elif rec.get("type") == "assistant":
            for b in (rec.get("message") or {}).get("content") or []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text" and b.get("text", "").strip():
                    lines.append(f"### [{stamp}] ASSISTANT\n{_clip(b['text'], opts.max_text)}")
                elif b.get("type") == "tool_use" and opts.tools:
                    lines.append(_tool_line(stamp, b, d, opts))
        for ln in lines:
            if not ln:
                continue
            ln = redact(ln)
            if pat and not pat.search(ln):
                continue
            d.entries.append(ln)
    return d


def _tool_line(stamp: str, block: dict, d: Digest, opts: DigestOptions) -> str:
    name = str(block.get("name", "?"))
    inp = block.get("input") or {}
    if name == "Bash":
        d.commands += 1
        desc = inp.get("description", "")
        return f"- [{stamp}] $ {_clip(str(inp.get('command', '')), opts.max_cmd)}" + (
            f"   # {desc}" if desc else ""
        )
    if name in ("Edit", "Write", "NotebookEdit", "MultiEdit"):
        fp = str(inp.get("file_path") or inp.get("notebook_path") or "?")
        d.files_written[fp] = d.files_written.get(fp, 0) + 1
        return f"- [{stamp}] {name} {fp}"
    if name in ("Read", "Glob", "Grep"):
        return f"- [{stamp}] {name} {inp.get('file_path') or inp.get('pattern') or ''}"
    if name == "Agent":
        desc = inp.get("description") or ""
        return f"- [{stamp}] Agent ({desc}): {_clip(str(inp.get('prompt') or ''), 600)}"
    if name == "Workflow":
        d.workflows.append(str(inp.get("script") or inp.get("scriptPath") or inp.get("name") or ""))
        head = inp.get("scriptPath") or inp.get("name") or f"inline script #{len(d.workflows)}"
        return f"- [{stamp}] Workflow {head}: {_clip(str(inp.get('script') or ''), 600)}"
    return f"- [{stamp}] {name}"


def render(info: SessionInfo, d: Digest) -> str:
    head = [
        f"# Session {info.session_id}",
        f"- title: {info.title or '(none)'}",
        f"- time: {info.start} .. {info.end}",
        f"- cwd: {info.cwd}   branch: {info.branch}",
        f"- human prompts: {info.human_prompts}   compactions: {info.compactions}   shell commands: {d.commands}",
        "",
        "## Files written (Edit/Write count)",
        "",
    ]
    head += [f"- {fp} ({n})" for fp, n in sorted(d.files_written.items())] or ["- (none)"]
    head += ["", "## Timeline", ""]
    return "\n".join(head + d.entries) + "\n"


def split(text: str, max_chars: int) -> list[str]:
    """Cut a digest at entry boundaries into chunks of at most `max_chars`."""
    parts, cur = [], ""
    for block in re.split(r"(?m)^(?=### \[|- \[)", text):
        if cur and len(cur) + len(block) > max_chars:
            parts.append(cur)
            cur = ""
        cur += block
    if cur:
        parts.append(cur)
    return parts
