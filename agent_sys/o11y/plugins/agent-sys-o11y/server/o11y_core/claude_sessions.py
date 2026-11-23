"""Tolerant reader of Claude Code's own live-session registry `<config>/sessions/<pid>.json` (undocumented).

Observed fields (2.1.246..2.1.325): pid, sessionId, cwd, startedAt, procStart (= /proc/<pid>/stat field 22),
version, kind, entrypoint, tmux, name, status (busy|idle|waiting|shell), waitingFor, updatedAt,
statusUpdatedAt, messagingSocketPath. Files can be empty or torn; stale files remain after SIGKILL.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path

from . import fleet

STATUSES = {"busy", "idle", "waiting", "shell"}


def config_dir() -> Path:
    return fleet.claude_config_dir()


def proc_start(pid: int) -> str | None:
    """Field 22 of /proc/<pid>/stat (start time in clock ticks), or None if the process is gone."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            stat = f.read().decode(errors="replace")
    except OSError:
        return None
    # comm (field 2) may contain spaces and ')'; split after the last ')'
    rest = stat[stat.rfind(")") + 2 :].split()
    return rest[19] if len(rest) > 19 else None


def _iso_ms(ms) -> str | None:
    if not isinstance(ms, (int, float)) or ms <= 0:
        return None
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def project_dir_name(cwd: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "-", cwd)


def transcript_path(cfg: Path, cwd: str | None, session_id: str | None) -> str | None:
    if not cwd or not session_id:
        return None
    name = project_dir_name(cwd)
    candidate = cfg / "projects" / name / f"{session_id}.jsonl"
    if candidate.exists() or len(name) <= 200:
        return str(candidate)
    # >200 chars: Claude Code truncates and appends a hash; find it by session id
    for p in (cfg / "projects").glob(f"{name[:200]}*/{session_id}.jsonl"):
        return str(p)
    return str(candidate)


def read_all(cfg: Path | None = None) -> tuple[list[dict], list[str]]:
    """Return (normalised session dicts, problems). Never raises on bad files."""
    cfg = cfg or config_dir()
    out, problems = [], []
    sdir = cfg / "sessions"
    try:
        names = sorted(os.listdir(sdir))
    except FileNotFoundError:
        return [], [f"{sdir}: missing"]
    for name in names:
        if not re.fullmatch(r"\d+\.json", name):
            continue
        path = sdir / name
        try:
            raw = path.read_text(encoding="utf-8")
            d = json.loads(raw) if raw.strip() else None
        except (OSError, ValueError) as exc:
            problems.append(f"{path}: unreadable ({exc.__class__.__name__})")
            continue
        if not isinstance(d, dict) or not isinstance(d.get("pid"), int) or not d.get("sessionId"):
            problems.append(f"{path}: empty or missing pid/sessionId")
            continue
        pid = d["pid"]
        live_start = proc_start(pid)
        recorded = d.get("procStart") or d.get("procStartFt")
        alive = live_start is not None and (recorded is None or str(recorded) == live_start)
        status = d.get("status") if d.get("status") in STATUSES else "unknown"
        out.append(
            {
                "session_id": d["sessionId"],
                "pid": pid,
                "proc_start": str(recorded) if recorded is not None else None,
                "alive": alive,
                "crashed": not alive,
                "name": d.get("name"),
                "cwd": d.get("cwd"),
                "tmux": d.get("tmux"),
                "transcript_path": transcript_path(cfg, d.get("cwd"), d["sessionId"]),
                "status": status,
                "waiting_for": d.get("waitingFor"),
                "status_since": _iso_ms(d.get("statusUpdatedAt")),
                "started_at": _iso_ms(d.get("startedAt")),
                "updated_at": _iso_ms(d.get("updatedAt")),
                "version": d.get("version"),
                "kind": d.get("kind"),
                "config_dir": str(cfg),
            }
        )
    return out, problems


def session_of_pid(pid: int, cfg: Path | None = None) -> str | None:
    """Current session id of the claude process `pid` (follows /clear, unlike CLAUDE_CODE_SESSION_ID in MCP servers)."""
    path = (cfg or config_dir()) / "sessions" / f"{pid}.json"
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d.get("sessionId") if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None
