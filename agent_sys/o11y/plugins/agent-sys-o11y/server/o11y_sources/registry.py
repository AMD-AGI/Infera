"""Fleet sessions, via o11y_core: the trigger mod's `registry/sessions.json` (registry.sessions.v1) when
present and valid, else `registry/launches.jsonl` joined with Claude Code's own live registry
`<claude config>/sessions/<pid>.json`. Only role main/child sessions are analysed."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from o11y_core import api, claude_sessions

from . import Problems


@dataclass
class Session:
    session_id: str
    pid: int | None = None
    name: str = ""
    role: str = "other"  # main | child | other
    parent: str | None = None
    cwd: str = ""
    transcript_path: str = ""
    status: str = ""  # busy | idle | waiting | shell | unknown
    waiting_for: str | None = None
    status_since: str | None = None  # RFC 3339
    alive: bool | None = None
    tmux: str = ""
    lineage: list[str] = field(default_factory=list)
    messaging_socket: str | None = (
        None  # Claude Code's messagingSocketPath (pool notify attribution)
    )

    @property
    def status_since_epoch(self) -> float | None:
        try:
            return datetime.fromisoformat(self.status_since.replace("Z", "+00:00")).timestamp()
        except (AttributeError, ValueError):
            return None

    @property
    def label(self) -> str:
        return self.name or self.session_id[:8]


def read_fleet(root: Path, cfg: dict, problems: Problems) -> list[Session]:
    cdir = cfg["sources"]["claude_config_dir"]
    _, probs = claude_sessions.read_all(Path(cdir))
    for p in probs:
        if "empty" in p or "unreadable" in p:
            continue  # empty/torn files are normal for this undocumented registry
        problems.add("claude_registry", p.split(":", 1)[0], "(file)", p.split(":", 1)[-1].strip())
    saved = os.environ.get("CLAUDE_CONFIG_DIR")
    os.environ["CLAUDE_CONFIG_DIR"] = cdir  # core reads the Claude registry from here
    try:
        rows = api.fleet_sessions(Path(root))
    finally:
        if saved is None:
            os.environ.pop("CLAUDE_CONFIG_DIR", None)
        else:
            os.environ["CLAUDE_CONFIG_DIR"] = saved
    out = []
    sockets = _messaging_sockets(Path(cdir))
    for r in rows:
        if r.get("role") not in ("main", "child"):
            continue
        out.append(
            Session(
                session_id=r["session_id"],
                pid=r.get("pid"),
                name=r.get("name") or "",
                role=r["role"],
                parent=r.get("parent_session_id"),
                cwd=r.get("cwd") or "",
                transcript_path=r.get("transcript_path") or "",
                status=r.get("status") or "unknown",
                waiting_for=r.get("waiting_for"),
                status_since=r.get("status_since"),
                alive=r.get("alive"),
                tmux=r.get("tmux") or "",
                messaging_socket=r.get("messaging_socket")
                or sockets.get((r.get("pid"), r["session_id"])),
                lineage=list(r.get("lineage") or []),
            )
        )
    return out


def _messaging_sockets(config_dir: Path) -> dict:
    """(pid, sessionId) -> messagingSocketPath from Claude Code's own `<config>/sessions/<pid>.json`."""
    out = {}
    sdir = config_dir / "sessions"
    try:
        names = os.listdir(sdir)
    except OSError:
        return out
    for name in names:
        if not name.endswith(".json"):
            continue
        try:
            d = json.loads((sdir / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and isinstance(d.get("pid"), int) and d.get("messagingSocketPath"):
            out[(d["pid"], d.get("sessionId"))] = d["messagingSocketPath"]
    return out
