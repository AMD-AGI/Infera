"""Fleet sessions from the trigger's `registry/sessions.json` rows (registry.sessions.v1), the only judge of
alive / crashed / status / lineage (review A4). Pure: the job reads the file through the door's api and passes
the rows in. History location (design §4): the row's transcript path, else `<config dir>/projects/*/<sid>.jsonl`
(cached per session id) — never derived from a slug."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from datetime import datetime

_TRANSCRIPTS: dict[str, str] = {}


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
    crashed: bool = False
    tmux: str = ""
    lineage: list[str] = field(default_factory=list)
    messaging_socket: str | None = None  # Claude Code's messagingSocketPath (pool attribution)
    launch_dir: str = ""

    @property
    def status_since_epoch(self) -> float | None:
        try:
            return datetime.fromisoformat(self.status_since.replace("Z", "+00:00")).timestamp()
        except (AttributeError, ValueError):
            return None

    @property
    def label(self) -> str:
        return self.name or self.session_id[:8]


def transcript_path(session_id: str, config_dir: str, given: str | None = None) -> str:
    if given and os.path.isfile(given):
        return given
    key = f"{config_dir}|{session_id}"
    if not os.path.isfile(_TRANSCRIPTS.get(key) or ""):
        hits = glob.glob(
            os.path.join(
                glob.escape(config_dir), "projects", "*", f"{glob.escape(session_id)}.jsonl"
            )
        )
        _TRANSCRIPTS[key] = hits[0] if len(hits) == 1 else ""
    return _TRANSCRIPTS[key]


def fleet_sessions(rows: list[dict], default_config_dir: str) -> list[Session]:
    """Main and child sessions of the registry rows, with their history located."""
    out = []
    for r in rows:
        if (
            not isinstance(r, dict)
            or r.get("role") not in ("main", "child")
            or not r.get("session_id")
        ):
            continue
        out.append(
            Session(
                session_id=r["session_id"],
                pid=r.get("pid"),
                name=r.get("name") or "",
                role=r["role"],
                parent=r.get("parent_session_id"),
                cwd=r.get("cwd") or "",
                transcript_path=transcript_path(
                    r["session_id"],
                    r.get("config_dir") or default_config_dir,
                    r.get("transcript_path"),
                ),
                status=r.get("status") or "unknown",
                waiting_for=r.get("waiting_for"),
                status_since=r.get("status_since"),
                alive=r.get("alive"),
                crashed=bool(r.get("crashed")),
                tmux=r.get("tmux") or "",
                lineage=list(r.get("lineage") or []),
                messaging_socket=r.get("messaging_socket"),
                launch_dir=r.get("launch_dir") or r.get("cwd") or "",
            )
        )
    return out
