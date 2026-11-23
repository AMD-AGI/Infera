"""Machine pool (`<pool_dir>/outbox.jsonl`, `grants/*.json`, `requests/*.json`; data.md §4).

outbox line: `{ts, req, type, msg, notify, cluster}`; `notify` = session name or
`uds:/run/user/<uid>/cc-socks/<pid>.sock`. Types mapped to events (design §8.3):
submitted -> requested (info); granted -> granted (info); released -> released (info); ended -> released
only when no `released` of the same request follows (held back one pass); lost / bad-node / idle-released ->
warn. fulfilled, done, renewing, cleared-foreign, expired and idle-warning are not reported (the first two
repeat a grant / release; idle-warning precedes idle-released).
"""

from __future__ import annotations

import json
import os
import pwd
import re
from pathlib import Path

from . import Problems

OUTBOX_REQUIRED = ["ts", "req", "type"]
GRANT_REQUIRED = ["status", "nodes"]
NODE_REQUIRED = ["node", "jobid"]

TYPE_MAP = {
    "submitted": ("requested", "INFO"),
    "granted": ("granted", "INFO"),
    "released": ("released", "INFO"),
    "ended": (
        "released",
        "INFO",
    ),  # deferred: dropped when a `released` of the same request follows
    "lost": ("lost", "WARN"),
    "bad-node": ("bad_node", "WARN"),
    "idle-released": ("idle_released", "WARN"),
}
OUTBOX_TYPES = {"ts": (str,), "req": (str,), "type": (str,)}


def parse_outbox(path: str, lines, problems: Problems) -> list[dict]:
    out = []
    for off, raw in lines:
        try:
            d = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(d, dict):
            continue
        if not problems.require("pool", path, d, OUTBOX_REQUIRED, OUTBOX_TYPES):
            continue
        d["_offset"] = off
        out.append(d)
    return out


def read_grants(pool_dir: str, problems: Problems) -> list[dict]:
    """Current grants with their nodes, joined with the request (purpose, requester.notify)."""
    out = []
    gdir = Path(pool_dir) / "grants"
    if not gdir.is_dir():
        return out
    for f in sorted(gdir.glob("*.json")):
        try:
            g = json.loads(f.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if not isinstance(g, dict) or not problems.require("pool", str(f), g, GRANT_REQUIRED):
            continue
        req = {}
        rp = Path(pool_dir) / "requests" / f.name
        try:
            req = json.loads(rp.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            pass
        nodes = []
        for n in g.get("nodes") or []:
            if isinstance(n, dict) and problems.require("pool", str(f), n, NODE_REQUIRED):
                nodes.append(
                    {
                        "node": n["node"],
                        "jobid": str(n["jobid"]),
                        "since": n.get("since"),
                        "gpu_util": n.get("gpu_util"),
                        "partition": n.get("partition"),
                    }
                )
        out.append(
            {
                "id": g.get("id") or f.stem,
                "cluster": g.get("cluster"),
                "status": g["status"],
                "kind": g.get("kind"),
                "nodes": nodes,
                "purpose": req.get("purpose"),
                "notify": (req.get("requester") or {}).get("notify"),
                "requester_user": (req.get("requester") or {}).get("user"),
                "until": req.get("until"),
                "updated": g.get("updated"),
            }
        )
    return out


def requester_user(pool_dir: str, req_id: str) -> str | None:
    """`requester.user` of `requests/<req_id>.json`, or None."""
    if not re.fullmatch(r"[A-Za-z0-9._-]+", req_id or ""):
        return None
    try:
        req = json.loads(
            (Path(pool_dir) / "requests" / f"{req_id}.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    r = req.get("requester") if isinstance(req, dict) else None
    return r.get("user") if isinstance(r, dict) else None


def current_user() -> str:
    try:
        return pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        return os.environ.get("USER", "")


def attribute(
    notify: str, req_user: str | None, sessions, uid: int | None = None, user: str | None = None
):
    """The ONE rule for "this pool record belongs to a session of this fleet" (used by o11y_log and o11y_facts).

    - The request must be this user's: a `uds:` notify path under `/run/user/<uid>/` names the uid, which must
      be ours; otherwise `requester.user` must equal our user name. Other users' requests are ignored.
    - The session must match exactly: a `uds:` notify must equal a fleet session's full messaging socket path
      (from Claude Code's registry), never just a trailing pid; a plain notify must equal a session name.
    """
    uid = os.getuid() if uid is None else uid
    user = current_user() if user is None else user
    notify = notify or ""
    if notify.startswith("uds:"):
        path = notify[4:]
        m = re.match(r"^/run/user/(\d+)/", path)
        if m:
            if int(m.group(1)) != uid:
                return None
        elif req_user != user:
            return None
        return next(
            (
                s
                for s in sessions
                if getattr(s, "messaging_socket", None) and s.messaging_socket == path
            ),
            None,
        )
    if not notify or req_user != user:
        return None
    return next((s for s in sessions if s.name and s.name == notify), None)
