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
import re
import stat
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
                "ours": request_is_ours(pool_dir, g.get("id") or f.stem),
                "until": req.get("until"),
                "updated": g.get("updated"),
            }
        )
    return out


def dir_problem(pool_dir: str) -> str | None:
    """Why the pool directory cannot be trusted (None = fine): not a symlink, a directory owned by this user,
    not group/other-writable (someone else could plant request files)."""
    try:
        st = os.lstat(pool_dir)
    except OSError as e:
        return f"unreadable ({e.__class__.__name__})"
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        return "not a directory (or a symlink)"
    if st.st_uid != os.getuid():
        return f"owned by uid {st.st_uid}"
    if st.st_mode & 0o022:
        return f"group/other-writable (mode {st.st_mode & 0o777:o})"
    return None


def request_is_ours(pool_dir: str, req_id: str) -> bool:
    """`requests/<req_id>.json` is a regular file (no symlink) owned by this uid. The request's own
    `requester.user` / notify text is self-reported and proves nothing; file ownership does."""
    if not re.fullmatch(r"[A-Za-z0-9._-]+", req_id or ""):
        return False
    try:
        st = os.lstat(Path(pool_dir) / "requests" / f"{req_id}.json")
    except OSError:
        return False
    return stat.S_ISREG(st.st_mode) and st.st_uid == os.getuid()


def attribute(notify: str, ours: bool, sessions, uid: int | None = None):
    """The ONE rule for "this pool record belongs to a session of this fleet" (o11y_log and o11y_facts).

    - The request must be ours: its request file is owned by this uid (`request_is_ours`); a `uds:` notify
      under `/run/user/<uid>/` must also name our uid.
    - The session must match exactly: a `uds:` notify must equal a fleet session's full messaging socket path
      (from the registry), never just a trailing pid; a plain notify must equal a session name.
    """
    uid = os.getuid() if uid is None else uid
    notify = notify or ""
    if not ours or not notify:
        return None
    if notify.startswith("uds:"):
        path = notify[4:]
        m = re.match(r"^/run/user/(\d+)/", path)
        if m and int(m.group(1)) != uid:
            return None
        return next((s for s in sessions if getattr(s, "messaging_socket", None) == path), None)
    return next((s for s in sessions if s.name and s.name == notify), None)
