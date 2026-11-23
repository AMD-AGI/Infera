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
from pathlib import Path

from . import Problems, owned_json, owned_problem

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
    """Current grants with their nodes, joined with the request of the SAME id (the grant file's stem): a grant
    file not ours, or whose `id` differs from its file name, is skipped and reported."""
    out = []
    gdir = Path(pool_dir) / "grants"
    for f in sorted(gdir.glob("*.json")) if gdir.is_dir() else []:
        g, why = owned_json(f, external=True)
        if why or not isinstance(g, dict) or g.get("id", f.stem) != f.stem:
            problems.add(
                "pool",
                str(f),
                "id" if not why else "(file)",
                why or "grant id differs from its file name",
            )
            continue
        if not problems.require("pool", str(f), g, GRANT_REQUIRED):
            continue
        req, _ = owned_json(Path(pool_dir) / "requests" / f"{f.stem}.json", external=True)
        ours = isinstance(req, dict)  # the request file passed the ownership rule
        req = req if ours else {}
        nodes = [
            {
                "node": n["node"],
                "jobid": str(n["jobid"]),
                "since": n.get("since"),
                "gpu_util": n.get("gpu_util"),
                "partition": n.get("partition"),
            }
            for n in g.get("nodes") or []
            if isinstance(n, dict) and problems.require("pool", str(f), n, NODE_REQUIRED)
        ]
        out.append(
            {
                "id": f.stem,
                "cluster": g.get("cluster"),
                "status": g["status"],
                "kind": g.get("kind"),
                "nodes": nodes,
                "purpose": req.get("purpose"),
                "notify": (req.get("requester") or {}).get("notify"),
                "ours": ours,
                "until": req.get("until"),
                "updated": g.get("updated"),
            }
        )
    return out


def trusted_dir(pool_dir: str | None, problems: Problems) -> bool:
    """The pool directory, its grants/ and requests/ and outbox.jsonl pass the ownership rule (someone else
    could otherwise plant records); an untrusted pool is reported once as a source problem and skipped."""
    if not pool_dir or not os.path.lexists(pool_dir):
        return False
    p = Path(pool_dir)
    checks = [
        (pool_dir, "dir"),
        (str(p / "grants"), "dir"),
        (str(p / "requests"), "dir"),
        (str(p / "outbox.jsonl"), "file"),
    ]
    why = next(
        (
            w
            for path, kind in checks
            if os.path.lexists(path) and (w := owned_problem(path, kind, external=True))
        ),
        None,
    )
    if why:
        problems.add("pool", pool_dir, "(directory)", f"not trusted: {why}")
    return why is None


def request_is_ours(pool_dir: str, req_id: str) -> bool:
    """`requests/<req_id>.json` passes the ownership rule. The request's own `requester.user` / notify text is
    self-reported and proves nothing; file ownership does."""
    if not re.fullmatch(r"[A-Za-z0-9._-]+", req_id or "") or req_id.startswith("."):
        return False
    return (
        owned_problem(str(Path(pool_dir) / "requests" / f"{req_id}.json"), "file", external=True)
        is None
    )


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
