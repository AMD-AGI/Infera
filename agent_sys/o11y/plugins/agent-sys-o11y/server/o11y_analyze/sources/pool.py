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
                "until": req.get("until"),
                "updated": g.get("updated"),
            }
        )
    return out
