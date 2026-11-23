"""The agent-door operations over the protocol files. Shared by the MCP server, the CLI and the analyzer."""

from __future__ import annotations

import copy
import hashlib
import os
import shlex
from pathlib import Path

from . import claude_sessions, fleet, schemas, store

SEVERITY = {"INFO": 9, "WARN": 13, "ERROR": 17}
SEVERITY_ALIASES = {"info": "INFO", "warning": "WARN", "warn": "WARN", "error": "ERROR"}
IDENTITY = None  # loaded lazily from runs.v1 x-identity


class Context:
    """Who is writing and where. Built once per process (MCP server) or per CLI call."""

    def __init__(
        self,
        root: Path | None = None,
        session_id: str | None = None,
        role: str | None = None,
        session_name: str | None = None,
    ):
        self.root = Path(root) if root else store.default_root()
        self.session_id = session_id or self._detect_session()
        self.role = role or self._detect_role()
        self.session_name = session_name

    @staticmethod
    def _detect_session() -> str:
        # An MCP server keeps the id it was spawned with after /clear, and CLAUDE_PID is not set for MCP
        # servers; the registry entry of the parent claude process (ppid) follows /clear, so prefer it.
        sid = claude_sessions.session_of_pid(os.getppid(), fleet.claude_config_dir())
        return sid or os.environ.get("CLAUDE_CODE_SESSION_ID") or "unknown"

    def _detect_role(self) -> str:
        """The fleet registry if this session is listed, else AGENT_SYS_O11Y, else other (design §6)."""
        doc = store.read_json(self.root / store.SESSIONS)
        if isinstance(doc, dict):
            for s in doc.get("sessions") or []:
                if (
                    isinstance(s, dict)
                    and s.get("session_id") == self.session_id
                    and s.get("role") in ("main", "child")
                ):
                    return s["role"]
        flag = os.environ.get(fleet.ACTIVATION_ENV_VAR)
        return {"true": "main", "child": "child"}.get(flag or "", "other")

    def writer(self, source: str = "written", origin: list | None = None) -> dict:
        w = {"session_id": self.session_id, "role": self.role, "source": source}
        if self.session_name:
            w["session_name"] = self.session_name
        if origin:
            w["from"] = origin
        return w


def _fail(problems: list[str], hint: str) -> dict:
    return {"ok": False, "problems": problems, "hint": hint}


# ------------------------------------------------------------------------------------------ ledger
def ledger_append(ctx: Context, record: dict) -> dict:
    """Validate and append one runs.v1 record. Server sets v, id, ts, writer (keeps caller's source/from)."""
    rec = copy.deepcopy(record)
    for k in ("v", "id", "ts"):
        rec.pop(k, None)
    given = rec.pop("writer", None) or {}
    source = (
        given.get("source")
        if given.get("source") in ("written", "extracted", "detected")
        else "written"
    )
    rec = {
        "v": 1,
        "kind": rec.pop("kind", None),
        "id": store.new_id("rec"),
        "ts": store.now(),
        "writer": ctx.writer(source, given.get("from")),
        **rec,
    }
    probs = schemas.problems("runs.v1", rec)
    if probs:
        return _fail(probs, "Fix these fields and call ledger_append again with the whole record.")
    try:
        store.append(ctx.root / store.LEDGER, rec)
    except store.LineTooLong as exc:
        return _fail([str(exc)], "Shorten long text fields (note, conclusion) and retry.")
    return {"ok": True, "id": rec["id"], "kind": rec["kind"], "ts": rec["ts"]}


def identity(rec: dict) -> tuple | None:
    """The supersede key of a ledger record (runs.v1 `x-identity`): a later valid record with the same key
    replaces the earlier one. None for kinds without an identity."""
    global IDENTITY
    if IDENTITY is None:
        IDENTITY = schemas.load("runs.v1").get("x-identity", {})
    keys = IDENTITY.get(rec.get("kind"))
    if not keys:
        return None
    vals = []
    for k in keys:
        if "|" in k:  # e.g. "at|ts"
            vals.append(next((rec.get(a) for a in k.split("|") if rec.get(a)), None))
        else:
            vals.append(rec.get(k))
    return (rec.get("kind"), *vals)


def read_ledger(root: Path, latest_only: bool = True) -> tuple[list[dict], int]:
    """Valid ledger records in file order; with latest_only, a later record with the same identity replaces
    the earlier one (kept at the later position). Returns (records, invalid_line_count)."""
    recs, invalid = [], 0
    v = schemas.validator("runs.v1")
    for _, rec, _ in store.read_jsonl(root / store.LEDGER):
        if rec is None or not v.is_valid(rec):
            invalid += 1
            continue
        recs.append(rec)
    if not latest_only:
        return recs, invalid
    keys = [identity(r) for r in recs]
    last = {k: i for i, k in enumerate(keys) if k is not None}
    keep = [r for i, (r, k) in enumerate(zip(recs, keys)) if k is None or last[k] == i]
    return keep, invalid


_identity = identity  # deprecated alias (o11y_analyze before the port); remove after one release


def ledger_query(
    ctx: Context,
    kind: str | None = None,
    run_id: str | None = None,
    task_id: str | None = None,
    since: str | None = None,
    limit: int = 20,
    latest_only: bool = True,
) -> dict:
    recs, invalid = read_ledger(ctx.root, latest_only)
    if kind:
        recs = [r for r in recs if r.get("kind") == kind]
    if run_id:
        recs = [r for r in recs if r.get("run_id") == run_id]
    if task_id:
        recs = [r for r in recs if r.get("task_id") == task_id]
    if since:
        recs = [r for r in recs if (r.get("at") or r.get("ts", "")) >= since]
    limit = max(1, min(int(limit or 20), 200))
    return {"ok": True, "count": len(recs), "records": recs[-limit:], "invalid_lines": invalid}


# ------------------------------------------------------------------------------------------ events
def event_report(
    ctx: Context,
    severity: str,
    event_name: str,
    body: str,
    attributes: dict | None = None,
    source: str = "agent",
    observed_ts: str | None = None,
    origin: list | None = None,
    writer_source: str = "written",
) -> dict:
    sev = SEVERITY_ALIASES.get(str(severity).lower(), str(severity).upper())
    name = event_name if str(event_name).startswith("o11y.") else f"o11y.{event_name}"
    attrs = dict(attributes or {})
    attrs.setdefault("o11y.source", source)
    attrs.setdefault("session.id", ctx.session_id)
    if not attrs.get(
        "o11y.dedup_key"
    ):  # an agent's report is its own fact: key it by who, what and the text
        digest = hashlib.sha1(f"{ctx.session_id}|{name}|{body}".encode()).hexdigest()[:16]
        attrs["o11y.dedup_key"] = f"agent:{digest}"
    rec = {
        "id": store.new_id("evt"),
        "v": 1,
        "ts": store.now(),
        "severity_text": sev,
        "severity_number": SEVERITY.get(sev, 0),
        "event_name": name,
        "body": body,
        "attributes": attrs,
        "writer": ctx.writer(writer_source, origin),
    }
    if observed_ts:
        rec["observed_ts"] = observed_ts
    probs = schemas.problems("event.v1", rec)
    if probs:
        return _fail(probs, "Fix these fields and call event_report again.")
    try:
        store.append(ctx.root / store.EVENTS, rec)
    except store.LineTooLong as exc:
        return _fail([str(exc)], "Shorten body/attributes and retry.")
    return {"ok": True, "id": rec["id"]}


# ------------------------------------------------------------------------------------------ sessions
def session_register(
    ctx: Context,
    name: str,
    role: str = "child",
    tmux_target: str | None = None,
    purpose: str | None = None,
    cwd: str | None = None,
    session_id: str | None = None,
) -> dict:
    """main: register this session. child: a session about to be launched (returns its env and launch
    command). declined: an existing session (session_id) the main says is not part of the fleet (spec §3.5);
    a child that already runs is registered with its session_id."""
    rec = {
        "v": 1,
        "id": store.new_id("launch"),
        "ts": store.now(),
        "role": role,
        "name": name,
        "fleet_dir": str(ctx.root),
        "writer": ctx.writer(),
    }
    if role == "main":
        rec["session_id"] = ctx.session_id
    else:
        rec["parent_session_id"] = ctx.session_id
        if session_id:
            rec["session_id"] = session_id
    for k, val in (("tmux_target", tmux_target), ("purpose", purpose), ("cwd", cwd)):
        if val:
            rec[k] = val
    probs = schemas.problems("registry.launch.v1", rec)
    if probs:
        return _fail(probs, "Fix these fields and call session_register again.")
    store.append(ctx.root / store.LAUNCHES, rec)
    out = {"ok": True, "id": rec["id"], "fleet_dir": str(ctx.root)}
    if role == "child" and not session_id:
        env = {
            fleet.ACTIVATION_ENV_VAR: "child",
            fleet.FLEET_DIR_ENV_VAR: str(ctx.root),
            fleet.PARENT_ENV_VAR: ctx.session_id,
        }
        out["env"] = env
        target = tmux_target or name
        flags = " ".join(f"-e {k}={shlex.quote(v)}" for k, v in env.items())
        out["launch_example"] = (
            f"tmux new-session -d -s {shlex.quote(target)} -c {shlex.quote(cwd or os.getcwd())} "
            f"{flags} claude --name {shlex.quote(name)}"
        )
    return out


def launches(root: Path) -> list[dict]:
    v = schemas.validator("registry.launch.v1")
    return [
        r for _, r, _ in store.read_jsonl(root / store.LAUNCHES) if r is not None and v.is_valid(r)
    ]


def fleet_sessions(root: Path) -> list[dict]:
    """Fleet = sessions named in launches (main by session id, children by name/parent) joined with the live
    Claude registry. Falls back to registry/sessions.json written by the trigger mod when present."""
    doc = store.read_json(root / store.SESSIONS)
    if isinstance(doc, dict) and schemas.validator("registry.sessions.v1").is_valid(doc):
        return doc["sessions"]
    live, _ = claude_sessions.read_all(fleet.claude_config_dir())
    by_id = {s["session_id"]: s for s in live}
    by_name = {s["name"]: s for s in live if s.get("name")}
    out, seen = [], set()
    for rec in launches(root):
        if rec["role"] == "declined":
            continue
        s = by_id.get(rec.get("session_id")) or (
            None if rec["role"] == "main" else by_name.get(rec["name"])
        )
        if s is None or s["session_id"] in seen:
            continue
        seen.add(s["session_id"])
        out.append(
            {
                **s,
                "role": rec["role"],
                "parent_session_id": rec.get("parent_session_id"),
                "purpose": rec.get("purpose"),
            }
        )
    return out


def fleet_status(ctx: Context) -> dict:
    sessions = fleet_sessions(ctx.root)
    recs, invalid = read_ledger(ctx.root)
    task = next((r for r in reversed(recs) if r["kind"] == "task"), None)
    best = next((r for r in reversed(recs) if r["kind"] == "best"), None)
    ended = {r["run_id"] for r in recs if r["kind"] == "run_end"}
    open_runs = [
        {"run_id": r["run_id"], "theme": r["theme"], "since": r.get("at") or r["ts"]}
        for r in recs
        if r["kind"] == "run_start" and r["run_id"] not in ended
    ]
    plan = next((r for r in reversed(recs) if r["kind"] == "plan"), None)
    events = [e for _, e, _ in store.read_jsonl(ctx.root / store.EVENTS) if isinstance(e, dict)]
    alerts = [
        {
            "ts": e.get("ts"),
            "severity": e.get("severity_text"),
            "event": e.get("event_name"),
            "body": e.get("body"),
        }
        for e in events
        if e.get("severity_text") in ("WARN", "ERROR")
    ][-5:]
    triggers = (
        sorted((ctx.root / "triggers").glob("*.json")) if (ctx.root / "triggers").is_dir() else []
    )
    last_trigger = store.read_json(triggers[-1]) if triggers else None
    facts = {
        t: (store.read_json(ctx.root / p) or {}).get("generated_at") for t, p in store.FACTS.items()
    }
    return {
        "ok": True,
        "fleet_dir": str(ctx.root),
        "me": {"session_id": ctx.session_id, "role": ctx.role},
        "sessions": [
            {
                k: s.get(k)
                for k in (
                    "session_id",
                    "name",
                    "role",
                    "status",
                    "waiting_for",
                    "status_since",
                    "alive",
                    "tmux",
                    "purpose",
                )
            }
            for s in sessions
        ],
        "task": {k: task.get(k) for k in ("task_id", "title", "metric", "baseline", "target")}
        if task
        else None,
        "best": {k: best.get(k) for k in ("value", "run_ids", "stack")} if best else None,
        "open_runs": open_runs,
        "next": plan["next"] if plan else [],
        "recent_alerts": alerts,
        "last_trigger": {k: last_trigger.get(k) for k in ("ts", "kind", "outcome")}
        if isinstance(last_trigger, dict)
        else None,
        "facts_generated_at": facts,
        "ledger_invalid_lines": invalid,
    }


# ------------------------------------------------------------------------------------------ config
def load_config(root: Path, which: str) -> tuple[dict, list[str]]:
    """Config merged over schema defaults. Invalid file → defaults + problems (never raises)."""
    name = f"config.{which}.v1"
    base = schemas.defaults(name)
    doc = store.read_json(root / store.CONFIG[which])
    if doc is None:
        return base, []
    probs = schemas.problems(name, doc)
    if probs:
        return base, [f"{root / store.CONFIG[which]}: {p}" for p in probs]

    def merge(a: dict, b: dict) -> dict:
        out = dict(a)
        for k, v in b.items():
            out[k] = merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
        return out

    return merge(base, doc), []


def validate_file(path: Path, name: str) -> list[str]:
    """Problems of a JSONL (one record per line) or JSON file against schema `name`."""
    out = []
    if str(path).endswith(".jsonl"):
        n = 0
        for n, (_, rec, raw) in enumerate(store.read_jsonl(path), 1):
            if rec is None:
                out.append(f"line {n}: not valid JSON")
                continue
            out += [f"line {n}: {p}" for p in schemas.problems(name, rec)]
        size = path.stat().st_size if path.exists() else 0
        if size and not path.read_bytes().endswith(b"\n"):
            out.append(f"line {n + 1}: no trailing newline (partial record)")
        return out
    doc = store.read_json(path)
    if doc is None:
        return [f"{path}: missing or not valid JSON"]
    return schemas.problems(name, doc)
