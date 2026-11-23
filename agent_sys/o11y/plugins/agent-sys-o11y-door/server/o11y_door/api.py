"""The protocol's whole operations: the only entry the door (MCP server, CLI, backstop) and the analyzers use.

Every JSONL record of the fleet is written here: ledger records (`ledger_append`), events (`event_report`,
analyzers only) and session registrations (`session_register`). Readers go through `ledger_query`,
`read_ledger`, `read_registry` and `load_config`.
"""

from __future__ import annotations

import copy
import hashlib
import os
import shlex
from pathlib import Path

from o11y_protocol import fleet, schemas, store

SEVERITY = {"INFO": 9, "WARN": 13, "ERROR": 17}
SEVERITY_ALIASES = {"info": "INFO", "warning": "WARN", "warn": "WARN", "error": "ERROR"}
SERVER_SET = {"v", "id", "ts", "writer"}
_IDENTITY: dict | None = None  # runs.v1 x-identity, loaded lazily


class Context:
    """Who writes, and into which fleet.

    `session_id`: the writing session. The door passes the id of its parent claude process (an MCP server's
    own `CLAUDE_CODE_SESSION_ID` is the one it was spawned with and does not follow `/clear`); without one,
    `CLAUDE_CODE_SESSION_ID` (correct for Bash and hook subprocesses), else "unknown".
    `root`: the fleet root; default per `fleet.fleet_dir` (the session's pin first).
    `role`: default from the fleet registry, else `AGENT_SYS_O11Y`, else "other".
    """

    def __init__(
        self,
        root: Path | None = None,
        session_id: str | None = None,
        role: str | None = None,
        session_name: str | None = None,
    ):
        self.session_id = session_id or os.environ.get("CLAUDE_CODE_SESSION_ID") or "unknown"
        self.root = Path(root) if root else fleet.fleet_dir(session_id=self.session_id)
        self.role = role or self._detect_role()
        self.session_name = session_name

    def _detect_role(self) -> str:
        for s in read_registry(self.root)[0]:
            if s.get("session_id") == self.session_id and s.get("role") in ("main", "child"):
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


def _fail(problems: list[str], hint: str, **extra) -> dict:
    return {"ok": False, "problems": problems, "hint": hint, **extra}


# ------------------------------------------------------------------------------------------ ledger
def ledger_kinds() -> list[str]:
    return list(schemas.load("runs.v1")["properties"]["kind"]["enum"])


def kind_schema(kind: str) -> dict | None:
    """Every field of one ledger kind (refs inlined, per-kind rules kept): sent back with a refusal."""
    if kind not in ledger_kinds():
        return None
    full = schemas.load("runs.v1")
    defs = {**schemas.load("common.v1")["$defs"], **full["$defs"]}

    def inline(node):
        if isinstance(node, dict):
            if "$ref" in node:
                target = copy.deepcopy(defs[node["$ref"].rsplit("/", 1)[-1]])
                rest = {k: v for k, v in node.items() if k != "$ref"}
                return inline({**target, **rest})
            return {k: inline(v) for k, v in node.items()}
        if isinstance(node, list):
            return [inline(x) for x in node]
        return node

    body = copy.deepcopy(full["$defs"][kind])
    common = {k: v for k, v in full["properties"].items() if k not in SERVER_SET | {"kind"}}
    out = {
        "kind": kind,
        "required": body.get("required", []),
        "properties": inline({**common, **body.get("properties", {})}),
    }
    if body.get("allOf"):
        out["conditional_rules"] = inline(body["allOf"])
    return out


def ledger_append(ctx: Context, record: dict) -> dict:
    """Validate and append one runs.v1 record; v, id, ts and writer are set here (an analyzer's
    `writer.source` / `writer.from` are kept). A refusal carries the one-line problems and the kind's schema."""
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
        extra = {"schema": kind_schema(rec["kind"])} if isinstance(rec["kind"], str) else {}
        if not extra.get("schema"):
            extra = {"kinds": ledger_kinds()}
        return _fail(
            probs, "Fix these fields and call ledger_append again with the whole record.", **extra
        )
    try:
        store.append(ctx.root / store.LEDGER, rec)
    except store.LineTooLong as exc:
        return _fail([str(exc)], "Shorten long text fields (note, conclusion) and retry.")
    return {"ok": True, "id": rec["id"], "kind": rec["kind"], "ts": rec["ts"]}


def identity(rec: dict) -> tuple | None:
    """The supersede key of a ledger record (runs.v1 `x-identity`): a later valid record with the same key
    replaces the earlier one. None for kinds without an identity."""
    global _IDENTITY
    if _IDENTITY is None:
        _IDENTITY = schemas.load("runs.v1").get("x-identity", {})
    keys = _IDENTITY.get(rec.get("kind"))
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
    source: str = "detector",
    observed_ts: str | None = None,
    origin: list | None = None,
    writer_source: str = "detected",
) -> dict:
    """Append one event.v1 record (analyzers only; not offered to agents). The name must be in
    protocol/events.json, at one of its levels."""
    sev = SEVERITY_ALIASES.get(str(severity).lower(), str(severity).upper())
    entry = schemas.events().get(event_name)
    if entry is None:
        return _fail(
            [f"unknown event_name {event_name!r} (not in protocol/events.json)"],
            "Use a catalogued name.",
        )
    if sev not in entry.get("levels", [entry["level"]]):
        return _fail(
            [f"{event_name} is not reported at {sev}"],
            f"Use {entry.get('levels', [entry['level']])}.",
        )
    attrs = dict(attributes or {})
    attrs.setdefault("o11y.source", source)
    attrs.setdefault("session.id", ctx.session_id)
    if not attrs.get("o11y.dedup_key"):  # without a key the fact is who, what and the text
        digest = hashlib.sha1(f"{ctx.session_id}|{event_name}|{body}".encode()).hexdigest()[:16]
        attrs["o11y.dedup_key"] = f"auto:{digest}"
    rec = {
        "id": store.new_id("evt"),
        "v": 1,
        "ts": store.now(),
        "severity_text": sev,
        "severity_number": SEVERITY.get(sev, 0),
        "event_name": event_name,
        "body": body,
        "attributes": attrs,
        "writer": ctx.writer(writer_source, origin),
    }
    if observed_ts:
        rec["observed_ts"] = observed_ts
    probs = schemas.problems("event.v1", rec)
    if probs:
        return _fail(probs, "Fix these fields and report the event again.")
    try:
        store.append(ctx.root / store.EVENTS, rec)
    except store.LineTooLong as exc:
        return _fail([str(exc)], "Shorten body/attributes and retry.")
    return {"ok": True, "id": rec["id"]}


# ------------------------------------------------------------------------------------------ sessions
def session_register(
    ctx: Context,
    decision: str,
    name: str | None = None,
    purpose: str | None = None,
    session_id: str | None = None,
    tmux_target: str | None = None,
    cwd: str | None = None,
) -> dict:
    """decision child: a child about to be launched (returns its env and launch argv) or, with session_id, a
    running session adopted as a child. not_in_fleet: the main declares session_id not part of the fleet
    (spec §3.5). Both need a purpose; a child needs a name or a tmux target (join keys)."""
    rec = {
        "v": 1,
        "id": store.new_id("launch"),
        "ts": store.now(),
        "decision": decision,
        "fleet_dir": str(ctx.root),
        "writer": ctx.writer(),
    }
    rec["parent_session_id"] = ctx.session_id
    if session_id:
        rec["session_id"] = session_id
    for k, val in (
        ("name", name),
        ("purpose", purpose),
        ("tmux_target", tmux_target),
        ("cwd", cwd),
    ):
        if val:
            rec[k] = val
    probs = schemas.problems("registry.launch.v1", rec)
    if probs:
        return _fail(probs, "Fix these fields and call session_register again.")
    store.append(ctx.root / store.LAUNCHES, rec)
    out = {"ok": True, "id": rec["id"], "fleet_dir": str(ctx.root)}
    if decision == "child" and not session_id:
        env = {
            fleet.ACTIVATION_ENV_VAR: "child",
            fleet.FLEET_DIR_ENV_VAR: str(ctx.root),
            fleet.PARENT_ENV_VAR: ctx.session_id,
        }
        out["env"] = env
        target = tmux_target or name
        argv = ["tmux", "new-session", "-d", "-s", target, "-c", cwd or os.getcwd()]
        for k, v in env.items():
            argv += ["-e", f"{k}={v}"]
        argv += ["claude"] + (["--name", name] if name else [])
        out["argv"] = argv
        out["launch_example"] = shlex.join(argv)
    return out


def launches(root: Path) -> list[dict]:
    v = schemas.validator("registry.launch.v1")
    return [
        r for _, r, _ in store.read_jsonl(root / store.LAUNCHES) if r is not None and v.is_valid(r)
    ]


def read_registry(root: Path) -> tuple[list[dict], str | None]:
    """The trigger's registry (registry/sessions.json), the only judge of liveness and status (review A4).
    Returns (sessions, problem): absent or invalid → ([], one line saying so); nothing is recomputed."""
    path = root / store.SESSIONS
    doc = store.read_json(path)
    if doc is None:
        return [], f"{path}: absent or not JSON"
    probs = schemas.problems("registry.sessions.v1", doc, limit=1)
    if probs:
        return [], f"{path}: {probs[0]}"
    return doc["sessions"], None


def fleet_status(ctx: Context) -> dict:
    sessions, registry_problem = read_registry(ctx.root)
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
    keys = (
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
    return {
        "ok": True,
        "fleet_dir": str(ctx.root),
        "me": {"session_id": ctx.session_id, "role": ctx.role},
        "sessions": [{k: s.get(k) for k in keys} for s in sessions],
        "registry_problem": registry_problem,
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
def load_config(
    root: Path, which: str, state_dir: Path | None = None
) -> tuple[dict, list[str], bool]:
    """(config, problems, problems_are_new). Omitted keys take the schema defaults. An invalid document,
    including an unknown key, is refused with one-line problems (did-you-mean for keys) and the last valid
    document stays in effect (review F5). Each consumer keeps its own last-valid copy under `state_dir`
    (default `<root>/state/config/`), so "report once" is per consumer: problems_are_new is True only the
    first time a given invalid document is seen."""
    name = f"config.{which}.v1"
    path = root / store.CONFIG[which]
    keep_dir = state_dir or root / "state" / "config"
    last_valid = keep_dir / f"{which}.json"
    seen = keep_dir / f"{which}.invalid.sha1"
    base = schemas.defaults(name)

    def merge(a: dict, b: dict) -> dict:
        out = dict(a)
        for k, v in b.items():
            out[k] = merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
        return out

    raw = path.read_bytes() if path.exists() else None
    doc = store.read_json(path) if raw is not None else None
    if raw is None:
        return base, [], False
    probs = (
        [f"{path}: not valid JSON"]
        if doc is None
        else [f"{path}: {p}" for p in schemas.problems(name, doc)]
    )
    if not probs:
        store.write_json(last_valid, doc)
        return merge(base, doc), [], False
    digest = hashlib.sha1(raw).hexdigest()
    new = (seen.read_text().strip() if seen.exists() else "") != digest
    if new:
        store.ensure_dir(keep_dir)
        store.write_text(seen, digest)
    kept = store.read_json(last_valid)
    return (merge(base, kept) if isinstance(kept, dict) else base), probs, new


def validate_file(path: Path, name: str) -> list[str]:
    """Problems of a JSONL (one record per line) or JSON file against schema `name`."""
    out = []
    if str(path).endswith(".jsonl"):
        n = 0
        for n, (_, rec, _raw) in enumerate(store.read_jsonl(path), 1):
            if rec is None:
                out.append(f"line {n}: not valid JSON")
                continue
            out += [f"line {n}: {p}" for p in schemas.problems(name, rec)]
        if path.exists() and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
            out.append(f"line {n + 1}: no trailing newline (partial record)")
        return out
    doc = store.read_json(path)
    if doc is None:
        return [f"{path}: missing or not valid JSON"]
    return schemas.problems(name, doc)
