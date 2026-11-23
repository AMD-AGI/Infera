"""The door's four agent-facing operations as tool specs and one dispatcher, without the MCP SDK: the MCP
server lists the specs and calls `call`; the CLI (`bin/py -m o11y_door tool <name> '<json>'`) calls `call` too,
so both have the same inputs, validation and results."""

from __future__ import annotations

from . import api

# Kept short: every listed byte is paid on each request of a fleet session when Claude Code's tool search is off
# (spec §12 criterion 23; measured per tool, research: tool shares). A refused append returns the problems and
# the kind's full schema, so the fields are learnt from the first refusal, not from here.
LEDGER_DOC = (
    "Record one fact in the o11y run ledger (never edit runs.jsonl). record = the kind's fields; "
    "a refusal returns the problems and the kind's schema: fix and retry."
)

SPECS = [
    dict(
        name="ledger_append",
        description=LEDGER_DOC,
        input_schema={
            "type": "object",
            "required": ["kind", "record"],
            "properties": {"kind": {"enum": api.ledger_kinds()}, "record": {"type": "object"}},
        },
    ),
    dict(
        name="ledger_query",
        description="Read ledger records, latest version of each fact; filters: kind, run_id, task_id, since.",
        input_schema={
            "type": "object",
            "properties": {
                "kind": {"type": "string"},
                "run_id": {"type": "string"},
                "task_id": {"type": "string"},
                "since": {"type": "string"},
                "limit": {"type": "integer"},
            },
        },
        read_only=True,
    ),
    dict(
        name="session_register",
        description="Declare a child you launch or adopt (session_id), or a session not_in_fleet; needs purpose.",
        input_schema={
            "type": "object",
            "required": ["decision", "purpose"],
            "properties": {
                "decision": {"enum": ["child", "not_in_fleet"]},
                "purpose": {"type": "string"},
                "name": {"type": "string"},
                "session_id": {"type": "string"},
                "tmux_target": {"type": "string"},
                "cwd": {"type": "string"},
            },
        },
    ),
    dict(
        name="fleet_status",
        description="Fleet summary: sessions, task, best, open runs, plan, warnings.",
        input_schema={"type": "object", "properties": {}},
        read_only=True,
    ),
]
TOOL_SCHEMAS = {t["name"]: t["input_schema"] for t in SPECS}


def _input_problems(name: str, args: dict) -> list[str]:
    import jsonschema

    v = jsonschema.Draft202012Validator(TOOL_SCHEMAS[name])
    return [f"/{'/'.join(map(str, e.absolute_path))}: {e.message}" for e in v.iter_errors(args)]


def call(ctx: api.Context, name: str, args: dict) -> dict:
    if name not in TOOL_SCHEMAS:
        return {"ok": False, "problems": [f"unknown tool {name}"]}
    if name == "ledger_append" and not isinstance(args.get("record"), dict):
        # tolerate the flat form {kind, field: ...}; the record is validated against runs.v1 either way
        args = {"kind": args.get("kind"), "record": {k: v for k, v in args.items() if k != "kind"}}
    probs = _input_problems(name, args)
    if probs:
        return {"ok": False, "problems": probs, "hint": f"fix the arguments and call {name} again"}
    if name == "ledger_append":
        return api.ledger_append(ctx, {**args["record"], "kind": args["kind"]})
    if name == "ledger_query":
        if (
            args.get("kind") and args["kind"] not in api.ledger_kinds()
        ):  # not an enum in the listing (cost)
            return {"ok": False, "problems": [f"/kind: not one of {api.ledger_kinds()}"]}
        return api.ledger_query(
            ctx,
            args.get("kind"),
            args.get("run_id"),
            args.get("task_id"),
            args.get("since"),
            args.get("limit", 20),
        )
    if name == "session_register":
        return api.session_register(
            ctx,
            args["decision"],
            args.get("name"),
            args.get("purpose"),
            args.get("session_id"),
            args.get("tmux_target"),
            args.get("cwd"),
        )
    return api.fleet_status(ctx)
