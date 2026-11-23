"""`bin/py -m o11y_door` — the script door over the same operations as the MCP tools.

Examples:
  bin/py -m o11y_door ledger append '{"kind":"run_start","run_id":"r8","theme":"decode mem 0.95","point":"explore"}'
  bin/py -m o11y_door ledger append --file rec.json          # or '-' for stdin
  bin/py -m o11y_door ledger query --kind run_end --limit 5
  bin/py -m o11y_door session register --decision child --name proj-kf --purpose "KernelForge mHC"
  bin/py -m o11y_door fleet status
  bin/py -m o11y_door validate-file <fleet>/ledger/runs.jsonl  # schema guessed from the file name
  bin/py -m o11y_door config show analysis
  bin/py -m o11y_door mcp                                      # run the stdio MCP server
Exit codes: 0 ok, 1 invalid input (problems printed as JSON), 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import api

GUESS = {
    "runs.jsonl": "runs.v1",
    "events.jsonl": "event.v1",
    "launches.jsonl": "registry.launch.v1",
    "sessions.json": "registry.sessions.v1",
    "general.json": "facts.general.v1",
    "detail.json": "facts.detail.v1",
    "kernelforge.json": "facts.kernelforge.v1",
    "trigger.json": "config.trigger.v1",
    "display.json": "config.display.v1",
    "health.json": "otel_health.v1",
}


def _guess(path: Path) -> str | None:
    if path.name == "analysis.json":
        return "requests.v1" if path.parent.name == "requests" else "config.analysis.v1"
    return GUESS.get(path.name)


def _out(result: dict) -> int:
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0 if result.get("ok", True) else 1


def _load_record(arg: str | None, file: str | None) -> dict:
    if file:
        text = sys.stdin.read() if file == "-" else Path(file).read_text(encoding="utf-8")
    elif arg:
        text = arg
    else:
        raise SystemExit("ledger append: give the record as JSON text or --file")
    rec = json.loads(text)
    if not isinstance(rec, dict):
        raise SystemExit("ledger append: the record must be a JSON object")
    return rec


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bin/py -m o11y_door", description=__doc__.split("\n")[0])
    p.add_argument(
        "--fleet-dir",
        help="fleet root (default: the session's pin, else $AGENT_SYS_O11Y_FLEET_DIR, "
        "else $AGENT_SYS_HOME/state/o11y/fleets/<slug of the launch dir>)",
    )
    p.add_argument(
        "--role",
        choices=["main", "child", "analyzer", "other"],
        help="writer role (default: fleet registry, else AGENT_SYS_O11Y, else other)",
    )
    p.add_argument("--session-id", help="writer session id (default: $CLAUDE_CODE_SESSION_ID)")
    sub = p.add_subparsers(dest="cmd", required=True)

    led = sub.add_parser("ledger").add_subparsers(dest="op", required=True)
    a = led.add_parser("append")
    a.add_argument("record", nargs="?")
    a.add_argument("--file")
    a.add_argument("--source", choices=["written", "extracted", "detected"])
    q = led.add_parser("query")
    q.add_argument("--kind")
    q.add_argument("--run-id")
    q.add_argument("--task-id")
    q.add_argument("--since")
    q.add_argument("--limit", type=int, default=20)
    q.add_argument("--all-versions", action="store_true", help="do not collapse superseded records")

    se = sub.add_parser("session").add_subparsers(dest="op", required=True)
    g = se.add_parser("register")
    g.add_argument("--decision", required=True, choices=["child", "not_in_fleet"])
    g.add_argument("--name")
    g.add_argument("--purpose")
    g.add_argument(
        "--subject-session-id",
        dest="subject",
        help="a running session (adopted child or not_in_fleet)",
    )
    g.add_argument("--tmux-target")
    g.add_argument("--cwd")

    sub.add_parser("fleet").add_subparsers(dest="op", required=True).add_parser("status")

    vf = sub.add_parser("validate-file")
    vf.add_argument("path")
    vf.add_argument(
        "--schema", help="schema name, e.g. runs.v1 (default: guessed from the file name)"
    )

    cf = sub.add_parser("config").add_subparsers(dest="op", required=True).add_parser("show")
    cf.add_argument("which", choices=["trigger", "analysis", "display"])

    sub.add_parser("mcp")

    args = p.parse_args(argv)
    root = Path(args.fleet_dir) if args.fleet_dir else None
    if args.cmd == "mcp":
        from .mcp_server import run

        run(root)
        return 0
    ctx = api.Context(root=root, session_id=args.session_id, role=args.role)

    if args.cmd == "ledger" and args.op == "append":
        rec = _load_record(args.record, args.file)
        if args.source:
            rec.setdefault("writer", {})["source"] = args.source
        return _out(api.ledger_append(ctx, rec))
    if args.cmd == "ledger":
        return _out(
            api.ledger_query(
                ctx,
                args.kind,
                args.run_id,
                args.task_id,
                args.since,
                args.limit,
                not args.all_versions,
            )
        )
    if args.cmd == "session":
        return _out(
            api.session_register(
                ctx,
                args.decision,
                args.name,
                args.purpose,
                args.subject,
                args.tmux_target,
                args.cwd,
            )
        )
    if args.cmd == "fleet":
        return _out(api.fleet_status(ctx))
    if args.cmd == "validate-file":
        path = Path(args.path)
        name = args.schema or _guess(path)
        if not name:
            print(f"cannot guess the schema of {path.name}; pass --schema", file=sys.stderr)
            return 2
        probs = api.validate_file(path, name)
        return _out({"ok": not probs, "schema": name, "problems": probs})
    if args.cmd == "config":
        cfg, probs, _new = api.load_config(
            ctx.root, args.which, state_dir=ctx.root / "state" / "config-cli"
        )
        return _out({"ok": not probs, "config": cfg, "problems": probs})
    return 2


if __name__ == "__main__":
    sys.exit(main())
