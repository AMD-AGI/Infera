"""`bin/py -m o11y_door` — the script door: the same operations, inputs and results as the MCP tools.

Examples:
  bin/py -m o11y_door ledger append '{"kind":"run_start","run_id":"r8","theme":"decode mem 0.95","point":"explore"}'
  bin/py -m o11y_door ledger append --file rec.json          # or '-' for stdin
  bin/py -m o11y_door tool ledger_query '{"kind":"run_end","limit":5}'
  bin/py -m o11y_door tool session_register '{"decision":"child","name":"proj-kf","purpose":"KernelForge mHC"}'
  bin/py -m o11y_door tool fleet_status
  bin/py -m o11y_door validate-file <fleet>/ledger/runs.jsonl  # schema guessed from the file name
  bin/py -m o11y_door config show analysis
Exit codes: 0 ok, 1 invalid input (problems printed as JSON), 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import api, tools

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


def _json_object(text: str, what: str) -> dict:
    try:
        doc = json.loads(text)
    except ValueError as e:
        raise SystemExit(f"{what}: not valid JSON ({e})") from None
    if not isinstance(doc, dict):
        raise SystemExit(f"{what}: must be a JSON object")
    return doc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bin/py -m o11y_door", description=__doc__.split("\n")[0])
    p.add_argument("--fleet-dir", help="fleet root (default: as for the MCP server, design §4)")
    p.add_argument("--session-id", help="writer session id (default: $CLAUDE_CODE_SESSION_ID)")
    p.add_argument(
        "--role",
        choices=["main", "child", "analyzer", "other"],
        help="writer role (default: registry row, pin, AGENT_SYS_O11Y)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ledger").add_subparsers(dest="op", required=True).add_parser("append")
    a.add_argument("record", nargs="?", help="the record as JSON, with its kind")
    a.add_argument("--file", help="read the record from a file ('-' = stdin)")
    t = sub.add_parser("tool", help=f"one of {', '.join(tools.TOOL_SCHEMAS)}")
    t.add_argument("name", choices=list(tools.TOOL_SCHEMAS))
    t.add_argument("args", nargs="?", default="{}", help="the tool's arguments as JSON")
    vf = sub.add_parser("validate-file")
    vf.add_argument("path")
    vf.add_argument(
        "--schema", help="schema name, e.g. runs.v1 (default: guessed from the file name)"
    )
    cf = sub.add_parser("config").add_subparsers(dest="op", required=True).add_parser("show")
    cf.add_argument("which", choices=["trigger", "analysis", "display"])

    args = p.parse_args(argv)
    ctx = api.Context(
        root=Path(args.fleet_dir) if args.fleet_dir else None,
        session_id=args.session_id,
        role=args.role,
    )
    if args.cmd == "ledger":
        if not (args.record or args.file):
            raise SystemExit("ledger append: give the record as JSON text or --file")
        text = args.record or (
            sys.stdin.read() if args.file == "-" else Path(args.file).read_text("utf-8")
        )
        rec = _json_object(text, "ledger append")
        return _out(
            tools.call(ctx, "ledger_append", {"kind": rec.pop("kind", None), "record": rec})
        )
    if args.cmd == "tool":
        return _out(tools.call(ctx, args.name, _json_object(args.args, args.name)))
    if args.cmd == "validate-file":
        path = Path(args.path)
        name = args.schema or _guess(path)
        if not name:
            print(f"cannot guess the schema of {path.name}; pass --schema", file=sys.stderr)
            return 2
        probs = api.validate_file(path, name)
        return _out({"ok": not probs, "schema": name, "problems": probs})
    cfg, probs, _new = api.load_config(
        ctx.root, args.which, state_dir=ctx.root / "state" / "config-cli"
    )
    return _out({"ok": not probs, "config": cfg, "problems": probs})


if __name__ == "__main__":
    sys.exit(main())
