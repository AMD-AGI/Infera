"""PostToolUse backstop for the o11y run ledger (Write|Edit|Bash), started by hooks/backstop.sh for fleet members.

The ledger must be written only through the `ledger_append` MCP tool (or `bin/py -m o11y_door ledger append`).
This hook checks the ledger file itself, not the command line, so indirect writers (python scripts, sed -i,
echo >>) are caught too. Changed file: validate the new lines and detect edits of earlier lines; on a problem
exit 2 so Claude reads the hint (stderr) next to the tool result. A valid appended record is accepted.

Cost (it runs after every Write, Edit and Bash call of a fleet session): the unchanged-ledger path imports only
`os` and `sys` (plus `json` for a pinned main session's pin), then compares one stat against a one-line cursor.
Ledger: <fleet root>/ledger/runs.jsonl; fleet root as everywhere (design §4): AGENT_SYS_O11Y_FLEET_DIR (children),
else the session's pin $prefix/state/o11y/sessions/<sid>.json, else <prefix>/state/o11y/fleets/<slug(launch dir)>.
"""

import os
import sys

PLUGIN_ROOT = os.environ.get("CLAUDE_PLUGIN_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
HINT = (
    "Use the ledger_append tool (or `bin/py -m o11y_door ledger append '<json>'`) to record ledger facts; "
    "to correct a record, append a new record with the same identity (run_id, op_id, ...) instead of editing."
)


def fleet_root(payload: str) -> str:
    explicit = os.environ.get("AGENT_SYS_O11Y_FLEET_DIR")
    if explicit:
        return explicit
    import json

    sys.path.insert(0, os.path.join(PLUGIN_ROOT, "protocol", "py"))
    from o11y_protocol import fleet

    try:
        session_id = json.loads(payload or "{}").get("session_id")
    except ValueError:
        session_id = None
    return str(fleet.fleet_dir(session_id=session_id or os.environ.get("CLAUDE_CODE_SESSION_ID")))


def read_cursor(path: str) -> list[str]:
    """`ino size mtime_ns offset prefix_sha1` written after the last check (private to the backstop)."""
    try:
        with open(path) as f:
            return f.read().split()
    except OSError:
        return []


def main() -> int:
    payload = sys.stdin.read()
    base = fleet_root(payload)
    ledger = os.path.join(base, "ledger", "runs.jsonl")
    try:
        st = os.stat(ledger)
    except FileNotFoundError:
        return 0
    cursor_path = os.path.join(base, "state", "backstop", "cursor")
    cursor = read_cursor(cursor_path)
    if cursor[:3] == [str(st.st_ino), str(st.st_size), str(st.st_mtime_ns)]:
        return 0

    import hashlib
    import json

    sys.path.insert(0, os.path.join(PLUGIN_ROOT, "protocol", "py"))
    from o11y_protocol import schemas

    try:
        hook = json.loads(payload or "{}")
    except ValueError:
        hook = {}
    with open(ledger, "rb") as f:
        data = f.read()
    known = int(cursor[3]) if len(cursor) == 5 and cursor[0] == str(st.st_ino) else 0
    known = known if len(data) >= known else 0
    rewritten = bool(known) and hashlib.sha1(data[:known]).hexdigest() != cursor[4]
    start = 0 if rewritten else known
    end = data.rfind(b"\n") + 1
    lineno = data[:start].count(b"\n")
    problems = []
    for raw in data[start:end].split(b"\n")[:-1]:
        lineno += 1
        if not raw.strip():
            continue
        try:
            rec = json.loads(raw)
        except ValueError:
            problems.append(f"line {lineno}: not valid JSON")
            continue
        problems += [f"line {lineno}: {p}" for p in schemas.problems("runs.v1", rec, limit=5)]
    if end < len(data):
        problems.append(f"line {lineno + 1}: incomplete last line (no trailing newline)")
    if rewritten:
        problems.insert(0, "earlier ledger lines were modified in place; the ledger is append-only")
    tool = hook.get("tool_name")
    target = (hook.get("tool_input") or {}).get("file_path") or ""
    if tool in ("Write", "Edit") and os.path.abspath(target) == os.path.abspath(ledger):
        problems.insert(0, f"{tool} was used on the ledger file directly")

    os.makedirs(os.path.dirname(cursor_path), mode=0o700, exist_ok=True)
    tmp = f"{cursor_path}.tmp.{os.getpid()}"
    with open(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
        f.write(
            f"{st.st_ino} {st.st_size} {st.st_mtime_ns} {end} {hashlib.sha1(data[:end]).hexdigest()}\n"
        )
    os.replace(tmp, cursor_path)
    if not problems:
        return 0
    sys.stderr.write(f"o11y run ledger {ledger}:\n- " + "\n- ".join(problems[:20]) + f"\n{HINT}\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
