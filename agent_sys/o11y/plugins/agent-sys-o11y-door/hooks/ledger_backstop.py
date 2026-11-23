"""PostToolUse backstop for the o11y run ledger (Write|Edit|Bash).

The ledger must be written only through the `ledger_append` MCP tool (or `py.sh -m o11y_door ledger append`).
This hook checks the ledger file itself, not the command line, so indirect writers (python scripts, sed -i,
echo >>) are caught too. Unchanged file: one stat, exit 0. Changed file: validate new lines and detect edits
of earlier lines; on a problem exit 2 so Claude reads the hint (stderr) next to the tool result.
Ledger path: <fleet root>/ledger/runs.jsonl, the fleet root resolved like every component (design §4): the
session's pin $AGENT_SYS_HOME/state/o11y/sessions/<session_id>.json, else $AGENT_SYS_O11Y_FLEET_DIR, else
$AGENT_SYS_HOME/state/o11y/fleets/<slug($CLAUDE_PROJECT_DIR)>. Sessions outside every fleet have no ledger
there, so the hook exits 0 after a pin read and one stat (stdlib only; jsonschema loads only on a change).
"""

import hashlib
import json
import os
import sys

PLUGIN_ROOT = os.environ.get("CLAUDE_PLUGIN_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, os.path.join(PLUGIN_ROOT, "protocol", "py"))

HINT = (
    "Use the ledger_append tool (or `py.sh -m o11y_door ledger append '<json>'`) to record ledger facts; "
    "to correct a record, append a new record with the same identity (run_id, op_id, ...) instead of editing."
)


def main() -> int:
    try:
        hook = json.load(sys.stdin)
    except ValueError:
        return 0
    from o11y_core import fleet  # stdlib-only module: cheap on the common path

    env = dict(os.environ)
    env.setdefault("CLAUDE_PROJECT_DIR", hook.get("cwd") or os.getcwd())
    session_id = hook.get("session_id") or env.get("CLAUDE_CODE_SESSION_ID")
    base = str(fleet.fleet_dir(env, session_id=session_id))
    ledger = os.path.join(base, "ledger", "runs.jsonl")
    try:
        st = os.stat(ledger)
    except FileNotFoundError:
        return 0
    state_path = os.path.join(
        base, "state", "backstop", hashlib.sha1(ledger.encode()).hexdigest() + ".json"
    )
    try:
        with open(state_path) as f:
            state = json.load(f)
    except (OSError, ValueError):
        state = {}
    if (state.get("ino"), state.get("size"), state.get("mtime_ns")) == (
        st.st_ino,
        st.st_size,
        st.st_mtime_ns,
    ):
        return 0

    from o11y_core import (
        schemas,
    )  # imported only when the ledger changed (keeps the common path fast)

    with open(ledger, "rb") as f:
        data = f.read()
    known = (
        state.get("offset", 0)
        if state.get("ino") == st.st_ino and len(data) >= state.get("offset", 0)
        else 0
    )
    rewritten = bool(known) and hashlib.sha1(data[:known]).hexdigest() != state.get("prefix_sha1")
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

    os.makedirs(os.path.dirname(state_path), mode=0o700, exist_ok=True)
    tmp = f"{state_path}.tmp.{os.getpid()}"
    with open(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
        json.dump(
            {
                "ino": st.st_ino,
                "size": st.st_size,
                "mtime_ns": st.st_mtime_ns,
                "offset": end,
                "prefix_sha1": hashlib.sha1(data[:end]).hexdigest(),
            },
            f,
        )
    os.replace(tmp, state_path)
    if not problems:
        return 0
    sys.stderr.write(f"o11y run ledger {ledger}:\n- " + "\n- ".join(problems[:20]) + f"\n{HINT}\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
