#!/bin/sh
# PostToolUse entry of the ledger backstop. Precheck without starting python (it runs after every Write, Edit
# and Bash call of every session on the host): a session is a fleet member only if it was launched as main or
# child (AGENT_SYS_O11Y) or the trigger pinned it at activation ($prefix/state/o11y/sessions/<sid>.json).
# CLAUDE_CODE_SESSION_ID in a hook's environment follows /clear (Claude Code docs, env-vars).
case "${AGENT_SYS_O11Y:-}" in
  true|child) ;;
  *)
    prefix=${AGENT_SYS_HOME:-${HOME:?}/.infera_agent_sys}
    [ -n "${CLAUDE_CODE_SESSION_ID:-}" ] && [ -e "$prefix/state/o11y/sessions/$CLAUDE_CODE_SESSION_ID.json" ] || exit 0
    ;;
esac
here=$(cd "$(dirname "$0")/.." && pwd)
exec "$here/bin/py" "$here/hooks/ledger_backstop.py"
