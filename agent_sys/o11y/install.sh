#!/usr/bin/env bash
# Install agent_sys o11y: create the Python venv in agent_sys's prefix and print the steps that change your
# Claude Code configuration. Those steps run only when you ask with --register; this script never edits
# settings or keybindings files itself (spec §2 principle 6, §10).
#
#   ./install.sh               venv in $AGENT_SYS_HOME/share/o11y/venv, then print the user steps
#   ./install.sh --register    also run `claude plugin marketplace add` and `claude plugin install` (both plugins)
#                              into the current Claude config dir ($CLAUDE_CONFIG_DIR, else ~/.claude)
#   ./install.sh --build       also rebuild the two mods (needs node; the built files are committed)
#   ./install.sh --no-venv     skip the venv (py.sh then uses python3 from PATH)
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
claude_bin=${CLAUDE_BIN:-claude}
prefix=${AGENT_SYS_HOME:-${HOME:?}/.infera_agent_sys}
venv="$prefix/share/o11y/venv"
register=0 build=0 make_venv=1
for a in "$@"; do
  case "$a" in
    --register) register=1 ;;
    --build) build=1 ;;
    --no-venv) make_venv=0 ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

if [ "$build" = 1 ]; then
  (cd "$here" && npm ci --no-audit --no-fund && node build.mjs)
fi

if [ "$make_venv" = 1 ]; then
  if [ ! -x "$venv/bin/python" ]; then
    echo "== creating $venv"
    mkdir -p "$(dirname "$venv")"
    python3 -m venv "$venv"
  fi
  "$venv/bin/pip" install --quiet 'mcp>=2,<3' 'jsonschema>=4' 'pydantic>=2'
fi
"$here/plugins/agent-sys-o11y/server/py.sh" -c 'import mcp, jsonschema, pydantic' || {
  echo "python packages missing for plugins/agent-sys-o11y/server/py.sh (mcp>=2,<3, jsonschema, pydantic>=2)" >&2
  exit 1
}

if [ "$register" = 1 ]; then
  echo "== registering the plugins in ${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
  "$claude_bin" plugin marketplace add "$here"
  "$claude_bin" plugin install agent-sys-o11y@agent-sys-local
  "$claude_bin" plugin install agent-sys-o11y-dash@agent-sys-local
fi

cat <<EOF

agent_sys o11y: the venv is ready. The remaining steps change your Claude Code configuration, so they are yours:

 1. Register the plugins (or rerun this script with --register):
      $claude_bin plugin marketplace add $here
      $claude_bin plugin install agent-sys-o11y@agent-sys-local
      $claude_bin plugin install agent-sys-o11y-dash@agent-sys-local
    Development instead: claude --plugin-dir $here/plugins/agent-sys-o11y --plugin-dir $here/plugins/agent-sys-o11y-dash
 2. Let the o11y tools run without a prompt each time: add to permissions.allow in your user settings
      "mcp__plugin_agent-sys-o11y_o11y__*"
 3. Shortcuts (prefix ctrl+q): merge the "bindings" block of
      $here/plugins/agent-sys-o11y-dash/keybindings.snippet.json
    into your keybindings.json.

Start the main session with AGENT_SYS_O11Y=true (or use an agent_sys skill in it). Children are launched as
children by session_register (AGENT_SYS_O11Y=child).
EOF
