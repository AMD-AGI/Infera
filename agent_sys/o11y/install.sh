#!/usr/bin/env bash
# Thin wrapper: install o11y's venv through the env_mgr recipe (agent_sys/env_mgr/recipes/o11y.yaml) and
# print the user steps of README.md "Install". The plugin registration changes the user's Claude
# configuration, so it runs only with --register.
#
#   ./install.sh               venv via the recipe, then print the user steps
#   ./install.sh --register    also register the three plugins in the current Claude config dir
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
claude_bin=${CLAUDE_BIN:-claude}
prefix=${AGENT_SYS_HOME:-${HOME:?}/.infera_agent_sys}
register=0
for a in "$@"; do
  case "$a" in
    --register) register=1 ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

mkdir -p "$prefix"
(cd "$here/.." && uv run env-mgr install env_mgr/recipes/o11y.yaml --path "$prefix")

if [ "$register" = 1 ]; then
  "$claude_bin" plugin marketplace add "$here"
  for p in agent-sys-o11y-door agent-sys-o11y agent-sys-o11y-dash; do "$claude_bin" plugin install "$p@agent-sys-local"; done
fi

sed -n '/^## Install/,/^## Use/p' "$here/README.md" | sed '$d'
