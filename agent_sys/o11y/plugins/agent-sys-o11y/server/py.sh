#!/bin/sh
# Run python for the o11y server, CLI, backstop hook and analyzer.
# The venv lives in agent_sys's prefix (design §2): $AGENT_SYS_HOME/share/o11y/venv, the prefix being
# $AGENT_SYS_HOME, else $HOME/.infera_agent_sys (env_mgr/prefix.py). Without it, python3 from PATH.
# Usage: py.sh -m o11y_core.mcp_server | py.sh -m o11y_analyze --kind log | py.sh <script.py>
here=$(cd "$(dirname "$0")" && pwd)
prefix=${AGENT_SYS_HOME:-${HOME:?}/.infera_agent_sys}
py=python3
[ -x "$prefix/share/o11y/venv/bin/python" ] && py="$prefix/share/o11y/venv/bin/python"
PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}" exec "$py" "$@"
