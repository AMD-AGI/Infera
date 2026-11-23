#!/bin/sh
# Python for this plugin's processes (MCP server, CLI, backstop hook, analyzers).
# The venv lives in agent_sys's prefix: $AGENT_SYS_HOME/share/o11y/venv, the prefix being $AGENT_SYS_HOME,
# else $HOME/.infera_agent_sys (env_mgr/prefix.py); without it, python3 from PATH. PYTHONPATH gets this
# server dir and the protocol library (<plugin>/protocol -> agent_sys/o11y/protocol). `exec` keeps the
# caller's pid as our parent (the door finds its session through its ppid).
here=$(cd "$(dirname "$0")" && pwd)
prefix=${AGENT_SYS_HOME:-${HOME:?}/.infera_agent_sys}
py=python3
[ -x "$prefix/share/o11y/venv/bin/python" ] && py="$prefix/share/o11y/venv/bin/python"
PYTHONPATH="$here:$here/../protocol/py${PYTHONPATH:+:$PYTHONPATH}" exec "$py" "$@"
