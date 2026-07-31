#!/bin/bash
# Restart ONLY the sglang engine legs. The kvd daemon stays up on purpose:
# the question is whether a NEW engine process can reuse what the OLD one
# stored, so killing kvd would destroy the very thing under test.
pkill -9 -f "infera.engine.sglang" 2>/dev/null
pkill -9 -f "sglang.launch_server" 2>/dev/null
pkill -9 -f "sglang::" 2>/dev/null
sleep 8
echo "kvd alive: $(pgrep -fc 'infera.kvd')"
echo "engine procs left: $(pgrep -fc 'sglang' || echo 0)"
