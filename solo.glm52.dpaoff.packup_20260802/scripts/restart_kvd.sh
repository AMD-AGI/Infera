#!/usr/bin/env bash
# Restart ONLY the kvd daemon, leaving the engines alone. Runs ON a node.
#
# WHY THIS SCRIPT EXISTS
# ----------------------
# The obvious `pkill -9 -f infera.kvd` is WRONG and cost a prefill engine:
# `-f` matches a REGEX against the full cmdline, and `.` is a wildcard, so
# `infera.kvd` also matches the engine's own
#     python3 -m infera.engine.sglang ... --infera-kvd-socket /tmp/kvd/kvd.sock
#                                           ^^^^^^^^^^
# The engine dies silently alongside the daemon, the leg leaves etcd, and the
# router quietly drops to 1 active worker -- which looks like an engine crash
# with no crash in the log (no HSA error, no traceback, just a clean exit).
#
# Anchoring on the module-invocation form `-m infera\.kvd ` (escaped dot,
# trailing space) matches only the daemon.
set -u
CTR="${CTR:-bench_run}"
KVD_SOCK="${KVD_SOCK:-/tmp/kvd/kvd.sock}"

echo "engines before: $(docker exec "$CTR" bash -c 'ps -eo args | grep -c "[i]nfera.engine.sglang"' || echo 0)"

docker exec "$CTR" bash -c '
  pkill -9 -f -- "-m infera\.kvd " 2>/dev/null
  sleep 2
  n=$(ps -eo args | grep -c "[-]m infera\.kvd ") || true
  echo "  kvd procs left: ${n:-0} (want 0)"'

docker exec -d "$CTR" bash -c 'nohup /run_kvd.sh > /tmp/kvd.log 2>&1'
sleep 15
docker exec "$CTR" bash -c "test -S $KVD_SOCK && echo '  kvd socket OK' || { echo '  KVD FAILED'; tail -20 /tmp/kvd.log; }"

echo "engines after:  $(docker exec "$CTR" bash -c 'ps -eo args | grep -c "[i]nfera.engine.sglang"' || echo 0)"
echo "  ^ must be UNCHANGED. If it dropped, the pkill pattern hit the engine again."
