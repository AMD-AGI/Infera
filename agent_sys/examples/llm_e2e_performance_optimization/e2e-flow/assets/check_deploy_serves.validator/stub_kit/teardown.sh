#!/bin/sh
# Removes what THIS run created, identified by its own run tag, and nothing else.
set -eu
. "$(dirname "$0")/stub_env.sh"
killed=0
for role in router engine; do
  f="$E2E_KIT_WORK_ROOT/${role}.pid"
  [ -f "$f" ] || continue
  pid="$(cat "$f")"
  if kill "$pid" 2>/dev/null; then killed=$((killed + 1)); fi
  rm -f "$f"
done
echo "{\"run_tag\": \"${E2E_KIT_RUN_TAG}\", \"processes_stopped\": ${killed}}"
