#!/usr/bin/env bash
# Sequences the e2e-disag steps of ci_local.sh on 138 once disag-prep (PID $1) has exited:
# disag-build on the peer, then disag. Each step runs only if the previous one reported rc 0.
# Usage: disag_after_prep.sh PREP_PID OUT
set -uo pipefail
prep_pid="$1" out="$2"
W=/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522
PEER=crsuse2-m2m-140
REPO=/home/liyingli/bench_agentx/baseline/Infera
BRANCH=feat/sglang-mix-glm5.2-tp4-v0522

while kill -0 "$prep_pid" 2>/dev/null; do sleep 30; done
echo "$(date -u +%T) prep exited"
grep -q DISAG_PREP_RC=0 "$W/ci/disag-prep.log" || { echo "prep failed"; exit 1; }

echo "$(date -u +%T) build on $PEER"
ssh -n -o BatchMode=yes "$PEER" "bash $W/ci/ci_local_disag.sh disag-build x x $out" > "$W/ci/disag-build-140.log" 2>&1
grep -q 'DISAG_BUILD_RC=0' "$W/ci/disag-build-140.log" || { echo "build on $PEER failed"; exit 1; }

echo "$(date -u +%T) disag"
bash "$W/ci/ci_local_disag.sh" disag "$REPO" "$BRANCH" "$out" > "$W/ci/disag.log" 2>&1
echo "$(date -u +%T) disag exited: $(grep -o 'DISAG_RC=[0-9]*' "$W/ci/disag.log")"
