#!/usr/bin/env bash
# Task v0521cap2: on the node, wait until <round>'s A2 bench is recorded in status.tsv (its chain driver was killed so no A3
# starts), then exec stop_then_chain.yihou.sh <old-env> <new-env> C1 C2 [C3].
# Usage: bash after_a2_then.yihou.sh <round> <C2> <old-env> <new-env> C1 C2 [C3]
set -uo pipefail
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
r=$1; c=$2; shift 2
echo "$(date -u +%FT%TZ) waiting for $r A2 (C$c) to end"
until grep -qE "bench	(ok|FAIL).*agentx-c$c-A2" $R/$r/status.tsv 2>/dev/null; do sleep 60; done
grep -a "agentx-c$c-A2" $R/$r/status.tsv | tail -1
exec bash $H/stop_then_chain.yihou.sh "$@"
