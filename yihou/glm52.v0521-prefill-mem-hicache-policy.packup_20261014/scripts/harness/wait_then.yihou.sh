#!/usr/bin/env bash
# Task v0521cap2: wait until a round's chain log shows "chain DONE" (or STOP), then exec stop_then_chain.yihou.sh.
# Usage: bash wait_then.yihou.sh <old-round> <old-env> <new-env> C1 C2 [C3]
set -uo pipefail
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
old=$1; shift
echo "$(date -u +%FT%TZ) waiting for $old chain to finish"
until grep -qE "chain DONE|STOP" $R/$old.chain.log 2>/dev/null; do sleep 60; done
tail -1 $R/$old.chain.log
exec bash $H/stop_then_chain.yihou.sh "$@"
