#!/usr/bin/env bash
# Task v0521cap2 (login host, plain-ssh node 136): wait for wts64 (wts48 stack, C64 A2) to finish, stop the wts48 stack, then run
# pf94 (prefill mem 0.94) via chain_cap2l 48 48.
set -uo pipefail
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
ts(){ date -u +%FT%TZ; }
echo "$(ts) waiting for wts48 C64 A2"
until grep -qE "bench	(ok|FAIL).*agentx-c64-A2" $R/wts48/status.tsv 2>/dev/null; do sleep 60; done
grep -a "agentx-c64-A2" $R/wts48/status.tsv | tail -1
( cd $H && source $H/env.wts48-136.yihou.sh && bash $H/run.v3.yihou.sh stop ) || echo "$(ts) stop rc!=0 (checking)"
left=$(ssh -o BatchMode=yes crsuse2-m2m-136 "docker ps --format '{{.Names}}' | grep -c '^glm52-v0521-yihou-136-wts48-'")
[[ "$left" == 0 ]] || { echo "$(ts) STOP: $left wts48 containers left"; exit 1; }
echo "$(ts) wts48 stopped -> pf94"
exec bash $H/chain_cap2l.yihou.sh $H/env.pf94-136.yihou.sh 48 48 > $R/pf94.chain.log 2>&1 < /dev/null
