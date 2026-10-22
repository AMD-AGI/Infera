#!/usr/bin/env bash
# Task v0521cap2 (login host, node 136): after wts64 -> stop wts48 stack -> ofs0c (OffloaderV2 S0, sim OFF, correctness via chain_cxl)
# -> ofs0 (S0 perf, chain_cap2l 48 48) -> pf94 (chain_cap2l 48 48). Each step starts only if the previous one did not STOP.
set -uo pipefail
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
ts(){ date -u +%FT%TZ; }
echo "$(ts) waiting for wts48 C64 A2"
until grep -qE "bench	(ok|FAIL).*agentx-c64-A2" $R/wts48/status.tsv 2>/dev/null; do sleep 60; done
grep -a "agentx-c64-A2" $R/wts48/status.tsv | tail -1
( cd $H && source $H/env.wts48-136.yihou.sh && bash $H/run.v3.yihou.sh stop ) || echo "$(ts) stop rc!=0 (checking)"
left=$(ssh -o BatchMode=yes crsuse2-m2m-136 "docker ps --format '{{.Names}}' | grep -c '^glm52-v0521-yihou-136-wts48-'")
[[ "$left" == 0 ]] || { echo "$(ts) STOP: $left wts48 containers left"; exit 1; }
echo "$(ts) -> ofs0c (correctness)"
bash $H/chain_cxl.yihou.sh $H/env.ofs0c-136.yihou.sh > $R/ofs0c.chain.log 2>&1 < /dev/null
grep -q "cx DONE" $R/ofs0c.chain.log || { echo "$(ts) STOP: ofs0c did not finish"; exit 1; }
echo "$(ts) -> ofs0 (perf)"
bash $H/chain_cap2l.yihou.sh $H/env.ofs0-136.yihou.sh 48 48 > $R/ofs0.chain.log 2>&1 < /dev/null
echo "$(ts) ofs0 chain rc=$?; stop ofs0 stack -> pf94"
( cd $H && source $H/env.ofs0-136.yihou.sh && bash $H/run.v3.yihou.sh stop ) || echo "$(ts) stop rc!=0"
exec bash $H/chain_cap2l.yihou.sh $H/env.pf94-136.yihou.sh 48 48 > $R/pf94.chain.log 2>&1 < /dev/null
