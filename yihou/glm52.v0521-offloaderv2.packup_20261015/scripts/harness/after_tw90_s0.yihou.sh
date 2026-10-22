#!/usr/bin/env bash
# Task v0521cap2 (ON node 089, local ssh shim): after tw90 chain DONE -> stop tw90 stack -> ofs0c (S0 correctness, chain_cx) -> ofs0 (S0 perf
# chain_cap2 48 48). S0 moved here because node 136 was taken by another user (limou, ~10:35).
set -uo pipefail
N=/shared_nfs/yihou_glm52/v0521cap2_20261013
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export PATH=$N/shimbin:$PATH
ts(){ date -u +%FT%TZ; }
echo "$(ts) waiting for tw90 chain"
until grep -qE "chain DONE|STOP" $R/tw90.chain.log 2>/dev/null; do sleep 60; done
tail -1 $R/tw90.chain.log
( cd $H && source $H/env.tw90-089.yihou.sh && bash $H/run.v3.yihou.sh stop ) || echo "$(ts) stop rc!=0 (checking)"
left=$(docker ps --format '{{.Names}}' | grep -c '^glm52-v0521-yihou-089-tw90-')
[[ "$left" == 0 ]] || { echo "$(ts) STOP: $left tw90 containers left"; exit 1; }
for i in $(seq 1 60); do k=$(rocm-smi --showpids 2>/dev/null | grep -cE '^[0-9]+ '); [[ "$k" == 0 ]] && break; sleep 30; done
echo "$(ts) -> ofs0c"
bash $H/chain_cx.yihou.sh $H/env.ofs0c-089.yihou.sh > $R/ofs0c.chain.log 2>&1 < /dev/null
grep -q "cx DONE" $R/ofs0c.chain.log || { echo "$(ts) STOP: ofs0c did not finish"; exit 1; }
for i in $(seq 1 60); do k=$(rocm-smi --showpids 2>/dev/null | grep -cE '^[0-9]+ '); [[ "$k" == 0 ]] && break; sleep 30; done
echo "$(ts) -> ofs0"
exec bash $H/chain_cap2.yihou.sh $H/env.ofs0-089.yihou.sh 48 48 > $R/ofs0.chain.log 2>&1 < /dev/null
