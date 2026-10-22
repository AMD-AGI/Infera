#!/usr/bin/env bash
# Task g4tune 2026-10-19: run ON a held node. Wait for prep_node.yihou.sh to finish, verify the node, then start
# chain_cap2 <C1> <C2> for the given env plus the warm-up py-spy watcher. Usage: launch_after_prep.yihou.sh <env> <round> <key> <C1> <C2>
set -uo pipefail
ENVF=$1; RD=$2; K=$3; C1=$4; C2=$5
H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness; R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs
C=/shared_nfs/yihou_glm52/v0521dbg_20261010/cache/$K; W=/shared_nfs/yihou_glm52/g4tune_20261019
LOG=$W/launch.$RD.yihou.log
ts(){ date -u +%FT%TZ; }
cd /
{
echo "$(ts) waiting for prep_node $K"
for i in $(seq 1 120); do pgrep -f "prep_node.yihou.sh $K" > /dev/null || break; sleep 10; done
pgrep -f "prep_node.yihou.sh $K" > /dev/null && { echo "$(ts) STOP: prep still running after 20 min"; exit 1; }
img=$(docker images --no-trunc -q infera-sglang:v0521-yihou-1010-mc4340 | cut -c8-19)
sp=$(ls $C/aiter/prefill/9d755dd6*/ 2>/dev/null | grep -c '\.so$'); sd=$(ls $C/aiter/decode/9d755dd6*/ 2>/dev/null | grep -c '\.so$')
lk=$(find $C/aiter -name 'lock_*' 2>/dev/null | wc -l); kfd=$(rocm-smi --showpids 2>/dev/null | grep -cE '^[0-9]+ ')
echo "$(ts) img=$img so_prefill=$sp so_decode=$sd locks=$lk kfd=$kfd"
tail -3 /shared_nfs/yihou_glm52/v0521cap2_20261013/logs/prep.$K.yihou.log
[[ "$img" == 9d755dd6a7bf && "$sp" -ge 16 && "$sd" -ge 14 && "$lk" == 0 && "$kfd" == 0 ]] || { echo "$(ts) STOP: node check failed"; exit 1; }
[[ -e $R/$RD ]] && { echo "$(ts) STOP: run dir $R/$RD exists"; exit 1; }
NOW=$(date -u +%FT%T)
setsid nohup bash $H/chain_cap2.yihou.sh $ENVF $C1 $C2 > $R/$RD.chain.log 2>&1 < /dev/null &
setsid nohup bash $H/warmup_pyspy.yihou.sh $RD $K $NOW 300 > $R/$RD.pyspy.log 2>&1 < /dev/null &
echo "$(ts) launched chain_cap2 $C1 $C2 round=$RD"
} >> $LOG 2>&1
