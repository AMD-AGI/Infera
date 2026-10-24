#!/usr/bin/env bash
# run_llr (task llrepro 2026-10-23, ON 135): one llying-kit point: wait our GPUs 4-7 idle on both nodes, remove only
# OUR previous exited yihou-llr-* containers, up.sh, agentx.sh, save docker logs, docker stop (containers kept).
# Usage: run_llr.yihou.sh <site.env> <run-id>
set -uo pipefail
D=/shared_nfs/yihou_glm52/llrepro_20261023; K=$D/kit/sglang_2p1d_glm5.2
source "$1"; export RUN_ID=$2
ts(){ date -u +%FT%TZ; }
echo "$(ts) run_llr $RUN_ID CTR=$CTR IMAGE=$IMAGE"
for i in $(seq 1 120); do
  b5=$(docker ps -q | wc -l); b6=$(ssh -o BatchMode=yes $DECODE_SSH "docker ps -q | wc -l")
  [[ "$b5" == 0 && "$b6" == 0 ]] && break; echo "$(ts) waiting: running containers 135=$b5 136=$b6"; sleep 60; done
[[ "$b5" == 0 && "$b6" == 0 ]] || { echo "$(ts) STOP: nodes busy"; exit 1; }
for c in $(docker ps -a --filter status=exited --format '{{.Names}}' | grep "^$CTR-"); do docker rm "$c" >/dev/null && echo "rm old $c"; done
for c in $(ssh -o BatchMode=yes $DECODE_SSH "docker ps -a --filter status=exited --format '{{.Names}}'" | grep "^$CTR-"); do ssh -o BatchMode=yes $DECODE_SSH "docker rm $c" >/dev/null && echo "rm old 136:$c"; done
df -h / /mnt/m2m_nobackup | tail -2
echo "$(ts) up.sh"; bash $K/engine/up.sh; rc=$?; echo "$(ts) up rc=$rc"
if [[ $rc == 0 ]]; then echo "$(ts) agentx C$CONC"; bash $K/engine/agentx/agentx.sh; echo "$(ts) agentx rc=$?"; fi
RD=$(cat $D/work/current_run 2>/dev/null); mkdir -p "$RD/logs"
for c in $(docker ps -a --format '{{.Names}}' | grep "^$CTR-"); do docker logs "$c" > "$RD/logs/$c.log" 2>&1; docker stop -t 30 "$c" >/dev/null; done
for c in $(ssh -o BatchMode=yes $DECODE_SSH "docker ps -a --format '{{.Names}}'" | grep "^$CTR-"); do ssh -o BatchMode=yes $DECODE_SSH "docker logs $c" > "$RD/logs/$c.log" 2>&1; ssh -o BatchMode=yes $DECODE_SSH "docker stop -t 30 $c" >/dev/null; done
echo "$(ts) result: $RD/agentx/agentx_conc$CONC.json"; echo "$(ts) RUN_LLR_DONE"
