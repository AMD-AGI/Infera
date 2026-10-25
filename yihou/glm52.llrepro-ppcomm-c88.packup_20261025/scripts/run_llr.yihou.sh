#!/usr/bin/env bash
# run_llr (task llrepro 2026-10-23, ON 135): one llying-kit point: wait our GPUs 4-7 idle on both nodes, remove only
# OUR previous exited yihou-llr-* containers, up.sh, agentx.sh, save docker logs, docker stop (containers kept).
# Usage: run_llr.yihou.sh <site.env> <run-id>
# 10-25: WORK_DIR may be node-local (/mnt/m2m_nobackup/yihou_llrepro/work, /shared_nfs hit quota). current_run is read
# from $WORK_DIR; after the run the compact results (no agentx/tmp, no file > 300 MB) are copied to
# $D/work/runs/<run-id> on NFS when WORK_DIR is not already there.
set -uo pipefail
D=/shared_nfs/yihou_glm52/llrepro_20261023; K=$D/kit/sglang_2p1d_glm5.2
source "$1"; export RUN_ID=$2
ts(){ date -u +%FT%TZ; }
echo "$(ts) run_llr $RUN_ID CTR=$CTR IMAGE=$IMAGE WORK_DIR=$WORK_DIR"
for i in $(seq 1 120); do
  b5=$(docker ps -q | wc -l); b6=$(ssh -o BatchMode=yes $DECODE_SSH "docker ps -q | wc -l")
  [[ "$b5" == 0 && "$b6" == 0 ]] && break; echo "$(ts) waiting: running containers 135=$b5 136=$b6"; sleep 60; done
[[ "$b5" == 0 && "$b6" == 0 ]] || { echo "$(ts) STOP: nodes busy"; exit 1; }
for c in $(docker ps -a --filter status=exited --format '{{.Names}}' | grep "^$CTR-"); do docker rm "$c" >/dev/null && echo "rm old $c"; done
for c in $(ssh -o BatchMode=yes $DECODE_SSH "docker ps -a --filter status=exited --format '{{.Names}}'" | grep "^$CTR-"); do ssh -o BatchMode=yes $DECODE_SSH "docker rm $c" >/dev/null && echo "rm old 136:$c"; done
df -h / /mnt/m2m_nobackup /shared_nfs | tail -3
echo "$(ts) up.sh"; bash $K/engine/up.sh; rc=$?; echo "$(ts) up rc=$rc"
if [[ $rc == 0 ]]; then echo "$(ts) agentx C$CONC"; bash $K/engine/agentx/agentx.sh; echo "$(ts) agentx rc=$?"; fi
RD=$(cat $WORK_DIR/current_run 2>/dev/null); mkdir -p "$RD/logs"
for c in $(docker ps -a --format '{{.Names}}' | grep "^$CTR-"); do docker logs "$c" > "$RD/logs/$c.log" 2>&1; docker stop -t 30 "$c" >/dev/null; done
for c in $(ssh -o BatchMode=yes $DECODE_SSH "docker ps -a --format '{{.Names}}'" | grep "^$CTR-"); do ssh -o BatchMode=yes $DECODE_SSH "docker logs $c" > "$RD/logs/$c.log" 2>&1; ssh -o BatchMode=yes $DECODE_SSH "docker stop -t 30 $c" >/dev/null; done
if [[ "$RD" != $D/* ]]; then
  mkdir -p $D/work/runs/$RUN_ID && rsync -a --exclude 'agentx/tmp/' --max-size=300m "$RD/" $D/work/runs/$RUN_ID/ \
    && echo "$(ts) copied compact results to $D/work/runs/$RUN_ID" || echo "$(ts) WARN: copy to NFS failed (results stay in $RD)"
fi
echo "$(ts) result: $RD/agentx/agentx_conc$CONC.json"; echo "$(ts) RUN_LLR_DONE"
