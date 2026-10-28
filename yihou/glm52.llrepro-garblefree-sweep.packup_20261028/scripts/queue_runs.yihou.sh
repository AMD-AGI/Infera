#!/usr/bin/env bash
# queue_runs (task llrepro 2026-10-26, ON 135): run site files one after another so the nodes do not sit idle.
# Usage: queue_runs.yihou.sh <prev-run-tag> <tag1>:<site1> [<tag2>:<site2> ...]   (tag = run_llr log tag, e.g. llr27)
# Waits for RUN_LLR_DONE of <prev-run-tag>, then for each entry: run_llr + guard, wait for its RUN_LLR_DONE.
# A file STOP_QUEUE.yihou in the llrepro dir stops the queue before the next launch. Touches nothing else.
D=/shared_nfs/yihou_glm52/llrepro_20261023
ts(){ date -u +%FT%TZ; }
waitdone(){ for i in $(seq 1 960); do grep -q RUN_LLR_DONE $D/run_llr.$1.yihou.log 2>/dev/null && return 0; sleep 30; done; return 1; }
waitdone "$1" || { echo "$(ts) prev $1 not done"; exit 1; }; shift
for e in "$@"; do
  tag=${e%%:*}; site=${e#*:}
  [[ -e $D/STOP_QUEUE.yihou ]] && { echo "$(ts) STOP_QUEUE present; stop before $tag"; exit 0; }
  conc=$(grep -o 'CONC=[0-9]*' $site | head -1 | cut -d= -f2)
  echo "$(ts) launch $tag ($site, C$conc)"
  ( cd $D && setsid nohup bash $D/run_llr.yihou.sh $site $tag-c$conc > $D/run_llr.$tag.yihou.log 2>&1 < /dev/null & )
  ( setsid nohup bash $D/guard_llr.yihou.sh $tag > $D/guard.$tag.yihou.log 2>&1 < /dev/null & )
  sleep 60; waitdone "$tag"; echo "$(ts) $tag done"
done
echo "$(ts) QUEUE_DONE"
