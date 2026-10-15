#!/usr/bin/env bash
# Run sweep points strictly one after another on the held node (one model in memory at a time).
# Usage (node): bash serial-sweep.yihou.sh  "NAME BS ISL GRAPH SHAPE_STEP PROFILE" ...
set -u
WS=/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.decode-side-prefill-shapes.yihou_20261008
OUT=/mnt/m2m_nobackup/yihou/dsp-yihou/out
for spec in "$@"; do
    read -r NAME BS ISL GRAPH SHAPE_STEP PROFILE <<< "$spec"
    echo "$(date -u +%FT%TZ) start $NAME bs=$BS isl=$ISL graph=$GRAPH"
    NAME=$NAME BS=$BS ISL=$ISL GRAPH=$GRAPH SHAPE_STEP=$SHAPE_STEP PROFILE=$PROFILE EAGER="${EAGER:-}" OSL=${OSL:-300} \
        MAX_STEPS=${MAX_STEPS:-40} WARMUP=${WARMUP:-5} PSTART=${PSTART:-15} PNUM=${PNUM:-3} \
        bash "$WS/scripts/run-point.yihou.sh" || { echo "launch failed $NAME"; continue; }
    until [[ -f "$OUT/$NAME/exit_code" ]]; do sleep 30; done
    echo "$(date -u +%FT%TZ) end $NAME exit=$(cat "$OUT/$NAME/exit_code")"
    # Wait for the GPUs to be released before the next point (no overlap, no orphan ranks).
    while docker exec glm52-dsp-yihou pgrep -f multiprocessing.spawn >/dev/null 2>&1; do sleep 10; done
done
echo "$(date -u +%FT%TZ) sweep done"
