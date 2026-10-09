#!/usr/bin/env bash
# Wait for the validated launcher artifact, then run smoke and the single approved C32 point.
set -euo pipefail
W=/home/yihou/dev/git/infera.glm52.pd/bench/glm5p2_pd/results/yihou-expert8-pd
R="$W/rounds/004-json-override"
for ((i=0;i<240;i++)); do
    if [[ -s "$R/launch/router-health.json" && -s "$R/launch/workers.json" ]]; then
        if curl -fsS --max-time 5 http://10.245.157.237:28000/health >/dev/null; then
            break
        fi
    fi
    (( i < 239 )) || { printf 'launcher readiness deadline exceeded\n' >&2; exit 1; }
    sleep 15
done
python3 "$W/scripts/collect.yihou.py" --out "$R/before-smoke"
bash "$W/scripts/smoke.yihou.sh" "$R"
python3 "$W/scripts/collect.yihou.py" --out "$R/after-smoke"
bash "$W/scripts/bench_fixed.yihou.sh" "$W/rounds/003-fixed-c32/benchmark"
python3 "$W/scripts/collect.yihou.py" --out "$W/rounds/003-fixed-c32/after-benchmark"
printf 'PASS: operational smoke and fixed C32 point completed\n'
