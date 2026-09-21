#!/usr/bin/env bash
# Purpose: run the remaining P8D8 concurrency points back-to-back against the
#   ALREADY-RUNNING deployment, so no cluster time is lost waiting for a human to
#   notice that a point finished. Captures RDMA rail-fault counts either side of
#   each point, because a benchmark measured over a dead rail is not a measurement.
# Usage: ./sweep_driver.yihou.sh CONC [CONC ...]
#   Env: WAIT_FOR=<path to an expected result json>  -- if set, the driver blocks
#        until that file exists before starting (used to chain behind a point that
#        is already in flight).
# Artifacts: per point, $SWEEP/c<NNN>/ plus $SWEEP/rails-c<NNN>.txt; a running log
#   at $SWEEP/driver.log.
#
# IT DOES NOT LAUNCH OR STOP ENGINES. One deployment serves every point --
# relaunching per point would cost a ~33-minute HiCache release each time (measured
# 2026-09-18) and would add "a different engine process" as a variable to what is
# supposed to be a single-variable concurrency sweep.
set -uo pipefail

BENCH="${BENCH:-/home/yihou/dev/git/infera.yihou.glm52.p8p4/bench/glm5p2_pd}"
WS="${WS:-/home/yihou/dev/git/infera.glm52.view/glm52.p8d8-vs-p4d4_20260918}"
SWEEP="${SWEEP:-/mnt/m2m_nobackup/yihou_p8p4/sweep}"
LOGD="${LOGD:-/mnt/m2m_nobackup/yihou_p8p4/logs}"
SRVLOG="${SRVLOG:-/mnt/m2m_nobackup/yihou_p8p4/launch/sweep-final/server-logs/prefill-0.log}"
CACHE="${CACHE:-/mnt/m2m_nobackup/yihou_p8p4/agentx-cache}"
CONFIG="$WS/scripts/config.yihou.p8d8.sh"
TOPO="$WS/scripts/topology.yihou.tsv"

(( $# > 0 )) || { echo "usage: $0 CONC [CONC ...]" >&2; exit 2; }
mkdir -p "$SWEEP" "$LOGD" "$CACHE"
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$SWEEP/driver.log"; }

rails() { strings "$SRVLOG" 2>/dev/null |
    grep -cE "transport retry counter exceeded|wqe is not posted" || echo 0; }

if [[ -n "${WAIT_FOR:-}" ]]; then
    say "waiting for in-flight point to produce $WAIT_FOR"
    while [[ ! -s "$WAIT_FOR" ]]; do sleep 60; done
    say "in-flight point finished"
fi

for conc in "$@"; do
    tag="c$(printf '%03d' "$conc")"
    out="$SWEEP/$tag"
    if [[ -e "$out" ]]; then say "SKIP $tag: $out already exists"; continue; fi

    # Health gate: all three endpoints must answer before spending an hour.
    ok=1
    for u in http://10.245.153.247:29001/health \
             http://10.245.154.168:29002/health \
             http://10.245.153.247:28000/health; do
        code=$(curl -s -m 8 -o /dev/null -w "%{http_code}" "$u")
        [[ "$code" == 200 ]] || { say "ABORT $tag: $u returned $code"; ok=0; }
    done
    (( ok )) || { say "deployment unhealthy -- stopping the driver"; exit 1; }

    before=$(rails)
    say "START $tag (CONC=$conc, DURATION inherited 3600) rail_faults_before=$before"
    ( cd "$BENCH" && ./agentx_bench.sh "CONC=$conc" \
        "CONFIG=$CONFIG" "TOPOLOGY=$TOPO" "OUT_DIR=$out" \
        "AGENTX_CACHE_DIR=$CACHE" ) > "$LOGD/bench-$tag.log" 2>&1
    rc=$?
    after=$(rails)
    printf 'point=%s rail_faults_before=%s rail_faults_after=%s exit=%s\n' \
        "$tag" "$before" "$after" "$rc" > "$SWEEP/rails-$tag.txt"
    say "END $tag exit=$rc rail_faults_after=$after"

    if (( rc != 0 )); then
        # The mission accepts an out-of-memory failure at the top point as an end
        # state. Any non-zero exit stops the driver so a human reads the log rather
        # than letting later points run against a degraded server.
        say "point $tag exited non-zero -- stopping. Read $LOGD/bench-$tag.log"
        exit "$rc"
    fi
done
say "sweep driver finished: $*"
