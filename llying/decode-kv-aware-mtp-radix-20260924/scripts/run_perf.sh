#!/usr/bin/env bash
# Purpose: Run performance arms one after another, unattended: check GPUs are
#   free, launch, AgentX at CONC (full mode from the config), snapshot metrics
#   and the router log, stop, wait for VRAM release.
# Usage: [EVIDENCE_DIR=DIR] [AGENTX_EVIDENCE_DIR=DIR] [WAIT_FREE_TRIES=N] ./run_perf.sh ARM [ARM ...]
#   ARM = a-perf | b-perf | c-perf | ... (results/decrad/config.decrad.ARM.sh)
#   WAIT_FREE_TRIES: 15 s polls for free GPUs before each arm (default 40).
#   AGENTX_EVIDENCE_DIR: where the AgentX client writes <arm>-agentx-c<CONC>/
#     (the bulk, ~1.5 GB per arm); default EVIDENCE_DIR. It must be writable
#     from CONTROL_NODE. When it differs, EVIDENCE_DIR gets a symlink to it.
# Artifacts: EVIDENCE_DIR (default analysis/evidence/perf)/<arm>-launch/,
#   <arm>-agentx-c<CONC>/, <arm>-{prefill,decode}-{before,after}.prom,
#   <arm>-router.log, run.log
# The deployment is stopped on every exit path, including interruption.
set -uo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
H="$DIR/scripts/bench-harness"
R="$H/results/decrad"
E="${EVIDENCE_DIR:-$DIR/analysis/evidence/perf}"
AX="${AGENTX_EVIDENCE_DIR:-$E}"
TOPO="$R/topology.yihou.tsv"
CONC="${CONC:-40}"
INFERENCEX_DIR="${INFERENCEX_DIR:-$(cd "$DIR/../../bench/glm5p2_pd/.cache/InferenceX" && pwd)}"
AGENTX_CACHE_DIR="${AGENTX_CACHE_DIR:-$HOME/.cache/agentx-decrad}"
NODES=(crsuse2-m2m-135 crsuse2-m2m-138)
mkdir -p "$E" "$AX"

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$E/run.log"; }
vram() {  # GPUs 2-5 VRAM% on both nodes, space separated
    for n in "${NODES[@]}"; do
        ssh -o BatchMode=yes -o ConnectTimeout=10 "$n" \
            'rocm-smi --showmemuse --csv | grep -E "card[2-5]," | cut -d, -f2 | tr "\n" " "'
    done
}
wait_free() {
    for _ in $(seq 1 "${1:-40}"); do
        [[ "$(vram)" =~ ^(0 ){8}$ ]] && return 0
        sleep 15
    done
    return 1
}
current=""
stop_arm() {
    [[ -n "$current" ]] || return 0
    log "stopping $current"
    "$H/stop.sh" CONFIG="$R/config.decrad.$current.sh" TOPOLOGY="$TOPO" >>"$E/$current-stop.log" 2>&1
    wait_free && log "GPUs 2-5 released" || log "WARNING: VRAM not released after stop: $(vram)"
    current=""
}
trap 'stop_arm' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for arm in "$@"; do
    cfg="$R/config.decrad.$arm.sh"
    [[ -r "$cfg" ]] || { log "no config for $arm"; exit 2; }
    eval "$(bash -c "set -a; source '$cfg'; echo ENGINE_PORT_BASE=\$ENGINE_PORT_BASE CONTAINER_PREFIX=\$CONTAINER_PREFIX CONTROL_NODE=\$CONTROL_NODE; printf 'AGENTX_OVERRIDES=%q\n' \"\${AGENTX_OVERRIDES:-}\"")"
    read -r -a agentx_overrides <<< "$AGENTX_OVERRIDES"
    p_url="http://$(python3 "$H/tools/topology.py" node-ip "$TOPO" crsuse2-m2m-135):$ENGINE_PORT_BASE"
    d_url="http://$(python3 "$H/tools/topology.py" node-ip "$TOPO" crsuse2-m2m-138):$((ENGINE_PORT_BASE + 1))"
    snap() {
        curl -fsS --max-time 60 "$p_url/metrics" >"$E/$arm-prefill-$1.prom" 2>/dev/null
        curl -fsS --max-time 60 "$d_url/metrics" >"$E/$arm-decode-$1.prom" 2>/dev/null
    }

    wait_free "${WAIT_FREE_TRIES:-40}" || { log "GPUs 2-5 busy before $arm: $(vram); aborting"; exit 1; }
    log "== $arm launch"
    current="$arm"
    if ! "$H/launch.sh" CONFIG="$cfg" TOPOLOGY="$TOPO" REMOTE_BENCH_DIR="$H" \
        OUT_DIR="$E/$arm-launch" >"$E/$arm-launch.log" 2>&1; then
        log "$arm launch failed; see $arm-launch.log"
        stop_arm
        continue
    fi
    snap before
    log "== $arm AgentX C$CONC"
    out="$AX/$arm-agentx-c$CONC"
    timeout 9000 "$H/agentx_bench.sh" CONC="$CONC" CONFIG="$cfg" TOPOLOGY="$TOPO" \
        INFERENCEX_DIR="$INFERENCEX_DIR" AGENTX_CACHE_DIR="$AGENTX_CACHE_DIR" \
        OUT_DIR="$out" "${agentx_overrides[@]}" >"$E/$arm-agentx-c$CONC.log" 2>&1
    rc=$?
    [[ "$AX" == "$E" ]] || ln -sfn "$out" "$E/$arm-agentx-c$CONC"
    sleep 20  # same late-result guard as yihou's sweep
    if [[ -s "$out/agentx_conc$CONC.json" ]]; then
        log "$arm AgentX done (rc=$rc)"
    else
        log "$arm AgentX FAILED (rc=$rc); see $arm-agentx-c$CONC.log"
    fi
    snap after
    ssh -o BatchMode=yes "$CONTROL_NODE" "docker logs $CONTAINER_PREFIX-router 2>&1 | sed 's/\x1b\[[0-9;]*m//g'" \
        >"$E/$arm-router.log" 2>&1
    stop_arm
done
log "all arms finished"
