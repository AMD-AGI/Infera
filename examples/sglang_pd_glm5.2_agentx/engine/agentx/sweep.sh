#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# AgentX points one after another, the InferenceX way: every CONC gets a fresh
# deployment (up.sh with SIM_ACCEPT_LEN, by default the golden acceptance of
# MTP_STEPS: 2.99 / 3.33 / 3.61) -> agentx.sh -> down.sh,
# and the next point starts only after the deployment's GPUs have released
# their VRAM. Before each point it records who else is on the nodes (GPU
# processes, containers, load) into <sweep>/c<CONC>.env-check for review; it
# does not judge them. A watchdog stops the client when an engine dies or
# nothing completes for 15 minutes, so a broken deployment does not hold the
# nodes for the rest of the hour. Ends with one Markdown row per point
# (summarize.py) in <sweep>/summary.md. common.sh runs the kit under set -e, so
# a check that may fail ends in `|| true`.
#
# Usage: bash engine/agentx/sweep.sh CONC...   (e.g. 32 64 104; SWEEP_ID=<name>)
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../common.sh
source "$KIT/engine/common.sh"
(($#)) || die "usage: bash engine/agentx/sweep.sh CONC..."

sweep="$WORK_DIR/sweeps/${SWEEP_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$sweep"
# InferenceX's golden acceptance for the deployment's draft steps.
golden=([3]=2.99 [4]=3.33 [5]=3.61)
export SIM_ACCEPT_LEN="${SIM_ACCEPT_LEN:-${golden[$MTP_STEPS]}}"
where=(local)
[[ "$DECODE_NODE_IP" == "$PREFILL_NODE_IP" ]] || where+=(decode)
on() { if [[ "$1" == local ]]; then bash -c "$2"; else on_decode "$2"; fi; }
gpus_of() {
    if [[ "$1" == decode ]]; then
        local d="$DECODE_GPUS"
        ((DECODE_WORKERS == 1)) || d+=",$DECODE1_GPUS"
        echo "$d"
    else
        local g="$PREFILL0_GPUS"
        ((PREFILL_WORKERS == 1)) || g+=",$PREFILL1_GPUS"
        [[ "${where[*]}" == *decode* ]] || g+=",$DECODE_GPUS"
        echo "$g"
    fi
}
# busy WHERE: how many of the deployment's GPUs there still hold >= 1% VRAM.
busy() {
    on "$1" "rocm-smi --showmeminfo vram --csv" | awk -F, -v gpus="$(gpus_of "$1")" '
        BEGIN { n = split(gpus, g, ","); for (i = 1; i <= n; i++) want["card" g[i]] = 1 }
        $1 in want && $2 > 0 && 100 * $3 / $2 >= 1 { b++ }
        END { print b + 0 }'
}
watchdog() {
    local log="$1" last="" since=$SECONDS fault progress i
    while sleep 60; do
        fault="$( {
            for ((i = 0; i < PREFILL_WORKERS; i++)); do docker logs --since 70s "$CTR-prefill$i" 2>&1; done
            for c in "${DECODE_CONTAINERS[@]}"; do on_decode "docker logs --since 70s $c 2>&1"; done
        } | grep -aE 'Fatal Python error|crashed with exit code' | head -1)" || true
        progress="$(tr '\r' '\n' <"$log" | grep -aoE '(returned|done)=[0-9,]+' | tail -1)" || true
        [[ "$progress" == "$last" ]] || { last="$progress" since=$SECONDS; }
        if [[ -n "$fault" ]] || ((SECONDS - since > 900)); then
            echo "$(date -u +%T) watchdog: stopping AgentX (${fault:-nothing completed for 15 min, last $last})"
            docker stop "$CTR-agentx" >/dev/null 2>&1
            return
        fi
    done
}

for conc in "$@"; do
    label="c$conc"
    for w in "${where[@]}"; do
        for ((i = 0; i < 30; i++)); do
            (($(busy "$w") == 0)) && break
            sleep 60
        done
        (($(busy "$w") == 0)) || die "$label: the $w node's GPUs still hold VRAM after 30 min"
    done
    for w in "${where[@]}"; do
        echo "===== $w $(date -u +%FT%TZ)"
        on "$w" "rocm-smi --showmeminfo vram --csv; rocm-smi --showpids; docker stats --no-stream --format '{{.Name}} {{.CPUPerc}} {{.MemUsage}}'; uptime; ps -eo user,pcpu,etime,comm --sort=-pcpu | head -12" || true
    done >"$sweep/$label.env-check" 2>&1
    echo "$(date -u +%T) $label up"
    if RUN_ID="$(basename "$sweep")-$label" bash "$KIT/engine/up.sh" >"$sweep/$label.log" 2>&1; then
        watchdog "$sweep/$label.log" &
        wd=$!
        rc=0
        CONC="$conc" bash "$KIT/engine/agentx/agentx.sh" >>"$sweep/$label.log" 2>&1 || rc=$?
        echo "$(date -u +%T) $label agentx exited $rc"
        kill "$wd" 2>/dev/null || true
    else
        echo "$(date -u +%T) $label up failed (see $sweep/$label.log)"
    fi
    # The engines' counters at the end of the point, before down.sh removes them.
    run="$(cat "$WORK_DIR/current_run")"
    mkdir -p "$run/metrics"
    for ((i = 0; i < PREFILL_WORKERS; i++)); do
        curl -s -m 10 "http://$PREFILL_NODE_IP:$((PREFILL_PORT + 256 * i))/metrics" >"$run/metrics/prefill$i.prom" || true
    done
    for ((i = 0; i < DECODE_WORKERS; i++)); do
        curl -s -m 10 "http://$DECODE_NODE_IP:$((DECODE_PORT + 256 * i))/metrics" >"$run/metrics/decode$i.prom" || true
    done
    bash "$KIT/engine/down.sh" >>"$sweep/$label.log" 2>&1 || echo "$(date -u +%T) $label down.sh failed"
    echo "$run" >>"$sweep/runs"
    echo "$(date -u +%T) $label down"
done
python3 "$KIT/engine/agentx/summarize.py" $(cat "$sweep/runs") | tee "$sweep/summary.md"
