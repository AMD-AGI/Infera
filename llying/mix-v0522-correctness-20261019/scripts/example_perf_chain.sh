#!/usr/bin/env bash
# Runs AgentX C48 points with the example kit one after another on one node: up.sh, agentx.sh,
# down.sh, then waits for the GPUs' VRAM to come back before the next point.
# Usage (on the GPU node): example_perf_chain.sh SITE_ENV LABEL=KIT_DIR=ENGINE_EXTRA_ARGS ...
#   e.g. base=/path/to/examples/sglang_mix_glm5.2=--disable-custom-all-reduce
# Each point logs to $LOG_DIR/<LABEL>.log (LOG_DIR defaults to the SITE_ENV's directory).
set -uo pipefail
site="$1"; shift
log_dir="${LOG_DIR:-$(dirname "$site")}"
for point in "$@"; do
    label="${point%%=*}"; rest="${point#*=}"
    kit="${rest%%=*}"; extra="${rest#*=}"
    log="$log_dir/$label.log"
    echo "$(date -u +%T) $label start (kit=$kit extra=[$extra])" | tee -a "$log"
    (
        set -a; . "$site"; set +a
        cd "$kit" || exit 1
        SIM_ACCEPT_LEN=3.61 ENGINE_EXTRA_ARGS="$extra" bash engine/up.sh &&
            CONC=48 bash engine/agentx/agentx.sh
        bash engine/down.sh
    ) >> "$log" 2>&1
    echo "$(date -u +%T) $label done" | tee -a "$log"
    for _ in $(seq 1 180); do
        busy="$(rocm-smi --showmemuse --json | python3 -c '
import json, sys
d = json.load(sys.stdin)
print(" ".join(k for k, v in d.items() if k in ("card0", "card1", "card2", "card3")
               and float(v.get("GPU Memory Allocated (VRAM%)", 0)) > 1))')"
        [[ -z "$busy" ]] && break
        sleep 10
    done
done
