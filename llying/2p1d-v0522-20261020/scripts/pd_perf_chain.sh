#!/usr/bin/env bash
# AgentX points on the 2P1D kit one after another, the InferenceX way: every CONC gets a fresh
# deployment (SIM_ACCEPT_LEN=3.61) -> agentx.sh -> down.sh, and the next point starts only after
# both nodes' GPUs have released their VRAM. Before each point it records who else is on the
# nodes (GPU processes and VRAM, containers with CPU use, load) into LOG_DIR/c<CONC>.env-check
# for manual review; it does not judge them.
# Usage: pd_perf_chain.sh KIT SITE_ENV LOG_DIR CONC...
set -uo pipefail
kit="$1" site="$2" log_dir="$3"
shift 3
set -a
# shellcheck source=/dev/null
. "$site"
set +a
mkdir -p "$log_dir"
nodes=(localhost "$DECODE_SSH")
on() { if [[ "$1" == localhost ]]; then bash -c "$2"; else ssh -o BatchMode=yes "$1" "$2"; fi; }
vram() { on "$1" "rocm-smi --showmeminfo vram --csv | awk -F, 'NR>1 && \$2 != \"\" {printf \"%d \", 100 * \$3 / \$2}'"; }

for conc in "$@"; do
    label="c$conc"
    for n in "${nodes[@]}"; do
        until [[ -z "$(vram "$n" | tr ' ' '\n' | awk '$1 >= 1')" ]]; do sleep 60; done
    done
    {
        for n in "${nodes[@]}"; do
            echo "===== $n $(date -u +%FT%TZ)"
            on "$n" "rocm-smi --showmeminfo vram --csv; rocm-smi --showpids; docker stats --no-stream --format '{{.Name}} {{.CPUPerc}} {{.MemUsage}}'; uptime; ps -eo user,pcpu,etime,comm --sort=-pcpu | head -12"
        done
    } >"$log_dir/$label.env-check" 2>&1
    echo "$(date -u +%T) $label up"
    (cd "$kit" && SIM_ACCEPT_LEN=3.61 RUN_ID="$label-$(date -u +%m%dT%H%M)" bash engine/up.sh) >"$log_dir/$label.log" 2>&1 &&
        (cd "$kit" && CONC="$conc" bash engine/agentx/agentx.sh) >>"$log_dir/$label.log" 2>&1
    echo "$(date -u +%T) $label agentx exited $?"
    # The engines' counters at the end of the run, before down.sh removes them.
    run_dir="$(cat "$WORK_DIR/current_run")"
    mkdir -p "$run_dir/metrics"
    for ((i = 0; i < ${PREFILL_WORKERS:-2}; i++)); do
        curl -s -m 10 "http://$PREFILL_NODE_IP:$((27201 + 256 * i))/metrics" >"$run_dir/metrics/prefill$i.prom"
    done
    curl -s -m 10 "http://$DECODE_NODE_IP:27301/metrics" >"$run_dir/metrics/decode.prom"
    (cd "$kit" && bash engine/down.sh) >>"$log_dir/$label.log" 2>&1
    echo "$(date -u +%T) $label down"
done
