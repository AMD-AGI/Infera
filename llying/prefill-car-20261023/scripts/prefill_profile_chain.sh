#!/usr/bin/env bash
# One deployment for a prefill profile under AgentX load: wait for both nodes' VRAM -> up.sh (SIM_ACCEPT_LEN=3.61)
# -> AgentX at CONC for DURATION s in the background -> PROFILE_AFTER s after "Warmup complete", POST /start_profile
# to every prefill (NUM_STEPS steps, CPU + GPU) -> when AgentX ends, copy the traces out of the containers into
# LOG_DIR/traces/, save /metrics, down.sh. Records who else is on the nodes into LOG_DIR/<label>.env-check.
# Usage: prefill_profile_chain.sh KIT SITE_ENV LOG_DIR CONC [DURATION=1500] [PROFILE_AFTER=600] [NUM_STEPS=30]
set -uo pipefail
kit="$1" site="$2" log_dir="$3" conc="$4" duration="${5:-1500}" after="${6:-600}" steps="${7:-30}"
set -a
# shellcheck source=/dev/null
. "$site"
set +a
mkdir -p "$log_dir/traces"
label="prof-c$conc"
nodes=(localhost "$DECODE_SSH")
on() { if [[ "$1" == localhost ]]; then bash -c "$2"; else ssh -o BatchMode=yes "$1" "$2"; fi; }
vram() { on "$1" "rocm-smi --showmeminfo vram --csv | awk -F, 'NR>1 && \$2 != \"\" {printf \"%d \", 100 * \$3 / \$2}'"; }
for n in "${nodes[@]}"; do
    until [[ -z "$(vram "$n" | tr ' ' '\n' | awk '$1 >= 1')" ]]; do sleep 60; done
done
{
    for n in "${nodes[@]}"; do
        echo "===== $n $(date -u +%FT%TZ)"
        on "$n" "rocm-smi --showmeminfo vram --csv; rocm-smi --showpids; docker stats --no-stream --format '{{.Name}} {{.CPUPerc}} {{.MemUsage}}'; uptime"
    done
} >"$log_dir/$label.env-check" 2>&1
echo "$(date -u +%T) $label up"
if ! (cd "$kit" && SIM_ACCEPT_LEN=3.61 RUN_ID="$label-$(date -u +%m%dT%H%M)" bash engine/up.sh) >"$log_dir/$label.log" 2>&1; then
    echo "$(date -u +%T) $label up failed"
    (cd "$kit" && bash engine/down.sh) >>"$log_dir/$label.log" 2>&1
    exit 1
fi
run_dir="$(cat "$WORK_DIR/current_run")"
(cd "$kit" && CONC="$conc" DURATION="$duration" bash engine/agentx/agentx.sh) >>"$log_dir/$label.log" 2>&1 &
ax=$!
until tr '\r' '\n' <"$log_dir/$label.log" | grep -aq 'Warmup complete'; do
    kill -0 "$ax" 2>/dev/null || break
    sleep 30
done
echo "$(date -u +%T) $label warmup complete; profiling in ${after}s"
sleep "$after"
pid="prefill-$(date -u +%H%M%S)"
for ((i = 0; i < ${PREFILL_WORKERS:-1}; i++)); do
    curl -s -m 60 -X POST "http://$PREFILL_NODE_IP:$((27201 + 256 * i))/start_profile" -H 'Content-Type: application/json' \
        -d "{\"output_dir\": \"/root/.cache/$pid\", \"num_steps\": $steps, \"activities\": [\"CPU\", \"GPU\"], \"profile_id\": \"$pid-p$i\"}"
    echo
done
echo "$(date -u +%T) $label profile requested ($pid)"
wait "$ax"
echo "$(date -u +%T) $label agentx exited $?"
for ((i = 0; i < ${PREFILL_WORKERS:-1}; i++)); do
    docker cp "${CTR:-glm52_2p1d}-prefill$i:/root/.cache/$pid" "$log_dir/traces/$pid-p$i" 2>&1 | tail -1
done
ls -la "$log_dir"/traces/*/ 2>&1 | tail -8
mkdir -p "$run_dir/metrics"
for ((i = 0; i < ${PREFILL_WORKERS:-1}; i++)); do
    curl -s -m 10 "http://$PREFILL_NODE_IP:$((27201 + 256 * i))/metrics" >"$run_dir/metrics/prefill$i.prom"
done
curl -s -m 10 "http://$DECODE_NODE_IP:27301/metrics" >"$run_dir/metrics/decode.prom"
(cd "$kit" && bash engine/down.sh) >>"$log_dir/$label.log" 2>&1
echo "$(date -u +%T) $label down ($run_dir)"
