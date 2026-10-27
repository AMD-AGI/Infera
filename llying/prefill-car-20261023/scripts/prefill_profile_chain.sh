#!/usr/bin/env bash
# One deployment for a prefill profile under AgentX load: wait for both nodes' VRAM -> up.sh (SIM_ACCEPT_LEN=$SIM,
# default 3.61; SIM= for real acceptance) -> with LONGCTX_PROMPTS set, a long-context probe through the router (128
# requests of ~16k tokens at 32 in flight, garbled / runaway counts; needs real acceptance) -> AgentX at CONC for DURATION s in the background -> PROFILE_AFTER s after "Warmup complete", POST /start_profile
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
# Only this deployment's GPUs must be free: another job may hold the rest of a node.
busy() {
    on "$1" "rocm-smi --showmeminfo vram --csv" | awk -F, -v g="$2" 'BEGIN {n = split(g, a, ","); for (i = 1; i <= n; i++) w["card" a[i]] = 1}
        NR > 1 && ($1 in w) && $2 != "" && 100 * $3 / $2 >= 1 {print $1}'
}
p_gpus="$PREFILL0_GPUS"
[[ "${PREFILL_WORKERS:-2}" == 1 ]] || p_gpus+=",${PREFILL1_GPUS:-4,5,6,7}"
until [[ -z "$(busy localhost "$p_gpus")$(busy "$DECODE_SSH" "${DECODE_GPUS:-0,1,2,3,4,5,6,7}")" ]]; do sleep 60; done
{
    for n in "${nodes[@]}"; do
        echo "===== $n $(date -u +%FT%TZ)"
        on "$n" "rocm-smi --showmeminfo vram --csv; rocm-smi --showpids; docker stats --no-stream --format '{{.Name}} {{.CPUPerc}} {{.MemUsage}}'; uptime"
    done
} >"$log_dir/$label.env-check" 2>&1
echo "$(date -u +%T) $label up"
if ! (cd "$kit" && SIM_ACCEPT_LEN="${SIM-3.61}" RUN_ID="$label-$(date -u +%m%dT%H%M)" bash engine/up.sh) >"$log_dir/$label.log" 2>&1; then
    echo "$(date -u +%T) $label up failed"
    (cd "$kit" && bash engine/down.sh) >>"$log_dir/$label.log" 2>&1
    exit 1
fi
run_dir="$(cat "$WORK_DIR/current_run")"
if [[ -n "${LONGCTX_PROMPTS:-}" ]]; then
    timeout 3600 python3 "$(dirname "$0")/../../mix-v0522-correctness-20261019/scripts/longctx_probe.py" \
        "$LONGCTX_PROMPTS" "http://$PREFILL_NODE_IP:${ROUTER_PORT:-28300}" "${SERVED_MODEL:-glm5.2-mxfp4}" \
        "$log_dir/$label.longctx.jsonl" --n 128 --concurrency 32 >>"$log_dir/$label.log" 2>&1
    echo "$(date -u +%T) $label longctx probe exited $?: $(tail -1 "$log_dir/$label.log")"
fi
(cd "$kit" && CONC="$conc" DURATION="$duration" bash engine/agentx/agentx.sh) >>"$log_dir/$label.log" 2>&1 &
ax=$!
until tr '\r' '\n' <"$log_dir/$label.log" | grep -aq 'Warmup complete'; do
    kill -0 "$ax" 2>/dev/null || break
    sleep 30
done
echo "$(date -u +%T) $label warmup complete; profiling in ${after}s"
sleep "$after"
pid="prefill-$(date -u +%H%M%S)"
# PROFILE_RECORD_SHAPES=1 also records the operators' input shapes (GEMM M / N / K), at some profiling overhead.
shapes=""
[[ "${PROFILE_RECORD_SHAPES:-0}" != 1 ]] || shapes=', "record_shapes": true'
for ((i = 0; i < ${PREFILL_WORKERS:-1}; i++)); do
    curl -s -m 60 -X POST "http://$PREFILL_NODE_IP:$((27201 + 256 * i))/start_profile" -H 'Content-Type: application/json' \
        -d "{\"output_dir\": \"/root/.cache/$pid\", \"num_steps\": $steps, \"activities\": [\"CPU\", \"GPU\"], \"profile_id\": \"$pid-p$i\"$shapes}"
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
