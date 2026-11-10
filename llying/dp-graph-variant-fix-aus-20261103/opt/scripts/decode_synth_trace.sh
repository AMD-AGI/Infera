#!/usr/bin/env bash
# Decode traces at fixed per-rank batch without AgentX, short enough for the 30 min a job is exempt from preemption
# on AUS (AgentX needs ~40 min before its decode batch is representative): wait for both nodes' VRAM -> up.sh
# (SIM_ACCEPT_LEN=$SIM) -> pkg/scripts/synth_decode_load.py with stepped concurrency -> PROFILE_AT s into every stage,
# POST /start_profile to the decode (NUM_STEPS steps, CPU + GPU, all DP ranks) -> copy each trace out of the decode
# container at the end of its stage (and all of them again at the end), so a preemption later loses nothing -> down.sh.
# Usage: decode_synth_trace.sh KIT SITE_ENV LOG_DIR [STAGES=96:150,160:150,192:150] [PROFILE_AT=90] [NUM_STEPS=40]
set -uo pipefail
kit="$1" site="$2" log_dir="$3" stages="${4:-96:150,160:150,192:150}" at="${5:-90}" steps="${6:-40}"
set -a
# shellcheck source=/dev/null
. "$site"
set +a
mkdir -p "$log_dir/traces"
label="strace"
on() {
    if [[ "$1" == localhost || "$1" == "$(hostname)" || "$1" == "$(hostname -s)" ]]; then bash -c "$2"
    elif ! timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=8 "$1" true 2>/dev/null && [[ -n "${SLURM_JOB_ID:-}" ]]; then
        srun --jobid="$SLURM_JOB_ID" --overlap -N1 -n1 -w "$1" bash -c "$2"
    else ssh -o BatchMode=yes "$1" "$2"; fi
}
busy() {
    on "$1" "rocm-smi --showmeminfo vram --csv" | awk -F, -v g="$2" 'BEGIN {n = split(g, a, ","); for (i = 1; i <= n; i++) w["card" a[i]] = 1}
        NR > 1 && ($1 in w) && $2 != "" && 100 * $3 / $2 >= 1 {print $1}'
}
copy() { on "$DECODE_SSH" "docker cp ${CTR:-glm52_2p1d}-decode:/root/.cache/$1 $log_dir/traces/$1" 2>&1 | tail -1; }
for attempt in 1 2; do
    until [[ -z "$(busy localhost "$PREFILL0_GPUS")$(busy "$DECODE_SSH" "${DECODE_GPUS:-0,1,2,3,4,5,6,7}")" ]]; do sleep 60; done
    echo "$(date -u +%T) $label up (attempt $attempt)"
    if (cd "$kit" && SIM_ACCEPT_LEN="${SIM-3.61}" RUN_ID="$label-$(date -u +%m%dT%H%M)" bash engine/up.sh) >"$log_dir/$label.log" 2>&1; then
        break
    fi
    echo "$(date -u +%T) $label up failed (attempt $attempt)"
    cp "$log_dir/$label.log" "$log_dir/$label.log.failed$attempt"
    (cd "$kit" && bash engine/down.sh) >>"$log_dir/$label.log" 2>&1
    [[ $attempt == 2 ]] && exit 1
done
run_dir="$(cat "$WORK_DIR/current_run")"
python3 "$(dirname "$site")/../scripts/synth_decode_load.py" "http://$PREFILL_NODE_IP:${ROUTER_PORT:-28300}" "${SERVED_MODEL:-glm5.2-mxfp4}" \
    --stages "$stages" >>"$log_dir/$label.log" 2>&1 &
load=$!
pids=()
for st in ${stages//,/ }; do
    conc="${st%%:*}" secs="${st##*:}"
    sleep "$at"
    pid="decode-c$conc-$(date -u +%H%M%S)"
    curl -s -m 60 -X POST "http://$DECODE_NODE_IP:${DECODE_PORT:-27301}/start_profile" -H 'Content-Type: application/json' \
        -d "{\"output_dir\": \"/root/.cache/$pid\", \"num_steps\": $steps, \"activities\": [\"CPU\", \"GPU\"], \"profile_id\": \"$pid\"}"
    echo
    echo "$(date -u +%T) $label conc $conc: profile requested ($pid)"
    pids+=("$pid")
    sleep $((secs - at))
    copy "$pid"
done
wait "$load"
sleep 30
for pid in "${pids[@]}"; do copy "$pid"; done
ls -la "$log_dir"/traces/*/ 2>&1 | tail -12
mkdir -p "$run_dir/metrics"
curl -s -m 10 "http://$DECODE_NODE_IP:${DECODE_PORT:-27301}/metrics" >"$run_dir/metrics/decode.prom"
(cd "$kit" && bash engine/down.sh) >>"$log_dir/$label.log" 2>&1
echo "$(date -u +%T) $label down ($run_dir)"
