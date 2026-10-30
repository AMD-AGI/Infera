#!/usr/bin/env bash
# One deployment for a decode profile under AgentX load (data collection, not scored): wait for both nodes' VRAM ->
# up.sh (SIM_ACCEPT_LEN=$SIM) -> AgentX at CONC for DURATION s in the background -> PROFILE_AFTER s after
# "Warmup complete", POST /start_profile to the decode (NUM_STEPS steps, CPU + GPU; all DP ranks), a second request
# GAP s later -> when AgentX ends, copy the traces out of the decode container into LOG_DIR/traces/, save /metrics,
# down.sh. Adapted from prefill-car-20261023/scripts/prefill_profile_chain.sh.
# Usage: decode_profile_chain.sh KIT SITE_ENV LOG_DIR CONC [DURATION=1500] [PROFILE_AFTER=600] [NUM_STEPS=40] [GAP=300]
set -uo pipefail
kit="$1" site="$2" log_dir="$3" conc="$4" duration="${5:-1500}" after="${6:-600}" steps="${7:-40}" gap="${8:-300}"
set -a
# shellcheck source=/dev/null
. "$site"
set +a
mkdir -p "$log_dir/traces"
label="dprof-c$conc"
nodes=(localhost "$DECODE_SSH")
on() { if [[ "$1" == localhost ]]; then bash -c "$2"; else ssh -o BatchMode=yes "$1" "$2"; fi; }
busy() {
    on "$1" "rocm-smi --showmeminfo vram --csv" | awk -F, -v g="$2" 'BEGIN {n = split(g, a, ","); for (i = 1; i <= n; i++) w["card" a[i]] = 1}
        NR > 1 && ($1 in w) && $2 != "" && 100 * $3 / $2 >= 1 {print $1}'
}
# One retry: 10-29 two of three start-ups failed on the decode's Mooncake ibv_reg_mr (EINVAL, same 401,795,328 B region).
for attempt in 1 2; do
    until [[ -z "$(busy localhost "$PREFILL0_GPUS")$(busy "$DECODE_SSH" "${DECODE_GPUS:-0,1,2,3,4,5,6,7}")" ]]; do sleep 60; done
    {
        for n in "${nodes[@]}"; do
            echo "===== $n $(date -u +%FT%TZ)"
            on "$n" "rocm-smi --showmeminfo vram --csv; rocm-smi --showpids; docker stats --no-stream --format '{{.Name}} {{.CPUPerc}} {{.MemUsage}}'; uptime"
        done
    } >"$log_dir/$label.env-check" 2>&1
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
(cd "$kit" && CONC="$conc" DURATION="$duration" bash engine/agentx/agentx.sh) >>"$log_dir/$label.log" 2>&1 &
ax=$!
until tr '\r' '\n' <"$log_dir/$label.log" | grep -aq 'Warmup complete'; do
    kill -0 "$ax" 2>/dev/null || break
    sleep 30
done
echo "$(date -u +%T) $label warmup complete; profiling in ${after}s"
sleep "$after"
pids=()
for k in 1 2; do
    pid="decode-$(date -u +%H%M%S)"
    curl -s -m 60 -X POST "http://$DECODE_NODE_IP:27301/start_profile" -H 'Content-Type: application/json' \
        -d "{\"output_dir\": \"/root/.cache/$pid\", \"num_steps\": $steps, \"activities\": [\"CPU\", \"GPU\"], \"profile_id\": \"$pid\"}"
    echo
    echo "$(date -u +%T) $label profile requested ($pid)"
    pids+=("$pid")
    [[ $k == 2 ]] || sleep "$gap"
done
wait "$ax"
echo "$(date -u +%T) $label agentx exited $?"
# LOG_DIR exists only on the prefill node (10-29: /mnt/m2m_nobackup/liyingli_3p1d is node-local), so copy to the
# decode node's data disk first, then pull it here.
dtmp=/mnt/m2m_nobackup/liyingli_tmp
for pid in "${pids[@]}"; do
    on "$DECODE_SSH" "mkdir -p $dtmp && docker cp ${CTR:-glm52_2p1d}-decode:/root/.cache/$pid $dtmp/$pid" 2>&1 | tail -1
    ssh -n -o BatchMode=yes "$DECODE_SSH" "tar c -C $dtmp $pid" | tar x -C "$log_dir/traces" 2>&1 | tail -1
done
ls -la "$log_dir"/traces/*/ 2>&1 | tail -12
mkdir -p "$run_dir/metrics"
curl -s -m 10 "http://$PREFILL_NODE_IP:27201/metrics" >"$run_dir/metrics/prefill0.prom"
curl -s -m 10 "http://$DECODE_NODE_IP:27301/metrics" >"$run_dir/metrics/decode.prom"
(cd "$kit" && bash engine/down.sh) >>"$log_dir/$label.log" 2>&1
echo "$(date -u +%T) $label down ($run_dir)"
