#!/usr/bin/env bash
# One-node run with several DP2 engines (configs/dp2_engine_v3.sh), one per GPU pair, for a holder item:
#   P/D: NP prefill engines on pairs 0..NP-1, ND decode engines on the next pairs (NP + ND <= 4), Rust kv-aware router
#   agg: NA agg engines on pairs 0..NA-1 (NA > 0 selects agg), Rust kv-aware router with session affinity on every pick
# then an optional correctness gate (GATE=1: block acceptance on the decode / agg engines, GSM8K + 16k probes, stop if
# they fail; then restart only those engines with golden acceptance) and one AgentX point.
# Inputs: NP, ND, NA, CONC, FAST (0), DURATION (3600), P_ENV / D_ENV / A_ENV (dp2_engine_v3.sh inputs per role, e.g.
# "KV_DTYPE=fp8_e4m3 PSI=11"), GOLDEN (3.51; empty = real acceptance), SPEC_K, UMBP_GB (agg: start a UMBP server with
# this DRAM pool), MOUNTS, MC_OVERLAY, NEED_MIN, NOTE. Never edit while an item uses it (copy to pd_run_v4.sh).
set -uo pipefail
NP=${NP:-2} ND=${ND:-0} NA=${NA:-0}
[[ "${ITEM_NAME:-}" =~ -c([0-9]+)$ ]] && CONC=${CONC:-${BASH_REMATCH[1]}}
CONC=${CONC:-1536} FAST=${FAST:-0} DURATION=${DURATION:-3600} GATE=${GATE:-1}
GOLDEN=${GOLDEN-3.51}
RUN=${ITEM_NAME:-pd-run-v3}-$(date -u +%m%dT%H%M)
source /perf_apps/yihou/dsv41_kf_20261120/pd/scripts/lib.sh
MR_MODE=${MR_MODE:-$(rdma_mr_mode)}
NEED_MIN=${NEED_MIN:-150}
SITE=/usr/local/lib/python3.12/dist-packages
HQ=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/holder
CFG=$W/configs/dp2_engine_v3.sh
CTR=${CTR_PREFIX}eng
left_min() {
    local l d=0 h=0 m=0 t
    l="$(squeue -h -j "${SLURM_JOB_ID:?}" -o %L 2>/dev/null)"
    [[ "$l" == *-* ]] && { d=${l%%-*}; l=${l#*-}; }
    IFS=: read -ra t <<<"$l"
    if ((${#t[@]} == 3)); then h=${t[0]} m=${t[1]}; elif ((${#t[@]} == 2)); then m=${t[0]}; fi
    echo $((10#$d * 1440 + 10#$h * 60 + 10#$m))
}
if ((NA > 0)); then LAYOUT="agg ${NA}xDP2"; NG=$((2 * NA)); else LAYOUT="P/D ${NP}xDP2 : ${ND}xDP2"; NG=$((2 * (NP + ND))); fi
say "start on $(hostname -s), job $SLURM_JOB_ID ($(left_min) min left), amdgpu $(cat /sys/module/amdgpu/version 2>/dev/null), MR_MODE=$MR_MODE, $LAYOUT ($NG GPUs), CONC=$CONC golden=${GOLDEN:-real} | ${NOTE:-}"
if (( $(left_min) < NEED_MIN )); then
    cp "$HQ/running/$ITEM_NAME.sh" "$HQ/queue/"
    say "deferred: needs ~$NEED_MIN min, job $SLURM_JOB_ID has $(left_min); back to the queue"
    exec sleep 100000
fi
node_state before
ensure_images > "$R/pull.log" 2>&1 || { say "image pull failed"; exit 1; }
remove_ours
r="$(bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/clear_foreign_containers.sh "$SLURM_JOB_ID" 2>&1)"
[ -z "$r" ] || say "foreign containers: $r"
WATCH=""
trap '[ -n "$WATCH" ] && kill $WATCH 2>/dev/null; cleanup' EXIT
start_guard
MC_OVERLAY=${MC_OVERLAY:-$W/vllm-overlay/mooncake_connector_v2.py}
UMBP_SOCK=/dev/shm/llying-umbp/standalone.grpc.sock
xm=""; [ -n "${UMBP_GB:-}" ] && xm="-v /dev/shm/llying-umbp:/dev/shm/llying-umbp"
export EXTRA_DOCKER_ARGS="-v $MC_OVERLAY:$SITE/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:ro $xm ${MOUNTS:-}"
[ -n "${UMBP_GB:-}" ] && mkdir -p /dev/shm/llying-umbp
start_etcd || { say "etcd failed"; exit 1; }
start_engine_ctr eng || { say "engine container / infera install failed"; exit 1; }
cp "$CFG" "$W/configs/agg_tp4.sh" "$W/configs/spec_k.sh" "$W/configs/pd_mooncake_common.sh" "$R/"
mkdir -p "$R/vllm-stackdump" && chmod 777 "$R/vllm-stackdump"
{ echo "LAYOUT=$LAYOUT CONC=$CONC GOLDEN=${GOLDEN:-real} SPEC_K=${SPEC_K:-5}"; echo "P_ENV=${P_ENV:-}"; echo "D_ENV=${D_ENV:-}"
  echo "A_ENV=${A_ENV:-} UMBP_GB=${UMBP_GB:-}"; echo "MOUNTS=${MOUNTS:-} MC_OVERLAY=$MC_OVERLAY"; echo "NOTE=${NOTE:-}"; } > "$R/variant.txt"
if [ -n "${UMBP_GB:-}" ]; then
    docker exec -d "$CTR" bash -c "P=\$(python3 -c 'import mori, os; print(os.path.dirname(mori.__file__))') && \
        UMBP_STANDALONE_ADDRESS=unix://$UMBP_SOCK UMBP_DRAM_CAPACITY=$((UMBP_GB * 1000000000)) UMBP_DRAM_USE_HUGEPAGES=0 \
        UMBP_SSD_ENABLED=0 MORI_UMBP_LOG_LEVEL=INFO LD_LIBRARY_PATH=\$P:\$LD_LIBRARY_PATH \
        exec \$P/umbp_standalone_server unix://$UMBP_SOCK > $R/umbp.log 2>&1"
    for _ in $(seq 180); do grep -q "listening" "$R/umbp.log" 2>/dev/null && break; sleep 10; done
    grep -q "listening" "$R/umbp.log" || { say "UMBP server not ready: $(tail -3 "$R/umbp.log" | tr '\n' ' ')"; exit 1; }
    say "UMBP server ready (${UMBP_GB} GB DRAM)"
fi
common=(MR_MODE="$MR_MODE" GEM_SO="" MC_TOPO=pergpu STACK_DUMP_DIR="$R/vllm-stackdump" UMBP_SOCK="$UMBP_SOCK")
spec=(); [ -n "${SPEC_K:-}" ] && spec=(SPEC_K="$SPEC_K")
# start_engine NAME ROLE ENG GOLDEN ENVWORDS
start_engine() {
    local name=$1 role=$2 eng=$3 golden=$4 envw=$5 g=()
    [ -n "$golden" ] && g=(GOLDEN_AL="$golden")
    # shellcheck disable=SC2086
    exec_bg eng "$name" env "${common[@]}" "${spec[@]}" "${g[@]}" ROLE="$role" ENG="$eng" $envw \
        EXTRA_WORKER_ENV="VLLM_SERVER_DEV_MODE=1" bash "$W/scripts/ctr_worker.sh" "$CFG"
}
# kill_engine PORT: the process tree under the launcher serving PORT.
kill_engine() {
    docker exec "$CTR" bash -c 'kt() { local c; for c in $(pgrep -P "$1"); do kt "$c"; done; kill -9 "$1" 2>/dev/null; }
        for p in $(pgrep -f -- "infera.engine.vllm.*--port '"$1"'( |$)"); do kt "$p"; done' 2>/dev/null
}
names=() gen_names=()
if ((NA > 0)); then
    for ((e = 0; e < NA; e++)); do start_engine "agg$e" agg $e "" "${A_ENV:-}"; names+=("agg$e:$((30000 + e))"); gen_names+=("agg$e:$e"); sleep 15; done
else
    for ((e = 0; e < NP; e++)); do start_engine "prefill$e" prefill $e "" "${P_ENV:-}"; names+=("prefill$e:$((30000 + e))"); sleep 15; done
    for ((j = 0; j < ND; j++)); do e=$((NP + j)); start_engine "decode$j" decode $e "" "${D_ENV:-}"; names+=("decode$j:$((30000 + e))"); gen_names+=("decode$j:$e"); sleep 15; done
fi
t0=$SECONDS; pids=()
for nm in "${names[@]}"; do wait_worker "${nm%%:*}" "${nm##*:}" 7200 & pids+=($!); done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
if ((fail)); then
    for nm in "${names[@]}"; do n=${nm%%:*}; say "$n: $(grep -aE 'Error|error:|Traceback|assert' "$R/$n.log" | tail -2 | tr '\n' ' ' | head -c 400)"; done
    say "engines failed after $((SECONDS - t0)) s"; exit 1
fi
say "engines ready in $((SECONDS - t0)) s; KV: $(for nm in "${names[@]}"; do n=${nm%%:*}; echo -n "$n $(grep -aoE 'GPU KV cache size: [0-9,]+' "$R/$n.log" | head -1 | grep -oE '[0-9,]+$') "; done)"
if ((NA > 0)); then RARGS="--session-affinity both"; else
    RARGS="--kv-prefill-overlap-weight 20 --kv-decode-overlap-weight 2 --session-affinity prefill --pd-prefill-guard-release completion --decode-input-demand on"; fi
up_router() { router_up "$1" ROUTER_BACKEND=rust ROUTER_POLICY=kv-aware ROUTER_EXTRA_ARGS="$RARGS"; }
up_router router || { say "router failed"; exit 1; }
curl -s "http://127.0.0.1:$ROUTER_PORT/v1/workers" > "$R/workers.json"
say "router workers: $(python3 -c 'import json,sys; print(sorted((w.get("disagg_mode"), w.get("url"), w.get("dp_size")) for w in json.load(open(sys.argv[1]))["workers"]))' "$R/workers.json" 2>&1 | head -c 500)"
chat "What is the capital of France? Answer with one word." 64 > "$R/smoke1.json"
say "smoke: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); c=d["choices"][0]; print(c.get("finish_reason"), repr((c["message"].get("content") or "")[:30]))' "$R/smoke1.json" 2>&1 | head -c 200)"
if ((GATE)); then
    # probe_set greps $R/decode.log for P-side-ready timeouts; point it at the first generation engine's log.
    ln -sf "$R/${gen_names[0]%%:*}.log" "$R/decode.log"
    probe_set real
    g=$(grep -aoE 'correct=[0-9.]+' "$R/real-gsm8k.log" | tail -1 | cut -d= -f2)
    nd=$(grep -aoE '^needle: n=[0-9]+ correct=[0-9]+' "$R/real-longctx16k.log" | grep -oE '[0-9]+$')
    ne=$(cat "$R/real-longctx16k.jsonl" "$R/real-gsm8k.jsonl" 2>/dev/null | grep -c '"error"')
    say "probe gate: gsm8k ${g:-?} needle ${nd:-?}/32 errors $ne"
    if ! python3 -c "import sys; sys.exit(0 if float('${g:-0}') >= 0.92 and int('${nd:-0}') >= 30 and int('$ne') == 0 else 1)"; then
        say "probe gate FAILED: no AgentX point"; node_state after; say "end"; exit 1
    fi
    if [ -n "$GOLDEN" ]; then
        t1=$SECONDS; pids=()
        for gn in "${gen_names[@]}"; do
            n=${gn%%:*} e=${gn##*:}; cp "$R/$n.log" "$R/$n-probe.log"; kill_engine $((30000 + e))
        done
        sleep 20
        for gn in "${gen_names[@]}"; do
            n=${gn%%:*} e=${gn##*:}
            if ((NA > 0)); then start_engine "$n" agg $e "$GOLDEN" "${A_ENV:-}"; else start_engine "$n" decode $e "$GOLDEN" "${D_ENV:-}"; fi
            sleep 10
        done
        for gn in "${gen_names[@]}"; do n=${gn%%:*} e=${gn##*:}; wait_worker "$n" $((30000 + e)) 3600 & pids+=($!); done
        fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
        ((fail)) && { say "golden restart failed"; exit 1; }
        say "golden engines ready in $((SECONDS - t1)) s: $(grep -aoE 'synthetic_acceptance_length[^,]*' "$R/${gen_names[0]%%:*}.log" | head -1)"
        up_router router2 || { say "router failed"; exit 1; }
        chat "What is the capital of France? Answer with one word." 64 > "$R/smoke2.json"
    fi
fi
mkdir -p "$R/pyspy"
bash "$W/scripts/hang_watch_v2.sh" "$R" 2 &
WATCH=$!
metrics="http://127.0.0.1:$ROUTER_PORT/metrics"
for nm in "${names[@]}"; do metrics+=",http://127.0.0.1:${nm##*:}/metrics"; done
say "AgentX $( ((FAST)) && echo fast || echo formal) c$CONC starting ($(left_min) min left)"
if ((NA > 0)); then
    CONC=$CONC FAST=$FAST DURATION=$DURATION MODE=agg AGG_TP=$NG SIM_ACCEPT_LEN="$GOLDEN" METRICS_URLS="$metrics" \
        timeout $(( ($(left_min) - 5) * 60 )) bash "$W/scripts/agentx_formal.sh" "$R/agentx-c$CONC"
else
    CONC=$CONC FAST=$FAST DURATION=$DURATION MODE=pd PREFILL_WORKERS=$NP PREFILL_GPUS=2 DECODE_GPUS=$((2 * ND)) PREFILL_DP_ATTN=true \
        DECODE_DP_ATTN=true SIM_ACCEPT_LEN="$GOLDEN" METRICS_URLS="$metrics" \
        timeout $(( ($(left_min) - 5) * 60 )) bash "$W/scripts/agentx_formal.sh" "$R/agentx-c$CONC"
fi
rc=$?
say "AgentX c$CONC (FAST=$FAST) rc=$rc: $(grep -aE 'Saved aggregated|failed|ERROR' "$R/agentx-c$CONC/runner.log" | tail -3 | tr -s ' ' | tr '\n' ' ' | head -c 500)"
j=$(ls "$R"/agentx-c$CONC/agentx_conc*.json 2>/dev/null | head -1)
[ -n "$j" ] && say "result: $(python3 "$W/scripts/ax_summary.py" "$j" 2>&1 | tail -1 | head -c 500)"
say "hang watch: $(grep -c 'stall seen' "$R/pyspy/watch.log" 2>/dev/null) stalls"
for nm in "${names[@]}"; do n=${nm%%:*}; python3 "$W/scripts/logstats.py" "$R/$n.log" --label "$n" > "$R/logstats-$n.txt" 2>&1; done
node_state after
say "end"
