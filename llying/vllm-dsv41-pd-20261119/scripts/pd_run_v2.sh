#!/usr/bin/env bash
# One-node P/D run for a holder item: deploy NP prefill workers (DP2 each, GPUs 0-1 / 2-3) and one decode engine on
# GPUs 4-7, the Rust kv-aware router, then either correctness probes (MODE=probe, real acceptance) or one AgentX
# point (MODE=agentx). Items set the inputs below and `exec bash` this file; never edit it while an item uses it
# (copy to pd_run_v3.sh instead). v2 = v1 + PROBE_FIRST.
# Inputs (env):
#   ITEM_NAME (from the holder) | NP (2) | CONC (from a -c<N> suffix, else 768) | FAST (0) | DURATION (3600)
#   PCFG / DCFG                configs (default: pd1n_dpa_prefill_dp2.sh / pd1n_dpa_decode_dp4.sh)
#   P_ENV / D_ENV              extra K=V words for ctr_worker.sh on the prefill / decode (config inputs, e.g. SPEC_GOLDEN=1)
#   P_WENV / D_WENV            extra K=V for the vLLM process env (EXTRA_WORKER_ENV)
#   MOUNTS                     extra docker -v args for the engine container (overlays)
#   MC_OVERLAY                 the MooncakeConnector file mounted over the image's (default vllm-overlay/mooncake_connector_v2.py:
#                              STOPPED-prefill fix + ZMQ MAX_SOCKETS 65536)
#   DECODE_DP_ATTN (true)      AgentX metadata; SIM_ACCEPT_LEN (3.51; empty = real acceptance)
#   MODE (agentx | probe)      probe: GSM8K + 16k long-context probes, no AgentX
#   HANG_WATCH (1)             py-spy watchdog on stalled steps; vLLM stack dumps go to $R/vllm-stackdump
#   NOTE                       one line describing the variant, written to the summary
#   PROBE_FIRST (0)            1: start the decode with D_PROBE_ENV (real acceptance), run the correctness probes, stop if
#                              they fail (GSM8K < PROBE_MIN_GSM8K, needle < 30/32 or any error), else restart only the
#                              decode with D_ENV (golden) for the AgentX point
set -uo pipefail
NP=${NP:-2}
[[ "${ITEM_NAME:-}" =~ -c([0-9]+)$ ]] && CONC=${CONC:-${BASH_REMATCH[1]}}
CONC=${CONC:-768} FAST=${FAST:-0} DURATION=${DURATION:-3600} MODE=${MODE:-agentx} HANG_WATCH=${HANG_WATCH:-1}
RUN=${ITEM_NAME:-pd-run}-$(date -u +%m%dT%H%M)
source /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/lib.sh
MR_MODE=${MR_MODE:-$(rdma_mr_mode)}
PCFG=${PCFG:-$W/configs/pd1n_dpa_prefill_dp2.sh} DCFG=${DCFG:-$W/configs/pd1n_dpa_decode_dp4.sh}
SIM_ACCEPT_LEN=${SIM_ACCEPT_LEN-3.51}
if [ "$MODE" = probe ]; then NEED_MIN=${NEED_MIN:-40}; elif ((FAST)); then NEED_MIN=${NEED_MIN:-60}; elif [ "${PROBE_FIRST:-0}" = 1 ]; then NEED_MIN=${NEED_MIN:-180}; else NEED_MIN=${NEED_MIN:-150}; fi
SITE=/usr/local/lib/python3.12/dist-packages
HQ=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/holder
left_min() {  # minutes left in this job (squeue %L: [D-]HH:MM:SS or MM:SS)
    local l d=0 h=0 m=0 t
    l="$(squeue -h -j "${SLURM_JOB_ID:?}" -o %L 2>/dev/null)"
    [[ "$l" == *-* ]] && { d=${l%%-*}; l=${l#*-}; }
    IFS=: read -ra t <<<"$l"
    if ((${#t[@]} == 3)); then h=${t[0]} m=${t[1]}; elif ((${#t[@]} == 2)); then m=${t[0]}; fi
    echo $((10#$d * 1440 + 10#$h * 60 + 10#$m))
}
say "start on $(hostname -s), job $SLURM_JOB_ID ($(left_min) min left), amdgpu $(cat /sys/module/amdgpu/version 2>/dev/null), MR_MODE=$MR_MODE, ${NP}P1D, MODE=$MODE CONC=$CONC FAST=$FAST | ${NOTE:-}"
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
export EXTRA_DOCKER_ARGS="-v $MC_OVERLAY:$SITE/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:ro ${MOUNTS:-}"
start_etcd || { say "etcd failed"; exit 1; }
start_engine_ctr eng || { say "engine container / infera install failed"; exit 1; }
GEM_SO=""
if [ "$MR_MODE" = gem ]; then
    build_gem eng > "$R/gem-build.log" 2>&1 || { say "GEM allocator build failed: $(tail -3 "$R/gem-build.log" | tr '\n' ' ')"; exit 1; }
fi
cp "$PCFG" "$R/prefill-config.sh"; cp "$DCFG" "$R/decode-config.sh"; cp "$W/configs/pd_mooncake_common.sh" "$R/"
cp "$W/configs/pd1n_dpa_decode_dp4.sh" "$W/configs/agg_tp4.sh" "$R/" 2>/dev/null
mkdir -p "$R/vllm-stackdump" && chmod 777 "$R/vllm-stackdump"
{ echo "NP=$NP PCFG=$PCFG DCFG=$DCFG"; echo "P_ENV=${P_ENV:-} P_WENV=${P_WENV:-}"; echo "D_ENV=${D_ENV:-} D_WENV=${D_WENV:-}"
  echo "MOUNTS=${MOUNTS:-} MC_OVERLAY=$MC_OVERLAY"; echo "SIM_ACCEPT_LEN=$SIM_ACCEPT_LEN DECODE_DP_ATTN=${DECODE_DP_ATTN:-true} NOTE=${NOTE:-}"; } > "$R/variant.txt"
common=(MR_MODE="$MR_MODE" GEM_SO="$GEM_SO" MC_TOPO=pergpu)
dstack=(); ((HANG_WATCH)) && dstack=(STACK_DUMP_DIR="$R/vllm-stackdump")
t0=$SECONDS
# shellcheck disable=SC2086
exec_bg eng prefill0 env "${common[@]}" ${P_ENV:-} EXTRA_WORKER_ENV="VLLM_SERVER_DEV_MODE=1 ${P_WENV:-}" GPUS=0,1 PORT=30000 BOOT_PORT=28398 \
    DP_RPC_PORT=29610 DP_MASTER_PORT=29511 SCAN_PORT=40000 bash "$W/scripts/ctr_worker.sh" "$PCFG"
names=(prefill0:30000 decode:30001)
if ((NP == 2)); then
    sleep 20
    # shellcheck disable=SC2086
    exec_bg eng prefill1 env "${common[@]}" ${P_ENV:-} EXTRA_WORKER_ENV="VLLM_SERVER_DEV_MODE=1 ${P_WENV:-}" GPUS=2,3 PORT=30002 BOOT_PORT=28397 \
        DP_RPC_PORT=29612 DP_MASTER_PORT=29513 SCAN_PORT=42000 bash "$W/scripts/ctr_worker.sh" "$PCFG"
    names+=(prefill1:30002)
fi
sleep 20
decode_kill() {
    docker exec "${CTR_PREFIX}eng" bash -c 'kt() { local c; for c in $(pgrep -P "$1"); do kt "$c"; done; kill -9 "$1" 2>/dev/null; }
        for p in $(pgrep -f -- "infera.engine.vllm.*--port 30001"); do kt "$p"; done' 2>/dev/null
    local i used
    for i in $(seq 60); do
        used=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | awk -F, '$1 ~ /card[4-7]/ {s += $3} END {print int(s / 2^30)}')
        ((${used:-999} < 20)) && return 0
        sleep 5
    done
    say "decode GPUs still hold ${used} GiB after kill"
}
PROBE_FIRST=${PROBE_FIRST:-0}
d_env_now=${D_ENV:-}
((PROBE_FIRST)) && d_env_now=${D_PROBE_ENV:-PROBE_PHASE=1}
# shellcheck disable=SC2086
exec_bg eng decode env "${common[@]}" "${dstack[@]}" $d_env_now EXTRA_WORKER_ENV="${D_WENV:-}" bash "$W/scripts/ctr_worker.sh" "$DCFG"
pids=()
for nm in "${names[@]}"; do wait_worker "${nm%%:*}" "${nm##*:}" 7200 & pids+=($!); done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
if ((fail)); then
    for nm in "${names[@]}"; do n=${nm%%:*}; say "$n: $(grep -aE 'Error|error:|Traceback|assert' "$R/$n.log" | tail -3 | tr '\n' ' ' | head -c 600)"; done
    say "workers failed after $((SECONDS - t0)) s"; exit 1
fi
say "workers ready in $((SECONDS - t0)) s; decode spec: $(grep -aoE "'rejection_sample_method': '[a-z]+'|synthetic_acceptance_length[^,]*|'num_speculative_tokens': [0-9]+" "$R/decode.log" | sort -u | head -3 | tr '\n' ' '); KV: $(grep -aoE 'GPU KV cache size: [0-9,]+ tokens' "$R/prefill0.log" "$R/decode.log" | sort -u | sed "s#$R/##" | tr '\n' ' ')"
router_up router ROUTER_BACKEND=rust ROUTER_POLICY=kv-aware \
    ROUTER_EXTRA_ARGS="--kv-prefill-overlap-weight 20 --kv-decode-overlap-weight 2 --session-affinity prefill --pd-prefill-guard-release completion --decode-input-demand on" \
    || { say "router failed"; exit 1; }
curl -s "http://127.0.0.1:$ROUTER_PORT/v1/workers" > "$R/workers.json"
say "router workers: $(python3 -c 'import json,sys; print(sorted((w.get("disagg_mode"), w.get("url"), w.get("dp_size")) for w in json.load(open(sys.argv[1]))["workers"]))' "$R/workers.json" 2>&1 | head -c 400)"
chat "What is the capital of France? Answer with one word." 64 > "$R/smoke1.json"
say "smoke: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); c=d["choices"][0]; print(c.get("finish_reason"), repr((c["message"].get("content") or "")[:40]), d.get("usage"))' "$R/smoke1.json" 2>&1 | head -c 300)"
if ((HANG_WATCH)); then
    mkdir -p "$R/pyspy"
    wp=$(docker exec "${CTR_PREFIX}eng" pgrep -f "Worker_DP0|Worker_TP0" | tail -1)
    docker exec "${CTR_PREFIX}eng" timeout 30 "$W/wheels/extra/py-spy" dump --pid "${wp:-1}" > "$R/pyspy/selftest.txt" 2>&1
    say "py-spy self-test on pid ${wp:-none}: rc=$? $(grep -c 'Thread ' "$R/pyspy/selftest.txt") threads"
    bash "$W/scripts/hang_watch.sh" "$R" 2 &
    WATCH=$!
fi
if ((PROBE_FIRST)) && [ "$MODE" != probe ]; then
    probe_set real
    g=$(grep -aoE 'correct=[0-9.]+' "$R/real-gsm8k.log" | tail -1 | cut -d= -f2)
    nd=$(grep -aoE '^needle: n=[0-9]+ correct=[0-9]+' "$R/real-longctx16k.log" | grep -oE '[0-9]+$')
    ne=$(cat "$R/real-longctx16k.jsonl" "$R/real-gsm8k.jsonl" 2>/dev/null | grep -c '"error"')
    say "probe gate: gsm8k ${g:-?} needle ${nd:-?}/32 errors $ne; spec: $(grep -aoE 'Mean acceptance length: [0-9.]+' "$R/decode.log" | tail -1)"
    if ! python3 -c "import sys; sys.exit(0 if float('${g:-0}') >= float('${PROBE_MIN_GSM8K:-0.92}') and int('${nd:-0}') >= 30 and int('$ne') == 0 else 1)"; then
        say "probe gate FAILED: no AgentX point"; node_state after; say "end"; exit 1
    fi
    cp "$R/decode.log" "$R/decode-probe.log"
    [ -n "$WATCH" ] && { kill $WATCH 2>/dev/null; WATCH=""; }
    decode_kill
    t1=$SECONDS
    # shellcheck disable=SC2086
    exec_bg eng decode env "${common[@]}" "${dstack[@]}" ${D_ENV:-} EXTRA_WORKER_ENV="${D_WENV:-}" bash "$W/scripts/ctr_worker.sh" "$DCFG"
    wait_worker decode 30001 3600 || { say "decode (scoring) failed: $(grep -aE 'Error|error:|Traceback' "$R/decode.log" | tail -3 | tr '\n' ' ' | head -c 500)"; exit 1; }
    say "decode restarted for scoring in $((SECONDS - t1)) s; spec: $(grep -aoE "'rejection_sample_method': '[a-z]+'|synthetic_acceptance_length[^,]*|'num_speculative_tokens': [0-9]+" "$R/decode.log" | sort -u | head -3 | tr '\n' ' ')"
    router_up router2 ROUTER_BACKEND=rust ROUTER_POLICY=kv-aware \
        ROUTER_EXTRA_ARGS="--kv-prefill-overlap-weight 20 --kv-decode-overlap-weight 2 --session-affinity prefill --pd-prefill-guard-release completion --decode-input-demand on" \
        || { say "router failed"; exit 1; }
    chat "What is the capital of France? Answer with one word." 64 > "$R/smoke2.json"
    say "smoke (scoring decode): $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["choices"][0].get("finish_reason"), d.get("usage",{}).get("completion_tokens"))' "$R/smoke2.json" 2>&1 | head -c 200)"
    if ((HANG_WATCH)); then mkdir -p "$R/pyspy"; bash "$W/scripts/hang_watch.sh" "$R" 2 & WATCH=$!; fi
fi
if [ "$MODE" = probe ]; then
    probe_set real
    say "spec decode (decode leg): $(grep -aoE 'Mean acceptance length: [0-9.]+' "$R/decode.log" | tail -3 | tr '\n' ' ')"
    node_state after
    say "end"
    exit 0
fi
metrics="http://127.0.0.1:$ROUTER_PORT/metrics,http://127.0.0.1:30000/metrics,http://127.0.0.1:30001/metrics"
((NP == 2)) && metrics+=",http://127.0.0.1:30002/metrics"
say "AgentX $( ((FAST)) && echo fast || echo formal) c$CONC starting ($(left_min) min left)"
CONC=$CONC FAST=$FAST DURATION=$DURATION MODE=pd PREFILL_WORKERS=$NP PREFILL_GPUS=2 DECODE_GPUS=4 PREFILL_DP_ATTN=true \
    DECODE_DP_ATTN=${DECODE_DP_ATTN:-true} SIM_ACCEPT_LEN=$SIM_ACCEPT_LEN METRICS_URLS="$metrics" \
    timeout $(( ($(left_min) - 5) * 60 )) bash "$W/scripts/agentx_formal.sh" "$R/agentx-c$CONC"
rc=$?
say "AgentX c$CONC (FAST=$FAST) rc=$rc: $(grep -aE 'Saved aggregated|failed|ERROR' "$R/agentx-c$CONC/runner.log" | tail -4 | tr -s ' ' | tr '\n' ' ' | head -c 700)"
j=$(ls "$R"/agentx-c$CONC/agentx_conc*.json 2>/dev/null | head -1)
[ -n "$j" ] && say "result: $(python3 "$W/scripts/ax_summary.py" "$j" 2>&1 | tail -1 | head -c 600)"
((HANG_WATCH)) && say "hang watch: $(tr '\n' ' ' < "$R/pyspy/watch.log" | head -c 400); vLLM stack dumps: $(ls "$R/vllm-stackdump" | wc -l) files"
for e in prefill0 prefill1 decode; do
    [ -f "$R/$e.log" ] && python3 "$W/scripts/logstats.py" "$R/$e.log" --label "$e" > "$R/logstats-$e.txt" 2>&1
done
node_state after
say "end"
