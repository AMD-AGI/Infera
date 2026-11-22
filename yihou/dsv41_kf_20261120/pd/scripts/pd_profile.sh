#!/usr/bin/env bash
# kf: one-node Infera + vLLM P/D deployment of DSv4.1-Flash under AgentX load, with calibrated-profiler windows on
# every engine (UNSCORED). Derived from llying's pd_run_v3.sh (2026-11-20 02:53Z): NP prefill DP2 engines on GPU pairs
# 0..NP-1, ND decode DP2 engines on the next pairs, Rust kv-aware router, Mooncake (per-GPU NIC), golden acceptance.
# Run ON the node (ssh, inside our pool job), as yihou:
#   TAG=<fresh> NP=2 ND=2 CONC=768 GOLDEN=3.51 D_MODE=graph CALIB=1 STOP_BY=<ISO UTC> bash pd_profile.sh
# Inputs:
#   NP, ND              prefill / decode DP2 engines (NP + ND <= 4)
#   CONC                AgentX concurrency (fast mode = 1 warmup request per lane; unscored)
#   GOLDEN              synthetic acceptance length for decode (3.51); empty = block (real) acceptance
#   D_MODE              graph (vLLM default CUDA graphs) | eager (--enforce-eager on decode)
#   P_ENV / D_ENV       extra dp2_engine_v3.sh inputs per role (default: llying's B300 prefill settings on P)
#   CALIB               1 = profiler hook + fork SDK in every engine; 0 = plain deployment (smoke / A-B reference)
#   WINDOW_AT_S         seconds after AgentX start to open the windows (default 1500 = after warmup ramp)
#   P_STEPS / D_STEPS   execute_model calls per window (20 / 60); CALIB_MAX_WINDOW_S 30
#   STOP_BY             lease end, ISO-8601 UTC; the script refuses to start with < NEED_MIN (default 90) minutes left
set -uo pipefail
TAG=${TAG:?TAG}
[[ "$TAG" =~ ^[a-z0-9][a-z0-9._-]{0,50}$ ]] || { echo "unsafe TAG"; exit 2; }
NA=${NA:-0} NP=${NP:-2} ND=${ND:-2} CONC=${CONC:-768} GOLDEN=${GOLDEN-3.51} D_MODE=${D_MODE:-graph} CALIB=${CALIB:-1}
WINDOW_AT_S=${WINDOW_AT_S:-1500} P_STEPS=${P_STEPS:-20} D_STEPS=${D_STEPS:-60} NEED_MIN=${NEED_MIN:-90}
DURATION=${DURATION:-2400}
P_ENV=${P_ENV-UTIL=0.93 LONG_PREFILL=8192 MAXSEQS=64}
D_ENV=${D_ENV-}
A_ENV=${A_ENV-PSI=11 KV_DTYPE=fp8_ds_mla}  # kf: colocated (agg) engines = InferenceX DP2 shape (decode-proxy profiles, user 11-22)
# The agg proxy runs the PD decode's attention path: a Mooncake connector forces fp8_ds_mla KV + no AITER sparse-MLA fast path
# (leader E4b, rocm.py:80-109/167-172), so the proxy sets both (A_FASTPATH=1 keeps the colocated fast path instead).
A_FASTPATH=${A_FASTPATH:-0}
A_STEPS=${A_STEPS:-80}
RUN=$TAG
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
MR_MODE=${MR_MODE:-$(rdma_mr_mode)}
SITE=/usr/local/lib/python3.12/dist-packages
CFG=$W/configs/dp2_engine_v3.sh
CTR=${CTR_PREFIX}eng
left_min() { echo $(( ($(date -d "${STOP_BY:?STOP_BY}" +%s) - $(date +%s)) / 60 )); }
((NP + ND <= 4 && NA <= 4)) || { echo "NP + ND > 4 or NA > 4"; exit 2; }
((NA > 0)) && { NP=0; ND=0; D_MODE=${D_MODE}; }
[ "$D_MODE" = eager ] && D_ENV="$D_ENV EXTRA_ARGS=--enforce-eager" && A_ENV="$A_ENV EXTRA_ARGS=--enforce-eager"
say "start on $(hostname -s) ($(left_min) min left), amdgpu $(cat /sys/module/amdgpu/version 2>/dev/null), MR_MODE=$MR_MODE, P/D ${NP}xDP2:${ND}xDP2, CONC=$CONC golden=${GOLDEN:-block} decode=$D_MODE calib=$CALIB"
(( $(left_min) >= NEED_MIN )) || { say "refused: $(left_min) min left < $NEED_MIN"; exit 1; }
node_state before
ensure_images > "$R/pull.log" 2>&1 || { say "image pull failed"; exit 1; }
if ((CALIB)) && [ ! -f "$CALIB_LOCAL/build/BUILD_OK" ]; then
    # fork SDK built in this image on a16-19 (calib-build-a1619), staged next to W; copied to node-local disk
    mkdir -p "$CALIB_LOCAL/build" && cp -r "$W/../calib-build/lib" "$W/../calib-build/BUILD_OK" "$CALIB_LOCAL/build/" \
        || { say "calib SDK copy failed"; exit 1; }
fi
remove_ours
WATCH="" AXPID=""
trap '[ -n "$WATCH" ] && kill $WATCH 2>/dev/null; [ -n "$AXPID" ] && kill $AXPID 2>/dev/null; docker stop -t 30 ${CTR_PREFIX}agentx >/dev/null 2>&1; cleanup' EXIT
MC_OVERLAY=${MC_OVERLAY:-$W/vllm-overlay/mooncake_connector_v2.py}
export EXTRA_DOCKER_ARGS="-v $MC_OVERLAY:$SITE/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:ro -v $CALIB_LOCAL:$CALIB_LOCAL:ro ${MOUNTS:-}"
start_etcd || { say "etcd failed"; exit 1; }
start_engine_ctr eng || { say "engine container / infera install failed: $(tail -3 "$R/install-eng.log" | tr '\n' ' ')"; exit 1; }
GEM_SO=""
if [ "$MR_MODE" = gem ]; then
    build_gem eng > "$R/gem-build.log" 2>&1 || { say "GEM allocator build failed"; exit 1; }
fi
cp "$CFG" "$W/configs/agg_tp4.sh" "$W/configs/spec_k.sh" "$W/configs/pd_mooncake_common.sh" "$R/"
mkdir -p "$R/vllm-stackdump" && chmod 1777 "$R/vllm-stackdump"
{ echo "TAG=$TAG NP=$NP ND=$ND CONC=$CONC GOLDEN=${GOLDEN:-block} D_MODE=$D_MODE CALIB=$CALIB"; echo "P_ENV=$P_ENV"; echo "D_ENV=$D_ENV"
  echo "WINDOW_AT_S=$WINDOW_AT_S P_STEPS=$P_STEPS D_STEPS=$D_STEPS IMAGE=$IMAGE MC_OVERLAY=$MC_OVERLAY"; } > "$R/variant.txt"
common=(MR_MODE="$MR_MODE" GEM_SO="$GEM_SO" MC_TOPO=pergpu STACK_DUMP_DIR="$R/vllm-stackdump")
calib_env() {  # calib_env NAME STEPS -> words for EXTRA_WORKER_ENV
    ((CALIB)) || return 0
    local d="$R/calib/$1"; mkdir -p "$d"; chmod 1777 "$d"
    echo "CALIB_ENABLE=1 CALIB_SDK_BUILD=$CALIB_LOCAL/build CALIB_TRACE_DIR=$d CALIB_STEPS=$2 CALIB_MAX_WINDOW_S=30" \
         "ROCPROFILER_REGISTER_LIBRARY=$CALIB_LOCAL/build/lib/librocprofiler-sdk.so" \
         "ROCP_TOOL_LIBRARIES=$CALIB_LOCAL/build/lib/rocprofiler-sdk/librocprofiler-sdk-json-tool.so" \
         "ROCPROFILER_TOOL_CONTEXTS=HIP_API_BUFFERED,KERNEL_DISPATCH_BUFFERED,CODE_OBJECT ROCPROFILER_QUEUE_INTERPOSITION=0" \
         "ROCPROFILER_EXPERIMENTAL_GRAPH_COPYBUFFER=1 ROCPROFILER_TOOL_DISABLE_PERFETTO=1"
}
# start_engine NAME ROLE ENG GOLDEN ENVWORDS STEPS
start_engine() {
    local name=$1 role=$2 eng=$3 golden=$4 envw=$5 steps=$6 g=()
    [ -n "$golden" ] && g=(GOLDEN_AL="$golden")
    # shellcheck disable=SC2086
    exec_bg eng "$name" env "${common[@]}" "${g[@]}" ROLE="$role" ENG="$eng" $envw \
        EXTRA_WORKER_ENV="VLLM_SERVER_DEV_MODE=1 $([ "$role" = agg ] && [ "$A_FASTPATH" = 0 ] && echo VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=False) $(calib_env "$name" "$steps")" bash "$W/scripts/ctr_worker.sh" "$CFG"
}
names=() gen=()
for ((e = 0; e < NA; e++)); do start_engine "agg$e" agg $e "$GOLDEN" "$A_ENV" "$A_STEPS"; names+=("agg$e:$((30000 + e))"); gen+=("agg$e"); sleep 15; done
for ((e = 0; e < NP; e++)); do start_engine "prefill$e" prefill $e "" "$P_ENV" "$P_STEPS"; names+=("prefill$e:$((30000 + e))"); sleep 15; done
for ((j = 0; j < ND; j++)); do e=$((NP + j)); start_engine "decode$j" decode $e "$GOLDEN" "$D_ENV" "$D_STEPS"; names+=("decode$j:$((30000 + e))"); gen+=("decode$j"); sleep 15; done
t0=$SECONDS; pids=()
for nm in "${names[@]}"; do wait_worker "${nm%%:*}" "${nm##*:}" 7200 & pids+=($!); done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
if ((fail)); then
    for nm in "${names[@]}"; do n=${nm%%:*}; say "$n: $(grep -aE 'Error|error:|Traceback|assert' "$R/$n.log" | tail -2 | tr '\n' ' ' | head -c 400)"; done
    say "engines failed after $((SECONDS - t0)) s"; exit 1
fi
say "engines ready in $((SECONDS - t0)) s; KV: $(for nm in "${names[@]}"; do n=${nm%%:*}; echo -n "$n $(grep -aoE 'GPU KV cache size: [0-9,]+' "$R/$n.log" | head -1 | grep -oE '[0-9,]+$') "; done)"
say "gen spec: $(grep -aoE "'rejection_sample_method': '[a-z]+'|synthetic_acceptance_length[^,}]*" "$R/${gen[0]}.log" | head -2 | tr '\n' ' ')"
if ((NA > 0)); then RARGS="--session-affinity both"; else
RARGS="--kv-prefill-overlap-weight 20 --kv-decode-overlap-weight 2 --session-affinity prefill --pd-prefill-guard-release completion --decode-input-demand on"; fi
router_up router ROUTER_BACKEND=rust ROUTER_POLICY=kv-aware ROUTER_EXTRA_ARGS="$RARGS" || { say "router failed"; exit 1; }
curl -s "http://127.0.0.1:$ROUTER_PORT/v1/workers" > "$R/workers.json"
chat "What is the capital of France? Answer with one word." 64 > "$R/smoke1.json"
say "smoke: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); c=d["choices"][0]; print(c.get("finish_reason"), d.get("usage"))' "$R/smoke1.json" 2>&1 | head -c 300)"
mkdir -p "$R/pyspy"
bash "$W/scripts/hang_watch_v2.sh" "$R" 2 &
WATCH=$!
metrics="http://127.0.0.1:$ROUTER_PORT/metrics"
for nm in "${names[@]}"; do metrics+=",http://127.0.0.1:${nm##*:}/metrics"; done
say "AgentX fast c$CONC starting (unscored, $(left_min) min left)"
if ((NA > 0)); then
  CONC=$CONC FAST=1 DURATION=$DURATION MODE=agg AGG_TP=$((2 * NA)) SIM_ACCEPT_LEN="$GOLDEN" METRICS_URLS="$metrics" \
    timeout $(( ($(left_min) - 5) * 60 )) bash "$W/scripts/agentx_formal.sh" "$R/agentx-c$CONC" &
else
  CONC=$CONC FAST=1 DURATION=$DURATION MODE=pd PREFILL_WORKERS=$NP PREFILL_GPUS=2 DECODE_GPUS=$((2 * ND)) PREFILL_DP_ATTN=true \
    DECODE_DP_ATTN=true SIM_ACCEPT_LEN="$GOLDEN" METRICS_URLS="$metrics" \
    timeout $(( ($(left_min) - 5) * 60 )) bash "$W/scripts/agentx_formal.sh" "$R/agentx-c$CONC" &
fi
AXPID=$!
# engine-side load samples every 60 s (running / waiting / KV usage per engine) for the window choice and the report
( while kill -0 $AXPID 2>/dev/null; do
    for nm in "${names[@]}"; do
      echo "$(ts) ${nm%%:*} $(curl -s -m 5 "http://127.0.0.1:${nm##*:}/metrics" | grep -E '^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc)[{ ]' | sed -E 's/^vllm:([a-z_]+)\{[^}]*\}/\1/' | tr '\n' ' ')"
    done >> "$R/load.log"; sleep 60; done ) &
if ((CALIB)); then
    ta=$SECONDS
    while kill -0 $AXPID 2>/dev/null && ((SECONDS - ta < WINDOW_AT_S)); do sleep 20; done
    kill -0 $AXPID 2>/dev/null || { say "AgentX ended before the window"; exit 1; }
    say "opening windows at AgentX +$((SECONDS - ta)) s; load: $(tail -n $(( ${#names[@]} )) "$R/load.log" | tr '\n' ' ' | head -c 600)"
    for nm in "${names[@]}"; do touch "$R/calib/${nm%%:*}/start_window"; done
    # every DP rank writes steps-<pid>.json when its window closes (2 per engine)
    for _ in $(seq 60); do
        n=$(ls "$R"/calib/*/steps-*.json 2>/dev/null | wc -l); ((n >= 2 * ${#names[@]})) && break; sleep 5
    done
    say "windows closed: $(ls "$R"/calib/*/steps-*.json 2>/dev/null | wc -l) step files"
    for nm in "${names[@]}"; do touch "$R/calib/${nm%%:*}/finalize"; done
    for _ in $(seq 120); do
        n=$(ls "$R"/calib/*/finalized-* 2>/dev/null | wc -l); ((n >= 2 * ${#names[@]})) && break; sleep 10
    done
    say "finalized: $(ls "$R"/calib/*/finalized-* 2>/dev/null | wc -l) processes; traces $(du -sh "$R/calib" | cut -f1)"
    say "load after window: $(tail -n $(( ${#names[@]} )) "$R/load.log" | tr '\n' ' ' | head -c 600)"
    [ "${KEEP_LOAD:-0}" = 1 ] || { docker stop -t 30 ${CTR_PREFIX}agentx >/dev/null 2>&1; say "AgentX stopped after the windows (unscored)"; }
fi
wait $AXPID; rc=$?
say "AgentX rc=$rc: $(grep -aE 'Saved aggregated|failed|ERROR' "$R/agentx-c$CONC/runner.log" 2>/dev/null | tail -2 | tr -s ' ' | tr '\n' ' ' | head -c 300)"
say "hang watch: $(grep -c 'stall seen' "$R/pyspy/watch.log" 2>/dev/null) stalls"
for nm in "${names[@]}"; do n=${nm%%:*}; python3 "$W/scripts/logstats.py" "$R/$n.log" --label "$n" > "$R/logstats-$n.txt" 2>&1; done
node_state after
say "end"
