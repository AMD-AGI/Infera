#!/usr/bin/env bash
# Bisect the decode DP4 stall of the AgentX warmup (README 4.4) with a synthetic load, one deployment: prefill DP2 x 2
# (GPUs 0-3) + decode DP4 (GPUs 4-7), then for each decode variant restart only the decode engine and the router, and
# replay the warmup's shape (long random prompts, max_tokens=1, high concurrency, so the prefill queues and the decode
# holds hundreds of KV fetches) for up to ROUNDS rounds. A round stalls when vLLM logs a stalled step on the decode;
# its stacks are dumped with py-spy and the variant is marked STALL. V0 (the unchanged decode) must stall first, or the
# synthetic load does not reproduce the problem and the item stops.
# Inputs: VARIANTS (space-separated names of the case below), ROUNDS (2), LOAD_IN (24000), LOAD_N (2500), LOAD_C (640).
set -uo pipefail
RUN=${ITEM_NAME:-hangbisect}-$(date -u +%m%dT%H%M)
source /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/lib.sh
MR_MODE=${MR_MODE:-$(rdma_mr_mode)}
VARIANTS=${VARIANTS:-V0 V1nobreak V2block V3eager}
ROUNDS=${ROUNDS:-2} LOAD_IN=${LOAD_IN:-24000} LOAD_N=${LOAD_N:-2500} LOAD_C=${LOAD_C:-640}
SITE=/usr/local/lib/python3.12/dist-packages
PCFG=$W/configs/pd1n_dpa_prefill_dp2.sh DCFG=$W/configs/pd1n_dpa_decode_dp4_v2.sh
CTR=${CTR_PREFIX}eng
say "start on $(hostname -s), job $SLURM_JOB_ID, variants: $VARIANTS, rounds $ROUNDS, load in=$LOAD_IN n=$LOAD_N c=$LOAD_C"
node_state before
ensure_images > "$R/pull.log" 2>&1 || { say "image pull failed"; exit 1; }
remove_ours
r="$(bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/clear_foreign_containers.sh "$SLURM_JOB_ID" 2>&1)"
[ -z "$r" ] || say "foreign containers: $r"
trap cleanup EXIT
start_guard
export EXTRA_DOCKER_ARGS="-v $W/vllm-overlay/mooncake_connector.py:$SITE/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:ro"
start_etcd || { say "etcd failed"; exit 1; }
start_engine_ctr eng || { say "engine container / infera install failed"; exit 1; }
cp "$PCFG" "$R/prefill-config.sh"; cp "$DCFG" "$R/decode-config.sh"; cp "$W/configs/pd_mooncake_common.sh" "$W/configs/spec_k.sh" "$R/"
common=(MR_MODE="$MR_MODE" GEM_SO="" MC_TOPO=pergpu)
exec_bg eng prefill0 env "${common[@]}" EXTRA_WORKER_ENV="VLLM_SERVER_DEV_MODE=1" GPUS=0,1 PORT=30000 BOOT_PORT=28398 \
    DP_RPC_PORT=29610 DP_MASTER_PORT=29511 SCAN_PORT=40000 bash "$W/scripts/ctr_worker.sh" "$PCFG"
sleep 20
exec_bg eng prefill1 env "${common[@]}" EXTRA_WORKER_ENV="VLLM_SERVER_DEV_MODE=1" GPUS=2,3 PORT=30002 BOOT_PORT=28397 \
    DP_RPC_PORT=29612 DP_MASTER_PORT=29513 SCAN_PORT=42000 bash "$W/scripts/ctr_worker.sh" "$PCFG"

# Kill the decode engine only (the process tree under the launcher that serves port 30001).
decode_kill() {
    docker exec "$CTR" bash -c 'kt() { local c; for c in $(pgrep -P "$1"); do kt "$c"; done; kill -9 "$1" 2>/dev/null; }
        for p in $(pgrep -f -- "infera.engine.vllm.*--port 30001"); do kt "$p"; done' 2>/dev/null
    local i used
    for i in $(seq 60); do
        used=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | awk -F, '$1 ~ /card[4-7]/ {s += $3} END {print int(s / 2^30)}')
        ((${used:-999} < 20)) && return 0
        sleep 5
    done
    say "decode GPUs still hold ${used} GiB after kill"
}
# decode_start NAME ENV...: start the decode with config inputs ENV, wait until it serves; logs to decode-NAME.log.
decode_start() {
    local name=$1; shift
    exec_bg eng "decode-$name" env "${common[@]}" STACK_DUMP_DIR="$R/vllm-stackdump-$name" "$@" bash "$W/scripts/ctr_worker.sh" "$DCFG"
    mkdir -p "$R/vllm-stackdump-$name" && chmod 777 "$R/vllm-stackdump-$name"
    wait_worker "decode-$name" 30001 3600
}
PAT="No available shared memory broadcast block found"
# load_round TAG DLOG: one synthetic warmup-shaped load; echoes STALL / DONE / DEAD.
load_round() {
    local tag=$1 dlog=$2 base n t0=$SECONDS
    base=$(grep -ac "$PAT" "$dlog")
    docker exec -d "$CTR" bash -c "timeout 1500 vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
        --base-url http://127.0.0.1:$ROUTER_PORT --model $SERVED --tokenizer $MODEL --trust-remote-code \
        --dataset-name random --random-input-len $LOAD_IN --random-output-len 1 --random-range-ratio 0.5 \
        --num-prompts $LOAD_N --max-concurrency $LOAD_C --percentile-metrics ttft,e2el --seed $RANDOM \
        > $R/load-$tag.log 2>&1"
    sleep 30
    while ((SECONDS - t0 < 1500)); do
        n=$(grep -ac "$PAT" "$dlog")
        if ((n > base)); then
            mkdir -p "$R/pyspy-$tag"
            docker exec "$CTR" ps -eo pid,ppid,stat,etime,pcpu,args > "$R/pyspy-$tag/ps.txt" 2>&1
            for p in $(awk 'NR > 1 && $6 ~ /^VLLM::(Worker|EngineCore)_DP/ {print $1}' "$R/pyspy-$tag/ps.txt"); do
                docker exec "$CTR" timeout 30 "$W/wheels/extra/py-spy" dump --pid "$p" > "$R/pyspy-$tag/$p.txt" 2>&1 &
            done
            wait
            rocm-smi --showuse --showmemuse --csv > "$R/pyspy-$tag/rocm-smi.csv" 2>&1
            docker exec "$CTR" pkill -f "vllm bench serve" 2>/dev/null
            echo STALL; return
        fi
        grep -qaE "permanently unhealthy|EngineCore encountered a fatal error" "$dlog" && { docker exec "$CTR" pkill -f "vllm bench serve"; echo DEAD; return; }
        docker exec "$CTR" pgrep -f "vllm bench serve" >/dev/null || { echo DONE; return; }
        sleep 10
    done
    docker exec "$CTR" pkill -f "vllm bench serve" 2>/dev/null
    echo TIMEOUT
}

first=1
for v in $VARIANTS; do
    case "$v" in
        V0) venv=(SPEC_GOLDEN=1) ;;
        V1nobreak) venv=(SPEC_GOLDEN=1 EXTRA_WORKER_ENV=VLLM_USE_BREAKABLE_CUDAGRAPH=0) ;;
        V2block) venv=() ;;
        V3eager) venv=(SPEC_GOLDEN=1 "D_EXTRA_ARGS=--enforce-eager") ;;
        V4randdummy) venv=(SPEC_GOLDEN=1 EXTRA_WORKER_ENV=VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1) ;;
        V5noasync) venv=(SPEC_GOLDEN=1 "D_EXTRA_ARGS=--no-async-scheduling") ;;
        *) say "unknown variant $v"; continue ;;
    esac
    ((first)) || decode_kill
    t0=$SECONDS
    if ! decode_start "$v" "${venv[@]}"; then
        say "$v: decode failed to start: $(grep -aE 'Error|error:|Traceback' "$R/decode-$v.log" | tail -3 | tr '\n' ' ' | head -c 500)"
        first=0; continue
    fi
    if ((first)); then
        wait_worker prefill0 30000 3600 && wait_worker prefill1 30002 3600 || { say "prefill failed"; exit 1; }
    fi
    first=0
    say "$v: decode ready in $((SECONDS - t0)) s (${venv[*]:-block acceptance}); spec: $(grep -aoE "'rejection_sample_method': '[a-z]+'" "$R/decode-$v.log" | sort -u | tr '\n' ' ')"
    router_up "router-$v" ROUTER_BACKEND=rust ROUTER_POLICY=kv-aware \
        ROUTER_EXTRA_ARGS="--kv-prefill-overlap-weight 20 --kv-decode-overlap-weight 2 --session-affinity prefill --pd-prefill-guard-release completion --decode-input-demand on" \
        || { say "$v: router failed"; continue; }
    chat "What is the capital of France? Answer with one word." 64 > "$R/smoke-$v.json"
    say "$v: smoke $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["choices"][0].get("finish_reason"), d.get("usage",{}).get("completion_tokens"))' "$R/smoke-$v.json" 2>&1 | head -c 200)"
    res=""
    for rd in $(seq "$ROUNDS"); do
        o=$(load_round "$v-r$rd" "$R/decode-$v.log")
        res+="r$rd=$o "
        say "$v round $rd: $o; $(grep -aE 'Successful requests|Failed requests|Mean TTFT' "$R/load-$v-r$rd.log" | tr -s ' ' | tr '\n' ' ' | head -c 300)"
        [ "$o" = DONE ] || break
    done
    say "$v result: $res"
    if [ "$v" = V0 ] && [[ "$res" != *STALL* && "$res" != *DEAD* ]]; then
        say "V0 did not stall under the synthetic load: the load does not reproduce the warmup stall; stopping"
        break
    fi
done
node_state after
say "end"
