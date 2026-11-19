#!/usr/bin/env bash
# 33d / 35 unchanged (one-node 2P1D: prefill DP2 x 2 on GPUs 0-1 / 2-3, decode DP4 on GPUs 4-7 with golden acceptance
# 3.51, Mooncake one NIC per GPU + the vLLM STOPPED-prefill fix, Rust kv-aware router; formal AgentX c768), plus a host
# watchdog (scripts/hang_watch.sh) that dumps every engine process with py-spy when vLLM reports a stalled step.
# 32c and 33d both died in the AgentX warmup the same way: decode DP2 stuck in sample_tokens, DP0/1/3 in
# execute_dummy_batch, until the 300 s execute-model timeout. This run is to catch the stacks; if it does not stall it
# is the first formal baseline point.
set -uo pipefail
NP=2
[[ "${ITEM_NAME:-}" =~ -c([0-9]+)$ ]] && CONC=${BASH_REMATCH[1]}
FAST=0 NEED_MIN=150 RUNNER=agentx_formal.sh
RUN=${ITEM_NAME:-36-pd2p1d-golden-hangdiag}-$(date -u +%m%dT%H%M)
source /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/lib.sh
MR_MODE=${MR_MODE:-$(rdma_mr_mode)} CONC=${CONC:-768}
SITE=/usr/local/lib/python3.12/dist-packages
left_min() {
    local l d=0 h=0 m=0 t
    l="$(squeue -h -j "${SLURM_JOB_ID:?}" -o %L 2>/dev/null)"
    [[ "$l" == *-* ]] && { d=${l%%-*}; l=${l#*-}; }
    IFS=: read -ra t <<<"$l"
    if ((${#t[@]} == 3)); then h=${t[0]} m=${t[1]}; elif ((${#t[@]} == 2)); then m=${t[0]}; fi
    echo $((10#$d * 1440 + 10#$h * 60 + 10#$m))
}
say "start on $(hostname -s), job $SLURM_JOB_ID ($(left_min) min left), amdgpu $(cat /sys/module/amdgpu/version 2>/dev/null), MR_MODE=$MR_MODE, ${NP}P1D, CONC $CONC"
if (( $(left_min) < NEED_MIN )); then
    cp "/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/holder/running/$ITEM_NAME.sh" \
        "/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/holder/queue/"
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
export EXTRA_DOCKER_ARGS="-v $W/vllm-overlay/mooncake_connector.py:$SITE/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:ro"
start_etcd || { say "etcd failed"; exit 1; }
start_engine_ctr eng || { say "engine container / infera install failed"; exit 1; }
GEM_SO=""
if [ "$MR_MODE" = gem ]; then
    build_gem eng > "$R/gem-build.log" 2>&1 || { say "GEM allocator build failed: $(tail -3 "$R/gem-build.log" | tr '\n' ' ')"; exit 1; }
fi
PCFG=$W/configs/pd1n_dpa_prefill_dp2.sh DCFG=$W/configs/pd1n_dpa_decode_dp4_diag.sh
cp "$PCFG" "$R/prefill-config.sh"; cp "$DCFG" "$R/decode-config.sh"; cp "$W/configs/pd1n_dpa_decode_dp4.sh" "$R/"
cp "$W/configs/pd_mooncake_common.sh" "$R/"
mkdir -p "$R/vllm-stackdump" && chmod 777 "$R/vllm-stackdump"
common=(MR_MODE="$MR_MODE" GEM_SO="$GEM_SO" MC_TOPO=pergpu)
t0=$SECONDS
exec_bg eng prefill0 env "${common[@]}" EXTRA_WORKER_ENV="VLLM_SERVER_DEV_MODE=1" GPUS=0,1 PORT=30000 BOOT_PORT=28398 \
    DP_RPC_PORT=29610 DP_MASTER_PORT=29511 SCAN_PORT=40000 bash "$W/scripts/ctr_worker.sh" "$PCFG"
sleep 20
exec_bg eng prefill1 env "${common[@]}" EXTRA_WORKER_ENV="VLLM_SERVER_DEV_MODE=1" GPUS=2,3 PORT=30002 BOOT_PORT=28397 \
    DP_RPC_PORT=29612 DP_MASTER_PORT=29513 SCAN_PORT=42000 bash "$W/scripts/ctr_worker.sh" "$PCFG"
sleep 20
exec_bg eng decode env "${common[@]}" SPEC_GOLDEN=1 STACK_DUMP_DIR="$R/vllm-stackdump" bash "$W/scripts/ctr_worker.sh" "$DCFG"
pids=(); names=(prefill0:30000 decode:30001 prefill1:30002)
for nm in "${names[@]}"; do wait_worker "${nm%%:*}" "${nm##*:}" 7200 & pids+=($!); done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
if ((fail)); then
    for nm in "${names[@]}"; do n=${nm%%:*}; say "$n: $(grep -aE 'Error|error:|Traceback|assert' "$R/$n.log" | tail -3 | tr '\n' ' ' | head -c 600)"; done
    say "workers failed after $((SECONDS - t0)) s"; exit 1
fi
say "workers ready in $((SECONDS - t0)) s; decode spec: $(grep -aoE "'rejection_sample_method': '[a-z]+'|synthetic_acceptance_length[^,]*" "$R/decode.log" | sort -u | head -2 | tr '\n' ' ')"
router_up router ROUTER_BACKEND=rust ROUTER_POLICY=kv-aware \
    ROUTER_EXTRA_ARGS="--kv-prefill-overlap-weight 20 --kv-decode-overlap-weight 2 --session-affinity prefill --pd-prefill-guard-release completion --decode-input-demand on" \
    || { say "router failed"; exit 1; }
curl -s "http://127.0.0.1:$ROUTER_PORT/v1/workers" > "$R/workers.json"
say "router workers: $(python3 -c 'import json,sys; print(sorted((w.get("disagg_mode"), w.get("url"), w.get("dp_size")) for w in json.load(open(sys.argv[1]))["workers"]))' "$R/workers.json" 2>&1 | head -c 400)"
chat "What is the capital of France? Answer with one word." 64 > "$R/smoke1.json"
say "smoke (connectivity only under golden acceptance): $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["choices"][0].get("finish_reason"), d.get("usage"))' "$R/smoke1.json" 2>&1 | head -c 300)"
# py-spy self-test on one decode worker, then the stall watchdog.
mkdir -p "$R/pyspy"
wp=$(docker exec "${CTR_PREFIX}eng" pgrep -f "Worker_DP0" | head -1)
docker exec "${CTR_PREFIX}eng" timeout 30 "$W/wheels/extra/py-spy" dump --pid "${wp:-1}" > "$R/pyspy/selftest.txt" 2>&1
say "py-spy self-test on pid ${wp:-none}: rc=$? $(grep -c 'Thread ' "$R/pyspy/selftest.txt") threads, $(head -c 200 "$R/pyspy/selftest.txt" | tr '\n' ' ')"
bash "$W/scripts/hang_watch.sh" "$R" 2 &
WATCH=$!
metrics="http://127.0.0.1:$ROUTER_PORT/metrics,http://127.0.0.1:30000/metrics,http://127.0.0.1:30001/metrics,http://127.0.0.1:30002/metrics"
say "AgentX formal c$CONC starting ($(left_min) min left)"
CONC=$CONC FAST=$FAST DURATION=3600 MODE=pd PREFILL_WORKERS=$NP PREFILL_GPUS=2 DECODE_GPUS=4 PREFILL_DP_ATTN=true DECODE_DP_ATTN=true \
    SIM_ACCEPT_LEN=3.51 METRICS_URLS="$metrics" timeout $(( ($(left_min) - 5) * 60 )) bash "$W/scripts/$RUNNER" "$R/agentx-c$CONC"
rc=$?
say "AgentX c$CONC (FAST=$FAST) rc=$rc: $(grep -aE 'Saved aggregated|failed|ERROR' "$R/agentx-c$CONC/runner.log" | tail -4 | tr -s ' ' | tr '\n' ' ' | head -c 700)"
say "hang watch: $(tr '\n' ' ' < "$R/pyspy/watch.log" | head -c 600); vLLM stack dumps: $(ls "$R/vllm-stackdump" | wc -l) files"
node_state after
say "end"
