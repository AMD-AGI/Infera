#!/usr/bin/env bash
# One-node P/D with DP attention scored like InferenceX: prefill DP2 worker(s) (32-pd2p1d: two, GPUs 0-1 and 2-3;
# 33-pd1p1d: one, GPUs 0-1) + decode DP4 on GPUs 4-7 whose DSpark verifies with the golden acceptance length 3.51
# (SPEC_GOLDEN=1, InferenceX's AgentX throughput setting; the prefill keeps block rejection). Mooncake with one NIC per
# GPU and the vLLM STOPPED-prefill fix; Rust kv-aware router with the SGLang recipe's P/D flags; one AgentX fast point
# (CONC, default 32) through scripts/agentx_run.sh. Golden acceptance makes the output text meaningless, so only a
# connectivity smoke runs before the replay.
set -uo pipefail
case "${ITEM_NAME:-}" in *2p1d*) NP=2 ;; *) NP=1 ;; esac
RUN=${ITEM_NAME:-3x-pd-golden-agentx}-$(date -u +%m%dT%H%M)
source /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/lib.sh
MR_MODE=${MR_MODE:-$(rdma_mr_mode)} CONC=${CONC:-32}
SITE=/usr/local/lib/python3.12/dist-packages
left_min() {  # minutes left in this job (squeue %L: [D-]HH:MM:SS or MM:SS)
    local l d=0 h=0 m=0 t
    l="$(squeue -h -j "${SLURM_JOB_ID:?}" -o %L 2>/dev/null)"
    [[ "$l" == *-* ]] && { d=${l%%-*}; l=${l#*-}; }
    IFS=: read -ra t <<<"$l"
    if ((${#t[@]} == 3)); then h=${t[0]} m=${t[1]}; elif ((${#t[@]} == 2)); then m=${t[0]}; fi
    echo $((10#$d * 1440 + 10#$h * 60 + 10#$m))
}
say "start on $(hostname -s), job $SLURM_JOB_ID ($(left_min) min left), amdgpu $(cat /sys/module/amdgpu/version 2>/dev/null), MR_MODE=$MR_MODE, ${NP}P1D, CONC $CONC"
if (( $(left_min) < 60 )); then say "skipped: needs ~60 min"; exit 0; fi
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
GEM_SO=""
if [ "$MR_MODE" = gem ]; then
    build_gem eng > "$R/gem-build.log" 2>&1 || { say "GEM allocator build failed: $(tail -3 "$R/gem-build.log" | tr '\n' ' ')"; exit 1; }
fi
PCFG=$W/configs/pd1n_dpa_prefill_dp2.sh DCFG=$W/configs/pd1n_dpa_decode_dp4.sh
cp "$PCFG" "$R/prefill-config.sh"; cp "$DCFG" "$R/decode-config.sh"; cp "$W/configs/pd_mooncake_common.sh" "$R/"
common=(MR_MODE="$MR_MODE" GEM_SO="$GEM_SO" MC_TOPO=pergpu)
t0=$SECONDS
exec_bg eng prefill0 env "${common[@]}" EXTRA_WORKER_ENV="VLLM_SERVER_DEV_MODE=1" GPUS=0,1 PORT=30000 BOOT_PORT=28398 \
    DP_RPC_PORT=29610 DP_MASTER_PORT=29511 SCAN_PORT=40000 bash "$W/scripts/ctr_worker.sh" "$PCFG"
if ((NP == 2)); then
    sleep 20
    exec_bg eng prefill1 env "${common[@]}" EXTRA_WORKER_ENV="VLLM_SERVER_DEV_MODE=1" GPUS=2,3 PORT=30002 BOOT_PORT=28397 \
        DP_RPC_PORT=29612 DP_MASTER_PORT=29513 SCAN_PORT=42000 bash "$W/scripts/ctr_worker.sh" "$PCFG"
fi
sleep 20
exec_bg eng decode env "${common[@]}" SPEC_GOLDEN=1 bash "$W/scripts/ctr_worker.sh" "$DCFG"
pids=(); names=(prefill0:30000 decode:30001); ((NP == 2)) && names+=(prefill1:30002)
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
metrics="http://127.0.0.1:$ROUTER_PORT/metrics,http://127.0.0.1:30000/metrics,http://127.0.0.1:30001/metrics"
((NP == 2)) && metrics+=",http://127.0.0.1:30002/metrics"
say "AgentX fast c$CONC starting ($(left_min) min left)"
CONC=$CONC FAST=1 MODE=pd PREFILL_WORKERS=$NP PREFILL_GPUS=2 DECODE_GPUS=4 PREFILL_DP_ATTN=true DECODE_DP_ATTN=true \
    SIM_ACCEPT_LEN=3.51 METRICS_URLS="$metrics" timeout $(( ($(left_min) - 5) * 60 )) bash "$W/scripts/agentx_run.sh" "$R/agentx-c$CONC"
rc=$?
say "AgentX fast c$CONC rc=$rc: $(grep -aE 'Saved aggregated|failed|ERROR' "$R/agentx-c$CONC/runner.log" | tail -4 | tr -s ' ' | tr '\n' ' ' | head -c 700)"
node_state after
say "end"
