#!/usr/bin/env bash
# Infera + vLLM aggregated reference on InferenceX's AgentX metric: one TP2 worker (InferenceX MI355X vLLM main-curve
# layout, golden acceptance length 3.51; configs/agg_tp2_golden.sh) behind the Rust router, then one AgentX fast point
# (CONC, default 64: the InferenceX TP2 c64 settings agg_tp2.sh copies). Also brings the AgentX client up on this node
# independently of the P/D deployments.
set -uo pipefail
RUN=31-agg-tp2-agentx-fast-$(date -u +%m%dT%H%M)
source /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/lib.sh
CFG=$W/configs/agg_tp2_golden.sh CONC=${CONC:-64}
say "start on $(hostname -s), job $SLURM_JOB_ID, config $CFG, CONC $CONC"
node_state before
ensure_images > "$R/pull.log" 2>&1 || { say "image pull failed"; exit 1; }
remove_ours
r="$(bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/clear_foreign_containers.sh "$SLURM_JOB_ID" 2>&1)"
[ -z "$r" ] || say "foreign containers: $r"
trap cleanup EXIT
start_guard
start_etcd || { say "etcd failed"; exit 1; }
start_engine_ctr eng || { say "engine container / infera install failed"; exit 1; }
cp "$CFG" "$R/config.sh"
t0=$SECONDS
exec_bg eng worker0 bash "$W/scripts/ctr_worker.sh" "$CFG"
if ! wait_worker worker0 30000 7200; then
    say "worker failed after $((SECONDS - t0)) s: $(grep -aE 'Error|error:|Traceback' "$R/worker0.log" | tail -4 | tr '\n' ' ' | head -c 800)"
    exit 1
fi
say "worker ready in $((SECONDS - t0)) s; $(grep -aoE 'GPU KV cache size: [0-9,]+ tokens' "$R/worker0.log" | sort -u | tr '\n' ' ')"
router_up router ROUTER_BACKEND=rust ROUTER_POLICY=round-robin || { say "router failed"; exit 1; }
chat "What is the capital of France? Answer with one word." 512 > "$R/smoke1.json"
say "smoke1: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(repr((d["choices"][0]["message"].get("content") or "")[:100]))' "$R/smoke1.json" 2>&1 | head -c 300)"
say "AgentX fast c$CONC starting"
CONC=$CONC FAST=1 MODE=agg AGG_TP=2 SIM_ACCEPT_LEN=3.51 \
    METRICS_URLS="http://127.0.0.1:$ROUTER_PORT/metrics,http://127.0.0.1:30000/metrics" \
    timeout 5400 bash "$W/scripts/agentx_fast.sh" "$R/agentx-c$CONC"
rc=$?
say "AgentX fast c$CONC rc=$rc: $(grep -aE 'tok/s/gpu|Output token throughput|Request throughput|Saved aggregated|failed|ERROR' "$R/agentx-c$CONC/runner.log" | tail -6 | tr -s ' ' | tr '\n' ' ' | head -c 900)"
node_state after
say "end"
