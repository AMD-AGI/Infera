#!/usr/bin/env bash
# Drive the kit's cluster/cluster.gem.sh on this node (AUS MI355X, amdgpu 6.19) for the 1P1D example runs:
#   fresh deployment -> smoke -> accuracy gate (real verification) -> golden 3.51 -> formal AgentX at CONC.
# Every step logs one line to $W/logs/summary.log. Runs on the node (srun --overlap into the job).
#   INFERA_IMAGE (required), CONC (384), GATE (1: smoke + accuracy before AgentX; 0: decode starts on golden),
#   FAST (0), RUN, plus any cluster.gem.sh variable (UMBP, UMBP_GB, PREFILL_GPUS, DECODE_GPUS, EXTRA_ENGINE_ARGS ...).
set -uo pipefail
source "$(dirname "$0")/env.sh"
KIT=/perf_apps/liyingli/bench_agentx/upstream/Infera-vllm-1p1d-dsv41/examples/vllm_1p1d_dsv41_flash
: "${INFERA_IMAGE:?INFERA_IMAGE}"
CONC=${CONC:-384} GATE=${GATE:-1} FAST=${FAST:-0}
RUN=${RUN:-pd1p1d-c$CONC-$(date -u +%m%dT%H%M)}
export LOG_DIR=$W/logs/$RUN
mkdir -p "$LOG_DIR" && chmod 777 "$LOG_DIR"
say() { echo "$(ts) $RUN: $*" | tee -a "$W/logs/summary.log"; }

export INFERA_IMAGE PREFILL_NODE=$(hostname -s) PREFILL_IP=$(node_ip)
export MODEL_MOUNT=/perf_apps/data/models MODEL=$MODEL
export RDMA_IB_DEVICES=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7 MC_GID_INDEX=1
export CTR=${CTR_PREFIX}pd UMBP_DIR=/dev/shm/${CTR_PREFIX}umbp-pd ETCD_PORT=22379 ROUTER_PORT=$ROUTER_PORT
export INFERENCEX_DIR=$IX AGENTX_HF_HOME=$HFC AGENTX_RUNTIME_DIR=$NODE_CACHE/agentx-runtime
export FAST
wrap() { bash "$KIT/cluster/cluster.gem.sh" "$@"; }
cleanup() { docker logs "$CTR" > "$LOG_DIR/docker-ctr.log" 2>&1; wrap down > "$LOG_DIR/down.log" 2>&1; }
trap cleanup EXIT

say "start on $PREFILL_NODE job ${SLURM_JOB_ID:-?}, $(docker run --rm --entrypoint cat "$INFERA_IMAGE" /opt/infera/BUILD_INFO | tr '\n' ' '), P GPUs ${PREFILL_GPUS:-0,1} D GPUs ${DECODE_GPUS:-4,5}, UMBP=${UMBP:-1} ${UMBP_GB:-449} GB, c$CONC GATE=$GATE FAST=$FAST ${NOTE:-}"
wrap down > "$LOG_DIR/down0.log" 2>&1
t0=$SECONDS
if ((GATE)); then wrap up > "$LOG_DIR/up.log" 2>&1; else SPEC_MODE=golden wrap up > "$LOG_DIR/up.log" 2>&1; fi
rc=$?
kv() { for l in prefill decode; do echo -n "$l $(grep -aoE 'GPU KV cache size: [0-9,]+' "$LOG_DIR/$l.log" | head -1 | grep -oE '[0-9,]+$') "; done; }
((rc)) && { say "up FAILED after $((SECONDS - t0)) s: $(grep -a 'ERROR\|died\|never' "$LOG_DIR/up.log" | tail -3 | tr '\n' ' ' | head -c 600)"; exit 1; }
say "up in $((SECONDS - t0)) s; KV tokens/rank: $(kv)"

if ((GATE)); then
    wrap smoke > "$LOG_DIR/smoke.log" 2>&1; rc=$?
    say "smoke rc=$rc: $(sed -r 's/\x1B\[[0-9;]*m//g' "$LOG_DIR/smoke.log" | grep -aE '^  (ok|FAIL)' | sed 's/^  //' | tr '\n' ';' | head -c 900)"
    wrap accuracy > "$LOG_DIR/accuracy.log" 2>&1; rc=$?
    say "accuracy: $(grep -ahE '^(gsm8k|needle@)' "$LOG_DIR"/accuracy/*.log | sed -E 's/ prompt_tokens_p50.*completion_tokens_max=/ ctmax=/' | tr '\n' ';' | head -c 700) $(grep -a '^ACCURACY' "$LOG_DIR/accuracy.log")"
    ((rc)) && { say "accuracy gate FAILED: no AgentX point"; exit 1; }
    t1=$SECONDS
    wrap golden > "$LOG_DIR/golden.log" 2>&1 || { say "golden restart failed: $(tail -2 "$LOG_DIR/golden.log" | tr '\n' ' ')"; exit 1; }
    say "decode on golden 3.51 after $((SECONDS - t1)) s: $(grep -aoE 'synthetic_acceptance_length[^,}]*' "$LOG_DIR/decode.log" | head -1)"
fi
code=$(curl -s -o "$LOG_DIR/smoke-golden.json" -w '%{http_code}' -m 600 "http://$PREFILL_IP:$ROUTER_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello.\"}],\"max_tokens\":64}")
say "golden smoke HTTP $code; AgentX $( ((FAST)) && echo fast || echo formal) c$CONC starting"
t2=$SECONDS
wrap agentx "$CONC" > "$LOG_DIR/agentx.log" 2>&1; rc=$?
say "AgentX c$CONC rc=$rc in $(( (SECONDS - t2) / 60 )) min: $(sed -r 's/\x1B\[[0-9;]*m//g' "$LOG_DIR/agentx.log" | grep -aA4 '^AgentX c' | tr -s ' ' | tr '\n' ' ' | head -c 700)"
j=$(ls "$LOG_DIR"/agentx-c$CONC/agentx_conc*.json 2>/dev/null | head -1)
[ -n "$j" ] && say "result row: $(python3 /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/ax_summary.py "$j" | tail -1)"
say "end"
