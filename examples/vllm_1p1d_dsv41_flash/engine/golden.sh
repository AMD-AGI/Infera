#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: switch the decode leg between real DSpark verification and InferenceX's golden acceptance.
#         golden.sh golden   restart decode with synthetic acceptance at GOLDEN_AL (default 3.51)
#         golden.sh block    restart decode with real (block) rejection sampling
# why : AgentX scores throughput at a fixed acceptance length (golden_al_distribution: DeepSeek-V4.1-
#       Flash, DSpark K=5, thinking_on = 3.51) so a benchmark-tuned draft cannot inflate the result.
#       Synthetic acceptance commits drafted tokens WITHOUT checking them against the target, so any
#       correctness run (smoke, accuracy) must use block — it would score near zero under golden.
# how : only the decode leg restarts; prefill, UMBP and etcd stay up. Run with no traffic in flight:
#       a pull pending on the old decode is only abandoned after VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../common.sh"
MODE="${1:?usage: golden.sh golden|block}"
case "$MODE" in golden|block) ;; *) die "usage: golden.sh golden|block" ;; esac
require_env DECODE_NODE; require_env DECODE_IP; require_env PREFILL_NODE; require_env PREFILL_IP
require_env KIT_DIR; require_env DECODE_GPUS

on "$DECODE_NODE" "CTR=$CTR bash -c 'source $KIT_DIR/common.sh; reap_leg $DECODE_PORT'"
[ -f "$LOG_DIR/decode.log" ] && mv "$LOG_DIR/decode.log" "$LOG_DIR/decode-$(date -u +%H%M%S).log"
on "$DECODE_NODE" "CTR=$CTR INFERA_IMAGE=$INFERA_IMAGE MODEL=$MODEL MODEL_MOUNT=$MODEL_MOUNT SERVED=$SERVED \
  ETCD_IP=$PREFILL_IP ETCD_PORT=$ETCD_PORT DECODE_PORT=$DECODE_PORT LOG_DIR=$LOG_DIR UMBP_DIR=$UMBP_DIR \
  MR_MODE=$MR_MODE RDMA_IB_DEVICES=$RDMA_IB_DEVICES MC_GID_INDEX=$MC_GID_INDEX \
  ${RDMA_GPU_NICS:+RDMA_GPU_NICS=$RDMA_GPU_NICS} READY_TIMEOUT=${READY_TIMEOUT:-3600} ${GMU:+GMU=$GMU} \
  ${D_PSI:+D_PSI=$D_PSI} ${GOLDEN_AL:+GOLDEN_AL=$GOLDEN_AL} ${EXTRA_ENGINE_ARGS:+EXTRA_ENGINE_ARGS=\"$EXTRA_ENGINE_ARGS\"} \
  ROLE=decode MY_IP=$DECODE_IP GPUS=$DECODE_GPUS SPEC_MODE=$MODE bash $KIT_DIR/engine/leg.sh"
tries=$(( (${READY_TIMEOUT:-3600} + 14) / 15 ))
for i in $(seq 1 "$tries"); do
  on "$DECODE_NODE" "docker exec $CTR curl -sf -m5 http://$DECODE_IP:$DECODE_PORT/health >/dev/null 2>&1" && break
  [ "$i" = "$tries" ] && die "decode leg did not come back ($LOG_DIR/decode.log)"
  sleep 15
done
log "decode serving with spec=$MODE after $((i * 15))s: $(grep -aoE "'rejection_sample_method': '[a-z]+'|synthetic_acceptance_length[^,}]*" "$LOG_DIR/decode.log" | head -2 | tr '\n' ' ')"
# The router drops a worker whose registration lapsed while it was down; restart it so its
# first view is the re-registered decode.
on "$PREFILL_NODE" "CTR=$CTR ETCD_PORT=$ETCD_PORT ROUTER_PORT=$ROUTER_PORT LOG_DIR=$LOG_DIR TOKENIZER=${TOKENIZER:-$MODEL} \
  ${SESSION_AFFINITY:+SESSION_AFFINITY=$SESSION_AFFINITY} bash -c 'source $KIT_DIR/common.sh; start_router $PREFILL_IP'"
