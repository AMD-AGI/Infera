#!/bin/bash
# Native sglang_router in PD-disaggregation mode. Run INSIDE mtt_pd on the
# prefill node. Fronts the prefill+decode pair. No etcd/nats/infera.
set -u
P_IP="${P_IP:?prefill data-plane IP}"
D_IP="${D_IP:?decode data-plane IP}"
RPORT="${RPORT:-8100}"           # router port (:8000 often taken)
SGL_PORT="${SGL_PORT:-30000}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
OUT="${OUT:-/mnt/vast/c_huggingface/mtt_study}"
LOG="${LOG:-$OUT/router_$(date +%H%M%S).log}"
mkdir -p "$OUT"
echo "[router] :$RPORT  prefill=http://$P_IP:$SGL_PORT (bs $BOOTSTRAP_PORT)  decode=http://$D_IP:$SGL_PORT"
python3 -m sglang_router.launch_router --pd-disaggregation \
  --prefill "http://$P_IP:$SGL_PORT" "$BOOTSTRAP_PORT" \
  --decode "http://$D_IP:$SGL_PORT" \
  --policy round_robin \
  --host 0.0.0.0 --port "$RPORT" > "$LOG" 2>&1 &
echo $! > "$OUT/router.pid"
echo "[router] pid=$(cat $OUT/router.pid) -> tail -f $LOG"
