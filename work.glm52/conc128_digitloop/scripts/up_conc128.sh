#!/usr/bin/env bash
# Relaunch both PD legs at the exp07 1k/1k high-concurrency capacity, on the PATCHED image.
# Only difference vs packup exp07: image = pd-unified-waitevent (854ebf70 KV-race fix), and
# prefill node is chi2867 (exp07 used chi2878).
set -euo pipefail
IMAGE="${IMAGE:-infera/engine-sglang:pd-unified-waitevent}"
MODEL=/mnt/vast/xiaobo/models/GLM-5.2-MXFP4
SERVED=glm5.2-mxfp4
PREFILL_HOST="${PREFILL_HOST:-chi2867}"; PREFILL_IP="${PREFILL_IP:-10.2.122.44}"
DECODE_HOST="${DECODE_HOST:-chi2879}";  DECODE_IP="${DECODE_IP:-10.2.122.10}"
CTX="${CTX:-32768}"            # exp07: 1k/1k needs <=2k/req
CHUNK="${CHUNK:-65536}"        # exp07: ISL 8192 x TP8 -> 8192/rank under DPA
MAX_RUNNING="${MAX_RUNNING:-2048}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-128}"
DPA="${DPA:-1}"
DMABUF="${DMABUF:-0}"
TAG="${TAG:-c128}"
CTR=pd_uni
KIT=/mnt/vast/c_huggingface/glm52_longctx_pd
J(){ ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 root@149.28.124.225 "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 $1 '$2'"; }

echo "== 0. stop old legs (containers stay; image already correct) =="
for h in "$PREFILL_HOST" "$DECODE_HOST"; do
  # NOTE: no nested single quotes -- J() already wraps the remote cmd in '...'; a bash -c '...'
  # inside it silently breaks the quoting and the kill never runs (bit us once: new legs then
  # die instantly on "port_base ... not available").
  J "$h" "docker exec $CTR pkill -9 -f launch_server; docker exec $CTR pkill -9 -f sglang_router; true" || true
done
sleep 20

echo "== 1. verify image + stage leg script =="
for h in "$PREFILL_HOST" "$DECODE_HOST"; do
  J "$h" "docker inspect -f '{{.Config.Image}}' $CTR; docker cp $KIT/pd_leg_dpa_longctx.sh $CTR:/pd_leg_dpa.sh"
done

COMMON="MODEL=$MODEL SERVED=$SERVED DPA=$DPA DMABUF=$DMABUF CTX=$CTX CHUNK=$CHUNK MAX_RUNNING=$MAX_RUNNING CUDA_GRAPH_BS=$CUDA_GRAPH_BS"
echo "== 2. launch prefill leg ($PREFILL_HOST :30000) =="
J "$PREFILL_HOST" "docker exec -d $CTR env ROLE=prefill MY_IP=$PREFILL_IP PORT=30000 $COMMON LOG=$KIT/pd_prefill_30000_$TAG.log bash /pd_leg_dpa.sh"
echo "== 3. launch decode leg ($DECODE_HOST :30001) =="
J "$DECODE_HOST" "docker exec -d $CTR env ROLE=decode MY_IP=$DECODE_IP PORT=30001 $COMMON LOG=$KIT/pd_decode_30001_$TAG.log bash /pd_leg_dpa.sh"

echo "== launched (tag=$TAG). DPA cold start ~6-10 min. Watch:"
echo "   $KIT/pd_prefill_30000_$TAG.log  /  pd_decode_30001_$TAG.log  -> 'ready to roll'"
