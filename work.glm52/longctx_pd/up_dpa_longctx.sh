#!/usr/bin/env bash
# GLM-5.2-MXFP4 mooncake PD + DP-attention, LONG CONTEXT (ctx=131072) so a 65K/119K-token
# input fits. Adapted from packup 07_pd_mooncake_dpa_sweep/scripts/up_dpa.sh:
#   - prefill moved to chi2867 (10.2.122.44), decode stays chi2879 (10.2.122.10)
#   - CTX 32768 -> 131072, chunk set explicitly (long prefill, not the 1k/1k sweep shape)
#   - MAX_RUNNING 2048 -> 64 (long ctx; conc=32 target)
# The leg script itself (pd_leg_dpa.sh) is unchanged from exp07 — all deltas are env.
set -euo pipefail
IMAGE="${IMAGE:-infera/engine-sglang:pd-unified}"   # :pd-unified-waitevent = + mooncake wait_event patch
MODEL=/mnt/vast/xiaobo/models/GLM-5.2-MXFP4
SERVED=glm5.2-mxfp4
PREFILL_HOST="${PREFILL_HOST:-chi2867}"; PREFILL_IP="${PREFILL_IP:-10.2.122.44}"
DECODE_HOST="${DECODE_HOST:-chi2879}";  DECODE_IP="${DECODE_IP:-10.2.122.10}"
ROUTER_PORT="${ROUTER_PORT:-8002}"
CTX="${CTX:-131072}"
CHUNK="${CHUNK:-16384}"
MAX_RUNNING="${MAX_RUNNING:-64}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-64}"
DMABUF="${DMABUF:-0}"
DPA="${DPA:-1}"                       # 0 = plain TP8 PD (control arm for the long-ctx garbage bug)
CTR=pd_uni
KIT=/mnt/vast/c_huggingface/glm52_longctx_pd
J(){ ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 root@149.28.124.225 "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 $1 '$2'"; }

prep(){ local h="$1"
  J "$h" "docker rm -f $CTR >/dev/null 2>&1 || true; docker run -d --name $CTR --network=host --ipc=host --shm-size=32G --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband --group-add video --group-add render --cap-add=SYS_PTRACE --cap-add=IPC_LOCK --security-opt seccomp=unconfined --ulimit memlock=-1:-1 -v /mnt/vast:/mnt/vast --entrypoint \"\" $IMAGE sleep infinity >/dev/null"
  J "$h" "HL=\$(readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1); B=\$(basename \$HL); docker cp \$HL $CTR:/usr/lib/x86_64-linux-gnu/\$B; docker exec $CTR bash -lc \"cd /usr/lib/x86_64-linux-gnu && ln -sf \$B libionic.so.1 && ln -sf libionic.so.1 libionic.so && cd libibverbs && ln -sf ../\$B libionic-rdmav34.so && ldconfig 2>/dev/null; echo active_ports=\\\$(ibv_devinfo | grep -c PORT_ACTIVE)\""
}
echo "== 1. prep containers (libionic inject) =="; prep "$PREFILL_HOST"; prep "$DECODE_HOST"

echo "== 2. stage scripts =="
for h in "$PREFILL_HOST" "$DECODE_HOST"; do
  J "$h" "docker cp $KIT/pd_leg_dpa_longctx.sh $CTR:/pd_leg_dpa.sh; docker cp $KIT/longctx_probe.py $CTR:/tmp/lp.py; docker cp $KIT/probe.py $CTR:/tmp/probe.py 2>/dev/null || true"
done

# NOTE: `env VAR=... bash /script`, NOT `bash -lc '...'` — the login-shell -d form does not persist.
COMMON="MODEL=$MODEL SERVED=$SERVED DPA=$DPA DMABUF=$DMABUF CTX=$CTX CHUNK=$CHUNK MAX_RUNNING=$MAX_RUNNING CUDA_GRAPH_BS=$CUDA_GRAPH_BS FORCE_TCP=${FORCE_TCP:-0} NO_CHUNKED_PREFIX=${NO_CHUNKED_PREFIX:-0} NO_OVERLAP=${NO_OVERLAP:-0}"
echo "== 3. launch prefill DPA leg ($PREFILL_HOST :30000) =="
TAG="${TAG:-dpa$DPA}"
J "$PREFILL_HOST" "docker exec -d $CTR env ROLE=prefill MY_IP=$PREFILL_IP PORT=30000 $COMMON LOG=$KIT/pd_prefill_30000_$TAG.log bash /pd_leg_dpa.sh"
echo "== 4. launch decode DPA leg ($DECODE_HOST :30001) =="
J "$DECODE_HOST" "docker exec -d $CTR env ROLE=decode MY_IP=$DECODE_IP PORT=30001 $COMMON LOG=$KIT/pd_decode_30001_$TAG.log bash /pd_leg_dpa.sh"

echo "== legs launching (mooncake RDMA + DPA, ctx=$CTX). cold start ~5-10min. =="
echo "   watch: grep -aE 'ready to roll|Traceback|rror' $KIT/pd_{prefill_30000,decode_30001}.log"
echo "   then:  bash run_router.sh"
