#!/usr/bin/env bash
# GLM-5.2 mooncake PD (RDMA) on infera/engine-sglang:pd-unified. prefill=chi2878, decode=chi2879.
# Follows the pd-unified packup method: pd_uni container + libionic inject on both nodes, launch
# both legs (mooncake, dmabuf OFF), wait ready, then sglang_router mini-LB in-container. No MTP.
set -euo pipefail
IMAGE=infera/engine-sglang:pd-unified
MODEL=/mnt/vast/xiaobo/models/GLM-5.2-MXFP4
SERVED=glm5.2-mxfp4
PREFILL_HOST=chi2878; PREFILL_IP=10.2.122.3
DECODE_HOST=chi2879;  DECODE_IP=10.2.122.10
ROUTER_PORT=8002; CONC="${CONC:-64}"; DMABUF="${DMABUF:-0}"
CTR=pd_uni
KIT=/mnt/vast/c_huggingface/glm52_p2b
J(){ ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 root@149.28.124.225 "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 $1 '$2'"; }

# 1. container + libionic inject on both nodes
prep(){ local h="$1"
  J "$h" "docker rm -f $CTR >/dev/null 2>&1 || true; docker run -d --name $CTR --network=host --ipc=host --shm-size=32G --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband --group-add video --group-add render --cap-add=SYS_PTRACE --cap-add=IPC_LOCK --security-opt seccomp=unconfined --ulimit memlock=-1:-1 -v /mnt/vast:/mnt/vast --entrypoint \"\" $IMAGE sleep infinity >/dev/null"
  J "$h" "HL=\$(readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1); B=\$(basename \$HL); docker cp \$HL $CTR:/usr/lib/x86_64-linux-gnu/\$B; docker exec $CTR bash -lc \"cd /usr/lib/x86_64-linux-gnu && ln -sf \$B libionic.so.1 && ln -sf libionic.so.1 libionic.so && cd libibverbs && ln -sf ../\$B libionic-rdmav34.so && ldconfig 2>/dev/null; echo active_ports=\\\$(ibv_devinfo | grep -c PORT_ACTIVE)\""
}
echo "== 1. prep containers (libionic inject) =="; prep "$PREFILL_HOST"; prep "$DECODE_HOST"

# 2. stage pd_leg.sh into both containers
for h in "$PREFILL_HOST" "$DECODE_HOST"; do J "$h" "docker cp $KIT/pd_leg.sh $CTR:/pd_leg.sh"; done

# 3. launch both legs (decode on :30001 per packup convention to avoid same-port confusion)
echo "== 2. launch prefill leg ($PREFILL_HOST :30000) =="
J "$PREFILL_HOST" "docker exec -d $CTR bash -lc 'ROLE=prefill MY_IP=$PREFILL_IP MODEL=$MODEL SERVED=$SERVED PORT=30000 DMABUF=$DMABUF MAX_RUNNING=$CONC bash /pd_leg.sh'"
echo "== 3. launch decode leg ($DECODE_HOST :30001) =="
J "$DECODE_HOST" "docker exec -d $CTR bash -lc 'ROLE=decode MY_IP=$DECODE_IP MODEL=$MODEL SERVED=$SERVED PORT=30001 DMABUF=$DMABUF MAX_RUNNING=$CONC bash /pd_leg.sh'"

echo "== legs launching (mooncake RDMA, dmabuf=$DMABUF). Cold start ~5min. =="
echo "   After BOTH print 'ready to roll', start router:"
echo "   J $PREFILL_HOST \"docker exec -d $CTR bash -lc 'pkill -9 -f sglang_router; python3 -m sglang_router.launch_router --pd-disaggregation --prefill http://$PREFILL_IP:30000 8998 --decode http://$DECODE_IP:30001 --host 0.0.0.0 --port $ROUTER_PORT > /tmp/router.log 2>&1'\""
