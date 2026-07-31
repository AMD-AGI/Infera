#!/usr/bin/env bash
# ===========================================================================
# HISTORICAL ARTIFACT — this is the original one-shot launcher used in the
# 2026-07-30 session. **scripts/run.sh is this folder's canonical driver** and
# fully supersedes this file (preflight, trap handling, evidence collection,
# verdict). Kept for provenance; note it uses its own container name and kit
# directory, so do NOT mix the two in one session.
# ===========================================================================
# GLM-5.2-MXFP4 two-node PD (mooncake RDMA) + DP-attention, through the infera stack,
# with kv-aware routing and kvd switchable. Runs on the LAPTOP/jump-capable host.
#
#   prefill = chi2879 (10.2.122.10)   decode = chi2867 (10.2.122.44)
#
# KVAWARE/KVD=1 -> the configuration under test.
# KVAWARE/KVD=0 -> the 对拍 baseline (reproduces the 4/4 of the earlier
#                 PD+DPA sweep in the SEPARATE packup glm5.2.mxfp4.packup_20260727;
#                 nothing to do with the folder numbering here).
set -uo pipefail
IMAGE="${IMAGE:-infera/engine-sglang:pd-unified}"
MODEL="${MODEL:-/mnt/vast/xiaobo/models/GLM-5.2-MXFP4}"
SERVED="${SERVED:-glm5.2-mxfp4}"
PREFILL_HOST="${PREFILL_HOST:-chi2879}"; PREFILL_IP="${PREFILL_IP:-10.2.122.10}"
DECODE_HOST="${DECODE_HOST:-chi2867}";   DECODE_IP="${DECODE_IP:-10.2.122.44}"
ROUTER_PORT="${ROUTER_PORT:-8100}"; ETCD_PORT="${ETCD_PORT:-2379}"
KVAWARE="${KVAWARE:-1}"; KVD="${KVD:-1}"; HICACHE_GB="${HICACHE_GB:-16}"
POLICY="${POLICY:-kv-aware}"
TAG="${TAG:-g1}"
CTR="${CTR:-glm52_kvexp}"
KIT="${KIT:-/mnt/vast/c_huggingface/glm52_kvexp}"
KVD_SOCK=/tmp/kvd/kvd.sock
JUMP="${JUMP:-root@149.28.124.225}"

J(){ ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$JUMP" "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 $1 '$2'"; }

echo "===== 0. container prep + libionic inject ====="
for h in "$PREFILL_HOST" "$DECODE_HOST"; do
  J "$h" "docker rm -f $CTR >/dev/null 2>&1; docker run -d --name $CTR --network=host --ipc=host --shm-size=32G --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband --group-add video --group-add render --cap-add=SYS_PTRACE --cap-add=IPC_LOCK --security-opt seccomp=unconfined --ulimit memlock=-1:-1 -v /mnt/vast:/mnt/vast --entrypoint \"\" $IMAGE sleep infinity >/dev/null"
  J "$h" "HL=\$(readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1); B=\$(basename \$HL); docker cp \$HL $CTR:/usr/lib/x86_64-linux-gnu/\$B; docker exec $CTR bash -lc \"cd /usr/lib/x86_64-linux-gnu && ln -sf \$B libionic.so.1 && ln -sf libionic.so.1 libionic.so && cd libibverbs && ln -sf ../\$B libionic-rdmav34.so && ldconfig 2>/dev/null; echo $h active_ports=\\\$(ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE)\""
  # the net.py port-collision fix (infera bug found in the Qwen3 MVP round)
  J "$h" "docker cp $KIT/net_fixed.py $CTR:/opt/infera/infera/common/net.py && echo $h net_fix_applied"
done

echo "===== 1. etcd on prefill node ====="
J "$PREFILL_HOST" "docker rm -f ${CTR}_etcd >/dev/null 2>&1; docker run -d --name ${CTR}_etcd --network=host quay.io/coreos/etcd:v3.5.14 etcd --advertise-client-urls http://$PREFILL_IP:$ETCD_PORT --listen-client-urls http://0.0.0.0:$ETCD_PORT >/dev/null; sleep 4; curl -sf -m5 http://$PREFILL_IP:$ETCD_PORT/version && echo ' etcd_up'"

if [ "$KVD" = "1" ]; then
  echo "===== 2. kvd daemon on BOTH nodes (each engine talks to its local daemon) ====="
  for h in "$PREFILL_HOST" "$DECODE_HOST"; do
    J "$h" "docker exec -d $CTR bash -lc 'mkdir -p /tmp/kvd /tmp/kvd-long; rm -f $KVD_SOCK; python3 -m infera.kvd --socket $KVD_SOCK --max-bytes 16G --long-path /tmp/kvd-long --long-bytes 128G --log-level INFO > /tmp/kvd.log 2>&1'"
  done
  sleep 25
  for h in "$PREFILL_HOST" "$DECODE_HOST"; do
    J "$h" "docker exec $CTR bash -lc 'test -S $KVD_SOCK && echo $h kvd_socket_ok || { echo $h KVD_FAILED; tail -20 /tmp/kvd.log; }'"
  done
fi

echo "===== 3. stage leg script into both containers ====="
for h in "$PREFILL_HOST" "$DECODE_HOST"; do
  J "$h" "docker cp $KIT/glm52_leg.sh $CTR:/glm52_leg.sh && docker cp $KIT/probe.py $CTR:/tmp/probe.py && echo $h staged"
done

echo "===== 4. launch legs (prefill $PREFILL_IP:30000 | decode $DECODE_IP:30000) ====="
# NOTE: `docker exec -d $CTR env VAR=... bash /script` — the detached-shell gotcha: the
# `bash -lc '...'` detached form does not persist.
J "$PREFILL_HOST" "docker exec -d $CTR env ROLE=prefill MY_IP=$PREFILL_IP ETCD_IP=$PREFILL_IP MODEL=$MODEL SERVED=$SERVED PORT=30000 KVAWARE=$KVAWARE KVD=$KVD HICACHE_GB=$HICACHE_GB KV_PUB_PORT=5557 LOG=$KIT/pd_prefill_$TAG.log bash /glm52_leg.sh"
sleep 20
J "$DECODE_HOST" "docker exec -d $CTR env ROLE=decode MY_IP=$DECODE_IP ETCD_IP=$PREFILL_IP MODEL=$MODEL SERVED=$SERVED PORT=30000 KVAWARE=$KVAWARE KVD=$KVD HICACHE_GB=$HICACHE_GB KV_PUB_PORT=5557 LOG=$KIT/pd_decode_$TAG.log bash /glm52_leg.sh"

echo "===== 5. infera router on prefill node (policy=$POLICY) ====="
sleep 10
J "$PREFILL_HOST" "docker exec -d $CTR bash -lc 'python3 -m infera.server --host 0.0.0.0 --port $ROUTER_PORT --discovery-backend etcd --etcd-endpoint $PREFILL_IP:$ETCD_PORT --request-transport http --kv-event-transport zmq --router-policy $POLICY --router-tokenizer-path $MODEL > /tmp/router.log 2>&1'"

echo
echo "launched tag=$TAG kvaware=$KVAWARE kvd=$KVD policy=$POLICY"
echo "GLM-5.2 DP cold start ~8-15 min. Poll:"
echo "  J $PREFILL_HOST \"grep -c 'ready to roll' $KIT/pd_prefill_$TAG.log\""
echo "  J $DECODE_HOST  \"grep -c 'ready to roll' $KIT/pd_decode_$TAG.log\""
echo "Correctness: docker exec $CTR python3 /tmp/probe.py http://$PREFILL_IP:$ROUTER_PORT $SERVED"
