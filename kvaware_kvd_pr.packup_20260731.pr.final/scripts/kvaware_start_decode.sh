#!/usr/bin/env bash
# Bring up the DECODE side on chi2867. Runs ON chi2867.
# etcd lives on the prefill node (chi2879, 10.2.122.10).
set -u
IMAGE=infera/engine-sglang:kvaware-kvd
CTR=kvaware_kvd_final
KIT=/mnt/vast/c_huggingface/kvaware_kvd_final
MODEL=/mnt/vast/xiaobo/models/GLM-5.2-MXFP4
MY_IP=10.2.122.44
ETCD_IP=10.2.122.10
KVD_SOCK=/tmp/kvd/kvd.sock

echo "===== teardown ====="
docker rm -f $CTR >/dev/null 2>&1
pkill -9 -f 'infera.engine.sglang' 2>/dev/null
pkill -9 -f 'infera.kvd' 2>/dev/null
sleep 3
echo "  kfd pids: $(rocm-smi --showpids 2>/dev/null | grep -c 'PID' || echo 0)"

echo "===== container (entrypoint injects host libionic) ====="
HL=$(readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1)
docker run -d --name $CTR --network=host --ipc=host --shm-size=32G \
  --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
  --group-add video --group-add render --cap-add=SYS_PTRACE --cap-add=IPC_LOCK \
  --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
  -v /mnt/vast:/mnt/vast -v "$HL":/host-libionic/libionic.so:ro \
  $IMAGE sleep infinity >/dev/null
sleep 5
echo "  active RDMA ports in container: $(docker exec $CTR bash -c 'ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE')"

echo "===== kvd daemon ====="
docker exec $CTR bash -c "mkdir -p /tmp/kvd /tmp/kvd-long && rm -f $KVD_SOCK && printf '%s\n' '#!/bin/bash' 'exec python3 -m infera.kvd --socket $KVD_SOCK --max-bytes 64G --long-path /tmp/kvd-long --long-bytes 512G --log-level INFO' > /run_kvd.sh && chmod +x /run_kvd.sh"
docker exec -d $CTR bash -c 'nohup /run_kvd.sh > /tmp/kvd.log 2>&1'
sleep 20
docker exec $CTR bash -c "test -S $KVD_SOCK && echo '  kvd socket OK' || { echo '  KVD FAILED'; tail -20 /tmp/kvd.log; }"

echo "===== stage scripts ====="
for f in glm52_leg.sh probe.py stress_capture.py run_tests.sh; do
  docker cp "$KIT/$f" "$CTR:/tmp/$f" >/dev/null
done
docker exec $CTR bash -c 'cp /tmp/glm52_leg.sh /glm52_leg.sh'
echo "  staged"

echo "===== decode leg ====="
docker exec -d $CTR env ROLE=decode MY_IP=$MY_IP ETCD_IP=$ETCD_IP MODEL=$MODEL \
  SERVED=glm5.2-mxfp4 PORT=30000 KVAWARE=1 KVD=1 HICACHE_GB=16 KVD_SOCK=$KVD_SOCK \
  LOG=$KIT/decode_final.log bash /glm52_leg.sh
echo "  decode launched (cold start ~8-15 min)"
