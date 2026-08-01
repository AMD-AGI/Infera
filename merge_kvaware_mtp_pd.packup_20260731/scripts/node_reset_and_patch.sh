#!/usr/bin/env bash
# Per-node reset ritual + merged patch application. Runs ON the node.
#
# RDMA discipline: tear the old container down, confirm the GPUs actually went
# back to idle (a leg that still holds VRAM makes the next round OOM in a way
# that looks like a regression in the patch), then start fresh and patch.
#
#   ROLE=prefill|decode  MY_IP=<rail ip>  [ETCD=1]
set -u
ROLE="${ROLE:?}"
MY_IP="${MY_IP:?}"
ETCD="${ETCD:-0}"
IMAGE="${IMAGE:-infera/engine-sglang:kvaware-kvd}"
CTR="${CTR:-merge_g0}"
KIT=/mnt/vast/c_huggingface/merge_20260731
KVD_SOCK=/tmp/kvd/kvd.sock

echo "===== 1. teardown ====="
docker rm -f "$CTR" kvaware_kvd_final >/dev/null 2>&1
[ "$ETCD" = "1" ] && docker rm -f "${CTR}_etcd" kvaware_kvd_final_etcd >/dev/null 2>&1
pkill -9 -f 'infera.engine.sglang' 2>/dev/null
pkill -9 -f 'infera.server' 2>/dev/null
pkill -9 -f 'infera.kvd' 2>/dev/null
pkill -9 -f 'sglang.launch_server' 2>/dev/null
sleep 8

echo "===== 2. confirm GPUs released ====="
for i in $(seq 1 30); do
  n=$(rocm-smi --showpids 2>/dev/null | grep -cE '^[0-9]+' || true); n=${n:-0}
  [ "$n" -eq 0 ] && { echo "  GPUs idle after ${i}x2s"; break; }
  sleep 2
done
rocm-smi --showpids 2>/dev/null | grep -E '^[0-9]+' | head -5
echo "  (empty above = clean)"

echo "===== 3. fresh container ====="
HL=$(readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1)
docker run -d --name "$CTR" --network=host --ipc=host --shm-size=32G \
  --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
  --group-add video --group-add render --cap-add=SYS_PTRACE --cap-add=IPC_LOCK \
  --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
  -v /mnt/vast:/mnt/vast -v "$HL":/host-libionic/libionic.so:ro \
  "$IMAGE" sleep infinity >/dev/null
sleep 5

echo "===== 4. RDMA fabric check (inside container) ====="
np=$(docker exec "$CTR" bash -c 'ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE')
echo "  PORT_ACTIVE in container: $np (want 8)"
[ "$np" -ge 1 ] || { echo "  RDMA DEAD -- libionic injection failed" >&2; exit 1; }

echo "===== 5. apply merged patch set ====="
docker exec "$CTR" bash -c "rm -rf /tmp/merge_patches && mkdir -p /tmp/merge_patches"
docker cp "$KIT/patches/." "$CTR":/tmp/merge_patches/ >/dev/null
docker cp "$KIT/scripts/apply_sglang_dsa_patches.sh" "$CTR":/tmp/merge_patches/ >/dev/null
docker cp "$KIT/scripts/apply_all_in_container.sh" "$CTR":/tmp/merge_patches/ >/dev/null
docker exec "$CTR" bash /tmp/merge_patches/apply_all_in_container.sh || {
  echo "  PATCH FAILED" >&2; exit 1; }

echo "===== 6. stage run scripts ====="
docker cp "$KIT/scripts/glm52_leg.sh" "$CTR":/glm52_leg.sh >/dev/null
for f in probe.py stress_capture.py prefix_reuse.py; do
  docker cp "$KIT/scripts/$f" "$CTR":/tmp/$f >/dev/null
done

if [ "$ETCD" = "1" ]; then
  echo "===== 7. etcd ====="
  docker run -d --name "${CTR}_etcd" --network=host quay.io/coreos/etcd:v3.5.14 etcd \
    --advertise-client-urls "http://$MY_IP:2379" --listen-client-urls http://0.0.0.0:2379 >/dev/null
  sleep 5
  curl -sf -m5 "http://$MY_IP:2379/version" >/dev/null && echo "  etcd up" || echo "  ETCD FAILED"
fi

echo "===== 8. kvd daemon ====="
docker exec "$CTR" bash -c "mkdir -p /tmp/kvd /tmp/kvd-long && rm -f $KVD_SOCK && printf '%s\n' '#!/bin/bash' 'exec python3 -m infera.kvd --socket $KVD_SOCK --max-bytes 64G --long-path /tmp/kvd-long --long-bytes 512G --log-level INFO' > /run_kvd.sh && chmod +x /run_kvd.sh"
docker exec -d "$CTR" bash -c 'nohup /run_kvd.sh > /tmp/kvd.log 2>&1'
sleep 20
docker exec "$CTR" bash -c "test -S $KVD_SOCK && echo '  kvd socket OK' || { echo '  KVD FAILED'; tail -20 /tmp/kvd.log; }"

echo "===== node $(hostname) ready for $ROLE ====="
