#!/bin/bash
# REPORT-20261110 section 6: Mooncake VRAM registration on an amdgpu 6.19 node, dma-buf vs peer memory
# (MOONCAKE_DISABLE_HIP_DMABUF=0 / 1): one RDMA-loopback write GPU 0 -> GPU 4 on ionic_4 (MC_DISABLE_HIP=1, the one-node
# 4+4 path), Infera's mooncakeperf, inside the infera image (entrypoint swaps in the host libionic).
# Usage on the node: mc_dmabuf_test.sh [OUT_DIR]
OUT=${1:-/perf_apps/liyingli/bench_agentx/kopt-20261109/bench/out/mc_dmabuf}
mkdir -p $OUT
IMAGE=infera-sglang:v0522-fdbafb60
for dmabuf in 0 1; do
    docker rm -f llying-mc-test >/dev/null 2>&1
    docker run --rm --name llying-mc-test --network host --ipc host --shm-size 16g \
        --device /dev/kfd --device /dev/dri --device /dev/infiniband \
        --group-add video --group-add render --cap-add SYS_PTRACE --cap-add IPC_LOCK \
        --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 \
        -v /lib/x86_64-linux-gnu/libionic.so:/host-libionic/libionic.so:ro -v /perf_apps:/perf_apps \
        -e MOONCAKE_DISABLE_HIP_DMABUF=$dmabuf -e MC_DISABLE_HIP=1 -e MC_GID_INDEX=1 -e MC_MAX_MR_SIZE=1073741824 \
        -e MC_IB_TIMEOUT=18 -e RDMAV_FORK_SAFE=1 -e MC_ENABLE_DEST_DEVICE_AFFINITY=1 \
        $IMAGE bash -c '
ip=$(ip -4 -o addr show dev fenic | awk "{print \$4}" | cut -d/ -f1); dev=ionic_4
MP="python3 -m infera.tools.preflight.network.mooncakeperf"
sig=$(mktemp -d /tmp/mc.XXXX)
MC_TE_FILTERS=$dev $MP "{\"role\":\"target\",\"sig\":\"$sig\",\"hostname\":\"$ip:0\",\"host\":\"$(hostname -s)\",\"protocol\":\"rdma\",\"operation\":\"write\",\"loc\":\"gpu\",\"gpu\":4,\"dev\":\"$dev\"}" >"$sig/target.log" 2>&1 &
tp=$!
MC_TE_FILTERS=$dev timeout 150 $MP "{\"role\":\"initiator\",\"sig\":\"$sig\",\"protocol\":\"rdma\",\"operation\":\"write\",\"loc\":\"gpu\",\"gpu\":0,\"dev\":\"$dev\"}" >"$sig/init.log" 2>&1
echo "initiator rc=$?"
touch "$sig/done"; wait "$tp"; echo "target rc=$?"
python3 -c "import json; r = json.load(open(\"$sig/result.json\")); print(\"result\", r.get(\"gb_s\"), r.get(\"verified\"), r.get(\"reason\"))" 2>/dev/null || echo "no result"
echo "--- initiator log"; grep -a -i -E "dmabuf|dma-buf|reg|mr|error|fail|segment|fault" "$sig/init.log" | head -15
echo "--- target log"; grep -a -i -E "dmabuf|dma-buf|reg|mr|error|fail|segment|fault" "$sig/target.log" | head -15
' > $OUT/dmabuf$dmabuf.log 2>&1
    echo "== MOONCAKE_DISABLE_HIP_DMABUF=$dmabuf"; cat $OUT/dmabuf$dmabuf.log | cut -c1-220 | tail -36
done
