#!/bin/bash
# DRIVER619-20261113: mc_gem_test.py pairs on this node, in the production image. Pair i: target GPU i, initiator GPU i+4,
# both on NIC ionic_<i+4> (RDMA loopback, MC_DISABLE_HIP=1 so Mooncake does not take its intra-node HIP path).
# Usage on the node: gem_node.sh OUT_DIR [PAIRS]   PAIRS: "i:pool:gib:dmabuf_off:kfd_pin_fix" separated by spaces
OUT=$1
PAIRS=${2:-"0:gem:150:0:0 1:kfd:150:1:0 2:gem:198:0:0 3:kfd:150:0:1"}
K=/perf_apps/liyingli/bench_agentx/kfd619
IMAGE=${IMAGE:-infera-sglang:v0522-fdbafb60}
TAR=/perf_apps/liyingli/images/infera-sglang-v0522-fdbafb60.tar
mkdir -p "$OUT"
h=$(hostname -s)
echo "$(date -u +%T) $h amdgpu $(cat /sys/module/amdgpu/version)" | tee "$OUT/$h.node"
docker image inspect $IMAGE >/dev/null 2>&1 || { echo "loading $IMAGE"; docker load -i $TAR >/dev/null; }
run() {  # name gpu [docker run args...] image cmd...
    local name=$1 g=$2; shift 2
    docker rm -f $name >/dev/null 2>&1
    timeout 1500 docker run --rm --name $name --network host --ipc host --pid host \
        --device /dev/kfd --device /dev/dri --device /dev/infiniband --group-add video --group-add render \
        --cap-add IPC_LOCK --cap-add SYS_PTRACE --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
        -v /lib/x86_64-linux-gnu/libionic.so:/host-libionic/libionic.so:ro -v /perf_apps:/perf_apps \
        -e ROCR_VISIBLE_DEVICES=$g -e RDMAV_FORK_SAFE=1 -e MC_GID_INDEX=1 \
        -e MC_MAX_MR_SIZE=1073741824 -e MC_IB_TIMEOUT=18 "$@"
}
B=$K/gemalloc/build
if [ ! -s $B/libhip_gem_alloc.so ] || [ $K/gemalloc/hip_gem_alloc.c -nt $B/libhip_gem_alloc.so ]; then
    mkdir -p $B
    run llying-kfd619-build 0 $IMAGE bash -c "cd $B && (command -v gcc || command -v hipcc) && \
        \${CC:-gcc} -O2 -Wall -shared -fPIC -D__HIP_PLATFORM_AMD__ -I/opt/rocm/include ../hip_gem_alloc.c \
        -o libhip_gem_alloc.so -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64 -libverbs -lpthread -ldl && ls -la libhip_gem_alloc.so" \
        > "$OUT/$h.build.log" 2>&1
    cat "$OUT/$h.build.log"
    [ -s $B/libhip_gem_alloc.so ] || { echo "build failed"; exit 1; }
fi
[ "$PAIRS" = build ] && exit 0
for i in $(seq 60); do
    rocm-smi --showmemuse 2>/dev/null | awk '/VRAM%/ {if ($NF + 0 > 2) b = 1} END {exit b}' && break
    sleep 20
done
pids=()
for pr in $PAIRS; do
    IFS=: read -r i pool gib dmaoff pinfix hipoff <<<"$pr"
    hipoff=${hipoff:-1}
    j=$((i + 4)) nic=ionic_$((i + 4))
    tag="$h-p$i-$pool-$gib-dmaoff$dmaoff-pinfix$pinfix-hipoff$hipoff"
    sig=$(mktemp -d /tmp/llying-gem.XXXX)
    envs=(-e MOONCAKE_DISABLE_HIP_DMABUF=$dmaoff -e MC_HIP_KFD_PIN_FIX=$pinfix -e MC_TE_FILTERS=$nic -v $sig:$sig)
    [ "$hipoff" = 1 ] && envs+=(-e MC_DISABLE_HIP=1)
    pre=""; [ "$pool" = gem ] && pre="LD_PRELOAD=$B/libhip_gem_alloc.so"
    run llying-kfd619-t$i $i "${envs[@]}" $IMAGE bash -c \
        "$pre python3 $K/bench/mc_gem_test.py --role target --pool $pool --gib $gib --nic $nic --sig $sig --tag $tag; echo exit \$?" \
        > "$OUT/$tag.target.log" 2>&1 &
    pids+=($!)
    run llying-kfd619-i$i $j "${envs[@]}" $IMAGE bash -c \
        "$pre python3 $K/bench/mc_gem_test.py --role initiator --pool $pool --gib 6 --nic $nic --sig $sig --tag $tag; echo exit \$?" \
        > "$OUT/$tag.initiator.log" 2>&1 &
    pids+=($!)
done
wait "${pids[@]}"
docker ps -a --format '{{.Names}}' | grep '^llying-kfd619-' | xargs -r docker rm -f >/dev/null 2>&1
for f in "$OUT"/$h-*.log; do
    echo "== $(basename $f)"
    grep -a -E '^\{|exit |Segmentation|Segfault|fatal|hip_gem_alloc|Error|error' "$f" | grep -a -v "Ignore import" | cut -c1-260 | tail -14
done | tee "$OUT/$h.summary"
