#!/bin/bash
# DRIVER619-20261113: run kfd_pin_repro.py cases on this node, one GPU per case in parallel (each container sees only its
# GPU and uses its rail NIC ionic_<gpu>), inside the production image (its entrypoint swaps in the host libionic).
# Usage on the node: repro_node.sh OUT_DIR [CASES]   CASES: "gpu:alloc:reg:gib[:nics]" separated by spaces
OUT=$1
CASES=${2:-"0:kfd:dmabuf:40 1:kfd:peermem:40 2:kfd:dmabuf:150 3:kfd:peermem:150 4:gem:dmabuf:40 5:gem:dmabuf:150 6:kfd:peermem:40:all 7:gem:dmabuf:200"}
K=/perf_apps/liyingli/bench_agentx/kfd619
IMAGE=${IMAGE:-infera-sglang:v0522-fdbafb60}
TAR=/perf_apps/liyingli/images/infera-sglang-v0522-fdbafb60.tar
mkdir -p "$OUT"
h=$(hostname -s)
echo "$(date -u +%T) $h amdgpu $(cat /sys/module/amdgpu/version) kernel $(uname -r)" | tee "$OUT/$h.node"
docker image inspect $IMAGE >/dev/null 2>&1 || { echo "loading $IMAGE"; docker load -i $TAR >/dev/null; }
for i in $(seq 60); do
    rocm-smi --showmemuse 2>/dev/null | awk '/VRAM%/ {if ($NF + 0 > 2) b = 1} END {exit b}' && break
    sleep 20
done
rocm-smi --showmemuse 2>/dev/null | grep VRAM% | tee -a "$OUT/$h.node"
pids=()
for c in $CASES; do
    IFS=: read -r g alloc reg gib nics <<<"$c"
    [ "$nics" = all ] && nics=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
    nics=${nics:-ionic_$g}
    name=llying-kfd619-g$g
    tag="$h-g$g-$alloc-$reg-$gib"
    docker rm -f $name >/dev/null 2>&1
    timeout 1200 docker run --rm --name $name --network host --ipc host --pid host \
        --device /dev/kfd --device /dev/dri --device /dev/infiniband --group-add video --group-add render \
        --cap-add IPC_LOCK --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
        -v /lib/x86_64-linux-gnu/libionic.so:/host-libionic/libionic.so:ro -v /perf_apps:/perf_apps \
        -e ROCR_VISIBLE_DEVICES=$g -e RDMAV_FORK_SAFE=1 \
        $IMAGE bash -c "python3 $K/bench/kfd_pin_repro.py --gpu 0 --alloc $alloc --reg $reg --gib $gib --nics $nics --tag $tag; echo \"exit \$?\"" \
        > "$OUT/$tag.log" 2>&1 &
    pids+=($!)
done
wait "${pids[@]}"
echo "$(date -u +%T) $h done"
