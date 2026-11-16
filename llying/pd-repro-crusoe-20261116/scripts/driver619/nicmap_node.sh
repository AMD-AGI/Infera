#!/bin/bash
# Which ionic NIC is close to a GPU inside the VM: pinned host memory (initiator) -> GPU G memory (target, GEM) through
# NIC n, for every n; then the same with GPU memory as the source (GPU S -> GPU G over n). One sweep each (12 GiB target).
# Usage on the node: nicmap_node.sh OUT_DIR [TARGET_GPU] [SRC_GPU]
OUT=$1 G=${2:-4} S=${3:-0}; R=/home/liyingli/bench_agentx/pd-repro-20261116; K=$R/driver619; SO=$K/gemalloc/build/libhip_gem_alloc.so
IMAGE=${IMAGE:-infera-sglang:v0522-fdbafb60}; mkdir -p "$OUT"
lib=(); [ -e /lib/x86_64-linux-gnu/libionic.so ] && lib=(-v /lib/x86_64-linux-gnu/libionic.so:/host-libionic/libionic.so:ro)
run() {  # name gpu pool args...
    local name=$1 g=$2 pool=$3; shift 3; local pre=()
    [ "$pool" = gem ] && pre=(-e LD_PRELOAD=$SO -e SGLANG_AMD_GEM_ALLOCATOR=$SO)
    docker rm -f $name >/dev/null 2>&1
    timeout 300 docker run --rm --name $name --network host --ipc host --device /dev/kfd --device /dev/dri --device /dev/infiniband \
        --group-add video --group-add render --cap-add IPC_LOCK --security-opt seccomp=unconfined --ulimit memlock=-1:-1 "${lib[@]}" \
        -v $R:$R -v /tmp:/tmp -e ROCR_VISIBLE_DEVICES=$g -e RDMAV_FORK_SAFE=1 -e MC_GID_INDEX=1 -e MC_MAX_MR_SIZE=1073741824 \
        -e MC_IB_TIMEOUT=18 -e MC_DISABLE_HIP=1 -e MOONCAKE_DISABLE_HIP_DMABUF=0 -e MC_HIP_KFD_PIN_FIX=0 -e MC_BUF_GIB=6 "${pre[@]}" \
        $IMAGE python3 $K/mc_loop_sweep.py --pool $pool "$@"
}
for src in host gpu; do
    for n in 0 1 2 3 4 5 6 7; do
        sig=$(mktemp -d /tmp/llying-pdr-nic.XXXX); tag=$src-g$G-ionic_$n
        run llying-pdr-nic-t $G gem --role target --gib 12 --nic ionic_$n --sig $sig --tag $tag > "$OUT/$tag.target.log" 2>&1 &
        tp=$!
        if [ $src = host ]; then ip=host; else ip=gem; fi
        run llying-pdr-nic-i $S $ip --role initiator --nic ionic_$n --sig $sig --tag $tag > "$OUT/$tag.init.log" 2>&1
        wait $tp; rm -rf $sig
        echo "$tag: $(grep -a '"step": "sweep"' "$OUT/$tag.init.log" | sed -E 's/.*"big_gbs": ([0-9.]+).*"small_gbs": ([0-9.]+).*/big \1 GB\/s small \2 GB\/s/' | tr '\n' ';') $(grep -a 'reg_rc": -1\|error' "$OUT/$tag.init.log" | head -1 | cut -c1-100)"
    done
done | tee "$OUT/summary.txt"
