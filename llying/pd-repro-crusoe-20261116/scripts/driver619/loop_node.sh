#!/bin/bash
# mc_loop_sweep.py on this node, in the production image, with the deployment's Mooncake settings (MC_DISABLE_HIP=1,
# dma-buf registration, 1 GiB MRs, IB timeout 18, the destination GPU's NIC on both ends).
#   phase 1  sweep, one initiator per target, KFD (GPU 0 -> GPU 4, ionic_4) and GEM (GPU 1 -> GPU 5, ionic_5) at once
#   phase 2  load, 4 initiators (GPU 0-3) into one target (GPU 4, ionic_4) for 40 s: GEM, then KFD
# Usage on the node: loop_node.sh OUT_DIR [TARGET_GIB]
OUT=$1 GIB=${2:-120}
R=/home/liyingli/bench_agentx/pd-repro-20261116
K=$R/driver619
SO=$K/gemalloc/build/libhip_gem_alloc.so
IMAGE=${IMAGE:-infera-sglang:v0522-fdbafb60}
mkdir -p "$OUT"; h=$(hostname -s)
for i in $(seq 90); do
    [ -z "$(docker ps -q --filter name=glm52_2p1d-)" ] && rocm-smi --showmemuse 2>/dev/null | awk '/VRAM%/ {if ($NF + 0 > 2) b = 1} END {exit b}' && break
    sleep 20
done
echo "$(date -u +%T) $h start; containers: $(docker ps --format '{{.Names}}' | tr '\n' ' ')" | tee "$OUT/$h.node"
lib=()  # the image's own libionic: the host one needs GLIBC_2.38 (crusoe v3) and leaves no ionic device
run() {  # name gpu pool args...
    local name=$1 g=$2 pool=$3; shift 3
    local pre=(); [ "$pool" = gem ] && pre=(-e LD_PRELOAD=$SO -e SGLANG_AMD_GEM_ALLOCATOR=$SO)
    [ "$pool" = kfd ] && [ -n "${IPCFAKE:-}" ] && pre=(-e LD_PRELOAD=$K/ipcfake/libipcfake.so)
    docker rm -f $name >/dev/null 2>&1
    timeout 1500 docker run --rm --name $name --network host --ipc host \
        --device /dev/kfd --device /dev/dri --device /dev/infiniband --group-add video --group-add render \
        --cap-add IPC_LOCK --security-opt seccomp=unconfined --ulimit memlock=-1:-1 "${lib[@]}" -v $R:$R -v /tmp:/tmp \
        -e ROCR_VISIBLE_DEVICES=$g -e RDMAV_FORK_SAFE=1 -e MC_GID_INDEX=1 -e MC_MAX_MR_SIZE=1073741824 -e MC_IB_TIMEOUT=18 \
        -e MC_DISABLE_HIP=1 -e MOONCAKE_DISABLE_HIP_DMABUF=0 -e MC_HIP_KFD_PIN_FIX=0 -e MC_BUF_GIB=${BUF_GIB:-6} "${pre[@]}" \
        $IMAGE python3 $K/mc_loop_sweep.py --pool $pool "$@"
}
pair() {  # tag pool target_gpu nic initiators(gpus,comma) sweep load
    local tag=$1 pool=$2 tg=$3 nic=$4 ig=$5 sweep=$6 load=$7 sig; sig=$(mktemp -d /tmp/llying-pdr-loop.XXXX)
    IFS=, read -ra gs <<<"$ig"
    run llying-pdr-loop-$tag-t $tg $pool --role target --gib $GIB --nic $nic --sig $sig --initiators ${#gs[@]} --tag $tag \
        > "$OUT/$tag.target.log" 2>&1 &
    local p=($!) k=0
    for g in "${gs[@]}"; do
        run llying-pdr-loop-$tag-i$k $g $pool --role initiator --nic $nic --sig $sig --id $k --initiators ${#gs[@]} \
            --sweep $sweep --load $load --tag $tag > "$OUT/$tag.init$k.log" 2>&1 &
        p+=($!); k=$((k + 1))
    done
    wait "${p[@]}"; rm -rf $sig
}
pair sweep-kfd kfd 4 ionic_4 0 1 0 &
pair sweep-gem gem 5 ionic_5 1 1 0 &
wait
pair load-gem gem 4 ionic_4 0,1,2,3 0 40
pair load-kfd kfd 4 ionic_4 0,1,2,3 0 40
docker ps -a --format '{{.Names}}' | grep '^llying-pdr-loop' | xargs -r docker rm -f >/dev/null 2>&1
for f in "$OUT"/*.log; do echo "== $(basename $f)"; grep -a -E '^\{|Error|error|Traceback|fatal|Segmentation' "$f" | grep -av "Ignore import" | cut -c1-240 | tail -24; done | tee "$OUT/$h.summary"
echo "$(date -u +%T) $h done"
